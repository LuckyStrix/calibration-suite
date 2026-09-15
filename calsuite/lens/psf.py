"""PSF / coma / astigmatism: star or pinhole blob detection, second-moment
FWHM (major/minor), ellipticity, orientation relative to the radial
direction (sagittal vs meridional), and third-moment coma asymmetry.
Field maps across apertures are built by commands.py calling ``psf_field``
once per aperture's frame and collecting the per-star results.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from calsuite.fit import Analysis
from calsuite.lens.constants import (
    PSF_BACKGROUND_ANNULUS_FACTOR,
    PSF_MIN_SNR,
    PSF_MOMENT_WINDOW_ITERATIONS,
    PSF_MOMENT_WINDOW_MIN_PX,
    PSF_MOMENT_WINDOW_SIGMAS,
    PSF_SATURATION_FRACTION,
    PSF_WINDOW_RADIUS_PX,
)

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))  # ~2.3548, the standard Gaussian FWHM/sigma ratio


def find_blobs(plane: np.ndarray, *, min_snr: float = PSF_MIN_SNR) -> list:
    """``[(x, y), ...]`` centroid seeds for every connected region whose
    peak is at least ``min_snr`` robust-sigma above the local background.
    Background/noise are estimated from a large-scale median filter so a
    bright star doesn't bias its own neighborhood's noise estimate."""
    background = ndimage.median_filter(plane, size=25)
    residual = plane - background
    std = float(np.std(residual))
    if std == 0:
        return []
    mask = residual > (min_snr * std)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    centroids = ndimage.center_of_mass(np.clip(residual, 0, None), labeled, range(1, n + 1))
    return [(float(c[1]), float(c[0])) for c in centroids]  # (x, y)


def is_saturated(plane: np.ndarray, x0: float, y0: float, *, window: int = PSF_WINDOW_RADIUS_PX,
                  saturation_dn: float | None = None) -> bool:
    """True if the blob nearest ``(x0, y0)``'s window reaches
    ``saturation_dn * PSF_SATURATION_FRACTION`` -- see lens/constants.py's
    ``PSF_SATURATION_FRACTION`` docstring for why a clipped core biases the
    second-moment FWHM upward rather than just losing peak signal. A
    ``None`` (unknown) ``saturation_dn`` never flags anything -- callers
    that don't know the sensor's white level (e.g. a plain synthetic
    array in a test) get the old, unchecked behavior."""
    if saturation_dn is None:
        return False
    h, w = plane.shape
    xi0, xi1 = max(int(x0 - window), 0), min(int(x0 + window + 1), w)
    yi0, yi1 = max(int(y0 - window), 0), min(int(y0 + window + 1), h)
    sub = plane[yi0:yi1, xi0:xi1]
    if sub.size == 0:
        return False
    return float(sub.max()) >= saturation_dn * PSF_SATURATION_FRACTION


def _window_moments(plane: np.ndarray, cx: float, cy: float, window: int) -> tuple | None:
    """Raw moment sums in one square window, with the background estimated
    from an annulus *outside* it (falling back to the window's own median
    only when the annulus is empty). Returns
    ``(cx, cy, mxx, myy, mxy, xx, yy, weights, total)``."""
    h, w = plane.shape
    xi0, xi1 = max(int(cx - window), 0), min(int(cx + window + 1), w)
    yi0, yi1 = max(int(cy - window), 0), min(int(cy + window + 1), h)
    sub = plane[yi0:yi1, xi0:xi1].astype(np.float64)
    if sub.size == 0:
        return None

    outer = int(round(PSF_BACKGROUND_ANNULUS_FACTOR * window))
    axi0, axi1 = max(int(cx - outer), 0), min(int(cx + outer + 1), w)
    ayi0, ayi1 = max(int(cy - outer), 0), min(int(cy + outer + 1), h)
    ring = plane[ayi0:ayi1, axi0:axi1].astype(np.float64)
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[yi0 - ayi0 : yi1 - ayi0, xi0 - axi0 : xi1 - axi0] = False
    background = float(np.median(ring[ring_mask])) if ring_mask.any() else float(np.median(sub))

    yy, xx = np.mgrid[yi0:yi1, xi0:xi1].astype(np.float64)
    weights = np.clip(sub - background, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return None
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    dx, dy = xx - cx, yy - cy
    mxx = float((weights * dx * dx).sum() / total)
    myy = float((weights * dy * dy).sum() / total)
    mxy = float((weights * dx * dy).sum() / total)
    return cx, cy, mxx, myy, mxy, xx, yy, weights, total


def moments(plane: np.ndarray, x0: float, y0: float, *, window: int = PSF_WINDOW_RADIUS_PX) -> dict | None:
    """Intensity-weighted second/third moments of the blob nearest
    ``(x0, y0)``. Returns ``None`` if the background-subtracted flux is
    non-positive (nothing to measure). Saturation is checked separately
    (``is_saturated``, called by ``psf_field`` before this) since a clipped
    blob still has positive flux -- it's biased, not absent.

    The window is *iterated*: a first pass over the full ``window``, then a
    re-measure in a window sized to the blob's own measured sigma
    (``PSF_MOMENT_WINDOW_SIGMAS``), with the background from an annulus
    outside it. A single fixed window biases every blob smaller than
    itself: the sums are r^2-weighted, so noise in the far corners of a
    15 px window carries ~56x the leverage it has at the core of a 2 px
    blob, and the window's own median over-estimates the background of a
    blob that fills it. See PSF_MOMENT_WINDOW_SIGMAS for the measured
    error (a truth-2.0/1.0 star read 2.05/1.13, i.e. 13% too round).

    The returned ``window_px`` is the window the reported moments came
    from, and ``window_truncated`` says whether the blob is wide enough
    that even the full ``window`` clips its wings -- in which case the
    reported sigmas are biased low.
    """
    result = _window_moments(plane, x0, y0, window)
    if result is None:
        return None
    used_window = window
    for _ in range(PSF_MOMENT_WINDOW_ITERATIONS):
        sigma_major_est = math.sqrt(max(max(result[2], result[3]), 0.0))
        next_window = int(np.clip(round(PSF_MOMENT_WINDOW_SIGMAS * sigma_major_est), PSF_MOMENT_WINDOW_MIN_PX, window))
        if next_window == used_window:
            break
        refined = _window_moments(plane, result[0], result[1], next_window)
        if refined is None:
            break
        result, used_window = refined, next_window

    cx, cy, mxx, myy, mxy, xx, yy, weights, total = result
    dx, dy = xx - cx, yy - cy

    cov = np.array([[mxx, mxy], [mxy, myy]])
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    sigma_minor = math.sqrt(max(eigvals[0], 0.0))
    sigma_major = math.sqrt(max(eigvals[1], 0.0))
    major_vec = eigvecs[:, 1]  # (vx, vy), since cov was built in (x, y) order
    orientation_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))

    proj_major = dx * major_vec[0] + dy * major_vec[1]
    m3_major = float((weights * proj_major**3).sum() / total)
    coma = m3_major / sigma_major**3 if sigma_major > 0 else 0.0

    ellipticity = 1.0 - (sigma_minor / sigma_major) if sigma_major > 0 else 0.0

    return {
        "x": cx,
        "y": cy,
        "window_px": used_window,
        "window_truncated": bool(sigma_major * PSF_MOMENT_WINDOW_SIGMAS > window),
        "sigma_major": sigma_major,
        "sigma_minor": sigma_minor,
        "fwhm_major": sigma_major * FWHM_PER_SIGMA,
        "fwhm_minor": sigma_minor * FWHM_PER_SIGMA,
        "ellipticity": ellipticity,
        "orientation_deg": orientation_deg,
        "coma": coma,
        "total_flux": total,
    }


def classify_orientation(orientation_deg: float, x: float, y: float, center: tuple) -> dict:
    """Sagittal (major axis aligned with the radial direction from image
    center) vs meridional (aligned tangentially) -- the standard
    astro-optics distinction for off-axis blur. Both the PSF's orientation
    and the radial direction have a 180-degree ambiguity (an axis, not a
    ray), so the comparison is taken mod 180 and folded to [0, 90]."""
    radial_deg = float(np.degrees(np.arctan2(y - center[1], x - center[0])))
    diff = abs((orientation_deg - radial_deg) % 180.0)
    diff = min(diff, 180.0 - diff)
    return {
        "radial_angle_deg": radial_deg,
        "angle_from_radial_deg": diff,
        "orientation": "sagittal" if diff < 45.0 else "meridional",
    }


def psf_field(
    plane: np.ndarray,
    *,
    center: tuple | None = None,
    min_snr: float = PSF_MIN_SNR,
    saturation_dn: float | None = None,
) -> Analysis:
    """Detect every star/pinhole blob in ``plane`` and report its moments
    plus its sagittal/meridional classification relative to ``center``
    (defaults to the plane's own geometric center). ``saturation_dn``, when
    given (the frame's own ``raw.RawFrame.white_level``), excludes any blob
    whose core is clipped -- its FWHM/ellipticity would be biased, not
    absent, so it's dropped from ``stars`` rather than measured wrong; see
    lens/constants.py's ``PSF_SATURATION_FRACTION``. The record only
    refuses outright if *every* detected blob turned out saturated (the
    same "keep what's usable, refuse only if nothing is left" shape as
    ``mtf.mtf_field_grid``'s per-cell refusals)."""
    a = Analysis()
    h, w = plane.shape
    center = center or ((w - 1) / 2.0, (h - 1) / 2.0)

    seeds = find_blobs(plane, min_snr=min_snr)
    if not seeds:
        a.refuse("no_blobs_detected", f"no blob exceeded {min_snr} sigma above background", 0, min_snr)
        return a

    stars = []
    n_saturated = 0
    for x0, y0 in seeds:
        if is_saturated(plane, x0, y0, saturation_dn=saturation_dn):
            n_saturated += 1
            continue
        m = moments(plane, x0, y0)
        if m is None:
            continue
        m.update(classify_orientation(m["orientation_deg"], m["x"], m["y"], center))
        stars.append(m)

    if not stars:
        if n_saturated and n_saturated == len(seeds):
            a.refuse(
                "all_blobs_saturated",
                f"all {n_saturated} detected blob(s) had a clipped (saturated) core -- FWHM/ellipticity from a "
                "clipped PSF core is biased, not just noisy",
                n_saturated,
                0,
            )
        else:
            a.refuse("no_usable_blobs", "every detected blob had non-positive background-subtracted flux", 0, 1)
        return a

    a.result = {"stars": stars, "center": list(center), "n_stars": len(stars), "n_saturated_excluded": n_saturated}
    return a

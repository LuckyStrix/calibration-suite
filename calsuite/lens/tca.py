"""Lateral chromatic aberration: R and B radial scale vs G, from per-CFA-
plane ChArUco corner detections (``charuco.detect_all_planes``), fit to
lensfun's ``tca`` element.

lensfun's TCA models share distortion's r=1-at-half-short-edge convention
(see lens/constants.py's citation block) and, per the ``poly3`` formula
confirmed there (``Rd = Ru*(br*Ru^2 + cr*Ru + vr)``), a poly3 element with
only ``vr``/``vb`` set (cr=cb=br=bb=0, lensfun's own default) is
*identical* to the ``linear`` model. The locally-installed
``mil-canon.xml``'s own "Canon RF 50mm F1.8 STM" entry does exactly this
(``<tca model="poly3" ... vr="0.9998424" vb="1.0000271"/>``, no c/b
attributes), so this module follows the same convention: it always fits
the plain radial scale (equivalent to "linear" kr/kb) and reports it under
the poly3 element's vr/vb, for direct comparability with the vendor entry.
"""

from __future__ import annotations

import math

import numpy as np

from calsuite.fit import Analysis

TCA_MIN_MATCHED_CORNERS = 12
# Below this many ids shared across R/G/B detections in one view, a radial
# scale fit is too noisy to trust (few points, and any one mis-detected
# corner dominates a least-squares slope) -- comfortably below the ~45
# corners a full-coverage view of the 10x6 board can offer, but well above
# "a handful".


def _radii(corners: np.ndarray, center: tuple) -> np.ndarray:
    cx, cy = center
    return np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)


def _slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y = k*x with no intercept -- the closed form
    for fitting Rd = Ru * k (lensfun's "linear" TCA model)."""
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom else 1.0


def fit_tca(detections_per_view: list, image_size: tuple) -> Analysis:
    """``detections_per_view``: one dict per captured view, each
    ``{"R": charuco.Detection|None, "G1":..., "G2":..., "B":...}``
    (``charuco.detect_all_planes``'s return shape). Matches R/G/B corners
    by shared id within each view (a corner not seen on all three of
    R/green/B in that view contributes nothing -- TCA is a *difference*
    between channels, so a partial detection is not usable for it), pools
    every matched pair across all views, and fits a single radial scale
    per channel about the image center."""
    a = Analysis()
    width, height = image_size
    center = ((width - 1) / 2.0, (height - 1) / 2.0)

    r_g_all, r_r_all, r_b_all = [], [], []
    for view in detections_per_view:
        det_r, det_g1, det_g2, det_b = view.get("R"), view.get("G1"), view.get("G2"), view.get("B")
        if det_r is None or det_b is None or (det_g1 is None and det_g2 is None):
            continue
        det_g = det_g1 if det_g2 is None else det_g2 if det_g1 is None else det_g1
        common = set(det_r.ids.tolist()) & set(det_g.ids.tolist()) & set(det_b.ids.tolist())
        if not common:
            continue
        common = sorted(common)
        idx_r = {i: k for k, i in enumerate(det_r.ids.tolist())}
        idx_g = {i: k for k, i in enumerate(det_g.ids.tolist())}
        idx_b = {i: k for k, i in enumerate(det_b.ids.tolist())}
        r_corners = det_r.corners[[idx_r[i] for i in common]]
        g_corners = det_g.corners[[idx_g[i] for i in common]]
        b_corners = det_b.corners[[idx_b[i] for i in common]]
        r_g_all.append(_radii(g_corners, center))
        r_r_all.append(_radii(r_corners, center))
        r_b_all.append(_radii(b_corners, center))

    if not r_g_all:
        a.refuse("no_matched_corners", "no view had a corner detected on R, G and B alike", value=0, threshold=1)
        return a

    r_g = np.concatenate(r_g_all)
    r_r = np.concatenate(r_r_all)
    r_b = np.concatenate(r_b_all)

    if r_g.size < TCA_MIN_MATCHED_CORNERS:
        a.refuse(
            "too_few_matched_corners",
            f"only {r_g.size} R/G/B-matched corners across all views, need >= {TCA_MIN_MATCHED_CORNERS}",
            value=int(r_g.size),
            threshold=TCA_MIN_MATCHED_CORNERS,
        )
        return a

    kr = _slope_through_origin(r_g, r_r)
    kb = _slope_through_origin(r_g, r_b)
    resid_r = float(np.sqrt(np.mean((kr * r_g - r_r) ** 2)))
    resid_b = float(np.sqrt(np.mean((kb * r_g - r_b) ** 2)))

    a.result = {
        "model": "poly3",
        "vr": kr,
        "vb": kb,
        "cr": 0.0,
        "cb": 0.0,
        "br": 0.0,
        "bb": 0.0,
        "kr": kr,
        "kb": kb,
        "n_matched_corners": int(r_g.size),
        "n_views": len(r_g_all),
    }
    a.residuals = {"rms_px_r": resid_r, "rms_px_b": resid_b}
    return a


def scale_at_radius(kr: float, kb: float, radius_px: float) -> tuple:
    """Convenience for report.py: how far R and B sit from G, in px, at a
    given radius -- ``radius_px * (k - 1)``."""
    return radius_px * (kr - 1.0), radius_px * (kb - 1.0)


def image_center(image_size: tuple) -> tuple:
    width, height = image_size
    return (width - 1) / 2.0, (height - 1) / 2.0


def half_diagonal(image_size: tuple) -> float:
    width, height = image_size
    return math.hypot(width, height) / 2.0

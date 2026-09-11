"""Vignetting and flat fields, including the self-calibrating flat
(docs/design.md §4.3): the source is never perfectly uniform, so a set of
poses (rotations, plus small shifts) is shot and the lens's vignetting
``V`` and the source's own non-uniformity ``S`` are solved for *jointly*,
in the log domain, rather than assuming the source is uniform.

Model: ``observed_i(x, y) = V(x, y) * S(pose_i(x, y))``, i.e. vignetting is
fixed to the sensor (every pose sees the same ``V``) while the source's
pattern ``S`` is fixed in the world and appears transformed by each pose's
inverse in the *camera's* frame. In the log domain this is additive and
linear in ``V``'s and ``S``'s polynomial coefficients, so one joint
least-squares solve over every pose recovers both -- ``S`` is then a
display/source-uniformity map, exactly as the design doc's "by-product"
describes.

**Degeneracy refusal**: with rotation-only poses (no shift), a radially
symmetric term in ``S`` is *exactly* linearly dependent on ``V``'s own
radial term for every sample (see ``self_calibrate_flat``'s docstring for
the algebra) -- caught here via the design matrix's condition number, not
by inspecting the fitted coefficients after the fact (a tight residual on
a degenerate dataset is exactly the failure mode house rule 3 warns
against).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from calsuite.fit import Analysis
from calsuite.lens.constants import FLAT_COND_THRESHOLD, FLAT_GRID_SAMPLES, FLAT_V_RADIAL_ORDER

assert FLAT_V_RADIAL_ORDER == 2, "self_calibrate_flat's basis is hard-coded to the 2-term (r^2, r^4) case"

N_V_TERMS = 2  # v2, v4 -- exp(v2*r^2 + v4*r^4), r normalized to 1 at the image corner (see fit_pa)
N_S_TERMS = 6  # s0, su, sv, suu, suv, svv -- a full 2nd-order 2D polynomial


@dataclass(frozen=True)
class Pose:
    """One flat-field pose: camera rotated ``angle_deg`` about the image
    center (0/90/180/270 for the design doc's rotation set) plus a small
    world-frame shift, in the same normalized (image-corner = 1) units as
    the coordinate grid below."""

    angle_deg: float
    shift_u: float = 0.0
    shift_v: float = 0.0


def normalize_coords(x, y, shape: tuple):
    """Sensor pixel (x, y) -> centered coordinates normalized so the image
    corner is at radius 1 -- the same convention lens/constants.py cites
    for lensfun's "pa" vignetting model, chosen here so ``fit_pa`` needs no
    further rescaling."""
    rows, cols = shape
    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
    half_diag = math.hypot(cols, rows) / 2.0
    return (x - cx) / half_diag, (y - cy) / half_diag


def pose_transform(xn, yn, pose: Pose):
    """Sensor-frame normalized coords -> the world/source frame as seen
    through ``pose``: rotate by ``pose.angle_deg`` (camera rotation ->
    the world appears rotated the same way in the captured frame) then
    subtract the pose's shift. Used identically by the solver here and by
    ``synth.lens``'s renderer, so both sides of the round trip agree on
    what a "pose" does."""
    theta = math.radians(pose.angle_deg)
    ca, sa = math.cos(theta), math.sin(theta)
    u = ca * xn + sa * yn - pose.shift_u
    v = -sa * xn + ca * yn - pose.shift_v
    return u, v


def _design_row(xn, yn, pose: Pose) -> np.ndarray:
    r2 = xn * xn + yn * yn
    u, v = pose_transform(xn, yn, pose)
    return np.stack([r2, r2 * r2, np.ones_like(xn), u, v, u * u, u * v, v * v], axis=-1)


def sample_grid(shape: tuple, n: int = FLAT_GRID_SAMPLES):
    """``n x n`` evenly spaced (x, y) sensor-pixel sample points -- kept
    small and shared by every pose so the joint design matrix stays a few
    thousand rows regardless of the actual image resolution."""
    rows, cols = shape
    ys = np.linspace(0, rows - 1, n)
    xs = np.linspace(0, cols - 1, n)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return xx.ravel(), yy.ravel()


def _bilinear_sample(image: np.ndarray, xs, ys) -> np.ndarray:
    return ndimage.map_coordinates(image, [ys, xs], order=1, mode="nearest")


def self_calibrate_flat(images: list, poses: list) -> Analysis:
    """Jointly solve for vignetting ``V`` and source non-uniformity ``S``
    from ``images`` (a list of 2D arrays -- one CFA plane per pose, already
    black-subtracted and positive) and their matching ``poses``.

    **Why rotation-only is degenerate**: for a pure rotation (no shift),
    ``pose_transform`` is an orthonormal rotation, so ``u^2 + v^2 == xn^2 +
    yn^2`` identically, *for every angle*. That means the column
    combination ``1*[r^2 term] - 1*[u^2 term] - 1*[v^2 term]`` is exactly
    zero on every sample row the design matrix ever sees -- an exact null
    vector, i.e. a singular (here: ill-conditioned to the numerical
    precision of the check) design matrix, regardless of how many
    rotation-only poses are added. Any nonzero shift breaks the identity
    (the shift introduces a term linear in the *rotated* coordinates that
    varies with the rotation angle, which neither basis can reproduce on
    its own), which is why the design doc calls for shifts alongside the
    rotations.
    """
    a = Analysis()
    if len(images) != len(poses):
        raise ValueError(f"{len(images)} images but {len(poses)} poses")
    if not images:
        a.refuse("no_data", "no pose images given", value=0, threshold=1)
        return a

    shape = images[0].shape
    xs, ys = sample_grid(shape)
    xn, yn = normalize_coords(xs, ys, shape)

    rows, targets = [], []
    for image, pose in zip(images, poses, strict=True):
        vals = np.clip(_bilinear_sample(np.asarray(image, dtype=np.float64), xs, ys), 1e-9, None)
        rows.append(_design_row(xn, yn, pose))
        targets.append(np.log(vals))

    design = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    cond = float(np.linalg.cond(design))

    if not np.isfinite(cond) or cond > FLAT_COND_THRESHOLD:
        a.refuse(
            "degenerate_poses",
            "the pose set cannot separate vignetting from source non-uniformity (design matrix "
            f"condition number {cond:.3g} exceeds {FLAT_COND_THRESHOLD:.0e}) -- add poses with a "
            "shift, not just rotations",
            value=cond if np.isfinite(cond) else None,
            threshold=FLAT_COND_THRESHOLD,
        )
        return a

    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    v_coeffs = [float(c) for c in coeffs[:N_V_TERMS]]
    s_coeffs = [float(c) for c in coeffs[N_V_TERMS : N_V_TERMS + N_S_TERMS]]
    resid = float(np.sqrt(np.mean((design @ coeffs - target) ** 2)))

    a.result = {
        "v_coeffs": v_coeffs,  # [v2, v4]: V(r) = exp(v2*r^2 + v4*r^4), r=1 at the image corner
        "s_coeffs": s_coeffs,  # [s0, su, sv, suu, suv, svv]
        "image_shape": [int(shape[0]), int(shape[1])],
        "n_poses": len(images),
        "condition_number": cond,
        "poses": [{"angle_deg": p.angle_deg, "shift_u": p.shift_u, "shift_v": p.shift_v} for p in poses],
    }
    a.residuals = {"log_domain_rms": resid}
    return a


def evaluate_v_map(v_coeffs: list, shape: tuple) -> np.ndarray:
    """Full-resolution vignetting map from fitted ``v_coeffs`` -- used both
    for the ``.npz`` artifact (design doc: "full flat-field maps are kept")
    and by report.py's heatmap."""
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    xn, yn = normalize_coords(xx, yy, shape)
    r2 = xn * xn + yn * yn
    v2, v4 = v_coeffs
    return np.exp(v2 * r2 + v4 * r2 * r2)


def evaluate_s_map(s_coeffs: list, shape: tuple, pose: Pose) -> np.ndarray:
    """Full-resolution source map as it would appear *in this pose's
    image* (i.e. already run through ``pose_transform``) -- the
    display-uniformity by-product, at whichever pose the caller wants it
    referenced to."""
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    xn, yn = normalize_coords(xx, yy, shape)
    u, v = pose_transform(xn, yn, pose)
    s0, su, sv, suu, suv, svv = s_coeffs
    return np.exp(s0 + su * u + sv * v + suu * u * u + suv * u * v + svv * v * v)


def fit_pa(v_coeffs: list, *, r_max: float = 1.2, n: int = 200) -> dict:
    """Refit our log-domain ``V(r) = exp(v2*r^2 + v4*r^4)`` to lensfun's
    ``pa`` model, ``V(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6`` (see
    lens/constants.py's citation of ``mod-color.cpp``'s
    ``ModifyColor_Vignetting_PA``) -- both already share the same r=1-at-
    the-corner normalization, so this is a plain resample-and-linear-least-
    -squares refit, the same move as distortion.py's ptlens/poly3 refit.
    """
    r = np.linspace(0.0, r_max, n)
    v2, v4 = v_coeffs
    v_of_r = np.exp(v2 * r * r + v4 * r**4)
    target = v_of_r - 1.0
    design = np.stack([r**2, r**4, r**6], axis=1)
    (k1, k2, k3), *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = float(np.sqrt(np.mean((design @ [k1, k2, k3] - target) ** 2)))
    return {"k1": float(k1), "k2": float(k2), "k3": float(k3), "residual_rms": resid}


def relative_t_stop(known_t_stop: float, known_signal: float, other_signal: float) -> float:
    """docs/design.md §4.6: the relative T-stop of a second lens, from the
    ratio of mean signal each lens delivers under the *same* exposure time,
    ISO and target -- T-stop is defined so scene-referred signal falls off
    as ``1/T^2`` (the same square-law f-number itself follows for a
    lossless lens), so ``T_other = T_known * sqrt(known_signal /
    other_signal)``: a lens passing less light (lower signal) has a
    *higher* T-stop for the same f-number."""
    return known_t_stop * math.sqrt(known_signal / other_signal)

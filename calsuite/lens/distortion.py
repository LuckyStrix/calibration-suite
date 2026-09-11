"""Geometric distortion: Brown-Conrady (k1,k2,k3,p1,p2) via
``cv2.calibrateCamera`` over many ChArUco views, with per-image
reprojection RMS, visible-outlier dropping, a coverage refusal, and a
refit to lensfun's ``ptlens``/``poly3`` export models.

Pure analysis (house rule 5): everything here takes corner detections
(``BoardView``) and a board/image size, and returns a ``fit.Analysis``.
Capture, file I/O and the Store live in ``lens/commands.py``.

See ``lens/constants.py`` for the lensfun radius-normalization and
direction citations this module's ``refit_ptlens_poly3`` depends on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from calsuite.fit import Analysis, Refusal
from calsuite.lens import charuco
from calsuite.lens.constants import (
    COVERAGE_ANGULAR_BINS,
    COVERAGE_OUTER_FRACTION,
    COVERAGE_RADIAL_BINS,
    DISTORTION_FIT_MAX_RU,
    DISTORTION_MIN_CORNERS_PER_VIEW,
    DISTORTION_MIN_VIEWS,
    REPROJECTION_OUTLIER_FACTOR,
)


@dataclass(frozen=True)
class BoardView:
    """One captured (or synthetic) ChArUco detection ready for
    calibration: sensor-pixel corners plus the ids they correspond to
    (``charuco.detect_green`` produces exactly this shape)."""

    corners: np.ndarray  # (N, 2) float, sensor px
    ids: np.ndarray  # (N,)
    name: str = ""


def _reprojection_rms(objp, imgp, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = proj.reshape(-1, 2) - imgp.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def coverage_grid(views: list, image_size: tuple) -> np.ndarray:
    """``(COVERAGE_RADIAL_BINS, COVERAGE_ANGULAR_BINS)`` corner-count grid
    in polar coordinates about the image center, radius normalized to
    ``[0, 1]`` by the half-diagonal. Used both for the coverage refusal
    (only the outermost radial ring, i.e. the outer
    ``COVERAGE_OUTER_FRACTION`` of the field, matters for that) and for
    report.py's coverage heatmap."""
    width, height = image_size
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    half_diag = math.hypot(width, height) / 2.0
    pts = np.concatenate([v.corners for v in views], axis=0) if views else np.zeros((0, 2))
    grid = np.zeros((COVERAGE_RADIAL_BINS, COVERAGE_ANGULAR_BINS), dtype=np.int64)
    if pts.shape[0] == 0:
        return grid
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / half_diag
    theta = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    r_idx = np.clip((r * COVERAGE_RADIAL_BINS).astype(int), 0, COVERAGE_RADIAL_BINS - 1)
    theta_idx = np.clip(
        ((theta + math.pi) / (2 * math.pi) * COVERAGE_ANGULAR_BINS).astype(int), 0, COVERAGE_ANGULAR_BINS - 1
    )
    np.add.at(grid, (r_idx, theta_idx), 1)
    return grid


def _coverage_refusal(grid: np.ndarray):
    # COVERAGE_RADIAL_BINS spans r in [0, 1] in equal steps, so the last
    # `COVERAGE_OUTER_FRACTION` of that range is exactly the outermost
    # ceil(COVERAGE_RADIAL_BINS * COVERAGE_OUTER_FRACTION) rows -- with the
    # default 5 bins x 0.2 fraction that's the single outermost ring.
    n_outer_rows = max(1, round(COVERAGE_RADIAL_BINS * COVERAGE_OUTER_FRACTION))
    outer = grid[-n_outer_rows:, :]
    empty_sectors = int((outer.sum(axis=0) == 0).sum())
    if empty_sectors > 0:
        return Refusal(
            "coverage",
            f"{empty_sectors}/{COVERAGE_ANGULAR_BINS} angular sectors of the outer "
            f"{COVERAGE_OUTER_FRACTION * 100:.0f}% of the field have no detected corners",
            value=empty_sectors,
            threshold=0,
        )
    return None


def refit_ptlens_poly3(dist_coeffs: np.ndarray, camera_matrix: np.ndarray, image_size: tuple) -> dict:
    """Resample the OpenCV Brown-Conrady radial curve and refit it (plain
    linear least squares -- both lensfun forms are linear in their
    coefficients once written as ``Rd - Ru = ...``) to lensfun's
    ``ptlens`` (a, b, c) and ``poly3`` (k1) models, in lensfun's own
    Hugin-normalized radius (see lens/constants.py's citations for the
    r=1-at-half-short-edge convention and the Ru->Rd direction)."""
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    f_px = 0.5 * (fx + fy)
    width, height = image_size
    hugin_scale_px = min(width, height) / 2.0

    k1, k2, p1, p2, *rest = (list(np.asarray(dist_coeffs).ravel()) + [0.0, 0.0, 0.0, 0.0, 0.0])[:5]
    k3 = rest[0] if rest else 0.0

    ru_hugin = np.linspace(0.0, DISTORTION_FIT_MAX_RU, 400)
    ru_norm = ru_hugin * hugin_scale_px / f_px  # OpenCV's normalized-camera radius
    rd_norm = ru_norm * (1.0 + k1 * ru_norm**2 + k2 * ru_norm**4 + k3 * ru_norm**6)
    rd_hugin = rd_norm * f_px / hugin_scale_px

    target = rd_hugin - ru_hugin

    design_ptlens = np.stack([ru_hugin**4, ru_hugin**3, ru_hugin**2], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(design_ptlens, target, rcond=None)
    ptlens_resid = float(np.sqrt(np.mean((design_ptlens @ [a, b, c] - target) ** 2)))

    denom = float(np.sum(ru_hugin**6))
    k1_poly3 = float(np.sum(ru_hugin**3 * target) / denom) if denom else 0.0
    poly3_resid = float(np.sqrt(np.mean((k1_poly3 * ru_hugin**3 - target) ** 2)))

    return {
        "ptlens": {"a": float(a), "b": float(b), "c": float(c), "residual_hugin_rms": ptlens_resid},
        "poly3": {"k1": k1_poly3, "residual_hugin_rms": poly3_resid},
        "hugin_scale_px": hugin_scale_px,
        "f_px": f_px,
    }


def fit_distortion(board, views: list, image_size: tuple) -> Analysis:
    """Brown-Conrady fit over ``views`` (a list of ``BoardView``), with
    coverage and outlier-dropping refusals, refit to lensfun's export
    models. ``image_size`` is ``(width, height)`` in sensor pixels
    (``charuco.image_size_from_frame``)."""
    a = Analysis()

    usable = [v for v in views if len(v.ids) >= DISTORTION_MIN_CORNERS_PER_VIEW]
    if len(usable) < DISTORTION_MIN_VIEWS:
        a.refuse(
            "min_views",
            f"only {len(usable)} view(s) with >= {DISTORTION_MIN_CORNERS_PER_VIEW} corners, "
            f"need >= {DISTORTION_MIN_VIEWS}",
            value=len(usable),
            threshold=DISTORTION_MIN_VIEWS,
        )
        return a

    grid = coverage_grid(usable, image_size)
    coverage_refusal = _coverage_refusal(grid)

    objpoints = [charuco.object_points_for(board, v.ids).astype(np.float32) for v in usable]
    imgpoints = [v.corners.astype(np.float32) for v in usable]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    per_view_rms = [
        _reprojection_rms(op, ip, rv, tv, K, dist) for op, ip, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs, strict=True)
    ]
    median = float(np.median(per_view_rms)) if per_view_rms else 0.0
    keep = [i for i, e in enumerate(per_view_rms) if median == 0 or e <= REPROJECTION_OUTLIER_FACTOR * median]
    dropped_names = [usable[i].name or str(i) for i in range(len(usable)) if i not in keep]

    kept_views = usable
    if 0 < len(keep) < len(usable) and len(keep) >= DISTORTION_MIN_VIEWS:
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        per_view_rms = [
            _reprojection_rms(op, ip, rv, tv, K, dist)
            for op, ip, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs, strict=True)
        ]
        kept_views = [usable[i] for i in keep]

    refit = refit_ptlens_poly3(dist, K, image_size)

    a.result = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": [float(v) for v in np.asarray(dist).ravel()],
        "image_size": [int(image_size[0]), int(image_size[1])],
        "n_views_used": len(kept_views),
        "n_views_dropped": len(dropped_names),
        "dropped_views": dropped_names,
        "ptlens": refit["ptlens"],
        "poly3": refit["poly3"],
    }
    a.residuals = {
        "overall_rms_px": float(rms),
        "per_view_rms_px": [float(v) for v in per_view_rms],
        "per_view_names": [v.name for v in kept_views],
        "coverage_grid": grid.tolist(),
    }
    if coverage_refusal is not None:
        a.refusals.append(coverage_refusal)
    return a

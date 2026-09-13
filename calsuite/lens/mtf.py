"""Sharpness: slanted-edge e-SFR per ISO 12233. Per-row sub-pixel edge
location -> line fit (angle) -> 4x-oversampled ESF by projecting every
pixel onto its perpendicular distance from the fitted edge line ->
derivative (LSF) with a Hamming window -> FFT -> MTF normalized to DC.

Runs on a single CFA plane at a time (house rule 1): green and red/blue
MTF differ honestly (and that difference is itself informative --
longitudinal CA shows up as a focus-dependent MTF gap between channels).
"""

from __future__ import annotations

import numpy as np

from calsuite.fit import Analysis
from calsuite.lens.constants import (
    MTF_EDGE_MAX_ANGLE_DEG,
    MTF_EDGE_MIN_ANGLE_DEG,
    MTF_FIELD_GRID,
    MTF_MIN_CONTRAST,
    MTF_MIN_ROWS,
    MTF_OVERSAMPLE,
    MTF_SATURATION_FRACTION,
    R100_PIXEL_PITCH_MM,
)


def _row_edge_position(row: np.ndarray, midpoint: float) -> float | None:
    """Sub-pixel column where ``row`` crosses ``midpoint``, by linear
    interpolation between the two samples that bracket the first sign
    change -- ``None`` if the row never crosses it at all."""
    diff = row - midpoint
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if sign_changes.size == 0:
        return None
    i = int(sign_changes[0])
    y0, y1 = float(row[i]), float(row[i + 1])
    if y1 == y0:
        return float(i)
    frac = (midpoint - y0) / (y1 - y0)
    return float(i) + frac


def _fit_edge_line(roi: np.ndarray) -> tuple:
    """``(slope, intercept, positions, low, high)`` -- ``positions[r]`` is
    the fitted (not raw-detected) edge column for row ``r``, i.e.
    ``slope*r + intercept``."""
    low, high = float(np.percentile(roi, 5)), float(np.percentile(roi, 95))
    midpoint = 0.5 * (low + high)
    raw_positions = [_row_edge_position(roi[r, :], midpoint) for r in range(roi.shape[0])]
    rows_idx = np.array([r for r, p in enumerate(raw_positions) if p is not None], dtype=np.float64)
    cols_pos = np.array([p for p in raw_positions if p is not None], dtype=np.float64)
    if rows_idx.size < 2:
        return None, None, None, low, high
    slope, intercept = np.polyfit(rows_idx, cols_pos, 1)
    all_rows = np.arange(roi.shape[0], dtype=np.float64)
    return float(slope), float(intercept), slope * all_rows + intercept, low, high


def _oversampled_esf(roi: np.ndarray, positions: np.ndarray, oversample: int, window_px: int = 12) -> np.ndarray:
    rows, cols = roi.shape
    half_bins = window_px * oversample
    n_bins = 2 * half_bins + 1
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    col_idx = np.arange(cols, dtype=np.float64)
    for r in range(rows):
        dist = col_idx - positions[r]
        bin_idx = np.round(dist * oversample).astype(int) + half_bins
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        np.add.at(sums, bin_idx[valid], roi[r, valid])
        np.add.at(counts, bin_idx[valid], 1)
    esf = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
    nan_mask = np.isnan(esf)
    if nan_mask.any():
        idx = np.arange(n_bins)
        if nan_mask.all():
            return np.zeros(n_bins)
        esf[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], esf[~nan_mask])
    return esf


def _mtf50(freqs: np.ndarray, mtf: np.ndarray) -> float:
    below = np.where(mtf <= 0.5)[0]
    if below.size == 0:
        return float(freqs[-1])
    i = int(below[0])
    if i == 0:
        return float(freqs[0])
    f0, f1, m0, m1 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
    frac = (0.5 - m0) / (m1 - m0) if m1 != m0 else 0.0
    return float(f0 + frac * (f1 - f0))


def edge_sfr(roi: np.ndarray, *, oversample: int = MTF_OVERSAMPLE, saturation_dn: float | None = None) -> Analysis:
    """One ROI's e-SFR/MTF. ``roi`` is a single-plane 2D array (plane px,
    e.g. one cell of ``mtf_field_grid``'s grid) whose columns cross a
    single slanted edge in every row. ``saturation_dn``, when given (the
    frame's own ``raw.RawFrame.white_level``), refuses a ROI whose bright
    plateau reaches the sensor's saturation ceiling -- a clipped edge
    transition biases MTF50 *upward*, not down, so Michelson contrast alone
    (which a clipped ROI can still pass comfortably) doesn't catch it; see
    lens/constants.py's ``MTF_SATURATION_FRACTION`` for the numeric check
    this bit avoided.

    ``Analysis.result``: ``mtf50_cycles_per_plane_px``,
    ``mtf50_cycles_per_sensor_px`` (x0.5, since a plane pixel spans two
    sensor pixels), ``mtf50_lp_per_mm``, ``angle_deg``, ``contrast``.
    ``Analysis.residuals``: the full MTF curve (``freq_cycles_per_plane_px``,
    ``mtf``), for the report's chart.
    """
    a = Analysis()
    if roi.shape[0] < MTF_MIN_ROWS:
        a.refuse("too_few_rows", f"ROI has {roi.shape[0]} rows, need >= {MTF_MIN_ROWS}", roi.shape[0], MTF_MIN_ROWS)
        return a

    if saturation_dn is not None:
        roi_max = float(roi.max())
        ceiling = saturation_dn * MTF_SATURATION_FRACTION
        if roi_max >= ceiling:
            a.refuse(
                "saturated",
                f"ROI's brightest pixel ({roi_max:g}) reaches the sensor's saturation level "
                f"({saturation_dn:g}) -- a clipped edge transition biases MTF50, it doesn't just lose contrast",
                roi_max,
                ceiling,
            )
            return a

    slope, intercept, positions, low, high = _fit_edge_line(roi)
    if positions is None:
        a.refuse("no_edge_crossing", "no row in the ROI crosses the midpoint intensity", None, None)
        return a

    contrast = (high - low) / (high + low) if (high + low) else 0.0
    if contrast < MTF_MIN_CONTRAST:
        a.refuse("low_contrast", f"Michelson contrast {contrast:.3f} < {MTF_MIN_CONTRAST}", contrast, MTF_MIN_CONTRAST)
        return a

    angle_deg = float(np.degrees(np.arctan(slope)))
    if not (MTF_EDGE_MIN_ANGLE_DEG <= abs(angle_deg) <= MTF_EDGE_MAX_ANGLE_DEG):
        a.refuse(
            "edge_angle",
            f"edge angle {angle_deg:.2f} deg outside the usable [{MTF_EDGE_MIN_ANGLE_DEG}, "
            f"{MTF_EDGE_MAX_ANGLE_DEG}] deg range",
            angle_deg,
            [MTF_EDGE_MIN_ANGLE_DEG, MTF_EDGE_MAX_ANGLE_DEG],
        )
        return a

    esf = _oversampled_esf(roi, positions, oversample)
    lsf = np.diff(esf)
    lsf = lsf * np.hamming(lsf.size)
    spectrum = np.abs(np.fft.rfft(lsf))
    if spectrum[0] == 0:
        a.refuse("zero_dc", "LSF spectrum has zero DC component", 0.0, None)
        return a
    mtf = spectrum / spectrum[0]
    freqs = np.fft.rfftfreq(lsf.size, d=1.0 / oversample)  # cycles per plane-px directly

    mtf50_plane = _mtf50(freqs, mtf)
    mtf50_sensor = mtf50_plane * 0.5  # a plane px spans 2 sensor px (Wave 2B task prompt)
    lp_per_mm = mtf50_sensor / R100_PIXEL_PITCH_MM

    a.result = {
        "angle_deg": angle_deg,
        "contrast": float(contrast),
        "mtf50_cycles_per_plane_px": mtf50_plane,
        "mtf50_cycles_per_sensor_px": mtf50_sensor,
        "mtf50_lp_per_mm": lp_per_mm,
    }
    a.residuals = {
        "freq_cycles_per_plane_px": [float(f) for f in freqs],
        "mtf": [float(m) for m in mtf],
    }
    return a


def mtf_field_grid(
    plane: np.ndarray,
    *,
    grid: tuple = MTF_FIELD_GRID,
    oversample: int = MTF_OVERSAMPLE,
    saturation_dn: float | None = None,
) -> Analysis:
    """Split ``plane`` into a ``grid = (rows, cols)`` field grid (default
    3x5, docs/design.md §4.4's "5x3 field grid"), run ``edge_sfr`` on each
    cell, and collect an MTF50 map. A cell that refuses is recorded (its
    reason and position) in ``residuals`` rather than refusing the whole
    record -- a real target photograph legitimately loses edge contrast in
    some corners at wide field angles, and that's the "sweet spot"
    information this measurement exists to show, not a failure. The
    overall record only refuses if *every* cell failed. ``saturation_dn``
    is forwarded to every cell's ``edge_sfr`` call (see its docstring)."""
    a = Analysis()
    n_rows, n_cols = grid
    h, w = plane.shape
    cell_h, cell_w = h // n_rows, w // n_cols
    mtf50_map = [[None] * n_cols for _ in range(n_rows)]
    lp_map = [[None] * n_cols for _ in range(n_rows)]
    cell_refusals = []
    n_ok = 0
    for i in range(n_rows):
        for j in range(n_cols):
            roi = plane[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            cell_analysis = edge_sfr(roi, oversample=oversample, saturation_dn=saturation_dn)
            if cell_analysis.ok:
                n_ok += 1
                mtf50_map[i][j] = cell_analysis.result["mtf50_cycles_per_sensor_px"]
                lp_map[i][j] = cell_analysis.result["mtf50_lp_per_mm"]
            else:
                cell_refusals.append({"row": i, "col": j, "refusals": [r.to_dict() for r in cell_analysis.refusals]})

    if n_ok == 0:
        a.refuse("all_cells_failed", f"all {n_rows * n_cols} field-grid cells failed their own edge checks", 0, 1)
        return a

    a.result = {
        "grid": [n_rows, n_cols],
        "mtf50_cycles_per_sensor_px_map": mtf50_map,
        "mtf50_lp_per_mm_map": lp_map,
        "n_cells_ok": n_ok,
        "n_cells_total": n_rows * n_cols,
    }
    a.residuals = {"cell_refusals": cell_refusals}
    return a

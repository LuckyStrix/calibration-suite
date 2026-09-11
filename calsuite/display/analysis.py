"""Pure display analysis: measurement arrays in, ``fit.Analysis`` out
(docs/design.md house rule 5). No file I/O, no subprocess, no window or
backend calls here -- those live in ``display/backends/``, ``window.py``,
``osstate.py``, ``profile.py``, ``install_*.py``, ``validate.py`` and
``commands.py``.

Covers docs/design.md §5.2's "what gets measured": TRC, additivity,
primaries/white vs. EDID, black level/contrast, uniformity, warm-up drift,
and rolling-shutter PWM banding.
"""

from __future__ import annotations

import numpy as np

import colour

from calsuite.display import constants as dc
from calsuite.fit import Analysis

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def xyz_to_lab(xyz, white_xyz) -> np.ndarray:
    """Absolute XYZ (cd/m^2, or any consistent unit) -> CIE Lab, using
    `white_xyz`'s own chromaticity (not a fixed D50/D65) as the reference
    white -- every Lab comparison in this module is against a measurement
    made on the same display in the same session, so the display's own
    measured white is the right reference, not a canonical illuminant.
    """
    white_xyz = np.asarray(white_xyz, dtype=np.float64)
    y_white = white_xyz[1]
    xyz_n = np.asarray(xyz, dtype=np.float64) / y_white
    white_xy = colour.XYZ_to_xy(white_xyz / y_white)
    return colour.XYZ_to_Lab(xyz_n, illuminant=white_xy)


def delta_e00(lab1, lab2) -> float:
    return float(colour.delta_E(np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64)))


# ---------------------------------------------------------------------------
# TRC (tone response curve)
# ---------------------------------------------------------------------------


def trc_fit(channel_ramps: dict, black_y: float = 0.0) -> Analysis:
    """``channel_ramps``: ``{"r": {"levels": [...], "xyz": [[X,Y,Z], ...]},
    "g": {...}, "b": {...}}`` -- `levels` the input drive in [0, 1] for each
    step of ``display.patches.channel_ramp``, `xyz` the measured XYZ at
    that step (any consistent absolute unit; only the ratio to the
    channel's brightest step matters). `black_y` is the display's own
    measured black luminance (Y) -- subtracted from every step before
    fitting, since a real display's *raw* Y is `black_y + contribution *
    level**gamma`, not a pure power law, and fitting the raw (non-black-
    corrected) values would bias the recovered gamma low, worse the lower
    the display's contrast ratio (a real display's black is never exactly
    zero, and neither is the synthetic one's by default -- this matches
    how tone-response characterization is actually done, e.g. dispcal's
    own black-relative "contrast" curve, not raw luminance).

    Fits an effective gamma per channel: a least-squares line through
    log(Y_normalized) vs. log(level) over the ramp's interior points (0 <
    level < 1 -- the endpoints carry no information for a power-law fit
    and level=0 is undefined in log space), the standard way to summarize
    a TRC as one number. The full normalized LUT (Y/Y_max at each measured
    level) is returned alongside it for a LUT-based profile build.
    """
    analysis = Analysis()
    gammas, luts, r2s = {}, {}, {}
    for ch, data in channel_ramps.items():
        levels = np.asarray(data["levels"], dtype=np.float64)
        y = np.asarray(data["xyz"], dtype=np.float64)[:, 1] - black_y
        y_max = float(y[int(np.argmax(levels))])
        if y_max <= 0:
            analysis.refuse(
                f"trc_{ch}_zero_max",
                f"channel {ch}'s brightest ramp step measured zero/negative black-corrected luminance",
                y_max,
                0.0,
            )
            continue
        y_norm = np.clip(y / y_max, 1e-6, None)
        mask = (levels > 0.0) & (levels < 1.0)
        if int(mask.sum()) < 3:
            analysis.refuse(
                f"trc_{ch}_too_few_points",
                f"fewer than 3 usable interior points for channel {ch}'s gamma fit",
                int(mask.sum()),
                3,
            )
            continue
        log_level = np.log(levels[mask])
        log_y = np.log(y_norm[mask])
        slope, intercept = np.polyfit(log_level, log_y, 1)
        pred = slope * log_level + intercept
        ss_res = float(np.sum((log_y - pred) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2)) or 1e-12
        gammas[ch] = float(slope)
        r2s[ch] = float(1.0 - ss_res / ss_tot)
        luts[ch] = [float(v) for v in y_norm]
    analysis.result["effective_gamma"] = gammas
    analysis.result["lut"] = luts
    analysis.result["levels"] = {ch: [float(v) for v in channel_ramps[ch]["levels"]] for ch in channel_ramps}
    analysis.residuals["gamma_fit_r2"] = r2s
    return analysis


# ---------------------------------------------------------------------------
# additivity
# ---------------------------------------------------------------------------


def additivity(black_xyz, r_xyz, g_xyz, b_xyz, w_xyz) -> Analysis:
    """ΔE00 between measured white and black-corrected R+G+B (docs/
    design.md §5.2: "measured W vs. R + G + B. If it fails, the display
    needs a LUT profile rather than matrix/TRC").

    Each of `r_xyz`/`g_xyz`/`b_xyz`/`w_xyz` is a full-drive single-channel
    (or white) measurement, which already includes the display's black
    offset once; naively summing all three would count black three times,
    so each is black-corrected before summing and the black term is added
    back exactly once -- the standard way this check is done (e.g. by
    ArgyllCMS's dispcal). This function only *reports* the discrepancy;
    ``display/profile.py`` is where it drives an actual matrix-vs-LUT-vs-
    refuse decision, since that decision also depends on whether Argyll is
    available.
    """
    analysis = Analysis()
    black = np.asarray(black_xyz, dtype=np.float64)
    summed = (np.asarray(r_xyz, dtype=np.float64) - black)
    summed = summed + (np.asarray(g_xyz, dtype=np.float64) - black)
    summed = summed + (np.asarray(b_xyz, dtype=np.float64) - black)
    summed = summed + black
    w = np.asarray(w_xyz, dtype=np.float64)
    de = delta_e00(xyz_to_lab(w, w), xyz_to_lab(summed, w))
    analysis.result["additivity_de00"] = de
    analysis.result["recommend_lut"] = bool(de > dc.ADDITIVITY_DE00_MAX)
    analysis.result["threshold_de00"] = dc.ADDITIVITY_DE00_MAX
    analysis.residuals["measured_white_xyz"] = [float(v) for v in w]
    analysis.residuals["summed_rgb_xyz"] = [float(v) for v in summed]
    return analysis


# ---------------------------------------------------------------------------
# primaries / white vs. EDID nominal
# ---------------------------------------------------------------------------


def primaries_vs_edid(measured: dict, edid_chromaticity: dict) -> Analysis:
    """``measured``: ``{"r": xyz, "g": xyz, "b": xyz, "w": xyz, "k": xyz}``
    (full-drive single-channel, white and black measurements).
    ``edid_chromaticity``: ``{"r": (x, y), ...}``, e.g.
    ``devices.parse_edid(...).chromaticity`` -- the nominal claim this
    compares against (docs/design.md §5.2: "EDID says R (0.638, 0.334);
    measured (..., ...)").

    R/G/B chromaticity is computed from *black-corrected* XYZ (a channel's
    own primary light, not the panel's backlight-leakage black); white
    keeps its black-inclusive chromaticity, matching what EDID's own white
    point column describes (the color the eye/instrument sees at full
    white, black leakage included).
    """
    analysis = Analysis()
    black = np.asarray(measured["k"], dtype=np.float64)
    measured_xy = {}
    for ch in ("r", "g", "b", "w"):
        xyz = np.asarray(measured[ch], dtype=np.float64)
        corrected = xyz if ch == "w" else xyz - black
        y = max(float(corrected[1]), 1e-9)
        xy = colour.XYZ_to_xy(corrected / y)
        measured_xy[ch] = (float(xy[0]), float(xy[1]))
    analysis.result["measured_chromaticity"] = measured_xy
    analysis.result["edid_chromaticity"] = {k: tuple(float(c) for c in v) for k, v in edid_chromaticity.items()}
    analysis.result["delta_xy"] = {
        ch: (
            measured_xy[ch][0] - edid_chromaticity[ch][0],
            measured_xy[ch][1] - edid_chromaticity[ch][1],
        )
        for ch in ("r", "g", "b", "w")
        if ch in edid_chromaticity
    }
    return analysis


# ---------------------------------------------------------------------------
# black level / contrast
# ---------------------------------------------------------------------------


def black_and_contrast(black_xyz, white_xyz) -> Analysis:
    analysis = Analysis()
    y_black = float(np.asarray(black_xyz, dtype=np.float64)[1])
    y_white = float(np.asarray(white_xyz, dtype=np.float64)[1])
    analysis.result["black_luminance_cdm2"] = y_black
    analysis.result["white_luminance_cdm2"] = y_white
    if y_black <= 0:
        analysis.refuse(
            "black_level_nonpositive",
            "black measurement is zero or negative; contrast ratio is undefined",
            y_black,
            0.0,
        )
        analysis.result["contrast_ratio"] = None
    else:
        analysis.result["contrast_ratio"] = y_white / y_black
    return analysis


# ---------------------------------------------------------------------------
# uniformity
# ---------------------------------------------------------------------------


def uniformity(grid_xyz) -> Analysis:
    """``grid_xyz``: nested sequence, shape (n, n, 3) -- row-major XYZ
    measurements from ``display.patches.uniformity_grid()``. Every cell is
    compared against the center cell (design §5.2: "luminance and
    chromaticity on a 5x5 grid").
    """
    analysis = Analysis()
    grid = np.asarray(grid_xyz, dtype=np.float64)
    n = grid.shape[0]
    if grid.shape[:2] != (n, n) or grid.shape[2] != 3:
        analysis.refuse(
            "uniformity_grid_shape",
            f"expected a square (n, n, 3) grid, got {grid.shape}",
            list(grid.shape),
            None,
        )
        return analysis
    center = grid[n // 2, n // 2]
    luminance_pct = (grid[..., 1] / center[1] * 100.0).tolist()
    de00_grid = np.zeros((n, n))
    lab_center = xyz_to_lab(center, center)
    for i in range(n):
        for j in range(n):
            de00_grid[i, j] = delta_e00(lab_center, xyz_to_lab(grid[i, j], center))
    flat_pct = [v for row in luminance_pct for v in row]
    analysis.result["luminance_pct_of_center"] = luminance_pct
    analysis.result["de00_vs_center"] = de00_grid.tolist()
    analysis.result["luminance_uniformity_min_pct"] = float(min(flat_pct))
    analysis.result["luminance_uniformity_max_pct"] = float(max(flat_pct))
    analysis.result["de00_vs_center_max"] = float(de00_grid.max())
    return analysis


# ---------------------------------------------------------------------------
# warm-up drift
# ---------------------------------------------------------------------------


def warmup_drift(times_s, luminance) -> Analysis:
    """``times_s``/``luminance``: equal-length sequences from repeated
    measurements of one patch since power-on (design §5.2: "measure for 30
    min after power-on; tells you how long to wait before profiling").
    Reports the final sample's luminance and the earliest time whose
    luminance is within ``constants.WARMUP_STABLE_FRACTION`` of that final
    value *and stays within it* for the rest of the series.
    """
    analysis = Analysis()
    t = np.asarray(times_s, dtype=np.float64)
    y = np.asarray(luminance, dtype=np.float64)
    if len(t) < 2 or len(t) != len(y):
        analysis.refuse(
            "warmup_too_few_samples",
            "need at least 2 equal-length time/luminance samples to characterize warm-up drift",
            len(t),
            2,
        )
        return analysis
    y_final = float(y[-1])
    stable_time = None
    if y_final != 0:
        for i in range(len(t)):
            remaining = y[i:]
            if np.all(np.abs(remaining - y_final) <= dc.WARMUP_STABLE_FRACTION * abs(y_final)):
                stable_time = float(t[i])
                break
    analysis.result["final_luminance_cdm2"] = y_final
    analysis.result["initial_luminance_cdm2"] = float(y[0])
    analysis.result["initial_fraction_of_final"] = float(y[0] / y_final) if y_final else None
    analysis.result["stable_time_s"] = stable_time
    analysis.result["stable_fraction_threshold"] = dc.WARMUP_STABLE_FRACTION
    if stable_time is None:
        analysis.refuse(
            "warmup_never_stabilized",
            f"luminance never settled within {dc.WARMUP_STABLE_FRACTION:.0%} of its final value across the series",
            None,
            dc.WARMUP_STABLE_FRACTION,
        )
    return analysis


# ---------------------------------------------------------------------------
# rolling-shutter PWM banding
# ---------------------------------------------------------------------------


def pwm_banding(row_means, *, row_period_s: float | None = None) -> Analysis:
    """FFT-based rolling-shutter banding detection (design §5.2) on
    `row_means` (1D, one value per row of a fast-shutter frame of a flat
    white screen -- from ``synth.display.render_rolling_shutter_rows`` in
    tests, or a real camera backend's row means from a raw frame).

    Reports the dominant non-DC frequency in cycles/row; converts to Hz
    only when `row_period_s` (seconds between the start of consecutive
    rows' exposure -- the sensor's row readout time) is given, since
    cycles/row alone carries no time base (design's explicit "Hz only if
    readout time is given").
    """
    analysis = Analysis()
    y = np.asarray(row_means, dtype=np.float64)
    n = len(y)
    if n < 8:
        analysis.refuse("pwm_too_few_rows", "need at least 8 rows for a meaningful FFT", n, 8)
        return analysis
    detrended = y - y.mean()
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n, d=1.0)  # cycles per row
    spectrum[0] = 0.0  # drop DC
    peak_idx = int(np.argmax(spectrum))
    peak_mag = float(spectrum[peak_idx])
    rest = np.delete(spectrum, peak_idx)
    floor = float(np.median(rest)) if len(rest) else 0.0
    detected = floor > 0 and peak_mag >= dc.PWM_FFT_MIN_PROMINENCE * floor
    analysis.result["detected"] = bool(detected)
    analysis.result["cycles_per_row"] = float(freqs[peak_idx]) if detected else None
    analysis.result["prominence"] = (peak_mag / floor) if floor > 0 else None
    analysis.result["frequency_hz"] = (
        float(freqs[peak_idx] / row_period_s) if detected and row_period_s else None
    )
    return analysis

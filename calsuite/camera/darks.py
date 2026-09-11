"""Dark current vs. temperature, hot-pixel maps, and the "star eater" check
(docs/design.md §3.1). Pure analysis (house rule 5).

Dark current is fit per CFA channel: within each temperature bin, mean
(black-subtracted) signal vs. exposure time gives a dark rate in DN/s,
converted to e-/s via gain; across temperature bins, log2(dark rate) vs.
temperature gives a doubling temperature (silicon dark current roughly
doubles every few degrees C -- see synth/sensor.py's
``DARK_CURRENT_DOUBLING_C``, which this fit is checked against in tests).

Hot pixels are found on the raw (not per-channel-deinterleaved) visible
pixel grid -- a hot photosite is a property of one physical pixel regardless
of which CFA color filter sits over it, exactly like ``raw.optical_black``
reads individual mosaic samples directly rather than interpolating between
them (house rule 1's "never demosaics" is about interpolation, not about
this per-pixel read).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from calsuite import raw as rawmod
from calsuite.camera.constants import (
    DARK_MIN_EXPOSURES_PER_BIN,
    DARK_MIN_TEMP_BINS,
    DARK_TEMP_BIN_WIDTH_C,
    HOT_PIXEL_SIGMA_K,
    STAR_EATER_MIN_COUNT,
    STAR_EATER_RATIO_THRESHOLD,
)
from calsuite.camera.settings import LENR_KEY, setting_on_any
from calsuite.fit import Analysis
from calsuite.synth.sensor import DARK_CURRENT_DOUBLING_C

CHANNELS = ("R", "G1", "G2", "B")


def _black_for(black_dn, channel: str) -> float:
    return black_dn[channel] if isinstance(black_dn, dict) else black_dn


def _gain_for(gain_e_per_dn, channel: str) -> float | None:
    if gain_e_per_dn is None:
        return None
    return gain_e_per_dn[channel] if isinstance(gain_e_per_dn, dict) else gain_e_per_dn


def _temp_bin(temp_c: float) -> float:
    return round(temp_c / DARK_TEMP_BIN_WIDTH_C) * DARK_TEMP_BIN_WIDTH_C


def find_hot_pixels(frames: list, sigma_k: float = HOT_PIXEL_SIGMA_K) -> dict:
    """Stack ``frames`` (same exposure length; averaging suppresses read
    noise so a marginal hot pixel isn't lost in a single frame's shot
    noise) and flag pixels above a robust (MAD-based) sigma threshold on the
    *raw* visible pixel grid.

    Returns ``{"rows": int array, "cols": int array, "count": int,
    "threshold_dn": float, "median_dn": float}`` -- coordinates are into
    each frame's own ``visible`` sub-array (row/col 0 is the visible area's
    own origin, not the full ``cfa`` array's), so they line up directly with
    ``raw.planes(frame, area="visible")``'s arrays too.
    """
    stack = np.mean(
        [f.cfa[f.visible[0], f.visible[1]].astype(np.float64) for f in frames],
        axis=0,
    )
    median = float(np.median(stack))
    mad = float(np.median(np.abs(stack - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + sigma_k * robust_sigma
    hot = stack > threshold
    rows, cols = np.nonzero(hot)
    return {
        "rows": rows.astype(np.int32),
        "cols": cols.astype(np.int32),
        "count": int(hot.sum()),
        "threshold_dn": threshold,
        "median_dn": median,
    }


def analyze_darks(frames: list, black_dn, gain_e_per_dn=None) -> Analysis:
    """``frames``: a dark series spanning several exposure times and (for
    the doubling-temperature fit) ideally several ambient temperatures, one
    ISO. ``black_dn``: float or ``{channel: float}``. ``gain_e_per_dn``:
    float, ``{channel: float}``, or ``None`` (from ``camera.ptc`` -- when a
    device has no PTC record yet, dark current is still reported in DN/s and
    the doubling-temperature fit still runs, since a ratio of rates is
    unit-invariant; only the electron-domain numbers are omitted).

    Refuses outright (LENR corrupts every dark in the series, not just
    some) if long-exposure noise reduction is on for any frame.
    """
    a = Analysis()
    if setting_on_any(frames, LENR_KEY):
        a.refuse(
            "long_exposure_nr_on",
            "long-exposure noise reduction is enabled on one or more frames; it subtracts an "
            "in-camera dark from the raw data, so the raw signal no longer reflects the sensor's "
            "own dark current",
            value=True,
            threshold=False,
        )
        return a

    by_temp: dict = defaultdict(list)
    for f in frames:
        temp = f.meta.sensor_temp_c
        if temp is None or f.meta.exposure_s is None:
            continue
        by_temp[_temp_bin(temp)].append(f)

    channels_result = {ch: {"temp_bins": {}} for ch in CHANNELS}
    usable_bins_by_channel = {ch: [] for ch in CHANNELS}

    for temp, temp_frames in sorted(by_temp.items()):
        exposures = sorted({f.meta.exposure_s for f in temp_frames})
        if len(exposures) < DARK_MIN_EXPOSURES_PER_BIN:
            continue
        for ch in CHANNELS:
            points = []
            for f in temp_frames:
                plane = rawmod.planes(f, area="visible")[ch]
                points.append((f.meta.exposure_s, float(plane.mean()) - _black_for(black_dn, ch)))
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            fit = stats.linregress(x, y)
            dark_rate_dn_s = max(fit.slope, 0.0)
            gain = _gain_for(gain_e_per_dn, ch)
            dark_rate_e_s = dark_rate_dn_s * gain if gain is not None else None
            channels_result[ch]["temp_bins"][temp] = {
                "n_exposures": len(exposures),
                "dark_current_dn_per_s": dark_rate_dn_s,
                "dark_current_e_per_s": dark_rate_e_s,
                "r_value": fit.rvalue,
            }
            if dark_rate_dn_s > 0:
                # A ratio of rates (what the doubling-temperature fit below
                # actually uses) is unit-invariant, so DN/s is just as valid
                # a unit as e-/s here -- this bin stays usable even with no
                # gain available yet.
                usable_bins_by_channel[ch].append((temp, dark_rate_dn_s))

    for ch in CHANNELS:
        bins = usable_bins_by_channel[ch]
        if len(bins) >= DARK_MIN_TEMP_BINS:
            temps = np.array([t for t, _ in bins])
            log_rates = np.log2(np.array([r for _, r in bins]))
            fit = stats.linregress(temps, log_rates)
            # log2(rate) = slope * temp + c  =>  rate doubles every 1/slope degrees.
            doubling_c = 1.0 / fit.slope if fit.slope > 0 else None
            channels_result[ch]["doubling_temperature_c"] = doubling_c
            channels_result[ch]["doubling_fit_r_value"] = fit.rvalue
        else:
            channels_result[ch]["doubling_temperature_c"] = None

    if all(len(v) < DARK_MIN_TEMP_BINS for v in usable_bins_by_channel.values()):
        a.refuse(
            "insufficient_temperature_bins",
            f"fewer than {DARK_MIN_TEMP_BINS} usable temperature bins (each needing "
            f">= {DARK_MIN_EXPOSURES_PER_BIN} exposure times) -- can't fit a doubling temperature",
            value=max((len(v) for v in usable_bins_by_channel.values()), default=0),
            threshold=DARK_MIN_TEMP_BINS,
        )

    by_exposure: dict = defaultdict(list)
    for f in frames:
        by_exposure[f.meta.exposure_s].append(f)
    hot_pixel_summary = {}
    for exposure_s, exp_frames in sorted(by_exposure.items()):
        summary = find_hot_pixels(exp_frames)
        hot_pixel_summary[exposure_s] = {
            "count": summary["count"],
            "threshold_dn": summary["threshold_dn"],
            "median_dn": summary["median_dn"],
            "n_frames_stacked": len(exp_frames),
        }

    a.result = {
        "channels": channels_result,
        "hot_pixels_by_exposure_s": hot_pixel_summary,
        "reference_doubling_temperature_c": DARK_CURRENT_DOUBLING_C,
    }
    return a


def star_eater_check(short_darks: list, long_darks: list, sigma_k: float = HOT_PIXEL_SIGMA_K) -> Analysis:
    """Compares isolated hot-pixel counts between a long and a short dark
    series at the same temperature to answer design doc §3.1's "star eater"
    question: does the camera filter single bright pixels out of the raw
    data? A healthy sensor's dark current excess grows with exposure time,
    so a long dark should show clearly more hot pixels than a short one; if
    it doesn't, that's the signature of in-camera suppression.
    """
    a = Analysis()
    short = find_hot_pixels(short_darks, sigma_k=sigma_k)
    long = find_hot_pixels(long_darks, sigma_k=sigma_k)

    if long["count"] < STAR_EATER_MIN_COUNT:
        verdict = "inconclusive"
    elif long["count"] >= short["count"] * STAR_EATER_RATIO_THRESHOLD:
        verdict = "no"
    else:
        verdict = "yes"

    a.result = {
        "verdict": verdict,
        "short_hot_pixel_count": short["count"],
        "long_hot_pixel_count": long["count"],
        "ratio_threshold": STAR_EATER_RATIO_THRESHOLD,
        "min_count_for_verdict": STAR_EATER_MIN_COUNT,
    }
    return a

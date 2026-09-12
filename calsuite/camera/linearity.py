"""Full well + linearity, per CFA channel (docs/design.md §3.1): mean signal
vs. exposure time, up to clipping. A baseline slope (flux rate) is fit from
the low-signal points, every point's deviation from that line is reported in
percent, and the "linear range" is the contiguous run of low-to-high
exposures that stays within ``LINEARITY_DEVIATION_PCT`` of it.
"""

from __future__ import annotations

import numpy as np

from calsuite import raw as rawmod
from calsuite.camera.constants import (
    LINEARITY_CLIP_FRACTION,
    LINEARITY_DEVIATION_PCT,
    LINEARITY_LOW_SIGNAL_FRACTION,
    LINEARITY_MIN_POINTS,
)
from calsuite.fit import Analysis

CHANNELS = ("R", "G1", "G2", "B")


def _black_for(black_dn, channel: str) -> float:
    return black_dn[channel] if isinstance(black_dn, dict) else black_dn


def _analyze_channel(points: list, white_level: float, black: float = 0.0) -> dict:
    """``points``: ``[(exposure_s, mean_dn), ...]`` for one channel, any
    order -- ``mean_dn`` already black-subtracted by the caller. Returns a
    dict with the baseline fit, per-point deviation, the linear range, and
    saturation DN.

    ``black``: this channel's own black level, DN -- needed here because
    ``points``' means are black-subtracted but ``white_level`` is not.
    Comparing a black-subtracted mean directly against
    ``LINEARITY_CLIP_FRACTION * white_level`` (the raw/absolute scale) is
    wrong: a fully saturated pixel's black-subtracted mean tops out at
    ``white_level - black``, not ``white_level``, so the clip threshold
    must be taken as a fraction of that same black-subtracted range or a
    genuinely clipped point never crosses it.
    """
    points = sorted(points, key=lambda p: p[0])
    exposures = np.array([p[0] for p in points])
    means = np.array([p[1] for p in points])

    signal_range = white_level - black
    clip_thresh = LINEARITY_CLIP_FRACTION * signal_range
    unclipped_mask = means < clip_thresh
    n_clipped = int(np.count_nonzero(~unclipped_mask))

    low_mask = unclipped_mask & (means < LINEARITY_LOW_SIGNAL_FRACTION * signal_range)
    if np.count_nonzero(low_mask) < 2:
        # Not enough low-signal points to anchor a baseline -- fall back to
        # every unclipped point rather than refusing outright; the caller
        # decides whether the resulting deviations are trustworthy.
        low_mask = unclipped_mask

    x_low, y_low = exposures[low_mask], means[low_mask]
    # Forced through the origin: at zero exposure a flat's signal (already
    # black-subtracted by the caller) is by construction zero, and fitting
    # a free intercept here would let a single noisy low-signal point pull
    # the whole baseline off the true flux rate.
    slope = float(np.sum(x_low * y_low) / np.sum(x_low * x_low)) if np.any(x_low) else 0.0

    predicted = slope * exposures
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation_pct = np.where(predicted != 0, (means - predicted) / predicted * 100.0, 0.0)

    # Linear range: the longest contiguous prefix (starting at the shortest
    # exposure) of unclipped points within LINEARITY_DEVIATION_PCT.
    linear_len = 0
    for i in range(len(points)):
        if not unclipped_mask[i] or abs(deviation_pct[i]) > LINEARITY_DEVIATION_PCT:
            break
        linear_len = i + 1

    max_deviation_in_range = float(np.max(np.abs(deviation_pct[:linear_len]))) if linear_len else 0.0

    return {
        "n_points": len(points),
        "n_clipped": n_clipped,
        "flux_rate_dn_per_s": slope,
        "exposure_s": exposures.tolist(),
        "mean_dn": means.tolist(),
        "deviation_pct": deviation_pct.tolist(),
        "linear_range_n_points": linear_len,
        "linear_range_max_exposure_s": float(exposures[linear_len - 1]) if linear_len else None,
        "max_deviation_pct_in_range": max_deviation_in_range,
        "saturation_dn": float(white_level),
    }


def analyze_linearity(frames: list, black_dn, gain_e_per_dn=None) -> Analysis:
    """``frames``: a series of flats at increasing exposure times, constant
    flux, single ISO. ``black_dn``: float or ``{channel: float}``.
    ``gain_e_per_dn``: optional (float or ``{channel: float}``, from
    ``camera.ptc``) -- when given, full well is also reported in electrons.
    """
    a = Analysis()
    if len(frames) < LINEARITY_MIN_POINTS:
        a.refuse(
            "too_few_points",
            f"need at least {LINEARITY_MIN_POINTS} exposure steps, got {len(frames)}",
            value=len(frames),
            threshold=LINEARITY_MIN_POINTS,
        )
        return a

    white_level = min(f.white_level for f in frames)
    per_channel_points = {ch: [] for ch in CHANNELS}
    for f in frames:
        planes = rawmod.planes(f, area="visible")
        exposure_s = f.meta.exposure_s
        if exposure_s is None:
            continue
        for ch in CHANNELS:
            per_channel_points[ch].append((exposure_s, float(planes[ch].mean()) - _black_for(black_dn, ch)))

    channels_result = {}
    for ch in CHANNELS:
        result = _analyze_channel(per_channel_points[ch], white_level, _black_for(black_dn, ch))
        if gain_e_per_dn is not None:
            gain = gain_e_per_dn[ch] if isinstance(gain_e_per_dn, dict) else gain_e_per_dn
            # saturation_dn is the absolute (not black-subtracted) white
            # level, since that's the intuitive number for a report ("clips
            # at DN X"); full well in electrons is a *signal* quantity, so
            # black is subtracted here before converting.
            result["full_well_e"] = (result["saturation_dn"] - _black_for(black_dn, ch)) * gain
        channels_result[ch] = result
        if result["linear_range_n_points"] == 0:
            a.refuse(
                "no_linear_range",
                f"channel {ch}: no prefix of points stayed within {LINEARITY_DEVIATION_PCT}% of the "
                "baseline slope",
                value=result["max_deviation_pct_in_range"],
                threshold=LINEARITY_DEVIATION_PCT,
            )

    a.result = {"channels": channels_result, "deviation_threshold_pct": LINEARITY_DEVIATION_PCT}
    a.residuals = {ch: channels_result[ch]["deviation_pct"] for ch in CHANNELS}
    return a

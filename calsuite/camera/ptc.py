"""Photon transfer curve (Janesick), per CFA channel (docs/design.md §3.1):
pairs of flats at many signal levels, variance of the pair difference over 2
vs. mean signal, fit in the shot-noise-dominated region to get gain
(electrons/DN) and read-noise (from the fit's intercept).

For a pair of frames at the same mean signal, Var(A-B) = Var(A) + Var(B) =
2*Var(single frame), so Var(A-B)/2 equals a single frame's own variance --
which, per Janesick, is:

    Var(DN) = (mean_DN - black_DN) / gain_e_per_dn + read_noise_dn**2

linear in signal, slope 1/gain, intercept read_noise_dn**2. Differencing a
pair (rather than using one frame's variance directly) cancels each
channel's fixed pattern (PRNU/DSNU), which would otherwise inflate a single
frame's variance and bias the fit.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from calsuite import raw as rawmod
from calsuite.camera.constants import (
    PTC_CLIP_FRACTION_MAX,
    PTC_CLIP_NEAR_WHITE_FRACTION,
    PTC_MIN_LEVELS,
    PTC_MIN_SHOT_NOISE_LEVELS,
    PTC_RESIDUAL_FRACTION_MAX,
)
from calsuite.fit import Analysis

CHANNELS = ("R", "G1", "G2", "B")


def _black_for(black_dn, channel: float):
    return black_dn[channel] if isinstance(black_dn, dict) else black_dn


def _level_stats(pair, black_dn) -> dict:
    """One pair's (mean_dn, var_diff_dn, clipped_fraction) per channel."""
    fa, fb = pair
    white = min(fa.white_level, fb.white_level)
    pa, pb = rawmod.planes(fa, area="visible"), rawmod.planes(fb, area="visible")
    out = {}
    for ch in CHANNELS:
        a_arr, b_arr = pa[ch], pb[ch]
        clip_thresh = PTC_CLIP_NEAR_WHITE_FRACTION * white
        clipped = np.count_nonzero(a_arr >= clip_thresh) + np.count_nonzero(b_arr >= clip_thresh)
        clipped_fraction = clipped / (a_arr.size + b_arr.size)
        mean_dn = 0.5 * (float(a_arr.mean()) + float(b_arr.mean()))
        var_diff = float(np.var(a_arr - b_arr)) / 2.0
        out[ch] = {
            "mean_dn": mean_dn,
            "signal_dn": mean_dn - _black_for(black_dn, ch),
            "var_diff_dn2": var_diff,
            "clipped_fraction": clipped_fraction,
        }
    return out


def _fit_channel(points: list, channel_refusals: list) -> dict | None:
    """``points``: unclipped ``{"signal_dn", "var_diff_dn2"}`` dicts for one
    channel, already sorted by signal. Iteratively trims the highest-signal
    point while it departs from the running linear fit by more than
    ``PTC_RESIDUAL_FRACTION_MAX`` (PRNU/nonlinearity excess grows with
    signal, so it always shows up at the high end first), then fits the
    remaining "shot noise region". Returns ``None`` (with a refusal appended
    to ``channel_refusals``) if too few points survive.
    """
    if len(points) < PTC_MIN_LEVELS:
        channel_refusals.append(("too_few_levels", len(points), PTC_MIN_LEVELS))
        return None

    # Iteratively trim the highest-signal point while *any* point departs
    # from the running linear fit by more than PTC_RESIDUAL_FRACTION_MAX
    # (PRNU/nonlinearity excess grows with signal, so it always shows up at
    # the high end first) -- always the highest-signal point, not whichever
    # has the single worst residual, so every iteration strictly shrinks the
    # signal range (guaranteed to terminate) instead of chasing whichever
    # point the noise made look worst this iteration. Stops either once the
    # residuals settle (a real shot-noise region was found) or once trimming
    # further would go below the reporting floor -- in the latter case, if
    # the residuals never settled, that IS the "shot-noise region too
    # narrow" finding, not merely running out of points.
    kept = list(points)
    region_ok = False
    while True:
        x = np.array([p["signal_dn"] for p in kept])
        y = np.array([p["var_diff_dn2"] for p in kept])
        slope, intercept, _, _, _ = stats.linregress(x, y)
        predicted = slope * x + intercept
        rel_resid = np.abs(y - predicted) / np.maximum(np.abs(predicted), 1e-9)
        if rel_resid.max() <= PTC_RESIDUAL_FRACTION_MAX:
            region_ok = True
            break
        if len(kept) <= PTC_MIN_SHOT_NOISE_LEVELS:
            break
        kept.pop(int(np.argmax(x)))

    if not region_ok:
        # `len(kept)` is always exactly PTC_MIN_SHOT_NOISE_LEVELS here (the
        # loop only stops early -- via `break` above -- once the residuals
        # settle, which sets region_ok=True and skips this branch
        # entirely), so reporting "N vs threshold N" would read as a count
        # that failed its own floor, which is never true: the real failure
        # is that residuals never settled even once trimmed all the way
        # down to that floor. Report the quantity that actually failed the
        # comparison -- the worst remaining residual against its own
        # tolerance -- instead.
        channel_refusals.append(("narrow_shot_noise_region", float(rel_resid.max()), PTC_RESIDUAL_FRACTION_MAX))
        return None

    x = np.array([p["signal_dn"] for p in kept])
    y = np.array([p["var_diff_dn2"] for p in kept])
    fit = stats.linregress(x, y)
    slope, intercept = fit.slope, fit.intercept
    if slope <= 0:
        channel_refusals.append(("non_positive_slope", slope, 0.0))
        return None

    gain_e_per_dn = 1.0 / slope
    read_noise_dn2 = max(intercept, 0.0)
    read_noise_dn = float(np.sqrt(read_noise_dn2))
    read_noise_e = read_noise_dn * gain_e_per_dn

    # Relative-error propagation: gain = 1/slope -> d(gain)/gain = d(slope)/slope.
    gain_rel_err = abs(fit.stderr / slope) if slope else float("nan")
    # read_noise_dn = sqrt(intercept) -> d(rn)/rn = 0.5 * d(intercept)/intercept.
    if intercept > 0 and fit.intercept_stderr is not None:
        rn_rel_err = 0.5 * abs(fit.intercept_stderr / intercept)
    else:
        rn_rel_err = float("nan")

    predicted = slope * x + intercept
    residuals = (y - predicted).tolist()

    return {
        "gain_e_per_dn": gain_e_per_dn,
        "read_noise_dn": read_noise_dn,
        "read_noise_e": read_noise_e,
        "gain_uncertainty_e_per_dn": gain_e_per_dn * gain_rel_err if gain_rel_err == gain_rel_err else None,
        "read_noise_uncertainty_e": read_noise_e * rn_rel_err if rn_rel_err == rn_rel_err else None,
        "r_value": fit.rvalue,
        "n_levels_total": len(points),
        "n_levels_used": len(kept),
        "signal_dn_used": x.tolist(),
        "var_diff_dn2_used": y.tolist(),
        "residuals_dn2": residuals,
    }


def _channel_refusal_message(ch: str, check: str, value, threshold) -> str:
    """One per-channel refusal's human-readable message. Pulled out of
    ``analyze_ptc`` so it's unit-testable with no synthetic frames/noise
    involved (a refusal *condition* can legitimately depend on a noise
    realization; the *wording* of the message it produces should not need
    a lucky seed to pin down) -- see test_camera_ptc.py's
    test_channel_refusal_message_* tests.

    ``narrow_shot_noise_region`` gets its own phrasing because its
    ``value``/``threshold`` are a residual fraction vs. its tolerance, not
    a plain count vs. a floor (see the comment where that refusal is
    raised, in ``_fit_channel``) -- the generic "N vs threshold M" template
    below is only honest for the other checks, where value and threshold
    really are two counts/numbers on the same axis being compared.
    """
    if check == "narrow_shot_noise_region":
        return (
            f"channel {ch}: narrow shot noise region: even trimmed down to the "
            f"minimum {PTC_MIN_SHOT_NOISE_LEVELS} points, the worst residual "
            f"({value:.1%}) still exceeds the {threshold:.0%} tolerance"
        )
    return f"channel {ch}: {check.replace('_', ' ')} ({value} vs threshold {threshold})"


def analyze_ptc(pairs: list, black_dn) -> Analysis:
    """``pairs``: a list of ``(frame_a, frame_b)`` at increasing signal
    levels (ideally 20-30, design doc §3.1). ``black_dn``: a float or
    ``{channel: float}``, from ``camera.bias``.
    """
    a = Analysis()
    if len(pairs) < PTC_MIN_LEVELS:
        a.refuse(
            "too_few_levels",
            f"need at least {PTC_MIN_LEVELS} flat-pair levels, got {len(pairs)}",
            value=len(pairs),
            threshold=PTC_MIN_LEVELS,
        )
        return a

    per_pair = [_level_stats(pair, black_dn) for pair in pairs]

    channels_result = {}
    for ch in CHANNELS:
        all_points = [p[ch] for p in per_pair]
        n_clipped = sum(1 for p in all_points if p["clipped_fraction"] > PTC_CLIP_FRACTION_MAX)
        unclipped = sorted(
            (p for p in all_points if p["clipped_fraction"] <= PTC_CLIP_FRACTION_MAX),
            key=lambda p: p["signal_dn"],
        )
        channel_refusals: list = []
        fitted = _fit_channel(unclipped, channel_refusals)
        channels_result[ch] = {
            "n_pairs_total": len(all_points),
            "n_pairs_clipped": n_clipped,
            **({"fit": fitted} if fitted is not None else {}),
        }
        for check, value, threshold in channel_refusals:
            a.refuse(f"{check}[{ch}]", _channel_refusal_message(ch, check, value, threshold), value=value, threshold=threshold)

    a.result = {"channels": channels_result}
    a.residuals = {
        ch: channels_result[ch]["fit"]["residuals_dn2"]
        for ch in CHANNELS
        if "fit" in channels_result[ch]
    }
    return a

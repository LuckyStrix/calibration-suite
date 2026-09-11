"""Bulb timing (docs/design.md §3.1): a flat shot through the star tracker's
optocoupler path at several commanded bulb lengths, fit to a latency
(intercept) + scale-error (slope) model of the intervalometer's timing.
Pure analysis.

Like ``camera.shutter``, this pools all four CFA planes into one signal
number per frame (bulb duration, like shutter speed, is a property of the
whole exposure window) and has no absolute time reference, so the flux rate
is estimated the same cross-calibration way -- see ``camera.shutter``'s
docstring for the reasoning and its trade-off.

Known blind spot, sharper here than in ``camera.shutter``: the first-pass
flux rate is itself proportional to the *true* scale factor (since it's
fit from the same signal-vs-commanded data the scale error lives in), so
dividing signal by that rate to get an inferred "actual" time cancels almost
all of the true scale error back out -- ``scale_error_pct`` is only
trustworthy to the extent ``latency_s * sum(commanded)/sum(commanded**2)``
is small relative to 1 (i.e. the commanded lengths are long compared to the
latency), and even then it is measuring a *residual*, not an absolute scale
error, with no independent way to tell from this data alone. The intercept
(``latency_s``) is on firmer ground -- an additive term isn't degenerate
with the flux-rate calibration the way a multiplicative one is -- but it
too comes back divided by whatever fraction of the true scale error the
first pass failed to absorb. Treat both numbers as decent order-of-magnitude
diagnostics, not a precision timing measurement; a real latency/scale
separation needs an independent absolute-time reference (e.g. the ESP32
timestamping its own trigger).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from calsuite.camera.constants import BULB_MIN_POINTS
from calsuite.camera.shutter import _mean_signal_dn
from calsuite.fit import Analysis


def analyze_bulb_timing(frames: list, commanded_s: list, black_dn) -> Analysis:
    """``frames``/``commanded_s``: parallel lists -- ``commanded_s[i]`` is
    the bulb length commanded for ``frames[i]`` by the intervalometer (not
    read from EXIF: true BULB exposures are often not reported accurately,
    or at all, by the camera itself, which is exactly what this measures).
    """
    a = Analysis()
    if len(frames) != len(commanded_s):
        raise ValueError(f"frames ({len(frames)}) and commanded_s ({len(commanded_s)}) must be the same length")
    if len(frames) < BULB_MIN_POINTS:
        a.refuse(
            "too_few_points",
            f"need at least {BULB_MIN_POINTS} commanded bulb lengths, got {len(frames)}",
            value=len(frames),
            threshold=BULB_MIN_POINTS,
        )
        return a

    commanded = np.array(commanded_s, dtype=np.float64)
    signal_dn = np.array([_mean_signal_dn(f, black_dn) for f in frames])

    # First pass: cross-calibrate the flux rate assuming actual ~= commanded
    # (errors partly cancel across the sweep), forced through the origin.
    flux_rate = float(np.sum(commanded * signal_dn) / np.sum(commanded**2))
    actual_s = signal_dn / flux_rate

    # Second pass: latency (intercept) + scale error (slope) of actual vs.
    # commanded -- a free-intercept fit this time, since a latency is
    # exactly a nonzero intercept.
    fit = stats.linregress(commanded, actual_s)

    a.result = {
        "flux_rate_dn_per_s": flux_rate,
        "commanded_s": commanded.tolist(),
        "actual_s": actual_s.tolist(),
        "latency_s": fit.intercept,
        "scale_error_pct": (fit.slope - 1.0) * 100.0,
        "r_value": fit.rvalue,
    }
    a.residuals = {"actual_minus_fit_s": (actual_s - (fit.slope * commanded + fit.intercept)).tolist()}
    return a

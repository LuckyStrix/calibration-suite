"""Shutter accuracy (docs/design.md §3.1): measured signal vs. nominal
exposure time, across the camera's shutter speed range, at one constant
flux. Pure analysis.

Pools all four CFA planes into one signal number per frame rather than
fitting each channel separately -- shutter timing is a property of the
sensor's whole exposure window (mechanical or electronic), not of the color
filter over a given photosite (see camera/constants.py's note on
``SHUTTER_MIN_SPEEDS``).

There's no absolute photometric reference here, so the flux rate itself is
estimated by cross-calibration: a line is fit through (nominal_s, signal_dn)
for every speed, and *that* fitted rate is then used to convert every
speed's own signal back into an inferred actual exposure time. This assumes
the nominal times are unbiased on average (errors partly cancel across the
sweep), which is the same trade every practical shutter-accuracy test makes
without a separate absolute-time reference.

Known blind spot: a *uniform* proportional (multiplicative) timing error
across every speed is degenerate with "the light source was slightly
brighter/dimmer than assumed" and is invisible to this method -- the
cross-calibration slope simply absorbs it. What this method *can* detect is
any variation in % error across speeds (an additive lag, e.g. mechanical
curtain travel time, which matters proportionally more at fast/short
speeds) -- which is also the more common real-world failure mode.
"""

from __future__ import annotations

import numpy as np

from calsuite import raw as rawmod
from calsuite.camera.constants import SHUTTER_MIN_SPEEDS
from calsuite.fit import Analysis


def _mean_signal_dn(frame, black_dn) -> float:
    planes = rawmod.planes(frame, area="visible")
    black = float(np.mean(list(black_dn.values()))) if isinstance(black_dn, dict) else black_dn
    return float(np.mean([p.mean() for p in planes.values()])) - black


def analyze_shutter_accuracy(frames: list, black_dn) -> Analysis:
    """``frames``: one per nominal shutter speed (``meta.exposure_s``),
    constant flux, increasing or decreasing speed order doesn't matter."""
    a = Analysis()
    usable = [f for f in frames if f.meta.exposure_s]
    if len(usable) < SHUTTER_MIN_SPEEDS:
        a.refuse(
            "too_few_speeds",
            f"need at least {SHUTTER_MIN_SPEEDS} shutter speeds with known nominal exposure time, "
            f"got {len(usable)}",
            value=len(usable),
            threshold=SHUTTER_MIN_SPEEDS,
        )
        return a

    nominal_s = np.array([f.meta.exposure_s for f in usable])
    signal_dn = np.array([_mean_signal_dn(f, black_dn) for f in usable])

    # Cross-calibration slope (flux rate, DN/s), forced through the origin:
    # zero exposure means zero (black-subtracted) signal by construction.
    flux_rate = float(np.sum(nominal_s * signal_dn) / np.sum(nominal_s**2))
    actual_s = signal_dn / flux_rate
    pct_error = (actual_s - nominal_s) / nominal_s * 100.0

    order = np.argsort(nominal_s)
    a.result = {
        "flux_rate_dn_per_s": flux_rate,
        "nominal_s": nominal_s[order].tolist(),
        "actual_s": actual_s[order].tolist(),
        "pct_error": pct_error[order].tolist(),
        "max_abs_pct_error": float(np.max(np.abs(pct_error))),
    }
    a.residuals = {"pct_error": pct_error[order].tolist()}
    return a

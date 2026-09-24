"""ISO invariance and engineering dynamic range (docs/design.md §3.1): given
input-referred (electron-domain) read noise and full well at a set of ISOs,
find the lowest ISO past which read noise has effectively stopped falling,
and compute log2(full well / read noise) per ISO. Pure analysis; the actual
per-ISO read noise/full well numbers come from ``camera.bias``/``camera.ptc``
runs at each ISO -- this module only does the cross-ISO reasoning.
"""

from __future__ import annotations

import math

from calsuite.camera.constants import (
    CG_SWITCH_MIN_DROP_LOG2_PER_STOP,
    CG_SWITCH_SPIKE_RATIO,
    ISO_INVARIANCE_TOLERANCE_PCT,
    ISO_MIN_COUNT,
)
from calsuite.fit import Analysis


def find_conversion_gain_switches(read_noise_e: dict) -> list[dict]:
    """Candidate dual-conversion-gain switches in a read-noise-vs-ISO curve.

    A sensor that switches conversion gain at some ISO shows a *step* in
    input-referred read noise there (typically ~halved) with flat curve on
    either side -- unlike the smooth fall-then-flat of ordinary analog gain.
    Each adjacent ISO pair gets a per-stop log2 drop; a pair is flagged when
    that drop is large (``CG_SWITCH_MIN_DROP_LOG2_PER_STOP``) and a spike
    relative to its neighbours' drops (``CG_SWITCH_SPIKE_RATIO``).
    Advisory: the caller records it, it never refuses.
    """
    isos = sorted(read_noise_e, key=float)
    drops = []
    for lo, hi in zip(isos, isos[1:], strict=False):
        stops = math.log2(float(hi) / float(lo))
        drops.append(math.log2(read_noise_e[lo] / read_noise_e[hi]) / stops if stops > 0 else 0.0)
    switches = []
    for i, drop in enumerate(drops):
        neighbours = [max(drops[j], 0.0) for j in (i - 1, i + 1) if 0 <= j < len(drops)]
        if drop >= CG_SWITCH_MIN_DROP_LOG2_PER_STOP and all(drop >= CG_SWITCH_SPIKE_RATIO * n for n in neighbours):
            switches.append(
                {"from_iso": isos[i], "to_iso": isos[i + 1], "read_noise_drop_log2_per_stop": drop}
            )
    return switches


def analyze_iso_invariance(read_noise_e_by_iso: dict, full_well_e_by_iso, *, refused_ptc_isos: list | None = None) -> Analysis:
    """``read_noise_e_by_iso``: ``{iso: input-referred read noise in
    electrons}``. ``full_well_e_by_iso``: same shape, or a single float
    applied at every ISO (a truly ISO-invariant sensor's full well in
    electrons is roughly constant across ISO -- only its value in DN
    shrinks as gain increases), or empty/``None`` if no ``camera.linearity``
    record had one to give. ``refused_ptc_isos``: ISOs whose ``camera.ptc``
    record is on record as refused, purely to name a likely cause in the
    ``no_full_well`` message below -- optional, and never itself a reason
    to refuse (a refused camera.ptc at one ISO doesn't prevent computing
    the invariance curve from the *other*, ok, ISOs).

    Recommended ISO is the *lowest* ISO whose read noise is within
    ``ISO_INVARIANCE_TOLERANCE_PCT`` of the minimum across the sweep --
    "lowest" because raising ISO past the invariance point buys nothing
    (design doc's star-tracker sentence) but does cost highlight headroom.

    Refuses (house rule 3 -- these are findings, not reasons to save no
    record at all; ``camera/commands.py::_cmd_iso`` always reaches
    ``_save`` now, so both checks below land in a real, readable record)
    if there's too little data to characterize the curve at all, or no
    electron-domain full well to pair it with -- both can fire together.
    """
    a = Analysis()
    if len(read_noise_e_by_iso) < ISO_MIN_COUNT:
        a.refuse(
            "too_few_isos",
            f"need at least {ISO_MIN_COUNT} ISOs to characterize the invariance curve, "
            f"got {len(read_noise_e_by_iso)}",
            value=len(read_noise_e_by_iso),
            threshold=ISO_MIN_COUNT,
        )

    if not full_well_e_by_iso:
        if refused_ptc_isos:
            cause = f"upstream camera.ptc refused at ISO {', '.join(str(i) for i in refused_ptc_isos)}"
        else:
            cause = "no ok camera.linearity record has a known electron-domain full well"
        a.refuse("no_full_well", f"no exportable full well: {cause}")

    non_positive = sorted(iso for iso, rn in read_noise_e_by_iso.items() if not (rn and rn > 0))
    if non_positive:
        # A zero or missing read noise would become `min_read_noise_e`,
        # making the invariance tolerance 0.0 and recommending that ISO --
        # a confident recommendation built on the one number that wasn't
        # measured. (PTC's intercept used to clamp an unresolvable read
        # noise to exactly 0.0; it reports None now, but a caller can still
        # hand this a bad value.)
        a.refuse(
            "non_positive_read_noise",
            f"ISO {', '.join(str(i) for i in non_positive)} has no positive read noise to place on the "
            "invariance curve",
            value=non_positive,
            threshold=0.0,
        )

    if not a.ok:
        return a

    isos = sorted(read_noise_e_by_iso)
    read_noise_e = {iso: read_noise_e_by_iso[iso] for iso in isos}
    min_read_noise = min(read_noise_e.values())
    tolerance = min_read_noise * (1.0 + ISO_INVARIANCE_TOLERANCE_PCT / 100.0)

    recommended_iso = next(iso for iso in isos if read_noise_e[iso] <= tolerance)

    def full_well_at(iso):
        return full_well_e_by_iso[iso] if isinstance(full_well_e_by_iso, dict) else full_well_e_by_iso

    dynamic_range_stops = {
        iso: math.log2(full_well_at(iso) / read_noise_e[iso]) for iso in isos if read_noise_e[iso] > 0
    }

    a.result = {
        "read_noise_e_by_iso": read_noise_e,
        "min_read_noise_e": min_read_noise,
        "tolerance_pct": ISO_INVARIANCE_TOLERANCE_PCT,
        "recommended_iso": recommended_iso,
        "dynamic_range_stops_by_iso": dynamic_range_stops,
        "possible_conversion_gain_switches": find_conversion_gain_switches(read_noise_e),
    }
    return a

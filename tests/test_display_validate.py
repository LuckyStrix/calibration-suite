"""``display/validate.py``'s ``validate()`` -- no test file existed for this
module before the second bug hunt, despite it being the one check standing
between an installed profile and ``store.require_exportable`` (house rule
2/3): a profile record is only ``measured``-and-``ok`` if validation
passes. These are boundary tests for the mean/p95/max ΔE00 thresholds, plus
the "validated against the wrong profile" false-confidence case (a large,
systematic color error must refuse, not just a noisy one).
"""

from __future__ import annotations

import numpy as np

from calsuite.display import constants as dc
from calsuite.display import validate as V

WHITE_XYZ = (95.0, 100.0, 108.9)  # D65-ish


def _lab_targets(n: int) -> list:
    # A spread of neutral targets -- enough to give mean/p95/max distinct
    # values without pulling in colour-science just for this.
    return [(l_star, 0.0, 0.0) for l_star in np.linspace(10.0, 90.0, n)]


def _xyz_for_lab(lab, white_xyz):
    import colour

    lab_arr = np.asarray(lab, dtype=np.float64)
    white_xy = colour.XYZ_to_xy(np.asarray(white_xyz, dtype=np.float64) / white_xyz[1])
    xyz = colour.Lab_to_XYZ(lab_arr, illuminant=white_xy) * white_xyz[1]
    return xyz


def test_validate_refuses_on_length_mismatch():
    result = V.validate([WHITE_XYZ], _lab_targets(3), WHITE_XYZ, 1.0)
    assert not result.ok
    assert result.refusals[0].check == "validation_length_mismatch"


def test_validate_passes_for_an_exact_reproduction():
    targets = _lab_targets(6)
    measured = [_xyz_for_lab(t, WHITE_XYZ) for t in targets]
    result = V.validate(measured, targets, WHITE_XYZ, accuracy_de00_estimate=1.0)
    assert result.ok
    assert result.result["de00_mean"] < 0.01


def test_validate_mean_de00_boundary_just_inside_and_outside():
    """Every patch is off from its target by the same amount, so mean == p95
    == max -- pushing just past the (loosest) max threshold in one
    direction crosses all three at once; this isolates the *mean* check
    specifically by choosing an offset that clears mean's threshold but not
    quite the looser p95/max ones would still pass -- so instead offset
    every patch by the *same* small amount and check the boundary against
    `mean_thr` alone (p95==max==mean here, so this is the tightest check)."""
    accuracy = 1.0
    thr = accuracy * dc.VALIDATION_MEAN_DE00_MULTIPLIER
    targets = _lab_targets(6)

    def _mean_de00_for_l_offset(l_offset):
        measured = [_xyz_for_lab((l_star + l_offset, a, b), WHITE_XYZ) for l_star, a, b in targets]
        return V.validate(measured, targets, WHITE_XYZ, accuracy)

    # Bisect on the L* offset (ΔE00 isn't linear in ΔL*) until de00_mean
    # straddles `thr` within a tight tolerance, then check the refusal
    # fires on exactly the expected side of it.
    lo, hi = 0.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _mean_de00_for_l_offset(mid).result["de00_mean"] < thr:
            lo = mid
        else:
            hi = mid

    just_under = _mean_de00_for_l_offset(lo)
    assert just_under.result["de00_mean"] < thr
    assert all(r.check != "validation_mean_de00" for r in just_under.refusals)

    just_over = _mean_de00_for_l_offset(hi)
    assert just_over.result["de00_mean"] > thr
    assert any(r.check == "validation_mean_de00" for r in just_over.refusals)
    assert not just_over.ok


def test_validate_refuses_when_run_against_effectively_the_wrong_profile():
    """The false-confidence case docs/design.md house rule 3 exists for:
    validating against a systematically wrong profile (here, simulated as
    every measured patch landing near the display's white point instead of
    its target -- what "the profile did nothing" or "the wrong profile was
    installed" looks like) must not sneak through as `ok=True`."""
    targets = _lab_targets(8)
    measured = [np.asarray(WHITE_XYZ, dtype=np.float64) for _ in targets]  # every patch reproduces as white
    result = V.validate(measured, targets, WHITE_XYZ, accuracy_de00_estimate=1.0)
    assert not result.ok
    assert result.result["de00_mean"] > dc.VALIDATION_MEAN_DE00_MULTIPLIER

import pytest

from calsuite.camera import iso


def test_recommended_iso_at_the_elbow_of_a_falling_then_flat_sweep():
    # Read noise falls sharply with ISO up to 800, then flattens -- the
    # classic ISO-invariance signature. 800 should be recommended, not the
    # (marginally lower-noise) higher ISOs above it.
    read_noise_e_by_iso = {
        100: 6.0,
        200: 4.2,
        400: 3.1,
        800: 2.5,
        1600: 2.45,
        3200: 2.42,
        6400: 2.40,
    }
    full_well_e = 40000.0
    a = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e)
    assert a.ok, a.refusals
    assert a.result["recommended_iso"] == 800


def test_dynamic_range_stops_computed_per_iso():
    read_noise_e_by_iso = {100: 4.0, 400: 2.0, 1600: 2.0}
    full_well_e = 16384.0
    a = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e)
    # log2(16384/4) = log2(4096) = 12 exactly.
    assert a.result["dynamic_range_stops_by_iso"][100] == pytest.approx(12.0, abs=0.01)
    # log2(16384/2) = log2(8192) = 13 exactly.
    assert a.result["dynamic_range_stops_by_iso"][400] == pytest.approx(13.0, abs=0.01)


def test_refuses_with_too_few_isos():
    a = iso.analyze_iso_invariance({100: 3.0, 200: 2.5}, 40000.0)
    assert not a.ok
    assert a.refusals[0].check == "too_few_isos"


def test_per_iso_full_well_dict_supported():
    read_noise_e_by_iso = {100: 3.0, 400: 2.5, 1600: 2.4}
    full_well_e_by_iso = {100: 40000.0, 400: 39000.0, 1600: 38000.0}
    a = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e_by_iso)
    assert a.ok
    assert set(a.result["dynamic_range_stops_by_iso"]) == set(read_noise_e_by_iso)

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


def test_scale_invariance_of_recommended_iso_and_dynamic_range():
    """recommended_iso and dynamic_range_stops are ratio/dimensionless
    quantities: uniformly scaling every read-noise and full-well number by
    the same constant (e.g. a different linear unit) must not move the
    recommended ISO, or the stops figure, at all."""
    read_noise_e_by_iso = {100: 6.0, 200: 4.2, 400: 3.1, 800: 2.5, 1600: 2.45, 3200: 2.42}
    full_well_e = 40000.0
    a1 = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e)

    k = 3.7
    a2 = iso.analyze_iso_invariance({i: v * k for i, v in read_noise_e_by_iso.items()}, full_well_e * k)

    assert a1.ok and a2.ok
    assert a1.result["recommended_iso"] == a2.result["recommended_iso"]
    for iso_val in read_noise_e_by_iso:
        assert a1.result["dynamic_range_stops_by_iso"][iso_val] == pytest.approx(
            a2.result["dynamic_range_stops_by_iso"][iso_val], abs=1e-9
        )


def test_per_iso_full_well_dict_supported():
    read_noise_e_by_iso = {100: 3.0, 400: 2.5, 1600: 2.4}
    full_well_e_by_iso = {100: 40000.0, 400: 39000.0, 1600: 38000.0}
    a = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e_by_iso)
    assert a.ok
    assert set(a.result["dynamic_range_stops_by_iso"]) == set(read_noise_e_by_iso)


def test_refuses_a_non_positive_read_noise_instead_of_recommending_that_iso():
    """A zero read noise becomes `min_read_noise_e`, which makes the
    invariance tolerance 0.0 and recommends that ISO -- the star-tracker
    recommendation the report prints, built on the one number that wasn't
    measured. PTC's intercept used to clamp an unresolvable read noise to
    exactly 0.0 and hand it here (it reports None now), which happened on
    roughly a third of channel fits of clean synthetic data.
    """
    analysis = iso.analyze_iso_invariance({100: 0.0, 400: 3.2, 1600: 2.9}, 40000.0)
    assert not analysis.ok
    refusal = [r for r in analysis.refusals if r.check == "non_positive_read_noise"]
    assert refusal, [r.check for r in analysis.refusals]
    assert refusal[0].value == [100]

    # The same sweep with all three measured is unaffected.
    ok = iso.analyze_iso_invariance({100: 4.1, 400: 3.2, 1600: 2.9}, 40000.0)
    assert ok.ok
    assert ok.result["recommended_iso"] == 1600


def test_a_step_in_read_noise_is_flagged_as_a_conversion_gain_switch():
    # Flat, then halved in one step, then flat: the dual-gain signature.
    a = iso.analyze_iso_invariance({100: 3.0, 200: 3.0, 400: 3.0, 800: 1.5, 1600: 1.45, 3200: 1.4}, 40000.0)
    assert a.ok, a.refusals
    switches = a.result["possible_conversion_gain_switches"]
    assert [(s["from_iso"], s["to_iso"]) for s in switches] == [(400, 800)]
    assert switches[0]["read_noise_drop_log2_per_stop"] == pytest.approx(1.0, abs=0.01)


def test_a_smooth_falling_then_flat_curve_is_not_flagged():
    a = iso.analyze_iso_invariance(
        {100: 6.0, 200: 4.2, 400: 3.1, 800: 2.5, 1600: 2.45, 3200: 2.42, 6400: 2.40}, 40000.0
    )
    assert a.result["possible_conversion_gain_switches"] == []


def test_a_switch_is_found_when_isos_are_unevenly_spaced():
    # 400 -> 1600 is two stops; halving read noise across it is 0.5 log2/stop
    # between flat neighbours, still a spike -- and normalised per stop, not per step.
    a = iso.analyze_iso_invariance({100: 3.0, 400: 3.0, 1600: 1.5, 6400: 1.5}, 40000.0)
    (sw,) = a.result["possible_conversion_gain_switches"]
    assert (sw["from_iso"], sw["to_iso"]) == (400, 1600)
    assert sw["read_noise_drop_log2_per_stop"] == pytest.approx(0.5, abs=0.01)

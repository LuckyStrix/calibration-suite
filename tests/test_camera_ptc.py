from dataclasses import replace

import numpy as np
import pytest

from calsuite.camera import ptc
from calsuite.camera.constants import PTC_MIN_SHOT_NOISE_LEVELS, PTC_RESIDUAL_FRACTION_MAX
from calsuite.synth import sensor as synth_sensor


def _pairs_at_levels(
    signal_e_levels, gain, read_noise_e, black_dn, exposure_s=0.01, seed=0, shape=(256, 256), black_dn_by_channel=None
):
    model = synth_sensor.SensorModel(
        shape=shape,
        gain_e_per_dn=gain,
        read_noise_e=read_noise_e,
        black_dn=black_dn,
        black_dn_by_channel=black_dn_by_channel,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
        full_well_e=60000.0,
    )
    rng = np.random.default_rng(seed)
    pairs = []
    for signal_e in signal_e_levels:
        flux = signal_e / exposure_s
        a = synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
        b = synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
        pairs.append((a, b))
    return pairs


def test_gain_and_read_noise_recovered_within_design_budget():
    # docs/design.md §10: gain +/-2-3%. The design doc's separate "read
    # noise +/-5%" budget is delivered by camera.bias's direct bias-pair
    # measurement (test_camera_bias.py) -- PTC's own intercept-derived read
    # noise is an ordinary-least-squares extrapolation back to zero signal
    # across ~1.5 decades of range, and is heteroscedastic (the high-signal
    # points' variance estimates carry far larger absolute -- if not
    # relative -- noise, which dominates the fit's sum of squared errors and
    # swamps the intercept term long before it swamps the slope). Real
    # Janesick-method PTC write-ups treat the intercept read noise the same
    # way: a cross-check against the bias-pair number, not the precision
    # measurement itself. This test holds gain to the design budget and only
    # checks the intercept read noise is positive and roughly the right
    # order of magnitude.
    gain = 2.5
    read_noise_e = 3.0
    black_dn = 512.0
    levels = np.linspace(500, 32000, 24)
    pairs = _pairs_at_levels(levels, gain, read_noise_e, black_dn)

    a = ptc.analyze_ptc(pairs, black_dn)
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        fit = data["fit"]
        assert fit["gain_e_per_dn"] == pytest.approx(gain, rel=0.03)
        # The intercept either resolves read noise or it doesn't. It used
        # to be clamped with `max(intercept, 0.0)`, which turned a negative
        # intercept -- an unphysical result meaning the extrapolation
        # failed -- into a confident "0.000 electrons" with no uncertainty
        # and status ok, on 11 of 32 channel fits of clean synthetic data.
        if fit["read_noise_resolved"]:
            assert 0 < fit["read_noise_e"] < read_noise_e * 3
        else:
            assert fit["read_noise_e"] is None and fit["read_noise_dn"] is None


def test_gain_recovered_correctly_with_per_channel_black_dn():
    """analyze_ptc already black-subtracts per channel via `_black_for`
    (a dict black_dn, matching what camera.bias would hand it in real
    usage) rather than a flat mean -- and since Var(A-B) cancels any
    constant offset entirely regardless, a per-channel black level should
    never bias gain even if it were subtracted wrong. This pins both: pass
    the real per-channel truth and check gain still lands in the design
    budget for every channel, exactly like the flat-black-level test above."""
    gain = 2.5
    read_noise_e = 3.0
    black_dn_by_channel = {"R": 500.0, "G1": 508.0, "G2": 516.0, "B": 524.0}
    levels = np.linspace(500, 32000, 24)
    pairs = _pairs_at_levels(levels, gain, read_noise_e, 512.0, black_dn_by_channel=black_dn_by_channel)

    a = ptc.analyze_ptc(pairs, black_dn_by_channel)
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        fit = data["fit"]
        assert fit["gain_e_per_dn"] == pytest.approx(gain, rel=0.03)
        assert fit["read_noise_e"] is None or 0 < fit["read_noise_e"] < read_noise_e * 3


def test_refuses_with_too_few_levels():
    gain, read_noise_e, black_dn = 2.0, 3.0, 512.0
    levels = np.linspace(1000, 10000, 3)
    pairs = _pairs_at_levels(levels, gain, read_noise_e, black_dn)
    a = ptc.analyze_ptc(pairs, black_dn)
    assert not a.ok
    assert a.refusals[0].check == "too_few_levels"


def test_clipped_pairs_are_excluded_and_can_trigger_refusal():
    gain, read_noise_e, black_dn = 2.0, 3.0, 512.0
    # 4 good, unclipped levels + 4 levels driven hard into saturation (well
    # above the model's full_well_e) -- clipped levels should be dropped,
    # leaving too few (4 < PTC_MIN_LEVELS=6) for a trustworthy fit.
    good_levels = np.linspace(500, 10000, 4)
    clipped_levels = [200000.0] * 4  # far above full_well_e=60000 -- forces heavy clipping
    pairs = _pairs_at_levels(list(good_levels) + clipped_levels, gain, read_noise_e, black_dn)
    a = ptc.analyze_ptc(pairs, black_dn)
    assert not a.ok
    assert any("too_few_levels" in r.check for r in a.refusals)
    # and the clipped pairs were actually detected as clipped, not just
    # coincidentally excluded
    for _ch, data in a.result["channels"].items():
        assert data["n_pairs_clipped"] >= 4


def test_clipping_refusal_does_not_fire_on_good_data():
    gain, read_noise_e, black_dn = 2.0, 3.0, 512.0
    levels = np.linspace(500, 20000, 8)
    pairs = _pairs_at_levels(levels, gain, read_noise_e, black_dn)
    a = ptc.analyze_ptc(pairs, black_dn)
    assert a.ok
    for _ch, data in a.result["channels"].items():
        assert data["n_pairs_clipped"] == 0


def test_narrow_shot_noise_region_refusal_fires_on_curved_data():
    # Hand-crafted points where variance follows a power law (var ~
    # signal**1.8) rather than the PTC's linear model at every scale, so no
    # contiguous run ever settles within the residual tolerance down to the
    # reporting floor -- this is the "clipping never happened, but the
    # curve was never a line" failure mode, distinct from too-few-levels.
    signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0])
    var = signal**1.8
    points = [{"signal_dn": float(s), "var_diff_dn2": float(v), "clipped_fraction": 0.0} for s, v in zip(signal, var, strict=True)]
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is None
    assert refusals and refusals[0][0] == "narrow_shot_noise_region"


def _points_with_bad_tail(bad_count, mult=4.0):
    """8 points on a clean line (``var = 0.4 * signal + 9.0``) except the
    top ``bad_count`` (by signal, i.e. the ones the trim loop removes
    first) are inflated by ``mult`` so they blow well past
    ``PTC_RESIDUAL_FRACTION_MAX``. The number of *clean* points left once
    the trim removes all the bad ones is ``8 - bad_count`` -- this is how
    the three boundary tests below dial the size of the surviving
    shot-noise region directly, rather than the total input count (which
    ``PTC_MIN_LEVELS`` -- a different, larger floor -- already gates)."""
    signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0])
    var = 0.4 * signal + 9.0
    if bad_count:
        var[-bad_count:] = var[-bad_count:] * mult
    return [{"signal_dn": float(s), "var_diff_dn2": float(v), "clipped_fraction": 0.0} for s, v in zip(signal, var, strict=True)]


def test_narrow_shot_noise_region_boundary_below_floor_refuses_honestly():
    """Only 3 clean points survive once the trim removes every corrupted
    one (bad_count=5 leaves 8-5=3 < PTC_MIN_SHOT_NOISE_LEVELS=4) -- the
    trim loop can't go below the floor of 4, so it stops there with 1
    corrupted point still mixed in and refuses. Pins two things: the
    refusal actually fires (this genuinely is too narrow a region), and
    the message no longer claims "4 vs threshold 4" (a count that can
    never be below its own floor) -- it reports the residual that
    actually failed, which must be a true inequality against its
    threshold, not an equality."""
    points = _points_with_bad_tail(bad_count=PTC_MIN_SHOT_NOISE_LEVELS + 1)
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is None
    assert refusals and refusals[0][0] == "narrow_shot_noise_region"
    _, value, threshold = refusals[0]
    assert value > threshold  # a real inequality, not "N vs threshold N"
    assert threshold == PTC_RESIDUAL_FRACTION_MAX


def test_narrow_shot_noise_region_boundary_exactly_at_floor_passes():
    """Exactly PTC_MIN_SHOT_NOISE_LEVELS (4) clean points survive
    (bad_count=4 leaves 8-4=4). "At least 4" is inclusive -- landing
    exactly on the floor with a clean fit must PASS, not refuse."""
    points = _points_with_bad_tail(bad_count=PTC_MIN_SHOT_NOISE_LEVELS)
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is not None
    assert not refusals
    assert fitted["n_levels_used"] == PTC_MIN_SHOT_NOISE_LEVELS


def test_narrow_shot_noise_region_boundary_above_floor_passes():
    """5 clean points survive (bad_count=3 leaves 8-3=5) -- comfortably
    above the floor, and the trim stops before ever reaching it."""
    points = _points_with_bad_tail(bad_count=PTC_MIN_SHOT_NOISE_LEVELS - 1)
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is not None
    assert not refusals
    assert fitted["n_levels_used"] == PTC_MIN_SHOT_NOISE_LEVELS + 1


def test_channel_refusal_message_for_narrow_shot_noise_region_is_a_true_inequality():
    """``_channel_refusal_message`` (pulled out of ``analyze_ptc`` so this
    needs no synthetic frames or RNG at all -- a refusal *condition* can
    legitimately depend on a noise realization; its *wording* should not)
    must never render "4 vs threshold 4" -- a count sitting exactly on its
    own floor reported as if it failed a less-than comparison against
    itself (the original ISO-1600-demo bug, seed offset 7, channel G2).
    ``narrow_shot_noise_region``'s value/threshold are a residual fraction
    vs. its tolerance, so a real refusal's value must exceed its
    threshold, not equal it."""
    message = ptc._channel_refusal_message("G2", "narrow_shot_noise_region", 0.1147, PTC_RESIDUAL_FRACTION_MAX)
    assert f"{PTC_MIN_SHOT_NOISE_LEVELS} vs threshold {PTC_MIN_SHOT_NOISE_LEVELS}" not in message
    assert "11.5%" in message and "8%" in message


def test_channel_refusal_message_for_a_plain_count_check_uses_the_generic_template():
    # too_few_levels/non_positive_slope are genuine value-vs-threshold
    # comparisons, unlike narrow_shot_noise_region -- the generic template
    # is honest for these.
    message = ptc._channel_refusal_message("R", "too_few_levels", 3, ptc.PTC_MIN_LEVELS)
    assert message == f"channel R: too few levels (3 vs threshold {ptc.PTC_MIN_LEVELS})"


def _shift_dn(frame, offset):
    cfa = frame.cfa.astype(np.float64) + offset
    return replace(
        frame,
        cfa=np.clip(cfa, 0, 65535).astype(np.uint16),
        black_level=tuple(b + offset for b in frame.black_level),
        white_level=frame.white_level + offset,
    )


def test_black_offset_invariance():
    """Adding a constant DN offset to every frame, with the declared
    black_dn shifted to match, must leave gain AND read noise unchanged.

    Gain is a plain (free-intercept) OLS slope in signal_dn, which is
    shift-invariant in x almost by definition of linear regression -- so
    checking gain alone wouldn't actually exercise the black-handling
    code. read_noise_e comes from the fit's *intercept* (an extrapolation
    to signal_dn == 0), which DOES move if signal_dn is computed from the
    wrong black level -- that's the half of this test that can actually
    fail (confirmed by deliberately hardcoding the black subtraction in
    ptc._level_stats and re-running: read_noise_e diverges between the
    unshifted and shifted runs while gain does not, then restored).
    """
    gain, read_noise_e, black_dn = 2.5, 3.0, 512.0
    levels = np.linspace(500, 32000, 12)
    pairs = _pairs_at_levels(levels, gain, read_noise_e, black_dn, shape=(128, 128))
    a1 = ptc.analyze_ptc(pairs, black_dn)

    offset = 300.0
    pairs_shifted = [(_shift_dn(a, offset), _shift_dn(b, offset)) for a, b in pairs]
    a2 = ptc.analyze_ptc(pairs_shifted, black_dn + offset)

    assert a1.ok and a2.ok
    for ch in ptc.CHANNELS:
        f1, f2 = a1.result["channels"][ch]["fit"], a2.result["channels"][ch]["fit"]
        assert f1["gain_e_per_dn"] == pytest.approx(f2["gain_e_per_dn"], rel=1e-9)
        # The intercept moves with a wrong black level, so this is the half
        # of the test that can actually fail -- compare it when the fit
        # resolved it, and compare the raw intercept itself either way.
        assert f1["read_noise_resolved"] == f2["read_noise_resolved"]
        assert f1["read_noise_intercept_dn2"] == pytest.approx(f2["read_noise_intercept_dn2"], rel=1e-6)
        if f1["read_noise_resolved"]:
            assert f1["read_noise_e"] == pytest.approx(f2["read_noise_e"], rel=1e-6)


def test_narrow_shot_noise_region_does_not_fire_on_a_clean_line():
    signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0])
    var = 0.4 * signal + 9.0
    points = [{"signal_dn": float(s), "var_diff_dn2": float(v), "clipped_fraction": 0.0} for s, v in zip(signal, var, strict=True)]
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is not None
    assert not refusals

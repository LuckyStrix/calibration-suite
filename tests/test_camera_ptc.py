import numpy as np
import pytest

from calsuite.camera import ptc
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
        # Clamped at 0 rather than negative when the noisy intercept dips
        # below zero (see the comment above) -- still a valid outcome, not
        # a bug, so this only bounds the upper side.
        assert 0 <= fit["read_noise_e"] < read_noise_e * 3


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
        assert 0 <= fit["read_noise_e"] < read_noise_e * 3


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


def test_narrow_shot_noise_region_does_not_fire_on_a_clean_line():
    signal = np.array([100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0])
    var = 0.4 * signal + 9.0
    points = [{"signal_dn": float(s), "var_diff_dn2": float(v), "clipped_fraction": 0.0} for s, v in zip(signal, var, strict=True)]
    refusals: list = []
    fitted = ptc._fit_channel(points, refusals)
    assert fitted is not None
    assert not refusals

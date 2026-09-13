import numpy as np
import pytest

from calsuite.camera import bulb
from calsuite.synth import sensor as synth_sensor


def _frames_for_commanded(model, commanded_s, latency_s, scale, flux_e_per_s, rng):
    frames = []
    for c in commanded_s:
        actual_s = latency_s + scale * c
        frames.append(synth_sensor.frame(model, exposure_s=actual_s, flux_e_per_s=flux_e_per_s, temp_c=20.0, rng=rng))
    return frames


def test_latency_recovered_when_scale_error_is_near_zero():
    # bulb.py's docstring spells out a real blind spot: the first-pass
    # cross-calibration is fit from the same data the *scale* error lives
    # in, so it cancels almost all of that error back out, and the
    # intercept (latency) comes back divided by whatever fraction survived
    # -- so latency is only cleanly recoverable when the true scale error
    # is itself small (dividing by ~1 barely changes it), which this test
    # holds to. See test_scale_error_is_not_reliably_recovered below for
    # the blind spot itself.
    black_dn, gain, flux = 512.0, 2.0, 150.0
    latency_s, scale = 0.15, 1.0
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=1.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(0)
    commanded = [10.0, 20.0, 40.0, 80.0, 160.0]
    frames = _frames_for_commanded(model, commanded, latency_s, scale, flux, rng)

    a = bulb.analyze_bulb_timing(frames, commanded, black_dn)
    assert a.ok, a.refusals
    assert a.result["latency_s"] == pytest.approx(latency_s, abs=0.05)


def test_scale_error_is_not_reliably_recovered_by_this_method():
    # Documents the blind spot itself (bulb.py's docstring): even a large
    # true scale error comes back close to 0% once the first-pass flux-rate
    # calibration has absorbed almost all of it. This is a known limitation,
    # not a bug -- the assertion pins the *current, understood* behavior so
    # a future change to the method is a deliberate decision, not a silent
    # regression.
    black_dn, gain, flux = 512.0, 2.0, 150.0
    latency_s, scale = 0.15, 1.20  # a large, obviously-wrong 20% scale error
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=1.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(0)
    commanded = [10.0, 20.0, 40.0, 80.0, 160.0]
    frames = _frames_for_commanded(model, commanded, latency_s, scale, flux, rng)

    a = bulb.analyze_bulb_timing(frames, commanded, black_dn)
    assert a.ok, a.refusals
    assert abs(a.result["scale_error_pct"]) < 2.0


def test_flux_scale_covariance():
    """flux_rate_dn_per_s must scale exactly with the illumination flux;
    latency_s and scale_error_pct are cross-calibrated, dimensionless-in-
    flux quantities (see bulb.py's own docstring on the method's blind
    spot) and must not move when only the absolute flux level changes."""
    black_dn, gain = 512.0, 2.0
    latency_s, scale = 0.15, 1.0
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=0.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    # Short commanded lengths (unlike the other bulb tests' 10-160s) so the
    # 4x flux comparison below stays well under the model's default full
    # well (40000 e-) at every commanded length -- this property is about
    # flux scaling, not about probing clipping.
    commanded = [1.0, 2.0, 4.0, 8.0, 16.0]

    def run(flux):
        rng = np.random.default_rng(0)
        frames = _frames_for_commanded(model, commanded, latency_s, scale, flux, rng)
        return bulb.analyze_bulb_timing(frames, commanded, black_dn)

    a1 = run(400.0)
    a2 = run(1600.0)
    assert a1.ok and a2.ok
    assert a2.result["flux_rate_dn_per_s"] == pytest.approx(4 * a1.result["flux_rate_dn_per_s"], rel=1e-3)
    assert a1.result["latency_s"] == pytest.approx(a2.result["latency_s"], rel=1e-2)
    assert a1.result["scale_error_pct"] == pytest.approx(a2.result["scale_error_pct"], abs=0.05)


def test_refuses_with_too_few_points():
    black_dn, gain = 512.0, 2.0
    model = synth_sensor.SensorModel(shape=(32, 32), black_dn=black_dn, gain_e_per_dn=gain)
    rng = np.random.default_rng(1)
    commanded = [1.0, 2.0]
    frames = _frames_for_commanded(model, commanded, 0.0, 1.0, 5000.0, rng)
    a = bulb.analyze_bulb_timing(frames, commanded, black_dn)
    assert not a.ok
    assert a.refusals[0].check == "too_few_points"


def test_raises_on_mismatched_lengths():
    black_dn, gain = 512.0, 2.0
    model = synth_sensor.SensorModel(shape=(32, 32), black_dn=black_dn, gain_e_per_dn=gain)
    rng = np.random.default_rng(1)
    frames = _frames_for_commanded(model, [1.0, 2.0, 3.0], 0.0, 1.0, 5000.0, rng)
    with pytest.raises(ValueError):
        bulb.analyze_bulb_timing(frames, [1.0, 2.0], black_dn)

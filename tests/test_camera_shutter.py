from dataclasses import replace

import numpy as np

from calsuite.camera import shutter
from calsuite.synth import sensor as synth_sensor


def _frame_at_speed(model, nominal_s, actual_s, flux_e_per_s, rng):
    frame = synth_sensor.frame(model, exposure_s=actual_s, flux_e_per_s=flux_e_per_s, temp_c=20.0, rng=rng)
    return replace(frame, meta=replace(frame.meta, exposure_s=nominal_s))


def test_accurate_shutter_reports_near_zero_error():
    black_dn, gain, flux = 512.0, 2.0, 20000.0
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=2.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(0)
    speeds = [1 / 4000, 1 / 1000, 1 / 250, 1 / 60, 1 / 15]
    frames = [_frame_at_speed(model, s, s, flux, rng) for s in speeds]

    a = shutter.analyze_shutter_accuracy(frames, black_dn)
    assert a.ok, a.refusals
    assert a.result["max_abs_pct_error"] < 5.0


def test_detects_a_fixed_shutter_lag_worst_at_fast_speeds():
    # A fixed additive lag (e.g. mechanical curtain travel time) shows up
    # as a LARGER percent error at fast (short-nominal) speeds than at slow
    # ones. This is deliberately not a *uniform* proportional slowdown --
    # see shutter.py's docstring: a purely multiplicative scale error is
    # degenerate with "the light was slightly brighter than assumed" under
    # this method's cross-calibration and is fundamentally undetectable by
    # it, so a useful synthetic test has to inject an additive error
    # instead, exactly like a real mechanical shutter's fixed-lag failure
    # mode.
    black_dn, gain, flux = 512.0, 2.0, 20000.0
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=1.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(1)
    lag_s = 1e-4
    nominal_speeds = [1 / 2000, 1 / 500, 1 / 125, 1 / 30]
    frames = [_frame_at_speed(model, s, s + lag_s, flux, rng) for s in nominal_speeds]

    a = shutter.analyze_shutter_accuracy(frames, black_dn)
    assert a.ok, a.refusals
    errors = dict(zip(a.result["nominal_s"], a.result["pct_error"], strict=True))
    fastest, slowest = min(nominal_speeds), max(nominal_speeds)
    assert abs(errors[fastest]) > abs(errors[slowest])
    assert abs(errors[fastest]) > 10.0


def test_refuses_with_too_few_speeds():
    black_dn, gain = 512.0, 2.0
    model = synth_sensor.SensorModel(shape=(32, 32), black_dn=black_dn, gain_e_per_dn=gain)
    rng = np.random.default_rng(2)
    frames = [_frame_at_speed(model, s, s, 5000.0, rng) for s in [1 / 100, 1 / 50]]
    a = shutter.analyze_shutter_accuracy(frames, black_dn)
    assert not a.ok
    assert a.refusals[0].check == "too_few_speeds"

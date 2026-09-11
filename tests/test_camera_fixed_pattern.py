from dataclasses import replace

import numpy as np

from calsuite.camera import fixed_pattern
from calsuite.synth import sensor as synth_sensor


def _frames(model, exposure_s, flux, temp_c, n, seed):
    rng = np.random.default_rng(seed)
    return [synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=temp_c, rng=rng) for _ in range(n)]


def _add_periodic_row_band(frame, period_rows: int, amplitude_dn: float):
    """Add a fixed (same every frame, unlike synth.sensor's own
    ``row_banding_std_dn`` -- see that constructor arg's docstring: it's
    deliberately a *temporal*, frame-to-frame-independent artifact, so
    averaging several frames cancels it out exactly like it should cancel
    ordinary read noise) periodic row offset directly onto a frame's CFA
    array, for a test vehicle that a per-frame FFT peak detector can
    actually find."""
    rows = np.arange(frame.cfa.shape[0])
    band = amplitude_dn * np.sin(2 * np.pi * rows / period_rows)
    cfa = frame.cfa.astype(np.float64) + band[:, None]
    return replace(frame, cfa=np.clip(cfa, 0, 65535).astype(np.uint16))


def test_prnu_std_recovered_near_injected_value():
    prnu_std = 0.03
    model = synth_sensor.SensorModel(
        shape=(128, 128), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=2.0,
        prnu_std=prnu_std, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    a = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 6, seed=0),
        biases=_frames(model, 1e-4, 0.0, 20.0, 4, seed=2),
        black_dn=512.0,
    )
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        assert abs(data["prnu_std_pct"] - prnu_std * 100.0) < prnu_std * 100.0 * 0.5


def test_row_banding_detected_when_present():
    model = synth_sensor.SensorModel(
        shape=(128, 64), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=1.0, prnu_std=0.0,
        dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    plain_biases = _frames(model, 1e-4, 0.0, 20.0, 6, seed=3)
    banded_biases = [_add_periodic_row_band(f, period_rows=16, amplitude_dn=15.0) for f in plain_biases]

    a_band = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=banded_biases,
        black_dn=512.0,
    )
    a_plain = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=plain_biases,
        black_dn=512.0,
    )
    for ch in fixed_pattern.CHANNELS:
        assert (
            a_band.result["channels"][ch]["row_banding_peak_ratio"]
            > a_plain.result["channels"][ch]["row_banding_peak_ratio"]
        )


def test_refuses_with_too_few_frames():
    model = synth_sensor.SensorModel(shape=(32, 32))
    a = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 1, seed=1),
        flats=_frames(model, 0.5, 5000.0, 20.0, 1, seed=2),
        biases=_frames(model, 1e-4, 0.0, 20.0, 1, seed=3),
        black_dn=512.0,
    )
    assert not a.ok
    assert a.refusals[0].check == "insufficient_frames"

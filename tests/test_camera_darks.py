import numpy as np
import pytest
from dataclasses import replace

from calsuite.camera import darks
from calsuite.synth import sensor as synth_sensor


def _dark_series(temps_c, exposures_s, black_dn=512.0, gain=2.0, dark_at_20c=0.2, shape=(64, 64), seed=0):
    model = synth_sensor.SensorModel(
        shape=shape,
        gain_e_per_dn=gain,
        black_dn=black_dn,
        dark_current_e_per_s_at_20c=dark_at_20c,
        read_noise_e=3.0,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(seed)
    frames = []
    for temp in temps_c:
        for exp in exposures_s:
            frames.append(synth_sensor.frame(model, exposure_s=exp, flux_e_per_s=0.0, temp_c=temp, rng=rng))
    return frames


def _set_lenr(frame, on: bool):
    meta = replace(frame.meta, settings={**frame.meta.settings, "long_exposure_nr": on})
    return replace(frame, meta=meta)


def test_dark_current_and_doubling_temperature_recovered():
    gain = 2.0
    black_dn = 512.0
    dark_at_20c = 0.5  # e-/s
    temps = [10.0, 16.0, 22.0, 28.0]
    exposures = [5.0, 15.0, 30.0, 60.0]
    frames = _dark_series(temps, exposures, black_dn=black_dn, gain=gain, dark_at_20c=dark_at_20c)

    a = darks.analyze_darks(frames, black_dn, gain_e_per_dn=gain)
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        assert data["doubling_temperature_c"] == pytest.approx(synth_sensor.DARK_CURRENT_DOUBLING_C, rel=0.3)
    # at least one channel's 20C-ish bin should be near the true rate
    any_bin = next(iter(a.result["channels"].values()))["temp_bins"]
    closest_temp = min(any_bin, key=lambda t: abs(t - 20.0))
    assert any_bin[closest_temp]["dark_current_e_per_s"] == pytest.approx(dark_at_20c, rel=0.3)


def test_refuses_when_lenr_on():
    frames = _dark_series([20.0], [5.0, 10.0, 15.0])
    frames = [_set_lenr(f, True) for f in frames]
    a = darks.analyze_darks(frames, 512.0, gain_e_per_dn=2.0)
    assert not a.ok
    assert a.refusals[0].check == "long_exposure_nr_on"


def test_does_not_refuse_when_lenr_off_or_unknown():
    frames = _dark_series([10.0, 20.0], [5.0, 15.0, 30.0])
    a = darks.analyze_darks(frames, 512.0, gain_e_per_dn=2.0)
    assert a.ok


def test_hot_pixel_map_exactly_finds_injected_pixels():
    model = synth_sensor.SensorModel(
        shape=(64, 64),
        gain_e_per_dn=2.0,
        black_dn=512.0,
        read_noise_e=2.0,
        dark_current_e_per_s_at_20c=0.05,
        dsnu_std_e_per_s=0.0,
        prnu_std=0.0,
        hot_pixel_fraction=0.02,
        hot_pixel_extra_e_per_s=5000.0,
        fixed_pattern_seed=3,
    )
    rng = np.random.default_rng(0)
    frames = [synth_sensor.frame(model, exposure_s=2.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(4)]

    result = darks.find_hot_pixels(frames)
    _, _, hot_mask = synth_sensor._fixed_pattern_maps(model)
    expected_rows, expected_cols = np.nonzero(hot_mask)

    found = set(zip(result["rows"].tolist(), result["cols"].tolist(), strict=True))
    expected = set(zip(expected_rows.tolist(), expected_cols.tolist(), strict=True))
    assert found == expected
    assert len(expected) > 0


def test_star_eater_check_says_no_when_hot_pixels_grow_with_exposure():
    model = synth_sensor.SensorModel(
        shape=(64, 64),
        gain_e_per_dn=2.0,
        black_dn=512.0,
        read_noise_e=2.0,
        dark_current_e_per_s_at_20c=0.05,
        dsnu_std_e_per_s=0.0,
        prnu_std=0.0,
        hot_pixel_fraction=0.02,
        hot_pixel_extra_e_per_s=3000.0,
        fixed_pattern_seed=3,
    )
    rng = np.random.default_rng(1)
    # Short exposure: the hot pixels' extra dark current (3000 e-/s * 1us)
    # hasn't accumulated enough to rise above read noise -- none detectable.
    # Long exposure: 10s of the same excess rate is unmistakable. A healthy
    # (non-suppressing) camera should show a large jump between the two.
    short = [synth_sensor.frame(model, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(8)]
    long = [synth_sensor.frame(model, exposure_s=10.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(8)]

    a = darks.star_eater_check(short, long)
    assert a.result["verdict"] == "no"


def test_star_eater_check_inconclusive_with_no_hot_pixels():
    model = synth_sensor.SensorModel(shape=(32, 32), hot_pixel_fraction=0.0, read_noise_e=2.0)
    rng = np.random.default_rng(2)
    short = [synth_sensor.frame(model, exposure_s=0.001, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(3)]
    long = [synth_sensor.frame(model, exposure_s=1.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(3)]

    a = darks.star_eater_check(short, long)
    assert a.result["verdict"] == "inconclusive"

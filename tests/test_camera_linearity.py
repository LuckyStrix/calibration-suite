import numpy as np
import pytest

from calsuite.camera import linearity
from calsuite.synth import sensor as synth_sensor


def _flats(exposures_s, flux_e_per_s, black_dn=512.0, gain=2.0, full_well_e=20000.0, seed=0, shape=(128, 128)):
    model = synth_sensor.SensorModel(
        shape=shape,
        gain_e_per_dn=gain,
        black_dn=black_dn,
        full_well_e=full_well_e,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
        read_noise_e=3.0,
    )
    rng = np.random.default_rng(seed)
    return [
        synth_sensor.frame(model, exposure_s=t, flux_e_per_s=flux_e_per_s, temp_c=20.0, rng=rng) for t in exposures_s
    ]


def test_full_well_and_linear_range_recovered():
    gain = 2.0
    black_dn = 512.0
    full_well_e = 20000.0
    flux = 4000.0  # e-/s
    # Exposures from well within the linear range up to well past clipping.
    exposures = np.linspace(0.2, 8.0, 12)
    frames = _flats(exposures, flux, black_dn=black_dn, gain=gain, full_well_e=full_well_e)

    a = linearity.analyze_linearity(frames, black_dn, gain_e_per_dn=gain)
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        assert data["linear_range_n_points"] >= 3
        assert data["max_deviation_pct_in_range"] <= linearity.LINEARITY_DEVIATION_PCT
        assert data["full_well_e"] == pytest.approx(full_well_e, rel=0.05)


def test_refuses_with_too_few_points():
    frames = _flats(np.linspace(0.5, 2.0, 3), 1000.0)
    a = linearity.analyze_linearity(frames, 512.0)
    assert not a.ok
    assert a.refusals[0].check == "too_few_points"


def test_no_linear_range_refusal_fires_on_all_saturated_data():
    # Every exposure drives the sensor past full well -- no prefix of
    # points can stay within the deviation tolerance because they're all
    # clipped from the very first point.
    frames = _flats(np.linspace(5.0, 20.0, 6), flux_e_per_s=50000.0, full_well_e=5000.0)
    a = linearity.analyze_linearity(frames, 512.0)
    assert not a.ok
    assert any(r.check == "no_linear_range" for r in a.refusals)


def test_no_linear_range_refusal_does_not_fire_on_good_data():
    frames = _flats(np.linspace(0.2, 8.0, 12), 4000.0, full_well_e=20000.0)
    a = linearity.analyze_linearity(frames, 512.0)
    assert a.ok

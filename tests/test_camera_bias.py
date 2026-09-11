import numpy as np
import pytest

from calsuite.camera import bias
from calsuite.synth import sensor as synth_sensor


def _bias_frames(n=6, black_dn=512.0, read_noise_e=4.0, gain=2.0, seed=0, exposure_s=1e-4):
    model = synth_sensor.SensorModel(
        shape=(64, 96),
        top_margin=8,
        left_margin=16,
        black_dn=black_dn,
        read_noise_e=read_noise_e,
        gain_e_per_dn=gain,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(seed)
    return [synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(n)]


def test_black_level_recovered_near_true_value():
    frames = _bias_frames(black_dn=512.0)
    a = bias.analyze_bias(frames)
    assert a.ok
    for _ch, value in a.result["black_level_dn"].items():
        assert value == pytest.approx(512.0, abs=2.0)


def test_read_noise_dn_recovered():
    gain = 2.0
    read_noise_e = 4.0
    frames = _bias_frames(read_noise_e=read_noise_e, gain=gain, n=40)
    a = bias.analyze_bias(frames)
    assert a.ok
    expected_dn = read_noise_e / gain
    for _ch, value in a.result["read_noise_dn"].items():
        assert value == pytest.approx(expected_dn, rel=0.25)


def test_read_noise_electrons_within_design_budget():
    # docs/design.md §10: read noise +/-5% with enough pairs.
    gain = 2.0
    read_noise_e = 4.0
    frames = _bias_frames(read_noise_e=read_noise_e, gain=gain, n=200)
    a = bias.analyze_bias(frames)
    bias.add_read_noise_electrons(a, gain)
    for _ch, value in a.result["read_noise_e"].items():
        assert value == pytest.approx(read_noise_e, rel=0.05)


def test_metadata_black_level_comparison_present():
    frames = _bias_frames(black_dn=500.0)
    a = bias.analyze_bias(frames)
    assert a.result["black_level_metadata_dn"] == pytest.approx(500.0)
    for _ch, discrepancy in a.result["black_level_discrepancy_dn"].items():
        assert abs(discrepancy) < 5.0


def test_refuses_with_too_few_frames():
    frames = _bias_frames(n=1)
    a = bias.analyze_bias(frames)
    assert not a.ok
    assert a.refusals[0].check == "insufficient_frames"


def test_refuses_when_frame_is_not_bias_exposure():
    frames = _bias_frames(n=4, exposure_s=1e-4)
    long_frame = _bias_frames(n=1, exposure_s=1.0)[0]
    frames_with_dark = frames[:-1] + [long_frame]
    a = bias.analyze_bias(frames_with_dark)
    assert not a.ok
    assert a.refusals[0].check == "not_bias_exposure"


def test_odd_frame_dropped_not_crashing():
    frames = _bias_frames(n=5)
    a = bias.analyze_bias(frames)
    assert a.ok
    assert a.result["n_pairs"] == 2

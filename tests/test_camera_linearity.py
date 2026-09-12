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


def test_clip_detection_compares_black_subtracted_signal_to_black_subtracted_range():
    # black=512, white_level=10512 (gain=2, full_well_e=20000): a point whose
    # RAW mean is pinned at white_level has a black-subtracted mean of
    # white_level - black = 10000. That point IS fully saturated and must be
    # flagged clipped. The old code compared the black-subtracted mean
    # directly against LINEARITY_CLIP_FRACTION * white_level (a RAW-level
    # threshold, ~10501), so a black-subtracted 10000 never crossed it --
    # every saturated point was silently counted as "unclipped".
    black = 512.0
    white_level = 10512.0
    # A perfectly linear rate of 4000 DN/s for the genuinely unclipped
    # points, plus two points pinned exactly at saturation (raw mean ==
    # white_level, i.e. black-subtracted mean == white_level - black ==
    # 10000).
    points = [
        (1.0, 4000.0),
        (2.0, 8000.0),
        (2.4, 9600.0),
        (3.0, 10000.0),  # raw mean == white_level -- fully saturated
        (3.5, 10000.0),  # also fully saturated (pinned)
    ]
    result = linearity._analyze_channel(points, white_level, black)
    assert result["n_clipped"] == 2

    # The corrupted baseline this bug produced: with too few low-signal
    # points, analyze_linearity's fallback baseline is fit from
    # `unclipped_mask` alone -- if that mask wrongly includes the two
    # saturated points, the "linear" flux-rate baseline is pulled far off
    # the true rate (exactly 4000 DN/s, by construction, for the 3
    # genuinely unclipped points).
    assert result["flux_rate_dn_per_s"] == pytest.approx(4000.0, rel=0.01)


def test_analyze_linearity_reports_clipped_points_and_uncorrupted_full_well(monkeypatch):
    # End-to-end (not just the private helper): force the low-signal
    # fallback path by only sampling exposures that are already deep in the
    # sensor's range, several of them clipped. gain=2.0, black_dn=512.0,
    # full_well_e=10000.0 -> white_level = 512 + 10000/2 = 5512 DN.
    monkeypatch.setattr(linearity, "LINEARITY_LOW_SIGNAL_FRACTION", 0.0)
    black_dn, gain, full_well_e = 512.0, 2.0, 10000.0
    flux = 4000.0  # e-/s -> saturates at 10000/4000 = 2.5s
    exposures = [1.6, 1.9, 2.2, 2.5, 2.8, 3.1]  # last few are clipped
    frames = _flats(exposures, flux, black_dn=black_dn, gain=gain, full_well_e=full_well_e)

    a = linearity.analyze_linearity(frames, black_dn, gain_e_per_dn=gain)
    for _ch, data in a.result["channels"].items():
        assert data["n_clipped"] >= 1
        # full_well_e is read straight from white_level (never corrupted by
        # this bug), but the baseline flux rate must not be dragged down by
        # clipped points sneaking into the fallback baseline fit.
        assert data["flux_rate_dn_per_s"] == pytest.approx(flux / gain, rel=0.1)

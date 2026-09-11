import numpy as np
import pytest

from calsuite.display import analysis
from calsuite.display import constants as dc
from calsuite.display import patches as patchesmod
from calsuite.synth.display import DisplayModel, render_rolling_shutter_rows


def _ramp_data(model, channel, steps=17):
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    levels = [i / (steps - 1) for i in range(steps)]
    xyz = []
    for level in levels:
        rgb = [0.0, 0.0, 0.0]
        rgb[idx] = level
        xyz.append(list(model.measure(tuple(rgb))))
    return {"levels": levels, "xyz": xyz}


def test_trc_fit_recovers_known_gammas():
    model = DisplayModel(gamma={"r": 2.2, "g": 2.4, "b": 1.8}, black_luminance_cdm2=0.0)
    ramps = {ch: _ramp_data(model, ch) for ch in ("r", "g", "b")}
    result = analysis.trc_fit(ramps)
    assert result.ok
    assert result.result["effective_gamma"]["r"] == pytest.approx(2.2, abs=0.03)
    assert result.result["effective_gamma"]["g"] == pytest.approx(2.4, abs=0.03)
    assert result.result["effective_gamma"]["b"] == pytest.approx(1.8, abs=0.03)


def test_trc_fit_refuses_too_few_points():
    result = analysis.trc_fit({"r": {"levels": [0.0, 1.0], "xyz": [[0, 0, 0], [1, 1, 1]]}})
    assert not result.ok
    assert result.refusals[0].check == "trc_r_too_few_points"


def test_additivity_passes_for_additive_display():
    model = DisplayModel(white_boost_frac=0.0)
    black, r, g, b, w = (model.measure(c) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    result = analysis.additivity(black, r, g, b, w)
    assert result.result["additivity_de00"] < dc.ADDITIVITY_DE00_MAX
    assert result.result["recommend_lut"] is False


def test_additivity_fails_and_recommends_lut_for_nonadditive_display():
    model = DisplayModel(white_boost_frac=0.3)
    black, r, g, b, w = (model.measure(c) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    result = analysis.additivity(black, r, g, b, w)
    assert result.result["additivity_de00"] > dc.ADDITIVITY_DE00_MAX
    assert result.result["recommend_lut"] is True


def test_primaries_vs_edid_recovers_chromaticity():
    model = DisplayModel()
    measured = {
        "r": model.measure((1, 0, 0)), "g": model.measure((0, 1, 0)), "b": model.measure((0, 0, 1)),
        "w": model.measure((1, 1, 1)), "k": model.measure((0, 0, 0)),
    }
    edid_chromaticity = {**model.primaries_xy, "w": model.white_xy}
    result = analysis.primaries_vs_edid(measured, edid_chromaticity)
    for ch in ("r", "g", "b", "w"):
        mx, my = result.result["measured_chromaticity"][ch]
        ex, ey = edid_chromaticity[ch]
        assert mx == pytest.approx(ex, abs=1e-3)
        assert my == pytest.approx(ey, abs=1e-3)


def test_black_and_contrast():
    model = DisplayModel(white_luminance_cdm2=250.0, black_luminance_cdm2=0.25)
    black = model.measure((0, 0, 0))
    white = model.measure((1, 1, 1))
    result = analysis.black_and_contrast(black, white)
    assert result.ok
    assert result.result["contrast_ratio"] == pytest.approx(white[1] / black[1])


def test_black_and_contrast_refuses_nonpositive_black():
    result = analysis.black_and_contrast([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert not result.ok


def test_uniformity_recovers_falloff():
    model = DisplayModel(nonuniformity_amplitude=0.25)
    grid_patches = patchesmod.uniformity_grid(n=5)
    grid_xyz = [[list(model.measure(p.rgb, position=p.position)) for p in row] for row in grid_patches]
    result = analysis.uniformity(grid_xyz)
    assert result.result["luminance_uniformity_min_pct"] < 100.0
    assert result.result["luminance_uniformity_max_pct"] == pytest.approx(100.0, abs=1e-6)


def test_warmup_drift_recovers_stable_time():
    model = DisplayModel(warmup_tau_s=200.0, warmup_initial_frac=0.8)
    times = [i * 60.0 for i in range(31)]
    luminance = [model.measure((1, 1, 1), t_s=t)[1] for t in times]
    result = analysis.warmup_drift(times, luminance)
    assert result.ok
    assert result.result["stable_time_s"] is not None
    assert result.result["initial_fraction_of_final"] == pytest.approx(0.8, abs=0.05)


def test_warmup_drift_refuses_too_few_samples():
    result = analysis.warmup_drift([0.0], [1.0])
    assert not result.ok


def test_pwm_banding_recovers_known_frequency():
    model = DisplayModel(pwm_hz=1000.0, pwm_duty=0.5)
    row_period_s = 0.0002
    rows = render_rolling_shutter_rows(model, (1, 1, 1), n_rows=500, row_period_s=row_period_s, exposure_s=1.0 / 8000)
    result = analysis.pwm_banding(rows, row_period_s=row_period_s)
    assert result.result["detected"] is True
    assert result.result["cycles_per_row"] == pytest.approx(0.2, abs=1e-3)
    assert result.result["frequency_hz"] == pytest.approx(1000.0, rel=1e-2)


def test_pwm_banding_reports_none_without_pwm():
    model = DisplayModel(pwm_hz=None)
    rows = render_rolling_shutter_rows(model, (1, 1, 1), n_rows=200, row_period_s=0.0002, exposure_s=1.0 / 8000)
    rng = np.random.default_rng(0)
    rows = rows + rng.normal(0, rows.mean() * 1e-4, size=rows.shape)  # avoid a literally-zero-variance FFT
    result = analysis.pwm_banding(rows, row_period_s=0.0002)
    assert result.result["detected"] is False


def test_pwm_banding_refuses_too_few_rows():
    result = analysis.pwm_banding([1.0] * 4)
    assert not result.ok

import numpy as np
import pytest

from calsuite.synth.display import DisplayModel, render_rolling_shutter_rows


def test_white_point_chromaticity_recovered():
    model = DisplayModel(white_xy=(0.3127, 0.3290))
    xyz = model.measure((1.0, 1.0, 1.0))
    xy = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    assert xy[0] == pytest.approx(0.3127, abs=1e-3)
    assert xy[1] == pytest.approx(0.3290, abs=1e-3)


def test_primary_chromaticity_recovered_black_corrected():
    model = DisplayModel(black_luminance_cdm2=0.5)
    black = model.measure((0.0, 0.0, 0.0))
    red = model.measure((1.0, 0.0, 0.0)) - black
    xy = red[0] / red.sum(), red[1] / red.sum()
    assert xy[0] == pytest.approx(model.primaries_xy["r"][0], abs=1e-3)
    assert xy[1] == pytest.approx(model.primaries_xy["r"][1], abs=1e-3)


def test_gamma_shapes_the_ramp():
    model = DisplayModel(gamma={"r": 2.2, "g": 2.2, "b": 2.2}, black_luminance_cdm2=0.0)
    y_half = model.measure((0.5, 0.0, 0.0))[1]
    y_full = model.measure((1.0, 0.0, 0.0))[1]
    assert y_half / y_full == pytest.approx(0.5**2.2, rel=1e-3)


def test_white_boost_breaks_additivity():
    additive = DisplayModel(white_boost_frac=0.0)
    boosted = DisplayModel(white_boost_frac=0.2)
    black_a = additive.measure((0, 0, 0))
    r_a, g_a, b_a, w_a = (additive.measure(c) for c in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    summed_a = (r_a - black_a) + (g_a - black_a) + (b_a - black_a) + black_a
    assert w_a[1] == pytest.approx(summed_a[1], rel=1e-6)

    black_b = boosted.measure((0, 0, 0))
    r_b, g_b, b_b, w_b = (boosted.measure(c) for c in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    summed_b = (r_b - black_b) + (g_b - black_b) + (b_b - black_b) + black_b
    assert w_b[1] > summed_b[1] * 1.05  # the boost only shows up when all three channels are on


def test_uniformity_falloff_darkens_corners():
    model = DisplayModel(nonuniformity_amplitude=0.3)
    center = model.measure((1, 1, 1), position=(0.5, 0.5))
    corner = model.measure((1, 1, 1), position=(0.0, 0.0))
    assert corner[1] < center[1]


def test_warmup_rises_toward_final_value():
    model = DisplayModel(warmup_tau_s=100.0, warmup_initial_frac=0.8)
    y0 = model.measure((1, 1, 1), t_s=0.0)[1]
    y_late = model.measure((1, 1, 1), t_s=1000.0)[1]
    y_final = model.measure((1, 1, 1), t_s=None)[1]
    assert y0 < y_late
    assert y_late == pytest.approx(y_final, rel=1e-4)


def test_render_rolling_shutter_rows_bands_at_known_frequency():
    model = DisplayModel(pwm_hz=1000.0, pwm_duty=0.5)
    row_period_s = 0.0002  # 200 microseconds/row
    rows = render_rolling_shutter_rows(
        model, (1.0, 1.0, 1.0), n_rows=500, row_period_s=row_period_s, exposure_s=1.0 / 8000
    )
    detrended = rows - rows.mean()
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(500, d=1.0)
    spectrum[0] = 0.0
    peak_freq = freqs[int(np.argmax(spectrum))]
    assert peak_freq == pytest.approx(model.pwm_hz * row_period_s, abs=1e-3)


def test_render_rolling_shutter_rows_flat_with_no_pwm():
    model = DisplayModel(pwm_hz=None)
    rows = render_rolling_shutter_rows(model, (1.0, 1.0, 1.0), n_rows=64, row_period_s=0.0002, exposure_s=1.0 / 8000)
    assert np.allclose(rows, rows[0])

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


def test_trc_fit_refuses_pure_noise():
    """A ramp whose measured Y is uncorrelated with the drive level used to
    come back `ok=True` with a specific-looking but meaningless
    "effective_gamma" -- there was no goodness-of-fit check at all. A real
    display's tone response is a smooth near-power-law; noise like this
    shouldn't fit at all."""
    rng = np.random.default_rng(0)
    levels = np.linspace(0.0, 1.0, 17)
    noisy_y = rng.uniform(0.01, 0.02, size=17)
    ramps = {"r": {"levels": levels.tolist(), "xyz": [[0.1, float(y), 0.1] for y in noisy_y]}}
    result = analysis.trc_fit(ramps, black_y=0.0)
    assert not result.ok
    assert result.refusals[0].check == "trc_r_poor_fit"
    assert "r" not in result.result["effective_gamma"]


def test_trc_fit_r2_boundary_just_inside_and_outside():
    """Construct a ramp with a known r^2 by mixing a perfect power-law
    signal with noise, and check the refusal straddles `TRC_MIN_R2`."""
    levels = np.linspace(0.01, 0.99, 15)
    true_y = levels**2.2
    rng = np.random.default_rng(3)

    def _fit(noise_scale):
        y = true_y + rng.normal(0.0, noise_scale, size=true_y.shape)
        y = np.clip(y, 1e-6, None)
        full_levels = np.concatenate([[0.0], levels, [1.0]])
        full_y = np.concatenate([[0.0], y, [1.0]])
        ramps = {"g": {"levels": full_levels.tolist(), "xyz": [[0, float(v), 0] for v in full_y]}}
        return analysis.trc_fit(ramps, black_y=0.0)

    good = _fit(1e-4)
    assert good.ok
    assert good.residuals["gamma_fit_r2"]["g"] > dc.TRC_MIN_R2

    bad = _fit(0.5)
    assert not bad.ok
    assert bad.refusals[0].check == "trc_g_poor_fit"
    assert bad.refusals[0].value < dc.TRC_MIN_R2


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


def test_uniformity_refuses_a_single_point_grid():
    """A 1x1 "grid" compares the center cell against itself -- a
    mathematically guaranteed perfect-uniformity result no matter how
    non-uniform the real panel is. This used to come back `ok=True` with
    100%/0-ΔE00 "uniformity" for any single measurement handed in."""
    grid = [[[100.0, 50.0, 20.0]]]
    result = analysis.uniformity(grid)
    assert not result.ok
    assert result.refusals[0].check == "uniformity_too_few_points"


def test_uniformity_does_not_refuse_at_the_minimum_grid_size():
    n = dc.UNIFORMITY_MIN_N
    model = DisplayModel(nonuniformity_amplitude=0.1)
    grid_patches = patchesmod.uniformity_grid(n=n)
    grid_xyz = [[list(model.measure(p.rgb, position=p.position)) for p in row] for row in grid_patches]
    result = analysis.uniformity(grid_xyz)
    assert result.ok
    assert all(r.check != "uniformity_too_few_points" for r in result.refusals)


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
    """Regression guard for a real false-positive detector bug (CLAUDE.md's
    seed sweep found ~30% of seeds failed this): "peak FFT bin / median of
    the rest" is an extreme-value statistic, and for i.i.d. (no-PWM) row
    noise its *expected* value alone already sits close to what
    PWM_FFT_MIN_PROMINENCE used to be (3.0) -- not a fixture artifact, since
    the ratio is scale-invariant in the noise amplitude and this reproduces
    at real-camera row counts too (verified empirically for 100-10000
    rows). Seed 6 with the exact fixture below is a concrete, deterministic
    reproducer of the old bug (prominence ~3.41 against the old 3.0
    threshold); the loop below sweeps several more seeds so this doesn't
    silently regress to another near-coin-flip threshold later.
    """
    model = DisplayModel(pwm_hz=None)
    rows = render_rolling_shutter_rows(model, (1, 1, 1), n_rows=200, row_period_s=0.0002, exposure_s=1.0 / 8000)
    for seed in (6, 0, 1, 2, 3, 4, 5, 7, 8, 9):
        rng = np.random.default_rng(seed)
        noisy = rows + rng.normal(0, rows.mean() * 1e-4, size=rows.shape)  # avoid a literally-zero-variance FFT
        result = analysis.pwm_banding(noisy, row_period_s=0.0002)
        assert result.result["detected"] is False, f"seed {seed}: false PWM detection, prominence={result.result['prominence']}"


def test_pwm_banding_refuses_too_few_rows():
    result = analysis.pwm_banding([1.0] * 4)
    assert not result.ok


def test_pwm_banding_detects_a_noise_free_square_wave():
    """A clean square-wave banding signal -- the most extreme PWM a panel
    can produce -- puts exact zeros in most FFT bins, so the median of the
    non-peak bins is exactly 0.0. The detector used to require `floor > 0`,
    which read "infinitely prominent peak" as "nothing there" and called
    this flicker-free. The suite's own round-trip test only escaped that by
    floating-point luck (its floor lands at ~1e-14 rather than 0).
    """
    rows = [100.0 if (i // 4) % 2 == 0 else 50.0 for i in range(64)]  # period 8 rows
    result = analysis.pwm_banding(rows, row_period_s=0.0002)
    assert result.result["detected"] is True
    assert result.result["cycles_per_row"] == pytest.approx(0.125, abs=1e-6)
    assert result.result["frequency_hz"] == pytest.approx(625.0, rel=1e-6)
    # Unbounded ratio: reported as None rather than a made-up finite number.
    assert result.result["prominence"] is None


def test_uniformity_refuses_a_wrong_rank_grid_instead_of_crashing():
    """A 2-D (luminance-only) grid used to pass the `shape[:2] == (n, n)`
    half of the shape check and then raise IndexError on `shape[2]` -- the
    `uniformity_grid_shape` refusal could never fire for the likeliest
    wrong-shaped input."""
    result = analysis.uniformity([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert not result.ok
    assert result.refusals[0].check == "uniformity_grid_shape"


def test_uniformity_refuses_an_even_sized_grid():
    """`patches.uniformity_grid` spans the screen edge-to-edge, so only an
    odd n has a cell at the screen center. For an even n, grid[n//2][n//2]
    is an off-center cell silently reported as "the center" every other
    cell is compared against."""
    result = analysis.uniformity(np.ones((4, 4, 3)))
    assert not result.ok
    assert result.refusals[0].check == "uniformity_grid_even_n"


def test_trc_fit_drops_a_sub_black_step_instead_of_refusing_the_channel():
    """Near-black instrument noise is ordinary: on a 1000:1 panel the first
    blue ramp step sits ~0.04 cd/m^2 above black, so a reading a hundredth
    of a cd/m^2 below the measured black is well within a colorimeter's
    repeatability. That step used to be *clipped* to 1e-6 and kept in the
    log-log fit, where log(1e-6) = -13.8 against a fit whose whole log
    range is ~6 wrecked r^2 and refused the channel as `poor_fit` -- naming
    the wrong cause for one unusable reading. It is now dropped, counted,
    and the rest of the ramp still fits.
    """
    model = DisplayModel(gamma={"r": 2.2, "g": 2.2, "b": 2.2}, black_luminance_cdm2=0.25)
    black_y = float(model.measure((0, 0, 0))[1])
    ramps = {ch: _ramp_data(model, ch) for ch in ("r", "g", "b")}
    # One near-black blue step reads 0.01 cd/m^2 *below* the measured black.
    ramps["b"]["xyz"][1] = [0.0, black_y - 0.01, 0.0]

    result = analysis.trc_fit(ramps, black_y=black_y)
    assert result.ok, [r.check for r in result.refusals]
    assert result.result["effective_gamma"]["b"] == pytest.approx(2.2, abs=0.1)
    assert result.result["steps_at_or_below_black"]["b"] == 1
    assert result.result["steps_at_or_below_black"]["r"] == 0
    # The LUT keeps every step, floored at 0 -- never a negative entry.
    assert min(result.result["lut"]["b"]) >= 0.0


def test_vcgt_correction_inverts_measured_response_to_target_gamma():
    """Feeding `vcgt_correction`'s own curve value back into the display
    model should make the *actual* light output follow the canonical
    `x ** target_gamma` curve, regardless of the panel's real per-channel
    gamma -- that's the entire point of a video-LUT correction."""
    model = DisplayModel(gamma={"r": 1.8, "g": 2.4, "b": 2.0}, black_luminance_cdm2=0.0)
    ramps = {ch: _ramp_data(model, ch) for ch in ("r", "g", "b")}
    trc = analysis.trc_fit(ramps)
    assert trc.ok

    result = analysis.vcgt_correction(trc, target_gamma=2.2, steps=64)
    assert result.ok
    assert result.result["target_gamma"] == 2.2

    idx = {"r": 0, "g": 1, "b": 2}
    for ch, i in idx.items():
        y_max = model.measure(tuple(1.0 if j == i else 0.0 for j in range(3)))[1]
        for x, driven in zip(result.result["input_levels"], result.result["curves"][ch], strict=True):
            rgb = [0.0, 0.0, 0.0]
            rgb[i] = driven
            y_norm = model.measure(tuple(rgb))[1] / y_max
            assert y_norm == pytest.approx(x**2.2, abs=0.01)


def test_vcgt_correction_refuses_when_a_channel_trc_was_refused():
    """`trc_fit` leaves a refused channel out of `result["lut"]`/["levels"]
    entirely -- a VCGT curve missing one of three channels would be a wrong
    color cast on every non-color-managed pixel, not a smaller correction,
    so this refuses outright rather than building two-thirds of a curve."""
    trc = analysis.Analysis(result={
        "lut": {"r": [0.0, 0.5, 1.0], "g": [0.0, 0.5, 1.0]},
        "levels": {"r": [0.0, 0.5, 1.0], "g": [0.0, 0.5, 1.0]},
    })
    result = analysis.vcgt_correction(trc)
    assert not result.ok
    assert "b" in result.refusals[0].message

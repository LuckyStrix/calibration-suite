from dataclasses import replace

import numpy as np
import pytest

from calsuite.camera import chart
from calsuite.synth import color as synth_color

# A plausible, diagonally-dominant, easily-invertible raw->XYZ matrix used
# across the color-wave tests as "ground truth" -- not a real camera's
# response, just something round-trip tests can recover exactly.
KNOWN_MATRIX = np.array(
    [
        [0.60, 0.20, 0.15],
        [0.15, 0.75, 0.10],
        [0.10, 0.15, 0.65],
    ]
)

# Corners (x, y), full-sensor px, for a 4x6 chart inside the default
# synth.color sensor's 480x720 visible area (top_margin=8, left_margin=16 --
# synth.sensor.SensorModel's own defaults): comfortably inside the visible
# window on every side.
CORNERS = [(60.0, 60.0), (660.0, 60.0), (660.0, 420.0), (60.0, 420.0)]


def _render_good(reference, **kwargs):
    rng = np.random.default_rng(0)
    return synth_color.render_chart(reference, KNOWN_MATRIX, CORNERS, rng=rng, signal_scale=20000.0, **kwargs)


def test_reference_colorchecker_layout():
    ref = chart.reference_colorchecker()
    assert ref.rows == 4
    assert ref.cols == 6
    assert len(ref.patches) == 24
    assert len(ref.neutral_indices()) == 6
    white_name = ref.patches[ref.brightest_neutral_index()].name
    assert "white" in white_name.lower()


def test_reference_from_csv_round_trip(tmp_path):
    # L*a*b* input (a*=b*=0 is neutral *by construction*, unlike an XYZ
    # triplet with X==Y==Z, which is NOT generally neutral -- true neutral
    # XYZ must be proportional to the illuminant white's own XYZ, not have
    # equal components) -- this also exercises reference_from_csv's L*a*b*
    # column path, which the XYZ path doesn't.
    csv_path = tmp_path / "custom.csv"
    csv_path.write_text(
        "# rows=1\n# cols=2\n# illuminant=D65\nname,L,a,b\nneutral gray,50,0,0\nvivid red,50,40,30\n",
        encoding="utf-8",
    )
    ref = chart.reference_from_csv(csv_path)
    assert ref.rows == 1 and ref.cols == 2
    assert ref.illuminant_xy == pytest.approx((0.3127, 0.3290))
    assert ref.patches[0].name == "neutral gray"
    assert ref.patches[0].is_neutral
    assert not ref.patches[1].is_neutral


def test_sample_chart_recovers_labeled_grid():
    ref = chart.reference_colorchecker()
    frame = _render_good(ref)
    sample = chart.sample_chart(frame, CORNERS, rows=ref.rows, cols=ref.cols)
    assert len(sample.patches) == 24
    named = chart.label_patches(sample, ref)
    assert [p.name for p in named.patches] == [p.name for p in ref.patches]
    # every patch should have sampled a non-trivial number of pixels on
    # every plane -- a corner/homography bug would show up as some patches
    # landing 0 px.
    for p in named.patches:
        for ch in chart.CHANNELS:
            assert p.n_px[ch] > 0


def test_no_refusals_on_clean_chart():
    ref = chart.reference_colorchecker()
    frame = _render_good(ref)
    sample = chart.label_patches(chart.sample_chart(frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    refusals = chart.refusals_for_fit(sample, ref, "matrix")
    assert refusals == []


def test_glare_refusal_fires_on_glare_and_not_otherwise():
    ref = chart.reference_colorchecker()
    white_name = ref.patches[ref.brightest_neutral_index()].name

    glare_frame = _render_good(ref, glare_patch_names=[white_name], glare_extra_fraction=0.5)
    sample = chart.label_patches(chart.sample_chart(glare_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    refusals = chart.refusals_for_fit(sample, ref, "matrix")
    assert any(r.check == "glare" for r in refusals)

    clean_frame = _render_good(ref)
    clean_sample = chart.label_patches(chart.sample_chart(clean_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    assert not any(r.check == "glare" for r in chart.refusals_for_fit(clean_sample, ref, "matrix"))


def test_uneven_lighting_refusal_fires_on_gradient_and_not_otherwise():
    ref = chart.reference_colorchecker()

    grad_frame = _render_good(ref, lighting_gradient_fraction=0.6)
    sample = chart.label_patches(chart.sample_chart(grad_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    refusals = chart.refusals_for_fit(sample, ref, "matrix")
    assert any(r.check == "uneven_lighting" for r in refusals)

    clean_frame = _render_good(ref)
    clean_sample = chart.label_patches(chart.sample_chart(clean_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    assert not any(r.check == "uneven_lighting" for r in chart.refusals_for_fit(clean_sample, ref, "matrix"))


def test_clipping_refusal_fires_on_clipped_patch_and_not_otherwise():
    ref = chart.reference_colorchecker()
    white_name = ref.patches[ref.brightest_neutral_index()].name

    clipped_frame = _render_good(ref, clip_patch_names=[white_name])
    sample = chart.label_patches(chart.sample_chart(clipped_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    refusals = chart.refusals_for_fit(sample, ref, "matrix")
    assert any(r.check == "clipping" for r in refusals)

    clean_frame = _render_good(ref)
    clean_sample = chart.label_patches(chart.sample_chart(clean_frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)
    assert not any(r.check == "clipping" for r in chart.refusals_for_fit(clean_sample, ref, "matrix"))


def test_too_few_patches_refusal_depends_on_model():
    ref = chart.reference_colorchecker()  # 24 patches
    frame = _render_good(ref)
    sample = chart.label_patches(chart.sample_chart(frame, CORNERS, rows=ref.rows, cols=ref.cols), ref)

    assert not any(r.check == "too_few_patches" for r in chart.refusals_for_fit(sample, ref, "matrix"))
    assert not any(r.check == "too_few_patches" for r in chart.refusals_for_fit(sample, ref, "rp2"))
    # rp3 needs 52 patches (4 x 13 terms); 24 is not enough.
    assert any(r.check == "too_few_patches" for r in chart.refusals_for_fit(sample, ref, "rp3"))


def _uniform_chart_sample(reference, *, black=2048.0, white_signal=8000.0, cv=0.005, glare=None):
    """A ChartSample built directly (no rendering): every patch's raw DN is
    black + white_signal * its reference Y, with a within-patch std of
    `cv` * signal. `glare` = (patch_name, cv) overrides one patch's CV.
    """
    channels = ("R", "G1", "G2", "B")
    patches = []
    for i, ref_patch in enumerate(reference.patches):
        signal = white_signal * ref_patch.XYZ[1]
        patch_cv = glare[1] if glare and ref_patch.name == glare[0] else cv
        patches.append(
            chart.PatchSample(
                name=ref_patch.name,
                row=i // reference.cols,
                col=i % reference.cols,
                mean={ch: black + signal for ch in channels},
                std={ch: patch_cv * signal for ch in channels},
                clipped_fraction={ch: 0.0 for ch in channels},
                n_px={ch: 400 for ch in channels},
            )
        )
    return chart.ChartSample(
        rows=reference.rows,
        cols=reference.cols,
        patches=patches,
        corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
        black_level=(black,) * 4,
        white_level=16383.0,
        black_level_by_channel={ch: black for ch in channels},
    )


def test_glare_check_is_not_diluted_by_the_black_pedestal():
    """The glare CV used to be std/mean on *raw* DN, so the black pedestal
    (2048 of 16383 on the R100) diluted it by signal/(signal + black). The
    same physical hot spot therefore measured several times weaker on a
    dark neutral than on a bright one and slipped under the threshold. The
    CV is now of the black-subtracted signal.
    """
    ref = chart.reference_colorchecker()
    neutrals = [ref.patches[i].name for i in ref.neutral_indices()]
    mid_neutral = neutrals[2]  # bright enough to clear GLARE_MIN_SIGNAL_FRACTION

    hot_spot_cv = 0.06  # a real +50%-over-a-quarter-of-the-patch hot spot
    sample = _uniform_chart_sample(ref, glare=(mid_neutral, hot_spot_cv))
    refusals = chart.refusals_for_fit(sample, ref, "matrix")
    glare = [r for r in refusals if r.check == "glare"]
    assert glare, [r.check for r in refusals]
    assert glare[0].value == pytest.approx(hot_spot_cv, rel=1e-6)
    assert mid_neutral in glare[0].message

    # The *reported* number is now the patch's true non-uniformity. On raw
    # DN it was diluted by signal / (signal + black) -- here that
    # understates a 6% hot spot as 3.5%, and the dilution gets worse the
    # darker the neutral, which is where a hot spot is proportionally
    # largest.
    signal = 8000.0 * ref.patches[ref.neutral_indices()[2]].XYZ[1]
    assert hot_spot_cv * signal / (signal + 2048.0) < 0.65 * hot_spot_cv

    # A clean chart still passes...
    assert not [r for r in chart.refusals_for_fit(_uniform_chart_sample(ref), ref, "matrix") if r.check == "glare"]

    # ...and the darkest neutrals are deliberately out of scope rather than
    # silently under-tested: their own shot noise is a several-percent CV,
    # so no fixed ceiling can separate glare from noise down there
    # (GLARE_MIN_SIGNAL_FRACTION).
    darkest = _uniform_chart_sample(ref, glare=(neutrals[-1], hot_spot_cv))
    assert not [r for r in chart.refusals_for_fit(darkest, ref, "matrix") if r.check == "glare"]


def test_uneven_lighting_refusal_names_the_axes_it_could_actually_test():
    """On a ColorChecker24 all six neutrals are patches 18-23: one row, six
    columns. The row column of the design matrix is then constant and
    perfectly collinear with the intercept, so a top-to-bottom gradient
    contributes nothing to the fitted spread -- a light above or below the
    chart is undetectable from these patches, and no arrangement of the fit
    changes that. What the refusal can do is not imply a 2-D check it
    didn't make.
    """
    ref = chart.reference_colorchecker()
    sample = _uniform_chart_sample(ref)
    # A 40% left-to-right gradient, 5x UNEVEN_LIGHTING_GRADIENT_MAX.
    lit = []
    for p in sample.patches:
        scale = 1.0 - 0.4 * (p.col / (ref.cols - 1))
        lit.append(
            chart.PatchSample(
                name=p.name, row=p.row, col=p.col,
                mean={ch: 2048.0 + (p.mean[ch] - 2048.0) * scale for ch in p.mean},
                std=p.std, clipped_fraction=p.clipped_fraction, n_px=p.n_px,
            )
        )
    gradient_sample = replace(sample, patches=lit)

    refusals = chart.refusals_for_fit(gradient_sample, ref, "matrix")
    uneven = [r for r in refusals if r.check == "uneven_lighting"]
    assert uneven, [r.check for r in refusals]
    assert "columns" in uneven[0].message
    assert "rows" not in uneven[0].message

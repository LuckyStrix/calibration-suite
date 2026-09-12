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

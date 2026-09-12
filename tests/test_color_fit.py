import numpy as np
import pytest

from calsuite import provenance as prov
from calsuite import store
from calsuite.camera import chart, color

KNOWN_MATRIX = np.array(
    [
        [0.60, 0.20, 0.15],
        [0.15, 0.75, 0.10],
        [0.10, 0.15, 0.65],
    ]
)
CORNERS = [(60.0, 60.0), (660.0, 60.0), (660.0, 420.0), (60.0, 420.0)]
ILLUMINANT_XY = (0.3127, 0.3290)  # D65

# Round-trip DeltaE00 threshold for these tests: comfortably tighter than
# color_constants.VALIDATION_MEAN_DE00_MAX (a real-fit acceptance bar), since
# this data has NO chart-photography imperfections at all (no glare, no
# gradient, a fixed known matrix) -- only ordinary sensor read/shot noise.
# House rule 4's "true 0.30 fits to 0.2997" move: the recovered fit should
# look nearly exact, not merely "passable".
ROUND_TRIP_DE00_MAX = 2.5


def _render_and_sample(reference, rng_seed=0, **render_kwargs):
    from calsuite.synth import color as synth_color

    rng = np.random.default_rng(rng_seed)
    frame = synth_color.render_chart(reference, KNOWN_MATRIX, CORNERS, rng=rng, signal_scale=20000.0, **render_kwargs)
    sample = chart.sample_chart(frame, CORNERS, rows=reference.rows, cols=reference.cols)
    return chart.label_patches(sample, reference)


def test_known_matrix_recovered_matrix_model():
    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)

    analysis = color.fit(sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix")
    assert analysis.ok
    assert analysis.result["validation_passed"]
    assert analysis.uncertainty["delta_e00_validation_mean"] < ROUND_TRIP_DE00_MAX

    M = np.array(analysis.result["matrix_raw_to_xyz"])
    assert M.shape == (3, 3)


def test_white_preserving_holds_white_patch_close():
    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)
    white_idx = ref.brightest_neutral_index()
    white_name = ref.patches[white_idx].name

    analysis = color.fit(
        sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix", white_preserving=True
    )
    assert analysis.ok
    rgb = color.black_subtracted_rgb(sample)
    M = np.array(analysis.result["matrix_raw_to_xyz"])
    predicted = M @ rgb[white_idx]
    target = np.array(ref.patches[white_idx].XYZ)
    # White-preserving is a heavily-upweighted soft constraint (WHITE_PRESERVING_WEIGHT),
    # not a hard equality -- but at that weight it should land very close.
    assert predicted == pytest.approx(target, rel=0.02)
    assert white_name  # sanity: a name was actually resolved


def test_root_polynomial_model_still_reports_plain_3x3():
    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)

    analysis = color.fit(sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="rp2")
    assert analysis.ok
    M = np.array(analysis.result["matrix_raw_to_xyz"])
    assert M.shape == (3, 3)
    assert "root_polynomial" in analysis.result
    assert analysis.result["root_polynomial"]["degree"] == 2
    assert len(analysis.result["root_polynomial"]["matrix"][0]) == 6  # rp2 has 6 terms per output channel


def test_held_out_patches_validated_not_fit():
    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)
    held_out = [ref.patches[i].name for i in range(0, 24, 6)]  # every 6th patch

    analysis = color.fit(
        sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix", held_out_names=held_out
    )
    assert analysis.result["validation_method"] == "held_out"
    assert analysis.result["n_patches_held_out"] == len(held_out)
    assert set(analysis.uncertainty["delta_e00_validation_per_patch"]) == set(held_out)
    assert set(analysis.residuals["delta_e00_per_patch"]).isdisjoint(held_out)


def test_glare_still_produces_a_best_effort_fit_but_refused():
    ref = chart.reference_colorchecker()
    white_name = ref.patches[ref.brightest_neutral_index()].name
    sample = _render_and_sample(ref, glare_patch_names=[white_name], glare_extra_fraction=0.5)

    analysis = color.fit(sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix")
    assert not analysis.ok
    assert any(r.check == "glare" for r in analysis.refusals)
    # A refusal is a finding, not an omission (house rule 3) -- the record
    # still carries a real matrix, just marked refused.
    assert "matrix_raw_to_xyz" in analysis.result


def test_record_provenance_reflects_validation(tmp_path):
    # Policy (docs/implementation-plan.md Wave 3 fix list item 2, store.py's
    # docstring): provenance names the *method* -- a chart WAS photographed,
    # so this is "measured" -- regardless of whether validation passed;
    # trustworthiness lives entirely in `status`. A command always uses
    # provenance="measured" here; it's `status` that require_exportable
    # actually gates on.
    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)
    analysis = color.fit(sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix")
    assert analysis.ok
    assert analysis.result["validation_passed"]

    device = {"kind": "camera", "model": "calsuite-synthetic", "id": "calsuite-synthetic-unknown"}
    record = store.Record.from_analysis(
        kind="camera.color",
        device=device,
        analysis=analysis,
        provenance="measured",
        method={"name": "camera.color.fit", "calsuite_version": "0.1.0", "params": {}},
    )
    assert record.status == "ok"
    assert record.provenance == "measured"
    assert prov.is_exportable(record.provenance)

    st = store.Store(tmp_path)
    path = st.save(record)
    loaded = st.load(path)
    assert loaded.result["matrix_raw_to_xyz"] == record.result["matrix_raw_to_xyz"]
    store.require_exportable(loaded)  # should not raise


def test_failed_validation_refuses_even_with_no_other_quality_issue(monkeypatch):
    # A chart with no glare/gradient/clipping issue can still fail
    # validation (a genuinely bad fit, an unlucky noise draw, ...) -- that
    # must refuse the record (status="refused"), not just set a quiet
    # validation_passed=False that require_exportable() never looks at.
    # Thresholds are monkeypatched to an impossible bar (>= 0 DeltaE00)
    # rather than hunting for noise parameters that fail validation without
    # tripping some *other* refusal first.
    from calsuite.camera import color_constants as cc

    monkeypatch.setattr(cc, "VALIDATION_MEAN_DE00_MAX", -1.0)
    monkeypatch.setattr(cc, "VALIDATION_P95_DE00_MAX", -1.0)
    monkeypatch.setattr(cc, "VALIDATION_MAX_DE00_MAX", -1.0)

    ref = chart.reference_colorchecker()
    sample = _render_and_sample(ref)
    analysis = color.fit(sample, ref, illuminant_xy=ILLUMINANT_XY, illuminant_name="D65", model="matrix")

    assert not analysis.ok
    assert any(r.check == "validation_failed" for r in analysis.refusals)
    assert analysis.result["validation_method"] == "leave_one_out_linear_folds"
    assert not analysis.result["validation_passed"]

    record = store.Record.from_analysis(
        kind="camera.color",
        device={"kind": "camera", "model": "calsuite-synthetic", "id": "calsuite-synthetic-unknown"},
        analysis=analysis,
        provenance="measured",  # method-based: a chart was still photographed
        method={"name": "camera.color.fit", "calsuite_version": "0.1.0", "params": {}},
    )
    assert record.status == "refused"
    with pytest.raises(store.ExportRefused):
        store.require_exportable(record)

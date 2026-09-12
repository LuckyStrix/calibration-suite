from calsuite.camera import color_report
from calsuite.store import Record


def _record(result=None, residuals=None, uncertainty=None, status="ok", refusals=None):
    return Record(
        schema=1,
        id="camera.color-x",
        kind="camera.color",
        device={"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234", "firmware": ""},
        provenance="measured",
        status=status,
        refusals=refusals or [],
        result=result or {},
        residuals=residuals or {},
        uncertainty=uncertainty or {},
    )


def test_render_with_a_full_result_shows_matrix_and_smi():
    record = _record(
        result={
            "matrix_raw_to_xyz": [[0.6, 0.2, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.6]],
            "model": "matrix",
            "illuminant": "D65",
            "smi": 92.5,
            "mean_delta_e_ab": 1.234,
            "luther_ives_deviation": 0.001,
            "validation_method": "leave_one_out_linear_folds",
        },
        residuals={"delta_e00_per_patch": {"white": 0.5}},
        uncertainty={
            "delta_e00_validation_per_patch": {"white": 0.6},
            "delta_e00_validation_mean": 0.6,
            "delta_e00_validation_p95": 0.9,
            "delta_e00_validation_max": 1.1,
        },
    )
    html = color_report.render(record)
    assert html.startswith("<!doctype html>")
    assert "SMI = 92.5" in html
    assert "0.00100" in html  # luther-ives deviation, 5dp
    assert "leave_one_out_linear_folds" in html


def test_render_with_empty_result_dict_does_not_raise():
    # Regression guard: every optional numeric field this module displays
    # (SMI, mean dE*ab, Luther-Ives deviation) used to be spliced straight
    # into a `{:.1f}`/`{:.3f}`/`{:.5f}` format spec via a bare
    # `result.get(key, float('nan'))` -- safe only because the *default*
    # substitutes for an absent key, not for a key whose value is
    # genuinely `None`, which is exactly what a refused/partial record has.
    record = _record(result={"smi": None, "mean_delta_e_ab": None, "luther_ives_deviation": None})
    html = color_report.render(record)
    assert html.startswith("<!doctype html>")
    assert "not measured" in html


def test_render_with_refused_analysis_does_not_raise():
    record = _record(
        result={},
        status="refused",
        refusals=[{"check": "glare", "message": "glare on neutral patch", "value": 0.2, "threshold": 0.05}],
    )
    html = color_report.render(record)
    assert html.startswith("<!doctype html>")
    assert "Refused" in html
    assert "glare on neutral patch" in html


def test_render_matrix_with_a_none_cell_does_not_raise():
    record = _record(result={"matrix_raw_to_xyz": [[0.6, None, 0.1], [0.1, 0.7, 0.1], [0.1, 0.1, 0.6]]})
    html = color_report.render(record)
    assert html.startswith("<!doctype html>")
    assert "not measured" in html

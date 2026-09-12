from calsuite.camera import report
from calsuite.fit import Analysis
from calsuite.store import Record


def _record(kind, result, provenance="measured", status="ok", refusals=None):
    return Record(
        schema=1,
        id=f"{kind}-x",
        kind=kind,
        device={"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234", "firmware": ""},
        provenance=provenance,
        status=status,
        refusals=refusals or [],
        result=result,
    )


def test_report_contains_star_tracker_sentence_with_recommended_iso():
    iso_record = _record(
        "camera.iso",
        {"read_noise_e_by_iso": {100: 4.0, 800: 2.0}, "recommended_iso": 800, "dynamic_range_stops_by_iso": {}},
    )
    html = report.render_sensor_report(
        device={"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234"},
        iso_record=iso_record,
    )
    assert "is the invariance point; above it you're only losing headroom." in html
    assert "ISO 800" in html


def test_report_is_valid_looking_html_with_no_records():
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"})
    assert html.startswith("<!doctype html>")
    assert "is the invariance point; above it you're only losing headroom." in html


def test_report_includes_ptc_gain_table():
    ptc_record = _record(
        "camera.ptc",
        {
            "channels": {
                "R": {
                    "fit": {
                        "gain_e_per_dn": 2.5,
                        "read_noise_e": 3.0,
                        "n_levels_used": 10,
                        "signal_dn_used": [10.0, 20.0],
                        "var_diff_dn2_used": [5.0, 9.0],
                        "gain_uncertainty_e_per_dn": 0.05,
                        "read_noise_uncertainty_e": 0.1,
                    }
                }
            }
        },
    )
    html = report.render_sensor_report(
        device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, ptc_record=ptc_record
    )
    assert "2.5" in html
    assert "Error budget" in html


def test_report_shows_refusal_box_for_refused_record():
    a = Analysis()
    a.refuse("clipping", "too many clipped pixels")
    darks_record = _record("camera.darks", {"channels": {}}, status="refused", refusals=[r.to_dict() for r in a.refusals])
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, darks_record=darks_record)
    assert "Refused" in html
    assert "too many clipped pixels" in html


def test_report_with_empty_result_dicts_does_not_raise():
    # Regression guard: every section here is *present* (a record was
    # supplied) but carries an empty `.result` -- the shape a refusal that
    # returned before computing anything (or an older/partial record)
    # actually has. A bare f-string format spec on an absent value used to
    # raise TypeError; report.html.optional_number is what every numeric
    # field in this report now goes through instead.
    device = {"kind": "camera", "model": "Canon EOS R100", "id": "x"}
    html = report.render_sensor_report(
        device=device,
        bias_record=_record("camera.bias", {}),
        ptc_record=_record("camera.ptc", {}),
        linearity_record=_record("camera.linearity", {}),
        darks_record=_record("camera.darks", {}),
        iso_record=_record("camera.iso", {}),
        shutter_record=_record("camera.shutter", {}),
    )
    assert html.startswith("<!doctype html>")


def test_report_with_refused_ptc_record_does_not_raise():
    a = Analysis()
    a.refuse("too_few_levels", "not enough usable levels")
    ptc_record = _record("camera.ptc", {}, status="refused", refusals=[r.to_dict() for r in a.refusals])
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, ptc_record=ptc_record)
    assert "Refused" in html
    assert "not enough usable levels" in html


def test_report_survives_a_malformed_empty_fit_dict():
    # An empty `fit` dict (a hand-built or future-schema record missing
    # its own numeric keys) is treated the same as "no fit for this
    # channel" -- `_ptc_section` skips it (an empty dict is falsy) and
    # `_error_budget` used to reach straight for `fit["gain_e_per_dn"]`
    # with no None/missing-key guard at all, raising KeyError instead of
    # just contributing nothing to the error-budget table.
    ptc_record = _record("camera.ptc", {"channels": {"R": {"fit": {}}}})
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, ptc_record=ptc_record)
    assert html.startswith("<!doctype html>")

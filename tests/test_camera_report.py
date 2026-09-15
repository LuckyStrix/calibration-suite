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


def test_error_budget_all_zero_read_noise_reports_not_measured_instead_of_raising():
    # Regression guard for the crash CLAUDE.md's seed sweep found:
    # `read_noise_vals` all zero/falsy used to leave the `if v` filter in
    # `_error_budget`'s generator expression with nothing for `max()` to
    # look at, raising `ValueError: max() arg is an empty sequence`. This
    # is not a synthetic corner case -- `demo._run_sensor()` hits it on a
    # real fraction of random noise realizations, because a fit can
    # legitimately land on read_noise_e == 0.0 for a channel. Construct
    # the all-zero case directly rather than relying on a lucky seed.
    ptc_record = _record(
        "camera.ptc",
        {
            "channels": {
                "R": {
                    "fit": {
                        "gain_e_per_dn": 2.0,
                        "gain_uncertainty_e_per_dn": 0.05,
                        "read_noise_e": 0.0,
                        "read_noise_uncertainty_e": 0.1,
                        "signal_dn_used": [10.0],
                        "var_diff_dn2_used": [5.0],
                        "n_levels_used": 1,
                    }
                },
                "G1": {
                    "fit": {
                        "gain_e_per_dn": 2.1,
                        "gain_uncertainty_e_per_dn": 0.04,
                        "read_noise_e": 0.0,
                        "read_noise_uncertainty_e": 0.2,
                        "signal_dn_used": [10.0],
                        "var_diff_dn2_used": [5.0],
                        "n_levels_used": 1,
                    }
                },
            }
        },
    )
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, ptc_record=ptc_record)
    assert html.startswith("<!doctype html>")
    assert "read noise" in html
    assert "not measured" in html


def test_error_budget_all_zero_gain_reports_not_measured_instead_of_raising():
    # Same shape of bug, the other operand: `gain_vals` all zero used to
    # divide by zero inside `_error_budget`'s gain generator (that branch
    # has no `if v` filter at all in the original code), raising
    # `ZeroDivisionError` instead of degrading to "not measured".
    ptc_record = _record(
        "camera.ptc",
        {
            "channels": {
                "R": {
                    "fit": {
                        "gain_e_per_dn": 0.0,
                        "gain_uncertainty_e_per_dn": 0.05,
                        "read_noise_e": 3.0,
                        "read_noise_uncertainty_e": 0.1,
                        "signal_dn_used": [10.0],
                        "var_diff_dn2_used": [5.0],
                        "n_levels_used": 1,
                    }
                }
            }
        },
    )
    html = report.render_sensor_report(device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, ptc_record=ptc_record)
    assert html.startswith("<!doctype html>")
    assert "gain" in html
    assert "not measured" in html


def test_error_budget_still_computes_a_real_percentage_when_values_are_nonzero():
    # Non-regression check alongside the zero-denominator guards above:
    # a healthy fit (nonzero gain/read-noise values) must still produce a
    # real achieved percentage, not "not measured".
    entries = report._error_budget(
        Record(
            schema=1,
            id="camera.ptc-x",
            kind="camera.ptc",
            device={"kind": "camera", "model": "Canon EOS R100", "id": "x", "firmware": ""},
            provenance="measured",
            status="ok",
            refusals=[],
            result={
                "channels": {
                    "R": {
                        "fit": {
                            "gain_e_per_dn": 2.0,
                            "gain_uncertainty_e_per_dn": 0.1,
                            "read_noise_e": 3.0,
                            "read_noise_uncertainty_e": 0.3,
                        }
                    }
                }
            },
        ),
        None,
    )
    by_quantity = {e["quantity"]: e for e in entries}
    assert by_quantity["gain"]["achieved"] == 5.0
    assert by_quantity["read noise"]["achieved"] == 10.0


def test_fixed_pattern_section_renders_and_says_when_dsnu_is_unresolved():
    """`camera.fixed_pattern` records had no section in the report at all
    (and no command that could produce one)."""
    record = _record(
        "camera.fixed_pattern",
        result={
            "channels": {
                "R": {
                    "dsnu_std_dn": 1.25,
                    "dsnu_resolved": True,
                    "dsnu_temporal_floor_dn": 0.87,
                    "prnu_std_pct": 0.9,
                    "row_banding_peak_ratio": 2.1,
                    "col_banding_peak_ratio": 1.8,
                },
                "G1": {
                    "dsnu_std_dn": None,
                    "dsnu_resolved": False,
                    "dsnu_temporal_floor_dn": 0.91,
                    "prnu_std_pct": 1.0,
                    "row_banding_peak_ratio": 1.9,
                    "col_banding_peak_ratio": 1.7,
                },
            }
        },
    )
    html = report.render_sensor_report(
        device={"kind": "camera", "model": "Canon EOS R100", "id": "x"}, fixed_pattern_record=record
    )
    assert "Fixed pattern" in html
    assert "1.250 DN" in html
    assert "not resolved above the 0.910 DN temporal-noise floor" in html

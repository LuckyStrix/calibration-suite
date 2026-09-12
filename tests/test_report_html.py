import xml.etree.ElementTree as ET

from calsuite.report import html as reporthtml


def test_render_report_contains_key_elements():
    out = reporthtml.render_report(
        title="R100 Sensor Report",
        device={"model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234"},
        provenance="measured",
        status="ok",
        inputs=[{"name": "bias0001.cr3", "sha256": "deadbeef"}],
        error_budget=[{"quantity": "gain", "expected": 2.5, "achieved": 2.51, "unit": "%"}],
        sections=[{"heading": "Bias", "html": "<p>hello</p>"}],
    )
    assert out.startswith("<!doctype html>")
    assert "R100 Sensor Report" in out
    assert "Canon EOS R100" in out
    assert "deadbeef" in out
    assert "measured" in out
    assert "<p>hello</p>" in out
    assert "expected (estimate)" in out


def test_refusal_box_appears_only_when_refused():
    ok = reporthtml.render_report(title="t", device={"model": "m", "id": "i"}, provenance="measured", status="ok")
    assert "Refused" not in ok

    refused = reporthtml.render_report(
        title="t",
        device={"model": "m", "id": "i"},
        provenance="measured",
        status="refused",
        refusals=[{"check": "clipping", "message": "channel saturated", "value": 1.0, "threshold": 0.98}],
    )
    assert "Refused" in refused
    assert "channel saturated" in refused


def test_provenance_badges_are_visually_distinct():
    colors = {p: reporthtml.provenance_badge(p) for p in ("measured", "derived", "vendor", "nominal")}
    assert len(set(colors.values())) == 4


def test_escaping_in_device_model():
    out = reporthtml.render_report(
        title="t", device={"model": "<script>x</script>", "id": "i"}, provenance="measured", status="ok"
    )
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out


def test_inputs_table_is_well_formed_markup():
    out = reporthtml.render_report(
        title="t",
        device={"model": "m", "id": "i"},
        provenance="measured",
        status="ok",
        inputs=[{"name": "a.cr3", "sha256": "abc123"}],
    )
    table_html = out[out.index("<table>") : out.index("</table>") + len("</table>")]
    ET.fromstring(table_html)  # well-formed table markup, independent of the surrounding HTML5 page


def test_conditions_rendered_when_present():
    out = reporthtml.render_report(
        title="t",
        device={"model": "m", "id": "i"},
        provenance="measured",
        status="ok",
        conditions={"iso": 800, "aperture": "f/1.8"},
    )
    assert "iso" in out and "800" in out


def test_optional_number_formats_present_value():
    assert reporthtml.optional_number(1.23456) == "1.235"
    assert reporthtml.optional_number(0.5, "{:.0%}") == "50%"
    assert reporthtml.optional_number(2.0, unit="e-") == "2.000 e-"


def test_optional_number_none_is_not_measured():
    assert reporthtml.optional_number(None) == reporthtml.NOT_MEASURED
    assert reporthtml.optional_number(None, "{:.5f}", unit="px") == reporthtml.NOT_MEASURED


def test_optional_number_never_raises_on_a_bad_format_spec():
    # A value of the wrong type for its format spec must not crash a
    # report either -- falls back to plain str().
    assert reporthtml.optional_number("not a number", "{:.3f}") == "not a number"


def test_optional_text_none_is_not_measured():
    assert reporthtml.optional_text(None) == reporthtml.NOT_MEASURED
    assert reporthtml.optional_text(3, formatter=lambda v: f"#{v}") == "#3"


def test_error_budget_table_shows_not_measured_for_none_not_the_literal_none():
    out = reporthtml.error_budget_table([{"quantity": "gain", "expected": 2.5, "achieved": None, "unit": "%"}])
    assert "not measured" in out
    assert ">None<" not in out

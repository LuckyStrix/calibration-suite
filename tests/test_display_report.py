from calsuite.display import report as reportmod
from calsuite.fit import Analysis, Refusal


def test_render_includes_every_supplied_section():
    html = reportmod.render(
        title="Display report",
        device={"model": "CSOT T3", "id": "csot-t3-unknown"},
        provenance="measured",
        status="ok",
        trc_analysis=Analysis(result={"effective_gamma": {"r": 2.2}, "lut": {"r": [0.0, 1.0]}, "levels": {"r": [0.0, 1.0]}}),
        primaries_analysis=Analysis(
            result={
                "measured_chromaticity": {"r": (0.64, 0.33), "g": (0.3, 0.6), "b": (0.15, 0.06), "w": (0.31, 0.33)},
                "edid_chromaticity": {"r": (0.638, 0.334), "g": (0.3, 0.596), "b": (0.141, 0.058), "w": (0.312, 0.329)},
            }
        ),
        additivity_analysis=Analysis(result={"additivity_de00": 0.5, "threshold_de00": 2.0, "recommend_lut": False}),
        uniformity_analysis=Analysis(result={"luminance_pct_of_center": [[95, 100], [90, 100]], "de00_vs_center": [[1, 0], [2, 0]]}),
        warmup_analysis=Analysis(result={"stable_time_s": 300.0, "stable_fraction_threshold": 0.02}),
        pwm_analysis=Analysis(result={"detected": True, "cycles_per_row": 0.2, "frequency_hz": 1000.0}),
        validation_analysis=Analysis(result={"de00_mean": 1.0, "de00_p95": 2.0, "de00_max": 3.0}),
        backend_accuracy={"de00_estimate": 1.0, "basis": "test basis", "cross_checked_against": None},
    )
    assert "<title>Display report</title>" in html
    assert "Tone response curves" in html
    assert "Primaries and white vs. EDID" in html
    assert "Additivity" in html
    assert "Uniformity" in html
    assert "Warm-up drift" in html
    assert "PWM banding" in html
    assert "Validation" in html
    assert "Backend accuracy statement" in html
    assert "measured" in html  # provenance badge


def test_render_refusal_box_appears_when_refused():
    html = reportmod.render(
        title="Display report",
        device={"model": "x", "id": "y"},
        provenance="measured",
        status="refused",
        refusals=[{"check": "hdr_on", "message": "HDR is on", "value": True, "threshold": False}],
    )
    assert "Refused" in html
    assert "HDR is on" in html


def test_render_omits_sections_not_supplied():
    html = reportmod.render(title="Display report", device={"model": "x", "id": "y"}, provenance="nominal")
    assert "Tone response curves" not in html
    assert "Validation" not in html


def test_render_with_empty_result_dicts_does_not_raise():
    # Regression guard: a section whose *analysis was supplied* but whose
    # `.result` is empty (a refusal that returned before computing
    # anything, a step that was skipped, an older record missing a key)
    # used to crash formatting a bare `None` into `{de00:.3f}` --
    # report.html.optional_number is what every value below now goes
    # through. Every section is present here, deliberately with nothing in
    # its result, so this exercises every "optional value" branch at once.
    html = reportmod.render(
        title="Display report",
        device={"model": "x", "id": "y"},
        provenance="measured",
        status="ok",
        trc_analysis=Analysis(result={}),
        primaries_analysis=Analysis(result={}),
        additivity_analysis=Analysis(result={}),
        uniformity_analysis=Analysis(result={}),
        warmup_analysis=Analysis(result={}),
        pwm_analysis=Analysis(result={"detected": True}),  # detected with no cycles_per_row/frequency_hz
        validation_analysis=Analysis(result={}),
        backend_accuracy={},
    )
    assert "<html" in html
    assert "not measured" in html


def test_render_with_refused_analyses_does_not_raise():
    refusal = Refusal("x", "refused before computing anything")
    html = reportmod.render(
        title="Display report",
        device={"model": "x", "id": "y"},
        provenance="measured",
        status="refused",
        refusals=[refusal.to_dict()],
        additivity_analysis=Analysis(refusals=[refusal]),
        validation_analysis=Analysis(refusals=[refusal]),
    )
    assert "<html" in html
    assert "Refused" in html
    assert "not measured" in html

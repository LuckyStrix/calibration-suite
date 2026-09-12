"""Display HTML report (docs/design.md §5's report requirements): TRC
curves, a CIE xy chromaticity diagram (measured vs. EDID nominal),
additivity, uniformity heatmaps, warm-up drift, PWM banding, validation ΔE
bars, and the backend's own accuracy statement -- assembled from the
``fit.Analysis`` objects ``display/analysis.py``/``display/validate.py``
produce, using ``report.html.render_report`` for the page shell (provenance
badge, refusal box, inputs/error-budget tables) and ``report.svg`` for
every chart, exactly as every other area's report does.
"""

from __future__ import annotations

from calsuite.report import html as reporthtml
from calsuite.report import svg


def _chromaticity_diagram(measured_xy: dict, edid_xy: dict, *, width: int = 420, height: int = 420) -> str:
    """CIE xy chromaticity scatter/triangle: measured vs. EDID-nominal
    R/G/B gamut triangle. ``report.svg`` has no dedicated chromaticity-
    diagram primitive, so this reuses ``line_chart``'s axis/tick/legend
    machinery with ``mode="both"`` (lines connect each triangle's three
    primaries back to the first, points mark the vertices) rather than
    duplicating that code.
    """

    def _triangle(xy: dict, name: str) -> dict:
        order = ["r", "g", "b", "r"]  # repeats "r" to close the triangle
        return {"name": name, "x": [xy[c][0] for c in order], "y": [xy[c][1] for c in order]}

    series = [_triangle(measured_xy, "measured"), _triangle(edid_xy, "EDID nominal")]
    return svg.line_chart(
        series,
        width=width,
        height=height,
        title="CIE xy chromaticity: measured vs. EDID",
        x_label="x",
        y_label="y",
        mode="both",
    )


def build_sections(
    *,
    trc_analysis=None,
    primaries_analysis=None,
    additivity_analysis=None,
    uniformity_analysis=None,
    warmup_analysis=None,
    pwm_analysis=None,
    validation_analysis=None,
    backend_accuracy: dict | None = None,
) -> list:
    """Every argument is optional -- a report can cover any subset of
    these (a bare `display measure` run has no validation yet, a `display
    validate` run doesn't recompute TRC/uniformity), so each section only
    appears when its analysis was actually supplied.
    """
    sections = []

    if trc_analysis is not None:
        series = [
            {"name": ch, "x": trc_analysis.result.get("levels", {}).get(ch, []), "y": lut}
            for ch, lut in trc_analysis.result.get("lut", {}).items()
        ]
        gamma_items = "".join(
            f"<li>{ch}: effective gamma {g:.3f}</li>" for ch, g in trc_analysis.result.get("effective_gamma", {}).items()
        )
        chart = svg.line_chart(series, title="Tone response (normalized)", x_label="input level", y_label="Y / Y_max")
        sections.append({"heading": "Tone response curves", "html": f"<ul>{gamma_items}</ul>{chart}"})

    if primaries_analysis is not None:
        measured_xy = primaries_analysis.result.get("measured_chromaticity", {})
        edid_xy = primaries_analysis.result.get("edid_chromaticity", {})
        rows = [
            {
                "channel": ch,
                "measured": f"{measured_xy[ch][0]:.4f}, {measured_xy[ch][1]:.4f}" if ch in measured_xy else "-",
                "edid": f"{edid_xy[ch][0]:.4f}, {edid_xy[ch][1]:.4f}" if ch in edid_xy else "-",
            }
            for ch in ("r", "g", "b", "w")
        ]
        table = reporthtml.table(
            rows, [("channel", "channel"), ("measured", "measured (x, y)"), ("edid", "EDID nominal (x, y)")]
        )
        diagram = _chromaticity_diagram(measured_xy, edid_xy) if edid_xy and measured_xy else ""
        sections.append({"heading": "Primaries and white vs. EDID", "html": table + diagram})

    if additivity_analysis is not None:
        de00 = reporthtml.optional_number(additivity_analysis.result.get("additivity_de00"))
        threshold = reporthtml.optional_number(additivity_analysis.result.get("threshold_de00"))
        recommend_lut = additivity_analysis.result.get("recommend_lut")
        if recommend_lut is None:
            verdict = "not evaluated"
        elif recommend_lut:
            verdict = "recommends a LUT profile"
        else:
            verdict = "additive enough for a matrix/TRC profile"
        html = f"<p>Measured white vs. black-corrected R+G+B: ΔE00 = {de00} (threshold {threshold}) &mdash; {verdict}.</p>"
        sections.append({"heading": "Additivity (measured W vs. R+G+B)", "html": html})

    if uniformity_analysis is not None:
        lum = svg.heatmap(uniformity_analysis.result.get("luminance_pct_of_center", []), title="Luminance (% of center)")
        de = svg.heatmap(uniformity_analysis.result.get("de00_vs_center", []), title="ΔE00 vs. center")
        sections.append({"heading": "Uniformity", "html": lum + de})

    if warmup_analysis is not None:
        stable = warmup_analysis.result.get("stable_time_s")
        if stable is not None:
            fraction = reporthtml.optional_number(warmup_analysis.result.get("stable_fraction_threshold", 0.0), "{:.0%}")
            note = f"<p>Stable within {fraction} of final luminance after {stable:.0f}s.</p>"
        else:
            note = "<p>Luminance did not stabilize within the measured series.</p>"
        sections.append({"heading": "Warm-up drift", "html": note})

    if pwm_analysis is not None:
        if pwm_analysis.result.get("detected"):
            cycles = reporthtml.optional_number(pwm_analysis.result.get("cycles_per_row"), "{:.4f}")
            note = f"<p>PWM banding detected at {cycles} cycles/row"
            hz = pwm_analysis.result.get("frequency_hz")
            note += f" ({hz:.1f} Hz).</p>" if hz is not None else " (no readout time given, so no Hz figure).</p>"
        else:
            note = "<p>No PWM banding detected.</p>"
        sections.append({"heading": "Rolling-shutter PWM banding", "html": note})

    if validation_analysis is not None:
        r = validation_analysis.result
        # A validation that refused before computing anything (e.g. a
        # measured/target-count mismatch) has an *empty* result -- shown as
        # an explicit "not measured" paragraph rather than a bar chart with
        # a suspiciously perfect zero for a DeltaE00 that was never computed.
        de00_keys = ("de00_mean", "de00_p95", "de00_max")
        if any(r.get(k) is None for k in de00_keys):
            sections.append(
                {"heading": "Validation", "html": "<p>ΔE00 not measured (the validation step did not complete).</p>"}
            )
        else:
            bars = svg.bar_chart([("mean ΔE00", r["de00_mean"]), ("p95 ΔE00", r["de00_p95"]), ("max ΔE00", r["de00_max"])])
            sections.append({"heading": "Validation", "html": bars})

    if backend_accuracy is not None:
        cross = backend_accuracy.get("cross_checked_against") or "none"
        de00_estimate = reporthtml.optional_number(backend_accuracy.get("de00_estimate"))
        html = (
            f"<p>{backend_accuracy.get('basis', '')} &mdash; ΔE00 estimate "
            f"{de00_estimate}; cross-checked against: {cross}.</p>"
        )
        sections.append({"heading": "Backend accuracy statement", "html": html})

    return sections


def render(
    *,
    title: str,
    device: dict,
    provenance: str,
    status: str = "ok",
    refusals: list | None = None,
    conditions: dict | None = None,
    inputs: list | None = None,
    error_budget: list | None = None,
    **section_kwargs,
) -> str:
    """The full report page: the shared shell from ``report.html.
    render_report`` (provenance badge, refusal box, inputs/error-budget
    tables) plus this module's own sections. `section_kwargs` forwards to
    ``build_sections`` (``trc_analysis=``, ``primaries_analysis=``, ...).
    """
    sections = build_sections(**section_kwargs)
    return reporthtml.render_report(
        title=title,
        device=device,
        provenance=provenance,
        status=status,
        refusals=refusals,
        conditions=conditions,
        inputs=inputs,
        error_budget=error_budget,
        sections=sections,
    )

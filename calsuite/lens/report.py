"""Single self-contained HTML report per lens, built on the foundation's
``report.html``/``report.svg`` (never edited here -- Wave 2B doesn't own
``report/``). Sections are added only for whichever records are actually
available, so a report built before every measurement exists still renders
something useful.
"""

from __future__ import annotations

import numpy as np

from calsuite.constants import ESTIMATED_ACCURACY
from calsuite.lens import tca
from calsuite.lens.flats import fit_pa
from calsuite.report import html as reporthtml
from calsuite.report import svg


def _coverage_section(record) -> str:
    grid = record.residuals.get("coverage_grid")
    if not grid:
        return ""
    chart = svg.heatmap(grid, title="Corner coverage (radial x angular bins)", width=460, height=380)
    rms_rows = [
        {"name": n, "value": v}
        for n, v in zip(
            record.residuals.get("per_view_names", []), record.residuals.get("per_view_rms_px", []), strict=False
        )
    ]
    bar = svg.bar_chart([(r["name"], r["value"]) for r in rms_rows]) if rms_rows else ""
    return chart + "<h3>Reprojection RMS per image (px)</h3>" + bar


def _distortion_curve_section(record, vendor_comparison: dict | None) -> str:
    ptlens = record.result.get("ptlens")
    if not ptlens:
        return ""
    ru = np.linspace(0.0, 1.2, 60)
    ours = ru * (ptlens["a"] * ru**3 + ptlens["b"] * ru**2 + ptlens["c"] * ru + 1.0) - ru
    series = [{"name": "ours (ptlens)", "x": ru.tolist(), "y": ours.tolist()}]
    if vendor_comparison and vendor_comparison.get("available") and vendor_comparison["vendor"]["distortion"]:
        vd = vendor_comparison["vendor"]["distortion"][0]
        if vd.get("model") == "ptlens":
            a, b, c = float(vd["a"]), float(vd["b"]), float(vd["c"])
            vend = ru * (a * ru**3 + b * ru**2 + c * ru + 1.0) - ru
            series.append({"name": "mil-canon.xml (vendor)", "x": ru.tolist(), "y": vend.tolist()})
    return svg.line_chart(series, title="Distortion: Rd - Ru vs Ru (Hugin-normalized)", x_label="Ru", y_label="Rd - Ru")


def _tca_section(record) -> str:
    if "vr" not in record.result:
        return ""
    # tca.fit_tca fits kr/kb against *raw sensor-pixel* radii (a dimensionless
    # ratio, so the fit itself doesn't care about scale -- see tca.py's
    # module docstring), which can run to a couple thousand px at the sensor
    # corner. Plotting over a fixed 0-1.2 range (a Hugin-normalized-looking
    # span left over from distortion.py's convention, which this module does
    # not use) showed only the first ~0.1% of the field: kr/kb values a few
    # parts in a thousand from 1.0 produce an "offset" too small to see over
    # such a short radius, making real chromatic aberration look like zero.
    # Use the record's own image_size (tca.fit_tca now stores it) to plot out
    # to the actual corner radius; fall back to the old short range only for
    # an older/partial record that predates that field.
    image_size = record.result.get("image_size")
    r_max = tca.half_diagonal(image_size) if image_size else 1.2
    r = np.linspace(0.0, r_max, 40)
    kr, kb = record.result["kr"], record.result["kb"]
    series = [
        {"name": "R vs G", "x": r.tolist(), "y": (kr * r - r).tolist()},
        {"name": "B vs G", "x": r.tolist(), "y": (kb * r - r).tolist()},
    ]
    return svg.line_chart(series, title="Lateral chromatic aberration", x_label="radius (G, px)", y_label="offset (px)")


def _flats_label(record) -> str:
    """``f/2.8``, plus focal length / focus distance when the record has them,
    so two flats at the same aperture but a different zoom stay distinguishable."""
    c = record.conditions
    label = f"f/{c.get('aperture')}"
    if c.get("focal_mm"):
        label += f" @ {c['focal_mm']:g} mm"
    if c.get("focus_distance_m") is not None:
        label += f", {c['focus_distance_m']:g} m"
    return label


def _flats_section(records_by_condition: dict) -> str:
    if not records_by_condition:
        return ""
    r = np.linspace(0.0, 1.2, 60)
    series = []
    ordered = sorted(
        records_by_condition.values(),
        key=lambda rec: (
            rec.conditions.get("focal_mm") or 0.0,
            rec.conditions.get("aperture") or 0.0,
            rec.conditions.get("focus_distance_m") or 0.0,
        ),
    )
    for record in ordered:
        v_coeffs = record.result.get("v_coeffs")
        if not v_coeffs:
            continue
        pa = fit_pa(v_coeffs)
        v_of_r = 1.0 + pa["k1"] * r**2 + pa["k2"] * r**4 + pa["k3"] * r**6
        series.append({"name": _flats_label(record), "x": r.tolist(), "y": v_of_r.tolist()})
    if not series:
        return ""
    return svg.line_chart(series, title="Vignetting (pa model)", x_label="r (1 = corner)", y_label="V(r)")


def _mtf_section(record) -> str:
    grid = record.result.get("mtf50_lp_per_mm_map")
    if not grid:
        return ""
    return svg.heatmap(grid, title="MTF50 field map (lp/mm)")


def _psf_section(record) -> str:
    """A small hand-rolled inline SVG (report/svg.py has no ellipse
    primitive, and it's foundation code Wave 2B doesn't extend) showing
    each detected blob as an ellipse at its fitted orientation."""
    stars = record.result.get("stars")
    if not stars:
        return ""
    width, height = 480, 360
    xs = [s["x"] for s in stars]
    ys = [s["y"] for s in stars]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys) or 1
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = y1 if y1 > y0 else y0 + 1
    scale = min((width - 40) / (x1 - x0), (height - 40) / (y1 - y0))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    for s in stars:
        cx = 20 + (s["x"] - x0) * scale
        cy = 20 + (s["y"] - y0) * scale
        rx = max(2.0, s["sigma_major"] * scale)
        ry = max(2.0, s["sigma_minor"] * scale)
        color = "#dc2626" if s["orientation"] == "sagittal" else "#2563eb"
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'transform="rotate({s["orientation_deg"]:.1f} {cx:.1f} {cy:.1f})" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_lens_report(
    *,
    device: dict,
    distortion_record=None,
    tca_record=None,
    flats_records_by_condition: dict | None = None,
    mtf_record=None,
    psf_record=None,
    vendor_comparison: dict | None = None,
) -> str:
    """Assemble the lens report from whichever records are available.
    ``status``/``refusals``/``provenance`` are taken from ``distortion_record``
    when present (the "headline" measurement) so a refused distortion fit
    is visible at the top of the page; a lens with only, say, an MTF record
    still gets a report, just without that framing."""
    primary = distortion_record or tca_record or mtf_record or psf_record
    sections = []

    if distortion_record is not None:
        sections.append({"heading": "Coverage & reprojection", "html": _coverage_section(distortion_record)})
        sections.append(
            {"heading": "Distortion curve", "html": _distortion_curve_section(distortion_record, vendor_comparison)}
        )
    if tca_record is not None:
        sections.append({"heading": "Chromatic aberration", "html": _tca_section(tca_record)})
    if flats_records_by_condition:
        sections.append({"heading": "Vignetting", "html": _flats_section(flats_records_by_condition)})
    if mtf_record is not None:
        sections.append({"heading": "Sharpness (MTF50)", "html": _mtf_section(mtf_record)})
    if psf_record is not None:
        sections.append({"heading": "PSF field map", "html": _psf_section(psf_record)})

    achieved = {}
    if distortion_record is not None:
        achieved["distortion_rms_px"] = distortion_record.residuals.get("overall_rms_px")
    error_budget = [
        {"quantity": key, "expected": ESTIMATED_ACCURACY[key], "achieved": achieved.get(key, "-"), "unit": ""}
        for key in ("distortion_rms_px", "vignetting_pct", "mtf50_pct")
        if key in ESTIMATED_ACCURACY
    ]

    return reporthtml.render_report(
        title=f"Lens report: {device.get('model', '')}",
        device=device,
        provenance=getattr(primary, "provenance", "nominal") if primary else "nominal",
        status=getattr(primary, "status", "ok") if primary else "ok",
        refusals=getattr(primary, "refusals", []) if primary else [],
        sections=[s for s in sections if s["html"]],
        error_budget=error_budget,
    )

"""HTML report for a ``camera.color`` (or ``camera.ssf``) record --
``report.html.render_report`` supplies the page shell (provenance badge,
refusal box, inputs/error-budget tables); this module supplies the
color-specific sections: the fitted matrix, per-patch DeltaE00 bars (fit vs.
held-out/leave-one-out), a reference-vs-predicted swatch strip, and (Tier B
only) the Luther-Ives deviation and sensor metamerism index.

Takes a ``store.Record`` (already loaded/saved) plus, optionally, the
``chart.ReferenceChart`` used to fit it -- the record alone carries every
patch's *name* and DeltaE but not its XYZ, so the swatch section (which
needs to render an actual color) is skipped when no reference is given
rather than guessing.
"""

from __future__ import annotations

import html as _html

import numpy as np

from calsuite.camera import color_constants as cc
from calsuite.report import html as report_html
from calsuite.report import svg as report_svg


def _xyz_to_hex(xyz: np.ndarray, illuminant_xy: tuple) -> str:
    import colour

    rgb = colour.XYZ_to_sRGB(np.asarray(xyz, dtype=np.float64), illuminant=illuminant_xy)
    rgb = np.clip(rgb, 0.0, 1.0)
    r, g, b = (int(round(v * 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _swatch_strip(names: list, reference_hex: list, predicted_hex: list, *, width: int = 640) -> str:
    """Two rows of colored rectangles, reference on top and the matrix's
    own prediction underneath each patch -- a visual "does this look
    right" check no ΔE number alone gives you. Hand-built inline SVG (like
    ``report/svg.py``'s own style) since swatch grids aren't one of that
    module's chart types."""
    n = len(names)
    if n == 0:
        return ""
    cell_w = width / n
    cell_h = 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {2 * cell_h + 16}" '
        f'width="{width}" height="{2 * cell_h + 16}" role="img" aria-label="Reference vs predicted patch colors">'
    ]
    for i, (name, ref_hex, pred_hex) in enumerate(zip(names, reference_hex, predicted_hex, strict=True)):
        x = i * cell_w
        title = _html.escape(str(name), quote=True)
        parts.append(f'<rect x="{x:.1f}" y="0" width="{cell_w:.1f}" height="{cell_h}" fill="{ref_hex}"><title>{title} (reference)</title></rect>')
        parts.append(
            f'<rect x="{x:.1f}" y="{cell_h}" width="{cell_w:.1f}" height="{cell_h}" fill="{pred_hex}">'
            f"<title>{title} (predicted)</title></rect>"
        )
    parts.append(
        f'<text x="0" y="{2 * cell_h + 12}" font-size="11" fill="#374151">top row: reference &middot; '
        f"bottom row: this fit's prediction</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _matrix_table_html(matrix: list) -> str:
    if not matrix:
        return f"<p>{report_html.NOT_MEASURED}</p>"
    rows = "".join(
        "<tr>" + "".join(f"<td>{report_html.optional_number(v, '{:.5f}')}</td>" for v in row) + "</tr>" for row in matrix
    )
    return f"<table><tbody>{rows}</tbody></table>"


def render(record, reference=None) -> str:
    """``record``: a ``store.Record`` (or anything with the same attributes
    -- ``.kind``, ``.device``, ``.provenance``, ``.status``, ``.refusals``,
    ``.conditions``, ``.inputs``, ``.method``, ``.result``, ``.residuals``,
    ``.uncertainty``). ``reference``: the ``chart.ReferenceChart`` (or
    ``camera/ssf.py``'s reflectance-set patch names) used to fit it, for
    the swatch section -- optional."""
    result = record.result or {}
    residuals = record.residuals or {}
    uncertainty = record.uncertainty or {}
    sections = []

    if "matrix_raw_to_xyz" in result:
        model_line = f"model: {result.get('model', '?')} &middot; illuminant: {result.get('illuminant', '?')}"
        if result.get("white_preserving"):
            model_line += " &middot; white-preserving"
        sections.append(
            {
                "heading": "Fitted matrix (raw -> XYZ)",
                "html": f"<p>{model_line}</p>" + _matrix_table_html(result["matrix_raw_to_xyz"]),
            }
        )

    fit_de = residuals.get("delta_e00_per_patch") or residuals.get("delta_e00_per_sample")
    if fit_de:
        rows = sorted(fit_de.items(), key=lambda kv: -kv[1])
        sections.append({"heading": "DeltaE2000 on fit patches", "html": report_svg.bar_chart(rows)})

    val_de = uncertainty.get("delta_e00_validation_per_patch")
    if val_de:
        rows = sorted(val_de.items(), key=lambda kv: -kv[1])
        method = result.get("validation_method", "validation")
        heading = f"DeltaE2000 on held-out patches ({method})"
        html = report_svg.bar_chart(rows)
        if method == "leave_one_out_linear_folds":
            html = (
                "<p><em>Each fold below refits with a cheap linear (raw-XYZ-error) matrix, not the full "
                "DeltaE2000-refined matrix reported above -- a faithful but slightly more conservative "
                "generalization estimate, not an exact per-patch replay of the reported fit.</em></p>" + html
            )
        sections.append({"heading": heading, "html": html})

    if reference is not None and "matrix_raw_to_xyz" in result:
        illuminant_xy = tuple(result.get("illuminant_xy", (0.3127, 0.3290)))
        M = np.asarray(result["matrix_raw_to_xyz"], dtype=np.float64)
        names, ref_hex, pred_hex = [], [], []
        # `reference` gives each patch's true XYZ; without the original
        # sampled raw RGB (not stored on the record) the "predicted" row
        # can only show what the matrix does to the *reference's own*
        # XYZ run back through -- i.e. a self-consistency picture, not a
        # fresh camera reading. That's still useful (a badly-conditioned
        # matrix looks visibly wrong here even without new data).
        inv_m = np.linalg.pinv(M)
        for p in reference.patches:
            names.append(p.name)
            ref_hex.append(_xyz_to_hex(p.XYZ, illuminant_xy))
            raw_back = inv_m @ np.asarray(p.XYZ, dtype=np.float64)
            pred_xyz = M @ raw_back
            pred_hex.append(_xyz_to_hex(pred_xyz, illuminant_xy))
        sections.append({"heading": "Reference vs. predicted swatches", "html": _swatch_strip(names, ref_hex, pred_hex)})

    if "smi" in result or "mean_delta_e_ab" in result:
        smi = report_html.optional_number(result.get("smi"), "{:.1f}")
        mean_de_ab = report_html.optional_number(result.get("mean_delta_e_ab"))
        sections.append(
            {
                "heading": "Sensor metamerism index (ISO 17321-1)",
                "html": f"<p>SMI = {smi} (mean dE*ab over 18 chromatic patches: {mean_de_ab})</p>",
            }
        )
    if "luther_ives_deviation" in result:
        deviation = report_html.optional_number(result.get("luther_ives_deviation"), "{:.5f}")
        sections.append(
            {
                "heading": "Luther-Ives deviation",
                "html": f"<p>{deviation} (normalized RMS residual; 0 = exact linear recombination of the CIE CMFs)</p>",
            }
        )

    error_budget = []
    mean_de = uncertainty.get("delta_e00_validation_mean")
    if mean_de is not None:
        error_budget.append(
            {"quantity": "validation mean DeltaE2000", "expected": cc.VALIDATION_MEAN_DE00_MAX, "achieved": round(mean_de, 3)}
        )
    p95_de = uncertainty.get("delta_e00_validation_p95")
    if p95_de is not None:
        error_budget.append(
            {"quantity": "validation p95 DeltaE2000", "expected": cc.VALIDATION_P95_DE00_MAX, "achieved": round(p95_de, 3)}
        )
    max_de = uncertainty.get("delta_e00_validation_max")
    if max_de is not None:
        error_budget.append(
            {"quantity": "validation max DeltaE2000", "expected": cc.VALIDATION_MAX_DE00_MAX, "achieved": round(max_de, 3)}
        )

    title = f"{record.kind} report -- {record.device.get('model', '?')}"
    return report_html.render_report(
        title=title,
        device=record.device,
        provenance=record.provenance,
        status=record.status,
        refusals=record.refusals,
        conditions=record.conditions,
        inputs=record.inputs,
        error_budget=error_budget,
        sections=sections,
    )

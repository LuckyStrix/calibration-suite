"""Single self-contained HTML report page: inline CSS, no external
resources, no JS. Every area's report (camera bias/PTC/darks, lens
distortion/MTF, display measurement/validation, ...) funnels through
``render_report`` so every report in the suite looks and behaves the same:
a provenance badge, a red refusal box when the record's status is
"refused" (house rule 3), an inputs table with each input's SHA-256, and an
error-budget table comparing the design doc's stated accuracy estimate
(``constants.ESTIMATED_ACCURACY``) against what was actually achieved
(house rule 7: "accuracy is stated, not implied").
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone

# fg (text) / bg (chip) pair per provenance level -- distinguishable at a
# glance and roughly in "strength of evidence" order (green=strongest,
# gray=weakest), matching provenance.PROVENANCE's ordering.
PROVENANCE_STYLE = {
    "measured": ("#166534", "#dcfce7"),
    "derived": ("#1e40af", "#dbeafe"),
    "vendor": ("#92400e", "#fef3c7"),
    "nominal": ("#374151", "#e5e7eb"),
}


def _esc(text) -> str:
    return _html.escape(str(text), quote=True)


def provenance_badge(provenance: str) -> str:
    fg, bg = PROVENANCE_STYLE.get(provenance, PROVENANCE_STYLE["nominal"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'font-size:12px;font-weight:600;color:{fg};background:{bg};">{_esc(provenance)}</span>'
    )


def refusal_box(refusals: list) -> str:
    """Red box listing every refusal; empty string when there are none, so
    callers can always splice this in unconditionally."""
    if not refusals:
        return ""
    items = []
    for r in refusals:
        line = f'<li><strong>{_esc(r.get("check", ""))}</strong>: {_esc(r.get("message", ""))}'
        if r.get("threshold") is not None:
            line += f' (value={_esc(r.get("value"))}, threshold={_esc(r.get("threshold"))})'
        items.append(line + "</li>")
    return (
        '<div style="border:2px solid #dc2626;background:#fef2f2;border-radius:8px;'
        'padding:12px 16px;margin:16px 0;">'
        '<strong style="color:#991b1b;">Refused</strong>'
        f'<ul style="margin:8px 0 0 0;color:#7f1d1d;">{"".join(items)}</ul></div>'
    )


def table(rows: list, headers: list) -> str:
    """A plain HTML table. ``headers`` is an ordered list of
    ``(row_key, column_label)`` pairs; ``rows`` is a list of dicts."""
    thead = "".join(f"<th>{_esc(label)}</th>" for _, label in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{_esc(row.get(key, ''))}</td>" for key, _ in headers)
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{thead}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def inputs_table(inputs: list) -> str:
    return table(inputs, [("name", "input"), ("sha256", "SHA-256")])


def error_budget_table(entries: list) -> str:
    """``entries``: ``[{"quantity":.., "expected":.., "achieved":..,
    "unit":..}, ...]``. Shows the estimate and the achieved number side by
    side without editorializing about pass/fail -- the report states the
    numbers, the reader judges them (house rule 7)."""
    rows = [
        {
            "quantity": e.get("quantity", ""),
            "expected": f"{e.get('expected', '')} {e.get('unit', '')}".strip(),
            "achieved": f"{e.get('achieved', '')} {e.get('unit', '')}".strip(),
        }
        for e in entries
    ]
    return table(rows, [("quantity", "quantity"), ("expected", "expected (estimate)"), ("achieved", "achieved")])


_CSS = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 0; padding: 24px;
       background: #f9fafb; color: #111827; }
.report { max-width: 960px; margin: 0 auto; background: #ffffff; border: 1px solid #e5e7eb;
          border-radius: 10px; padding: 24px 28px; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
h2 { font-size: 16px; margin: 28px 0 8px 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
.device-header { color: #4b5563; font-size: 13px; margin-bottom: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px 0; font-size: 13px; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid #e5e7eb; }
th { color: #4b5563; font-weight: 600; }
.conditions { font-size: 13px; color: #374151; }
.conditions ul { margin: 4px 0 0 0; padding-left: 18px; }
.section { margin-top: 12px; }
"""


def render_report(
    *,
    title: str,
    device: dict,
    provenance: str,
    status: str = "ok",
    refusals: list | None = None,
    conditions: dict | None = None,
    inputs: list | None = None,
    error_budget: list | None = None,
    sections: list | None = None,
    generated_at: str | None = None,
) -> str:
    """Assemble a full HTML document string.

    ``sections`` is a list of ``{"heading": str, "html": str}``; callers
    build each section's inner HTML with ``report.svg``'s chart functions
    plus ``table()``, and this function only supplies the page shell (title,
    device header, provenance badge, refusal box, inputs/error-budget
    tables) so every report in the suite is visually consistent.
    """
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    conditions_html = ""
    if conditions:
        items = "".join(f"<li>{_esc(k)}: {_esc(v)}</li>" for k, v in conditions.items())
        conditions_html = f'<div class="conditions"><strong>Conditions</strong><ul>{items}</ul></div>'

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        '<div class="report">',
        f"<h1>{_esc(title)}</h1>",
        f'<div class="device-header">{_esc(device.get("model", ""))} '
        f'(id {_esc(device.get("id", ""))}) &middot; generated {_esc(generated_at)} &middot; '
        f"status {_esc(status)} &middot; {provenance_badge(provenance)}</div>",
        refusal_box(refusals or []),
        conditions_html,
    ]

    if inputs:
        parts.append("<h2>Inputs</h2>")
        parts.append(inputs_table(inputs))
    if error_budget:
        parts.append("<h2>Error budget</h2>")
        parts.append(error_budget_table(error_budget))
    for section in sections or []:
        parts.append(f'<h2>{_esc(section["heading"])}</h2>')
        parts.append(f'<div class="section">{section["html"]}</div>')

    parts.append("</div></body></html>")
    return "".join(parts)

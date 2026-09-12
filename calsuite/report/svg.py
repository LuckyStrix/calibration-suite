"""Pure-string SVG chart rendering -- no plotting library, no JavaScript.
Just enough for the HTML reports: line/scatter plots (optional log axes,
multiple series + legend), a labeled heatmap grid with a colorbar, and a
horizontal bar chart. Every text node is escaped; every coordinate is
written at fixed precision so output is deterministic.

Same "small hand-rolled SVG string builder" style as
CIS_Stockroom_Inventory_System's ``stockroom/reports.py:bar_chart`` --
presentation attributes (``fill=``, ``stroke=``) rather than CSS classes,
so a chart renders correctly even if pasted somewhere with no stylesheet.
report/html.py embeds these directly into a single self-contained page.
"""

from __future__ import annotations

import math

_MARGIN = {"top": 40, "right": 24, "bottom": 48, "left": 64}

# A qualitative palette with decent pairwise separation for up to 8 series.
# Beyond that, colors repeat with a dashed stroke (_series_style) rather
# than silently reusing an identical line style for two different series.
_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d")


def _escape(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v: float) -> str:
    return f"{v:.3g}"


def _series_style(index: int) -> tuple:
    color = _COLORS[index % len(_COLORS)]
    dash = "" if index < len(_COLORS) else "6,3"
    return color, dash


def _linear_scale(domain, rng):
    d0, d1 = domain
    r0, r1 = rng
    span = (d1 - d0) or 1.0

    def scale(v):
        return r0 + (v - d0) / span * (r1 - r0)

    return scale


def _log_scale(domain, rng):
    d0, d1 = domain
    if d0 <= 0 or d1 <= 0:
        raise ValueError("log scale requires a strictly positive domain")
    l0, l1 = math.log10(d0), math.log10(d1)
    r0, r1 = rng
    span = (l1 - l0) or 1.0

    def scale(v):
        if v <= 0:
            raise ValueError(f"log scale cannot plot non-positive value {v}")
        return r0 + (math.log10(v) - l0) / span * (r1 - r0)

    return scale


def _nice_ticks(lo, hi, n=5):
    """A handful of roughly-evenly-spaced linear tick values across
    [lo, hi] -- not a full "nice numbers" algorithm, just enough to label
    an axis without cluttering it."""
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(1, n - 1)
    return [lo + i * step for i in range(n)]


def _log_ticks(lo, hi, n=5):
    if lo <= 0 or hi <= 0:
        raise ValueError("log ticks require a strictly positive range")
    if hi == lo:
        return [lo]
    ratio = hi / lo
    return [lo * ratio ** (i / (n - 1)) for i in range(n)]


def _svg_open(width, height, title) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_escape(title or "chart")}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
    )


def line_chart(
    series: list,
    *,
    width: int = 640,
    height: int = 400,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    log_x: bool = False,
    log_y: bool = False,
    mode: str = "line",
) -> str:
    """``series``: ``[{"name": str, "x": [...], "y": [...]}]`` (equal-length
    x/y per series). Multiple series share one pair of axes and get a
    legend. ``mode`` is ``"line"``, ``"scatter"``, or ``"both"``.

    A point whose ``x`` or ``y`` is ``None`` (an optional per-point value a
    partial or refused record doesn't have for every sample) is dropped
    rather than plotted or left to blow up the axis-bounds computation --
    ``min()``/``max()`` over a list containing ``None`` raises ``TypeError``
    the moment Python tries to compare it against a real number.
    """
    series = [
        {
            **s,
            "x": [x for x, y in zip(s.get("x", []), s.get("y", []), strict=True) if x is not None and y is not None],
            "y": [y for x, y in zip(s.get("x", []), s.get("y", []), strict=True) if x is not None and y is not None],
        }
        for s in series
    ]
    if not series or not any(s["x"] for s in series):
        return _svg_open(width, height, title) + "</svg>"

    all_x = [v for s in series for v in s["x"]]
    all_y = [v for s in series for v in s["y"]]
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    if x0 == x1:
        x0, x1 = (x0 * 0.9, x1 * 1.1) if log_x else (x0 - 1, x1 + 1)
    if y0 == y1:
        y0, y1 = (y0 * 0.9, y1 * 1.1) if log_y else (y0 - 1, y1 + 1)

    plot_left, plot_top = _MARGIN["left"], _MARGIN["top"]
    plot_right, plot_bottom = width - _MARGIN["right"], height - _MARGIN["bottom"]

    sx = _log_scale((x0, x1), (plot_left, plot_right)) if log_x else _linear_scale((x0, x1), (plot_left, plot_right))
    sy = _log_scale((y0, y1), (plot_bottom, plot_top)) if log_y else _linear_scale((y0, y1), (plot_bottom, plot_top))

    parts = [_svg_open(width, height, title)]
    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-size="15" '
            f'font-weight="600" fill="#111827">{_escape(title)}</text>'
        )

    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#9ca3af"/>')

    x_ticks = _log_ticks(x0, x1) if log_x else _nice_ticks(x0, x1)
    for tv in x_ticks:
        px = sx(tv)
        parts.append(f'<line x1="{px:.2f}" y1="{plot_bottom}" x2="{px:.2f}" y2="{plot_bottom + 5}" stroke="#9ca3af"/>')
        parts.append(
            f'<text x="{px:.2f}" y="{plot_bottom + 18}" font-size="10" text-anchor="middle" '
            f'fill="#4b5563">{_escape(_fmt(tv))}</text>'
        )
    y_ticks = _log_ticks(y0, y1) if log_y else _nice_ticks(y0, y1)
    for tv in y_ticks:
        py = sy(tv)
        parts.append(f'<line x1="{plot_left - 5}" y1="{py:.2f}" x2="{plot_left}" y2="{py:.2f}" stroke="#9ca3af"/>')
        parts.append(
            f'<text x="{plot_left - 8}" y="{py + 3:.2f}" font-size="10" text-anchor="end" '
            f'fill="#4b5563">{_escape(_fmt(tv))}</text>'
        )

    if x_label:
        parts.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 8}" font-size="11" '
            f'text-anchor="middle" fill="#374151">{_escape(x_label)}</text>'
        )
    if y_label:
        mid_y = (plot_top + plot_bottom) / 2
        parts.append(
            f'<text x="14" y="{mid_y:.1f}" font-size="11" text-anchor="middle" fill="#374151" '
            f'transform="rotate(-90 14 {mid_y:.1f})">{_escape(y_label)}</text>'
        )

    for i, s in enumerate(series):
        color, dash = _series_style(i)
        pts = [(sx(x), sy(y)) for x, y in zip(s["x"], s["y"], strict=True)]
        if mode in ("line", "both") and len(pts) > 1:
            path = "M " + " L ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>')
        if mode in ("scatter", "both"):
            for px, py in pts:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}" fill-opacity="0.85"/>')

    if len(series) > 1:
        lx, ly = plot_left + 8, plot_top + 4
        for i, s in enumerate(series):
            color, _ = _series_style(i)
            yy = ly + i * 15
            parts.append(f'<rect x="{lx}" y="{yy}" width="10" height="10" fill="{color}"/>')
            parts.append(f'<text x="{lx + 14}" y="{yy + 9}" font-size="10" fill="#374151">{_escape(s.get("name", ""))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def scatter_chart(series: list, **kwargs) -> str:
    kwargs.setdefault("mode", "scatter")
    return line_chart(series, **kwargs)


def _ramp_color(t: float) -> str:
    """Blue -> white -> red diverging ramp, ``t`` in [0, 1]. Good enough
    for "which cells are high/low" at report-reading distance -- not a
    perceptually-uniform colormap, and not meant to be one."""
    t = min(1.0, max(0.0, t))
    if t < 0.5:
        u = t / 0.5
        r, g, b = 33 + u * (255 - 33), 102 + u * (255 - 102), 172 + u * (255 - 172)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 255 + u * (178 - 255), 255 + u * (24 - 255), 255 + u * (43 - 255)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def heatmap(
    grid,
    *,
    width: int = 480,
    height: int = 420,
    title: str = "",
    colorbar: bool = True,
    value_range: tuple | None = None,
) -> str:
    """``grid``: a 2D sequence of floats (row-major), where a cell may be
    ``None`` (e.g. ``lens.mtf_field_grid``'s map: a field-grid cell whose
    own edge check failed is recorded as ``None`` rather than dropping the
    whole map, per that function's own docstring) -- a ``None`` cell is
    excluded from the auto min/max and drawn as a flat "no data" gray
    rather than colored, instead of raising when it's compared against a
    number. ``value_range`` overrides the auto min/max -- pass it to keep
    two heatmaps' colors comparable (e.g. PRNU maps from two different
    ISOs)."""
    rows = [list(r) for r in grid]
    nrows = len(rows)
    ncols = len(rows[0]) if nrows else 0
    flat = [v for row in rows for v in row if v is not None]
    lo, hi = value_range if value_range is not None else ((min(flat), max(flat)) if flat else (0.0, 1.0))
    if hi == lo:
        hi = lo + 1.0

    margin_right = 70 if colorbar else 20
    margin_left, margin_top, margin_bottom = 50, 40, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    cell_w = plot_w / max(1, ncols)
    cell_h = plot_h / max(1, nrows)

    parts = [_svg_open(width, height, title)]
    if title:
        parts.append(
            f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" font-size="14" '
            f'font-weight="600" fill="#111827">{_escape(title)}</text>'
        )
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            x, y = margin_left + c * cell_w, margin_top + r * cell_h
            if v is None:
                fill = "#d1d5db"  # neutral gray -- "no data for this cell", not a colored (0 = blue) value
                title = "<title>no data</title>"
            else:
                t = min(1.0, max(0.0, (v - lo) / (hi - lo)))
                fill = _ramp_color(t)
                title = ""
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}">{title}</rect>'
            )

    if colorbar:
        bar_x = width - margin_right + 16
        bar_top, bar_bottom = margin_top, margin_top + plot_h
        steps = 40
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            y0 = bar_bottom - t0 * (bar_bottom - bar_top)
            y1 = bar_bottom - t1 * (bar_bottom - bar_top)
            parts.append(f'<rect x="{bar_x}" y="{y1:.2f}" width="14" height="{(y0 - y1) + 0.5:.2f}" fill="{_ramp_color((t0 + t1) / 2)}"/>')
        parts.append(f'<text x="{bar_x + 18}" y="{bar_top + 4}" font-size="10" fill="#374151">{_escape(_fmt(hi))}</text>')
        parts.append(f'<text x="{bar_x + 18}" y="{bar_bottom}" font-size="10" fill="#374151">{_escape(_fmt(lo))}</text>')

    parts.append("</svg>")
    return "".join(parts)


def bar_chart(rows: list, *, width: int = 520, label_width: int = 170) -> str:
    """``rows``: ``[(label, value), ...]`` or ``[(label, value,
    display_str), ...]``. Horizontal bars in the given order -- sort
    before calling if a particular order matters. Same shape as
    ``CIS_Stockroom_Inventory_System/src/stockroom/reports.py``'s
    ``bar_chart``.

    ``value`` may be ``None`` (an optional quantity a refused/partial
    record doesn't have) -- drawn as a zero-length bar labeled "not
    measured" rather than raising on ``float(None)``.
    """
    if not rows:
        return ""
    norm = []
    for row in rows:
        if len(row) == 3:
            label, value, display = row
        else:
            label, value = row
            display = None
        if value is None:
            norm.append((label, 0.0, display if display is not None else "not measured"))
            continue
        norm.append((label, float(value), display if display is not None else _fmt(value)))

    bar_height, gap, top = 22, 8, 6
    height = top * 2 + len(norm) * (bar_height + gap) - gap
    plot_width = max(60, width - label_width - 60)
    biggest = max((v for _, v, _ in norm), default=0) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="Bar chart of {len(norm)} values">'
    ]
    for i, (label, value, display) in enumerate(norm):
        y = top + i * (bar_height + gap)
        length = max(1, round(plot_width * value / biggest))
        parts.append(
            f'<text x="0" y="{y + 15}" font-size="11" fill="#111827">{_escape(label)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{length}" height="{bar_height}" rx="3" fill="#2563eb"/>'
            f'<text x="{label_width + length + 8}" y="{y + 15}" font-size="11" fill="#111827">{_escape(display)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)

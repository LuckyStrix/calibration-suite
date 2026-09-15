"""The R100 sensor report (docs/design.md §3.1 "Deliverable"): one
self-contained HTML page built from whichever of this area's records are
available, via ``report.html.render_report`` + ``report.svg``. Every section
is optional -- a report can be rendered from a partial set of records (e.g.
before darks have been shot), and simply omits the sections it has no data
for.
"""

from __future__ import annotations

from calsuite.constants import ESTIMATED_ACCURACY
from calsuite.report import html as report_html
from calsuite.report import svg

STAR_TRACKER_SENTENCE_TEMPLATE = (
    "ISO {iso} is the invariance point; above it you're only losing headroom."
)
# The exact sentence docs/design.md §3.1 asks for, for the star tracker
# README, with the recommended ISO (camera.iso's result) substituted in.

CHANNEL_ORDER = ("R", "G1", "G2", "B")


def _ptc_section(record) -> dict | None:
    if record is None:
        return None
    channels = record.result.get("channels", {})
    series = []
    for ch in CHANNEL_ORDER:
        fit = channels.get(ch, {}).get("fit")
        if not fit:
            continue
        series.append({"name": ch, "x": fit["signal_dn_used"], "y": fit["var_diff_dn2_used"]})
    chart = svg.line_chart(
        series,
        title="Photon transfer curve",
        x_label="signal (DN, black-subtracted)",
        y_label="Var(A-B)/2 (DN^2)",
        log_x=True,
        log_y=True,
        mode="scatter",
    ) if series else ""

    rows = []
    for ch in CHANNEL_ORDER:
        fit = channels.get(ch, {}).get("fit")
        if not fit:
            continue
        rows.append(
            {
                "channel": ch,
                "gain": report_html.optional_number(fit.get("gain_e_per_dn")),
                "read_noise_e": report_html.optional_number(fit.get("read_noise_e")),
                "n_levels_used": report_html.optional_text(fit.get("n_levels_used")),
            }
        )
    table = report_html.table(rows, [("channel", "channel"), ("gain", "gain (e-/DN)"), ("read_noise_e", "read noise (e-)"), ("n_levels_used", "levels used")])
    return {"heading": "Photon transfer curve", "html": chart + table}


def _read_noise_vs_iso_section(iso_record) -> dict | None:
    if iso_record is None:
        return None
    by_iso = iso_record.result.get("read_noise_e_by_iso", {})
    if not by_iso:
        return None
    isos = sorted(by_iso, key=lambda k: float(k))
    chart = svg.line_chart(
        [{"name": "read noise", "x": [float(i) for i in isos], "y": [by_iso[i] for i in isos]}],
        title="Input-referred read noise vs. ISO",
        x_label="ISO",
        y_label="read noise (e-)",
        log_x=True,
        mode="both",
    )
    recommended = iso_record.result.get("recommended_iso")
    note = f"<p>Recommended ISO: <strong>{recommended}</strong></p>" if recommended is not None else ""
    return {"heading": "ISO invariance", "html": chart + note}


def _linearity_section(record) -> dict | None:
    if record is None:
        return None
    channels = record.result.get("channels", {})
    series = []
    for ch in CHANNEL_ORDER:
        data = channels.get(ch)
        if not data:
            continue
        series.append({"name": ch, "x": data["exposure_s"], "y": data["deviation_pct"]})
    if not series:
        return None
    chart = svg.line_chart(
        series,
        title="Linearity residuals",
        x_label="exposure (s)",
        y_label="deviation from baseline (%)",
        mode="both",
    )
    return {"heading": "Linearity", "html": chart}


def _darks_section(record) -> dict | None:
    if record is None:
        return None
    channels = record.result.get("channels", {})
    series = []
    for ch in CHANNEL_ORDER:
        bins = channels.get(ch, {}).get("temp_bins", {})
        if not bins:
            continue
        temps = sorted(bins, key=lambda t: float(t))
        series.append({"name": ch, "x": [float(t) for t in temps], "y": [bins[t]["dark_current_e_per_s"] for t in temps]})
    chart = (
        svg.line_chart(series, title="Dark current vs. temperature", x_label="sensor temp (C)", y_label="e-/s", log_y=True, mode="both")
        if series
        else ""
    )

    hot = record.result.get("hot_pixels_by_exposure_s", {})
    bar_rows = [(f"{exp}s", info["count"]) for exp, info in sorted(hot.items(), key=lambda kv: float(kv[0]))]
    bars = svg.bar_chart(bar_rows) if bar_rows else ""
    return {"heading": "Dark current + hot pixels", "html": chart + bars}


def _fixed_pattern_section(record) -> dict | None:
    """DSNU / PRNU / banding (design §3.1). `camera.fixed_pattern` records
    had no section here at all -- the analysis wasn't reachable from the
    CLI either, so nothing was being left out in practice; both halves of
    that gap are closed together."""
    if record is None:
        return None
    channels = record.result.get("channels", {})
    if not channels:
        return None
    rows = []
    for ch in CHANNEL_ORDER:
        data = channels.get(ch)
        if not data:
            continue
        dsnu = data.get("dsnu_std_dn")
        floor = data.get("dsnu_temporal_floor_dn")
        dsnu_text = (
            f"{dsnu:.3f} DN"
            if dsnu is not None
            else f"not resolved above the {floor:.3f} DN temporal-noise floor"
            if floor is not None
            else "not resolved"
        )
        # A banding ratio is None when its noise floor is exactly zero (an
        # unbounded ratio, which a record can't carry as a number).
        row_band = report_html.optional_number(data.get("row_banding_peak_ratio"), "{:.1f}")
        col_band = report_html.optional_number(data.get("col_banding_peak_ratio"), "{:.1f}")
        prnu = report_html.optional_number(data.get("prnu_std_pct"), "{:.2f}")
        rows.append(
            f"<tr><td>{ch}</td><td>{dsnu_text}</td><td>{prnu}%</td>"
            f"<td>{row_band}</td><td>{col_band}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>channel</th><th>DSNU</th><th>PRNU</th>"
        "<th>row banding peak/median</th><th>col banding peak/median</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    note = (
        "<p><em>DSNU is reported only when the pattern clears the temporal-noise floor that stacking N darks "
        "leaves behind (var/N), which a raw stacked-image std would otherwise report as the pattern "
        "itself.</em></p>"
    )
    return {"heading": "Fixed pattern (DSNU / PRNU / banding)", "html": table + note}


def _max_ratio_pct(numerators: list, denominators: list) -> float | None:
    """``max(n / d) * 100`` over paired numerators/denominators, skipping
    any pair whose denominator is zero -- a zero gain/read-noise value is a
    legitimate (if degenerate) fit result, not a schema gap, and it
    surfaces on a real fraction of noise realizations, not just hand-built
    test data. Returns ``None`` -- rendered as
    :data:`report.html.NOT_MEASURED` by ``error_budget_table`` -- when that
    leaves no valid pair (every denominator zero/absent, or the inputs were
    empty to begin with) instead of calling ``max()`` on an empty sequence
    or dividing by zero.
    """
    ratios = [n / d for n, d in zip(numerators, denominators, strict=False) if d]
    if not ratios:
        return None
    return max(ratios) * 100.0


def _error_budget(ptc_record, linearity_record) -> list:
    """Every ``fit`` sub-dict is read with ``.get(...)``, never direct
    indexing, and only entries where *every* value involved is present are
    kept -- a ``fit`` dict can legitimately exist but be missing a specific
    key (a hand-built/partial record, or a future schema change), and this
    must degrade to "no error-budget entry for that quantity" rather than
    a ``KeyError`` reaching straight into a chain of ``["fit"]["key"]``.

    A quantity whose values/uncertainties *are* present but whose ratio
    can't be formed (every denominator is zero, a degenerate-but-real fit
    outcome) still gets an entry -- via :func:`_max_ratio_pct` returning
    ``None`` -- so the report says "not measured" instead of either
    crashing or silently dropping the row.
    """
    entries = []
    if ptc_record is not None:
        fits = [c.get("fit") or {} for c in ptc_record.result.get("channels", {}).values()]

        gains = [f["gain_uncertainty_e_per_dn"] for f in fits if f.get("gain_uncertainty_e_per_dn") is not None]
        gain_vals = [f["gain_e_per_dn"] for f in fits if f.get("gain_e_per_dn") is not None]
        if gains and gain_vals:
            achieved_pct = _max_ratio_pct(gains, gain_vals)
            entries.append(
                {
                    "quantity": "gain",
                    "expected": ESTIMATED_ACCURACY["gain_pct"],
                    "achieved": None if achieved_pct is None else round(achieved_pct, 2),
                    "unit": "%",
                }
            )

        read_noise_vals = [f["read_noise_e"] for f in fits if f.get("read_noise_e") is not None]
        read_noise_unc = [f["read_noise_uncertainty_e"] for f in fits if f.get("read_noise_uncertainty_e") is not None]
        if read_noise_vals and read_noise_unc:
            achieved_pct = _max_ratio_pct(read_noise_unc, read_noise_vals)
            entries.append(
                {
                    "quantity": "read noise",
                    "expected": ESTIMATED_ACCURACY["read_noise_pct"],
                    "achieved": None if achieved_pct is None else round(achieved_pct, 2),
                    "unit": "%",
                }
            )
    return entries


def render_sensor_report(
    *,
    device: dict,
    bias_record=None,
    ptc_record=None,
    linearity_record=None,
    darks_record=None,
    iso_record=None,
    shutter_record=None,
    fixed_pattern_record=None,
) -> str:
    """Assemble the single-file sensor HTML report from whichever records
    are available. ``device``: a ``devices.DeviceRef.to_dict()``-shaped
    dict. Every ``*_record`` is a ``store.Record`` or ``None``.
    """
    # The "primary" record for the page shell's provenance/status/refusals:
    # ptc if we have it (the record most other numbers here hang off of),
    # else whichever record is available first, else a bare "no data" shell.
    primary = next(
        (
            r
            for r in (
                ptc_record, bias_record, linearity_record, darks_record, iso_record, shutter_record,
                fixed_pattern_record,
            )
            if r is not None
        ),
        None,
    )
    provenance = primary.provenance if primary else "nominal"
    records = [
        r
        for r in (
            bias_record, ptc_record, linearity_record, darks_record, iso_record, shutter_record,
            fixed_pattern_record,
        )
        if r
    ]
    # The page's status is pooled over *every* record shown on it, the same
    # way the refusals below are. Taking it from `primary` alone badged a
    # page "ok" while listing another record's refusals right underneath --
    # e.g. camera.ptc passed but camera.darks and camera.iso refused.
    status = "refused" if any(r.status == "refused" for r in records) else (primary.status if primary else "ok")
    refusals = list(primary.refusals) if primary else []
    for r in records:
        if r is not primary:
            refusals.extend(r.refusals)

    recommended_iso = iso_record.result.get("recommended_iso") if iso_record else None
    star_tracker_sentence = STAR_TRACKER_SENTENCE_TEMPLATE.format(iso=recommended_iso if recommended_iso is not None else "N")

    sections = []
    for section in (
        _ptc_section(ptc_record),
        _read_noise_vs_iso_section(iso_record),
        _linearity_section(linearity_record),
        _darks_section(darks_record),
        _fixed_pattern_section(fixed_pattern_record),
    ):
        if section is not None:
            sections.append(section)
    sections.append({"heading": "Star tracker recommendation", "html": f"<p>{star_tracker_sentence}</p>"})

    return report_html.render_report(
        title=f"{device.get('model', 'camera')} sensor report",
        device=device,
        provenance=provenance,
        status=status,
        refusals=refusals,
        error_budget=_error_budget(ptc_record, linearity_record),
        sections=sections,
    )

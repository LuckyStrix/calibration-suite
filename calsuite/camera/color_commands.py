"""`calsuite camera color ...` subcommands (camera color characterization,
docs/design.md section 3.2). Wired in from ``camera/commands.py`` so the
sensor and color halves of ``camera/`` can be built independently.

This is the *only* module in the color wave that touches a file, the
capture pipeline or the record store (house rule 5) -- ``chart.py``,
``color.py``, ``ssf.py`` and ``dcp.py`` are all pure analysis/encoding, and
every subcommand here is a thin "load -> call the pure function -> save"
wrapper.
"""

from __future__ import annotations

import numpy as np

from calsuite import __version__, config, devices, raw as rawmod, store
from calsuite.camera import color

# `color` stays imported here (unlike chart/color_report/dcp/ssf/manual/icc,
# each deferred into the one function that actually uses it) because
# `_add_fit`/`_add_spectral` -- called by `register()` -- need
# `color.MODELS` right now, to build the `--model` argument's `choices=`.
# That's fine: `color.py`'s own heavy dependency (scipy.optimize) is
# itself deferred inside the one function that needs it, so importing
# `color` here costs next to nothing.


def register(camera_subparsers) -> None:
    """Add the `color` subcommand to the `camera` command's subparsers."""
    p = camera_subparsers.add_parser("color", help="camera color characterization (chart fit, spectral, export, report)")
    sub = p.add_subparsers(dest="color_command")
    p.set_defaults(func=lambda args: p.print_help() or 1)

    _add_fit(sub)
    _add_spectral(sub)
    _add_export(sub)
    _add_report(sub)


def _corners_type(text: str) -> list:
    values = [float(v) for v in text.split(",")]
    if len(values) != 8:
        raise ValueError("--corners needs exactly 8 comma-separated numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return [(values[i], values[i + 1]) for i in range(0, 8, 2)]


def _illuminant_xy(name: str) -> tuple:
    import colour

    ill2 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
    if name not in ill2:
        raise ValueError(f"unknown illuminant {name!r} -- known names: {sorted(ill2.keys())}")
    return tuple(float(v) for v in ill2[name])


# ---------------------------------------------------------------------------
# fit (Tier A)
# ---------------------------------------------------------------------------


def _add_fit(sub) -> None:
    p = sub.add_parser("fit", help="fit raw->XYZ from a photographed chart (Tier A)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="from_dir", help="folder of raw captures (manual import; the first 'target' frame is used)")
    src.add_argument("--raw", dest="raw_file", help="a single raw file")
    p.add_argument("--corners", required=True, type=_corners_type, help="x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL patch centers, full-sensor px)")
    p.add_argument("--illuminant", required=True, help="e.g. D65, A, D50")
    p.add_argument("--model", choices=color.MODELS, default="matrix")
    p.add_argument("--reference", help="custom chart reference CSV; default is the ColorChecker24 built into colour-science")
    p.add_argument("--rows", type=int, help="chart rows (default: reference chart's own)")
    p.add_argument("--cols", type=int, help="chart cols (default: reference chart's own)")
    p.add_argument("--white-preserving", action="store_true")
    p.add_argument("--held-out", help="comma-separated patch names reserved for validation only")
    p.set_defaults(func=_cmd_fit)


def _load_target_frame(args):
    from calsuite.capture import manual

    if args.raw_file:
        return rawmod.load(args.raw_file)
    manifest = manual.scan_folder(args.from_dir)
    targets = manifest.by_role("target")
    if not targets:
        raise SystemExit(f"no 'target'-classified frame found under {args.from_dir}")
    return targets[0].frame  # already decoded by scan_folder -- no need to load() it a second time


def _load_reference(args):
    from calsuite.camera import chart

    if args.reference:
        return chart.reference_from_csv(args.reference)
    return chart.reference_colorchecker()


def _cmd_fit(args) -> int:
    from calsuite.camera import chart

    frame = _load_target_frame(args)
    reference = _load_reference(args)
    rows = args.rows or reference.rows
    cols = args.cols or reference.cols

    sample = chart.sample_chart(frame, args.corners, rows=rows, cols=cols)
    sample = chart.label_patches(sample, reference)

    illuminant_xy = _illuminant_xy(args.illuminant)
    held_out = [n.strip() for n in args.held_out.split(",")] if args.held_out else None

    analysis = color.fit(
        sample,
        reference,
        illuminant_xy=illuminant_xy,
        illuminant_name=args.illuminant,
        model=args.model,
        white_preserving=args.white_preserving,
        held_out_names=held_out,
    )

    # Provenance policy (store.py's docstring; docs/implementation-plan.md
    # Wave 3 fix list item 2): provenance names the *method* -- a chart WAS
    # photographed here, so this is always "measured", whether or not the
    # fit's own refusal/validation checks passed. `status` (set by
    # Record.from_analysis from analysis.ok, which color.fit() now sets to
    # False on a failed validation too) is what actually gates
    # store.require_exportable() -- not a second axis on provenance.
    record = store.Record.from_analysis(
        kind="camera.color",
        device=devices.camera_ref(frame.meta).to_dict(),
        analysis=analysis,
        provenance="measured",
        method={
            "name": "camera.color.fit",
            "calsuite_version": __version__,
            "params": {"model": args.model, "illuminant": args.illuminant, "white_preserving": args.white_preserving},
        },
        inputs=[{"name": frame.path, "sha256": frame.sha256}],
        conditions={"illuminant": args.illuminant, "reference_chart": reference.name},
    )
    st = store.Store(config.records_dir())
    path = st.save(record)
    print(f"saved {record.kind} record {record.id} ({record.status}, provenance={record.provenance}) -> {path}")
    if not analysis.ok:
        for r in record.refusals:
            print(f"  refused: {r['check']}: {r['message']}")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# spectral (Tier B)
# ---------------------------------------------------------------------------


def _add_spectral(sub) -> None:
    p = sub.add_parser("spectral", help="fit raw->XYZ from measured spectral sensitivities (Tier B, no chart shot)")
    p.add_argument("--ssf", required=True, help="CSV: nm,r,g,b")
    p.add_argument("--illuminant", required=True, help="e.g. D65, A, D50")
    p.add_argument("--model", choices=color.MODELS, default="matrix")
    p.add_argument("--reflectance-set", default="ISO 17321-1", help="colour-science SDS_COLOURCHECKERS key")
    p.add_argument("--white-preserving", action="store_true")
    p.add_argument("--camera-model", default="unknown-camera", help="device label -- no photograph is taken in this subcommand")
    p.set_defaults(func=_cmd_spectral)


def _cmd_spectral(args) -> int:
    from calsuite.camera import ssf as ssfmod

    ssf_obj = ssfmod.load_ssf_csv(args.ssf)
    illuminant_xy = _illuminant_xy(args.illuminant)

    analysis = ssfmod.fit_from_ssf(
        ssf_obj,
        illuminant_name=args.illuminant,
        illuminant_xy=illuminant_xy,
        model=args.model,
        reflectance_set_name=args.reflectance_set,
        white_preserving=args.white_preserving,
    )
    analysis.result["luther_ives_deviation"] = ssfmod.luther_ives_deviation(ssf_obj)
    try:
        smi = ssfmod.sensor_metamerism_index(ssf_obj)
        analysis.result["smi"] = smi["smi"]
        analysis.result["mean_delta_e_ab"] = smi["mean_delta_e_ab"]
    except ValueError:
        pass  # --reflectance-set doesn't carry ISO 17321-1's named chromatic patches -- SMI just isn't reported

    device = devices.DeviceRef(kind="camera", model=args.camera_model, id=devices.device_id(args.camera_model, None))
    record = store.Record.from_analysis(
        kind="camera.color",
        device=device.to_dict(),
        analysis=analysis,
        provenance="derived",  # Tier B: computed from SSFs + spectral data, never "measured" (no chart shot to validate against)
        method={
            "name": "camera.color.spectral",
            "calsuite_version": __version__,
            "params": {"model": args.model, "illuminant": args.illuminant, "reflectance_set": args.reflectance_set},
        },
        inputs=[{"name": args.ssf, "sha256": rawmod.sha256_file(args.ssf)}],
        conditions={"illuminant": args.illuminant},
    )
    st = store.Store(config.records_dir())
    path = st.save(record)
    print(f"saved {record.kind} record {record.id} ({record.status}, provenance={record.provenance}) -> {path}")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _add_export(sub) -> None:
    p = sub.add_parser("export", help="export a fitted record to DCP and/or ICC")
    p.add_argument("--record", required=True, help="path to a saved camera.color record JSON")
    p.add_argument("--record2", help="a second record (e.g. tungsten) for a dual-illuminant DCP")
    p.add_argument("--dcp", help="output .dcp path")
    p.add_argument("--icc", help="output .icc path")
    p.set_defaults(func=_cmd_export)


def _matrices_for_record(record) -> tuple:
    from calsuite.camera import dcp as dcpmod

    matrix = np.array(record.result["matrix_raw_to_xyz"], dtype=np.float64)
    raw_white = record.result.get("white_patch_raw_rgb")
    if raw_white is None:
        raise SystemExit(f"record {record.id} has no white_patch_raw_rgb -- cannot build DCP matrices")
    color_matrix, forward_matrix = dcpmod.build_forward_and_color_matrices(matrix, np.array(raw_white))
    illuminant_name = record.result.get("illuminant", "D65")
    try:
        code = dcpmod.illuminant_code(illuminant_name)
    except ValueError as exc:
        raise SystemExit(f"record {record.id}: {exc}") from exc
    return color_matrix, forward_matrix, code


def _cmd_export(args) -> int:
    from calsuite.camera import dcp as dcpmod
    from calsuite.formats import icc as iccmod

    st = store.Store(config.records_dir())
    record = st.load(args.record)
    store.require_exportable(record)

    color_matrix1, forward_matrix1, code1 = _matrices_for_record(record)
    kwargs = dict(
        unique_camera_model=record.device.get("model", "calsuite-camera"),
        profile_name=f"calsuite {record.device.get('model', '')}".strip(),
        color_matrix1=color_matrix1,
        calibration_illuminant1=code1,
        forward_matrix1=forward_matrix1,
    )
    if args.record2:
        record2 = st.load(args.record2)
        store.require_exportable(record2)
        color_matrix2, forward_matrix2, code2 = _matrices_for_record(record2)
        kwargs.update(color_matrix2=color_matrix2, calibration_illuminant2=code2, forward_matrix2=forward_matrix2)

    if args.dcp:
        dcpmod.write_dcp(args.dcp, **kwargs)
        print(f"wrote {args.dcp}")
    if args.icc:
        matrix = np.array(record.result["matrix_raw_to_xyz"], dtype=np.float64)
        raw_white = record.result.get("white_patch_raw_rgb")
        if raw_white is None:
            raise SystemExit(f"record {record.id} has no white_patch_raw_rgb -- cannot build an ICC colorant matrix")
        # Device RGB for this profile is white-balanced raw in [0, 1]
        # (raw / white_patch_raw_rgb), which is what puts the colorant
        # columns in ICC's own units and makes (1, 1, 1) land on D50.
        colorant_matrix = dcpmod.icc_colorant_matrix(matrix, np.array(raw_white))
        iccmod.write_profile(
            args.icc,
            device_class="scnr",
            description=f"calsuite {record.device.get('model', '')}".strip(),
            matrix=colorant_matrix,
            trc=1.0,  # linear -- this suite's matrices are already fit against linear raw values
        )
        print(f"wrote {args.icc}")
    if not args.dcp and not args.icc:
        print("nothing to do: pass --dcp and/or --icc")
        return 1
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _add_report(sub) -> None:
    p = sub.add_parser("report", help="render an HTML report for a saved camera.color record")
    p.add_argument("--record", required=True, help="path to a saved camera.color record JSON")
    p.add_argument("--out", required=True, help="output .html path")
    p.set_defaults(func=_cmd_report)


def _cmd_report(args) -> int:
    from calsuite.camera import chart, color_report

    st = store.Store(config.records_dir())
    record = st.load(args.record)
    reference = None
    ref_name = (record.result or {}).get("reference_chart")
    if ref_name:
        try:
            reference = chart.reference_colorchecker(ref_name)
        except Exception:
            reference = None  # a custom-CSV reference chart's name won't resolve here -- report just skips the swatch section
    html = color_report.render(record, reference=reference)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}")
    return 0

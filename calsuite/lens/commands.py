"""``calsuite lens`` subcommands: ``distortion``/``tca``/``flats``/``mtf``/
``psf``/``export``/``report``. All file, capture and Store I/O lives here
(house rule 5) -- every actual measurement is one of the pure functions in
``lens/distortion.py`` etc.

Manual import is first-class (docs/design.md §3.3), but every lens command
here takes a dedicated ``--from DIR`` of already-role-sorted captures (a
charuco session, a flats session, ...) rather than routing through
``capture.manual.scan_folder``: that module's ``classify()`` heuristic is
tuned for bias/dark/flat/target frames from *sensor* characterization
(camera/), where a mixed folder genuinely needs sorting. A folder of
slanted-edge frames, for instance, often has a near-50/50 mean signal that
lands inside ``classify()``'s "flat" DN band by construction -- not a bug
in that module, just a mismatch with lens/'s one-role-per-folder
convention, so this module reads every raw file in the folder directly
instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from calsuite import config, devices as devicesmod, raw as rawmod, store
from calsuite.lens.constants import DISPLAY_SQUARE_MM, R100_PIXEL_PITCH_MM as LENS_R100_PIXEL_PITCH_MM

# charuco/distortion/tca/flats/mtf/psf/export_lensfun/report are each
# imported locally, inside the one `_cmd_*` function that uses them --
# charuco and distortion pull in cv2 (a few hundred ms), and `register()`
# (called for every `calsuite ...` invocation, `--help`/`doctor` included)
# has no need to pay that cost just to build the argparse tree.

_RAW_EXTENSIONS = {".cr3", ".CR3", ".dng", ".DNG", ".nef", ".NEF"} | rawmod.NPZ_EXTENSIONS | {
    ext.upper() for ext in rawmod.NPZ_EXTENSIONS
}
# .npz (raw.save_npz's format) alongside real raw extensions -- lets a
# synthetic session (no camera attached) exercise this module's CLI
# commands exactly like a real folder of captures (docs/implementation-plan.md
# Wave 3 fix list item 4).


def _load_raw_folder(folder: Path) -> list:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix in _RAW_EXTENSIONS)
    return [rawmod.load(p) for p in paths]


def _device_refs(frames: list) -> tuple:
    meta = frames[0].meta
    return devicesmod.camera_ref(meta), devicesmod.lens_ref(meta)


def _method(name: str, params: dict) -> dict:
    from calsuite import __version__

    return {"name": name, "calsuite_version": __version__, "params": params}


def _print_refusals(analysis) -> None:
    for r in analysis.refusals:
        print(f"  REFUSED [{r.check}]: {r.message} (value={r.value}, threshold={r.threshold})")


def _store() -> store.Store:
    return store.Store(config.records_dir())


def _latest_ok_flats_by_aperture(st: "store.Store", device_id: str) -> dict:
    """The latest ``status == "ok"`` ``lens.flats`` record per aperture for
    ``device_id`` -- ``lens.flats`` sessions are per-aperture
    (docs/design.md §4.3: "Flats at each aperture"), unlike
    ``distortion``/``tca``/``mtf``/``psf`` which ``store.latest()`` alone
    already handles as a single record per device. Both ``_cmd_export`` and
    ``_cmd_report`` need this same aperture-keyed collection."""
    by_aperture: dict = {}
    for record in st.all("lens.flats", device_id):
        if record.status != "ok":
            continue
        aperture = record.conditions.get("aperture")
        if aperture is None:
            continue
        existing = by_aperture.get(aperture)
        if existing is None or record.created > existing.created:
            by_aperture[aperture] = record
    return by_aperture


# ---------------------------------------------------------------------------
# distortion
# ---------------------------------------------------------------------------


def _cmd_distortion(args) -> int:
    from calsuite.lens import charuco, distortion as distortionmod

    frames = _load_raw_folder(args.from_dir)
    if len(frames) < 2:
        print(f"need at least 2 raw files under {args.from_dir}, found {len(frames)}")
        return 1
    camera_ref, lens_ref = _device_refs(frames)

    board = charuco.build_board(square_length=args.square_mm)
    views = []
    for frame in frames:
        detection = charuco.detect_green(frame, board)
        if detection is None:
            print(f"  no board detected in {frame.path}, skipping")
            continue
        views.append(distortionmod.BoardView(corners=detection.corners, ids=detection.ids, name=Path(frame.path).name))

    image_size = charuco.image_size_from_frame(frames[0])
    analysis = distortionmod.fit_distortion(board, views, image_size)

    conditions = {"focal_mm": frames[0].meta.focal, "square_mm": args.square_mm}
    if args.distance is not None:
        conditions["focus_distance_m"] = args.distance
    conditions["focus_distance_caveat"] = (
        "distortion changes with focus distance; lensfun records per focal length only -- "
        "this record is only valid at the distance above (docs/design.md §4.1)"
    )

    record = store.Record.from_analysis(
        kind="lens.distortion",
        device=lens_ref.to_dict(),
        devices=[camera_ref.to_dict()],
        analysis=analysis,
        provenance="measured",
        method=_method("distortion.fit_distortion", {"square_mm": args.square_mm}),
        inputs=[{"name": Path(f.path).name, "sha256": f.sha256} for f in frames],
        conditions=conditions,
    )
    _store().save(record)
    _print_refusals(analysis)
    print(f"lens.distortion: {record.status} ({record.id})")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# tca
# ---------------------------------------------------------------------------


def _cmd_tca(args) -> int:
    from calsuite.lens import charuco, tca as tcamod

    frames = _load_raw_folder(args.from_dir)
    if not frames:
        print(f"no raw files under {args.from_dir}")
        return 1
    camera_ref, lens_ref = _device_refs(frames)
    board = charuco.build_board(square_length=args.square_mm)
    image_size = charuco.image_size_from_frame(frames[0])

    detections_per_view = [charuco.detect_all_planes(frame, board) for frame in frames]
    analysis = tcamod.fit_tca(detections_per_view, image_size)

    conditions = {"focal_mm": frames[0].meta.focal, "square_mm": args.square_mm}
    record = store.Record.from_analysis(
        kind="lens.tca",
        device=lens_ref.to_dict(),
        devices=[camera_ref.to_dict()],
        analysis=analysis,
        provenance="measured",
        method=_method("tca.fit_tca", {}),
        inputs=[{"name": Path(f.path).name, "sha256": f.sha256} for f in frames],
        conditions=conditions,
    )
    _store().save(record)
    _print_refusals(analysis)
    print(f"lens.tca: {record.status} ({record.id})")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# flats
# ---------------------------------------------------------------------------


def _parse_poses(spec: str, n: int) -> list:
    from calsuite.lens import flats as flatsmod

    parts = spec.split(",")
    if len(parts) != n:
        raise ValueError(f"--poses has {len(parts)} entries but {n} raw files were found")
    poses = []
    for part in parts:
        fields = part.split(":")
        angle = float(fields[0])
        shift_u = float(fields[1]) if len(fields) > 1 else 0.0
        shift_v = float(fields[2]) if len(fields) > 2 else 0.0
        poses.append(flatsmod.Pose(angle_deg=angle, shift_u=shift_u, shift_v=shift_v))
    return poses


def _cmd_flats(args) -> int:
    from calsuite.lens import flats as flatsmod

    frames = _load_raw_folder(args.from_dir)
    if not frames:
        print(f"no raw files under {args.from_dir}")
        return 1
    camera_ref, lens_ref = _device_refs(frames)
    try:
        poses = _parse_poses(args.poses, len(frames))
    except ValueError as exc:
        print(str(exc))
        return 1

    # Black-subtracted, per channel (raw.black_level_by_channel): the joint
    # V x S model `self_calibrate_flat` fits is *multiplicative* -- it works
    # on log(V . S) -- so an additive black pedestal isn't absorbed anywhere,
    # it flattens the recovered falloff. With a realistic 2048 DN pedestal
    # and ~3000 DN of signal, a true corner vignetting of V=0.741 came back
    # as 0.846 and the exported lensfun k1 was 40% low. The function's own
    # docstring asks for "already black-subtracted and positive" input; the
    # tests did the subtraction themselves, so only this caller skipped it.
    planes = []  # green plane: highest sample density
    for f in frames:
        black = rawmod.black_level_by_channel(f)["G1"]
        planes.append(np.clip(rawmod.planes(f)["G1"] - black, 1.0, None))
    analysis = flatsmod.self_calibrate_flat(planes, poses)

    conditions = {"focal_mm": frames[0].meta.focal, "aperture": args.aperture}
    if args.distance is not None:
        conditions["focus_distance_m"] = args.distance
    conditions["focus_distance_caveat"] = (
        "vignetting can change with focus distance, same as distortion (docs/design.md §4.1) -- "
        "this record is only valid at the distance above"
    )
    record = store.Record.from_analysis(
        kind="lens.flats",
        device=lens_ref.to_dict(),
        devices=[camera_ref.to_dict()],
        analysis=analysis,
        provenance="measured",
        method=_method("flats.self_calibrate_flat", {"poses": args.poses}),
        inputs=[{"name": Path(f.path).name, "sha256": f.sha256} for f in frames],
        conditions=conditions,
    )
    if analysis.ok:
        v_map = flatsmod.evaluate_v_map(analysis.result["v_coeffs"], tuple(analysis.result["image_shape"]))
        pa = flatsmod.fit_pa(analysis.result["v_coeffs"])
        record.result["pa"] = pa
        artifacts = {"flat_maps": {"V": v_map}}
    else:
        artifacts = None
    _store().save(record, artifacts=artifacts)
    _print_refusals(analysis)
    print(f"lens.flats: {record.status} ({record.id})")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# mtf
# ---------------------------------------------------------------------------


def _cmd_mtf(args) -> int:
    from calsuite.lens import mtf as mtfmod

    frames = _load_raw_folder(args.from_dir)
    if not frames:
        print(f"no raw files under {args.from_dir}")
        return 1
    camera_ref, lens_ref = _device_refs(frames)
    plane = rawmod.planes(frames[0])["G1"]
    grid = tuple(int(v) for v in args.grid.split("x"))
    analysis = mtfmod.mtf_field_grid(
        plane, grid=grid, saturation_dn=frames[0].white_level, pixel_pitch_mm=args.pixel_pitch_mm
    )

    conditions = {"focal_mm": frames[0].meta.focal, "aperture": frames[0].meta.aperture}
    record = store.Record.from_analysis(
        kind="lens.mtf",
        device=lens_ref.to_dict(),
        devices=[camera_ref.to_dict()],
        analysis=analysis,
        provenance="measured",
        method=_method("mtf.mtf_field_grid", {"grid": args.grid, "pixel_pitch_mm": args.pixel_pitch_mm}),
        inputs=[{"name": Path(frames[0].path).name, "sha256": frames[0].sha256}],
        conditions=conditions,
    )
    _store().save(record)
    _print_refusals(analysis)
    print(f"lens.mtf: {record.status} ({record.id})")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# psf
# ---------------------------------------------------------------------------


def _cmd_psf(args) -> int:
    from calsuite.lens import psf as psfmod

    frames = _load_raw_folder(args.from_dir)
    if not frames:
        print(f"no raw files under {args.from_dir}")
        return 1
    camera_ref, lens_ref = _device_refs(frames)
    plane = rawmod.planes(frames[0])["G1"]
    analysis = psfmod.psf_field(plane, saturation_dn=frames[0].white_level)

    conditions = {"focal_mm": frames[0].meta.focal, "aperture": frames[0].meta.aperture}
    record = store.Record.from_analysis(
        kind="lens.psf",
        device=lens_ref.to_dict(),
        devices=[camera_ref.to_dict()],
        analysis=analysis,
        provenance="measured",
        method=_method("psf.psf_field", {}),
        inputs=[{"name": Path(frames[0].path).name, "sha256": frames[0].sha256}],
        conditions=conditions,
    )
    _store().save(record)
    _print_refusals(analysis)
    print(f"lens.psf: {record.status} ({record.id})")
    return 0 if analysis.ok else 1


# ---------------------------------------------------------------------------
# export / report
# ---------------------------------------------------------------------------


def _cmd_export(args) -> int:
    from calsuite.lens import export_lensfun

    st = _store()
    distortion_record = st.latest("lens.distortion", args.device_id)
    tca_record = st.latest("lens.tca", args.device_id)
    flats_records = list(_latest_ok_flats_by_aperture(st, args.device_id).values())
    try:
        xml_text = export_lensfun.export_records(
            lens_model=args.lens_model,
            distortion_record=distortion_record if distortion_record and distortion_record.status == "ok" else None,
            tca_record=tca_record if tca_record and tca_record.status == "ok" else None,
            flats_records=flats_records,
        )
    except store.ExportRefused as exc:
        print(f"export refused: {exc}")
        return 1
    path = export_lensfun.write_lensfun(xml_text, out=args.out)
    print(f"wrote {path}")
    return 0


def _cmd_report(args) -> int:
    from calsuite.lens import export_lensfun, report as reportmod

    st = _store()
    distortion_record = st.latest("lens.distortion", args.device_id)
    tca_record = st.latest("lens.tca", args.device_id)
    mtf_record = st.latest("lens.mtf", args.device_id)
    psf_record = st.latest("lens.psf", args.device_id)
    flats_records_by_aperture = _latest_ok_flats_by_aperture(st, args.device_id)

    vendor_comparison = None
    if distortion_record is not None and distortion_record.status == "ok":
        vendor_comparison = export_lensfun.compare_with_vendor(
            distortion_record.result["ptlens"],
            tca_record.result if tca_record and tca_record.status == "ok" else None,
        )

    device = distortion_record.device if distortion_record else {"model": args.lens_model, "id": args.device_id}
    html = reportmod.render_lens_report(
        device=device,
        distortion_record=distortion_record,
        tca_record=tca_record,
        flats_records_by_aperture=flats_records_by_aperture,
        mtf_record=mtf_record,
        psf_record=psf_record,
        vendor_comparison=vendor_comparison,
    )
    out_path = Path(args.out) if args.out else config.records_dir() / args.device_id / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    parser = subparsers.add_parser("lens", help="lens calibration")
    sub = parser.add_subparsers(dest="lens_command")

    distortion_p = sub.add_parser("distortion", help="Brown-Conrady distortion fit from a ChArUco session")
    distortion_p.add_argument("--from", dest="from_dir", type=Path, required=True)
    distortion_p.add_argument("--square-mm", type=float, default=DISPLAY_SQUARE_MM)
    distortion_p.add_argument("--distance", type=float, default=None, help="focus distance in meters")
    distortion_p.set_defaults(func=_cmd_distortion)

    tca_p = sub.add_parser("tca", help="lateral chromatic aberration fit from a ChArUco session")
    tca_p.add_argument("--from", dest="from_dir", type=Path, required=True)
    tca_p.add_argument("--square-mm", type=float, default=DISPLAY_SQUARE_MM)
    tca_p.set_defaults(func=_cmd_tca)

    flats_p = sub.add_parser("flats", help="self-calibrating flat field from a pose session")
    flats_p.add_argument("--from", dest="from_dir", type=Path, required=True)
    flats_p.add_argument("--aperture", type=float, required=True)
    flats_p.add_argument(
        "--poses",
        type=str,
        required=True,
        help="comma-separated angle[:shift_u:shift_v] per raw file, in filename order, e.g. '0,90:0.05,180,270:0:0.05'",
    )
    flats_p.add_argument("--distance", type=float, default=None, help="focus distance in meters")
    flats_p.set_defaults(func=_cmd_flats)

    mtf_p = sub.add_parser("mtf", help="slanted-edge e-SFR field grid")
    mtf_p.add_argument("--from", dest="from_dir", type=Path, required=True)
    mtf_p.add_argument("--grid", type=str, default="3x5")
    mtf_p.add_argument(
        "--pixel-pitch-mm",
        type=float,
        default=LENS_R100_PIXEL_PITCH_MM,
        help="sensor pixel pitch in mm, for the cycles/px -> lp/mm conversion (default: the R100's)",
    )
    mtf_p.set_defaults(func=_cmd_mtf)

    psf_p = sub.add_parser("psf", help="star/pinhole PSF field map")
    psf_p.add_argument("--from", dest="from_dir", type=Path, required=True)
    psf_p.set_defaults(func=_cmd_psf)

    export_p = sub.add_parser("export", help="write lensfun XML for the latest ok distortion/tca records")
    export_p.add_argument("--device-id", required=True)
    export_p.add_argument("--lens-model", required=True)
    export_p.add_argument("--out", type=Path, default=None)
    export_p.set_defaults(func=_cmd_export)

    report_p = sub.add_parser("report", help="render the lens HTML report")
    report_p.add_argument("--device-id", required=True)
    report_p.add_argument("--lens-model", default="")
    report_p.add_argument("--out", type=Path, default=None)
    report_p.set_defaults(func=_cmd_report)

    parser.set_defaults(func=_not_built)


def _not_built(args) -> int:
    print("calsuite lens: pass a subcommand (distortion, tca, flats, mtf, psf, export, report)")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(1)

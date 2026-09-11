"""``calsuite camera`` subcommands (Wave 2A -- sensor characterization,
docs/design.md §3.1). This module is the *only* place in ``camera/`` that
touches files, the record store, or a capture backend (house rule 5): every
subcommand's job is to turn a folder of raw files (or, behind ``--capture``,
a tethered gphoto2 session) into loaded ``RawFrame``s, hand them to a pure
``analyze_*`` function, and save the resulting ``Analysis`` as a ``Record``.

``calsuite camera color`` is wired in from ``camera.color_commands`` (a
sibling area of this same wave, built independently) so the two halves of
``camera/`` combine into one CLI command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calsuite import __version__, config, devices as devicesmod, raw as rawmod, store
from calsuite.camera import bias, bulb, color_commands, darks, iso, linearity, ptc, report, settings, shutter
from calsuite.capture import gphoto2, manual

# -- shared helpers -----------------------------------------------------------


def _load_frames(entries) -> list:
    return [rawmod.load(e.path) for e in entries]


def _inputs(frames) -> list:
    return [{"name": Path(f.path).name, "sha256": f.sha256} for f in frames]


def _black_dn(frames) -> dict:
    """Per-channel black level, DN. Falls back to each frame's own metadata
    black level (rawpy's ``black_level_per_channel``, averaged to a single
    scalar -- see camera/bias.py's docstring on why a precise per-channel
    mapping isn't attempted here) when no measured ``camera.bias`` record is
    passed in; callers that have one should prefer it."""
    return float(sum(frames[0].black_level) / len(frames[0].black_level)) if frames[0].black_level else 0.0


def _latest_ok(st: store.Store, kind: str, device_id: str):
    record = st.latest(kind, device_id)
    return record if record is not None and record.status == "ok" else None


def _black_dn_from_store(st: store.Store, device_id: str, frames) -> float | dict:
    bias_record = _latest_ok(st, "camera.bias", device_id)
    if bias_record is not None and "black_level_dn" in bias_record.result:
        return bias_record.result["black_level_dn"]
    return _black_dn(frames)


def _gain_from_store(st: store.Store, device_id: str):
    ptc_record = _latest_ok(st, "camera.ptc", device_id)
    if ptc_record is None:
        return None
    channels = ptc_record.result.get("channels", {})
    gains = {ch: c["fit"]["gain_e_per_dn"] for ch, c in channels.items() if "fit" in c}
    return gains or None


def _print_refusals(analysis) -> None:
    for r in analysis.refusals:
        print(f"REFUSED [{r.check}]: {r.message}")


def _save(
    st: store.Store,
    *,
    kind: str,
    device_ref,
    analysis,
    method_name: str,
    conditions: dict,
    inputs: list,
    artifacts: dict | None = None,
) -> int:
    """Build and save a ``Record`` from an ``Analysis``, per the house rule
    that a refused analysis is still saved (status "refused"), just not
    with strong provenance.

    A refused record's provenance can't honestly be "measured" (house rule
    2: that label is earned only by *passing* the refusal checks), but none
    of the other three tiers (derived/vendor/nominal) describes "a fresh
    measurement attempt that failed its own checks" either -- "nominal" is
    used as the fallback because it's already the tier that means "don't
    build anything on this number", which is exactly the guarantee
    ``store.require_exportable`` needs regardless of which non-measured
    label is picked.
    """
    record = store.Record.from_analysis(
        kind=kind,
        device=device_ref.to_dict(),
        analysis=analysis,
        provenance="measured" if analysis.ok else "nominal",
        method={"name": method_name, "calsuite_version": __version__, "params": {}},
        conditions=conditions,
        inputs=inputs,
    )
    path = st.save(record, artifacts=artifacts)
    _print_refusals(analysis)
    print(f"{'ok' if analysis.ok else 'refused'}: {kind} -> {path}")
    return 0 if analysis.ok else 1


def _store() -> store.Store:
    return store.Store(config.records_dir())


def _maybe_capture(args, target_dir: Path) -> None:
    """Best-effort guided capture behind ``--capture``: never exercised by
    tests (``gphoto2.is_camera_connected()`` degrades to ``False`` with no
    camera and no gphoto2 binary present, per capture/gphoto2.py, so this
    path is safe to leave wired in). Captures ``args.count`` frames with
    whatever config the camera is already set to; it does not itself step
    through an ISO/exposure sweep -- that's still done by hand between
    captures, same as a plain tethered session."""
    if not getattr(args, "capture", False):
        return
    if not gphoto2.is_camera_connected():
        raise SystemExit("--capture given but no camera is connected (gphoto2 --auto-detect found none)")
    target_dir.mkdir(parents=True, exist_ok=True)
    count = getattr(args, "count", 10)
    print(f"capturing {count} frame(s) into {target_dir} ...")
    for i in range(count):
        path = gphoto2.capture_and_download(target_dir)
        print(f"  [{i + 1}/{count}] {path}")


def _resolve_from_dir(args) -> Path:
    if getattr(args, "capture", False):
        target = Path(args.from_dir) if args.from_dir else config.captures_dir() / args.subcommand_name
        _maybe_capture(args, target)
        return target
    if not args.from_dir:
        raise SystemExit("--from DIR is required (or pass --capture)")
    return Path(args.from_dir)


# -- bias ---------------------------------------------------------------------


def _cmd_bias(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("bias")
    if not entries:
        print(f"no bias frames found in {folder}")
        return 1

    by_iso: dict = {}
    for e in entries:
        by_iso.setdefault(e.iso, []).append(e)

    st = _store()
    rc = 0
    for iso_value, iso_entries in sorted(by_iso.items(), key=lambda kv: (kv[0] is None, kv[0])):
        frames = _load_frames(iso_entries)
        device_ref = devicesmod.camera_ref(frames[0].meta)
        analysis = bias.analyze_bias(frames)
        gain = _gain_from_store(st, device_ref.id)
        if analysis.ok and gain is not None:
            bias.add_read_noise_electrons(analysis, gain)
        conditions = {"iso": iso_value, "settings": settings.summarize(frames)}
        rc |= _save(
            st,
            kind="camera.bias",
            device_ref=device_ref,
            analysis=analysis,
            method_name="camera.bias",
            conditions=conditions,
            inputs=_inputs(frames),
        )
    return rc


# -- ptc ------------------------------------------------------------------


def _cmd_ptc(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("flat")
    if len(entries) < 2:
        print(f"need at least 2 flat frames (a pair) in {folder}, found {len(entries)}")
        return 1

    frames = _load_frames(entries)
    device_ref = devicesmod.camera_ref(frames[0].meta)
    st = _store()
    black_dn = _black_dn_from_store(st, device_ref.id, frames)

    n_pairs = len(frames) // 2
    pairs = [(frames[2 * i], frames[2 * i + 1]) for i in range(n_pairs)]
    analysis = ptc.analyze_ptc(pairs, black_dn)

    conditions = {"iso": frames[0].meta.iso, "settings": settings.summarize(frames)}
    return _save(
        st,
        kind="camera.ptc",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.ptc",
        conditions=conditions,
        inputs=_inputs(frames),
    )


# -- linearity --------------------------------------------------------------


def _cmd_linearity(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("flat")
    if not entries:
        print(f"no flat frames found in {folder}")
        return 1

    frames = _load_frames(entries)
    device_ref = devicesmod.camera_ref(frames[0].meta)
    st = _store()
    black_dn = _black_dn_from_store(st, device_ref.id, frames)
    gain = _gain_from_store(st, device_ref.id)
    analysis = linearity.analyze_linearity(frames, black_dn, gain_e_per_dn=gain)

    conditions = {"iso": frames[0].meta.iso, "settings": settings.summarize(frames)}
    return _save(
        st,
        kind="camera.linearity",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.linearity",
        conditions=conditions,
        inputs=_inputs(frames),
    )


# -- darks (+ hot pixels, + star-eater when both a short and long exposure
# group are present in the same folder) --------------------------------------


def _cmd_darks(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("dark")
    if not entries:
        print(f"no dark frames found in {folder}")
        return 1

    frames = _load_frames(entries)
    device_ref = devicesmod.camera_ref(frames[0].meta)
    st = _store()
    black_dn = _black_dn_from_store(st, device_ref.id, frames)
    gain = _gain_from_store(st, device_ref.id)
    analysis = darks.analyze_darks(frames, black_dn, gain_e_per_dn=gain)

    if analysis.ok:
        by_exposure: dict = {}
        for f in frames:
            by_exposure.setdefault(f.meta.exposure_s, []).append(f)
        if len(by_exposure) >= 2:
            exposures = sorted(by_exposure)
            star = darks.star_eater_check(by_exposure[exposures[0]], by_exposure[exposures[-1]])
            analysis.result["star_eater"] = star.result

    artifacts = {}
    by_exposure_all: dict = {}
    for f in frames:
        by_exposure_all.setdefault(f.meta.exposure_s, []).append(f)
    for exposure_s, exp_frames in by_exposure_all.items():
        hp = darks.find_hot_pixels(exp_frames)
        artifacts[f"hot_pixels_{exposure_s}s"] = {"rows": hp["rows"], "cols": hp["cols"]}

    conditions = {"iso": frames[0].meta.iso, "settings": settings.summarize(frames)}
    return _save(
        st,
        kind="camera.darks",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.darks",
        conditions=conditions,
        inputs=_inputs(frames),
        artifacts=artifacts,
    )


# -- shutter ------------------------------------------------------------------


def _cmd_shutter(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("flat") + manifest.by_role("target")
    if not entries:
        print(f"no candidate shutter-test frames found in {folder}")
        return 1

    frames = _load_frames(entries)
    device_ref = devicesmod.camera_ref(frames[0].meta)
    st = _store()
    black_dn = _black_dn_from_store(st, device_ref.id, frames)
    analysis = shutter.analyze_shutter_accuracy(frames, black_dn)

    conditions = {"iso": frames[0].meta.iso, "settings": settings.summarize(frames)}
    return _save(
        st,
        kind="camera.shutter",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.shutter",
        conditions=conditions,
        inputs=_inputs(frames),
    )


# -- bulb -----------------------------------------------------------------


def _cmd_bulb(args) -> int:
    folder = _resolve_from_dir(args)
    manifest = manual.scan_folder(folder)
    entries = manifest.by_role("flat") + manifest.by_role("target")
    if not args.commanded:
        raise SystemExit("--commanded SEC,SEC,... is required (the intervalometer's commanded bulb lengths, "
                          "in the same order as the frames in --from DIR)")
    commanded = [float(x) for x in args.commanded.split(",")]
    if len(entries) != len(commanded):
        print(f"{len(entries)} frame(s) in {folder} but {len(commanded)} --commanded value(s); must match 1:1")
        return 1

    frames = _load_frames(entries)
    device_ref = devicesmod.camera_ref(frames[0].meta)
    st = _store()
    black_dn = _black_dn_from_store(st, device_ref.id, frames)
    analysis = bulb.analyze_bulb_timing(frames, commanded, black_dn)

    conditions = {"iso": frames[0].meta.iso, "settings": settings.summarize(frames), "commanded_s": commanded}
    return _save(
        st,
        kind="camera.bulb",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.bulb",
        conditions=conditions,
        inputs=_inputs(frames),
    )


# -- iso invariance (derived from existing camera.ptc/camera.linearity records,
# not from fresh captures -- no --from here) ---------------------------------


def _cmd_iso(args) -> int:
    st = _store()
    ptc_records = [r for r in st.all(kind="camera.ptc", device_id=args.device_id) if r.status == "ok"]
    if not ptc_records:
        print(f"no ok camera.ptc records found for device {args.device_id}")
        return 1

    read_noise_e_by_iso = {}
    full_well_e_by_iso = {}
    for r in ptc_records:
        iso_value = r.conditions.get("iso")
        if iso_value is None:
            continue
        channels = r.result.get("channels", {})
        rn = [c["fit"]["read_noise_e"] for c in channels.values() if "fit" in c]
        if rn:
            read_noise_e_by_iso[iso_value] = max(rn)

    linearity_records = [r for r in st.all(kind="camera.linearity", device_id=args.device_id) if r.status == "ok"]
    for r in linearity_records:
        iso_value = r.conditions.get("iso")
        channels = r.result.get("channels", {})
        fw = [c["full_well_e"] for c in channels.values() if "full_well_e" in c]
        if iso_value is not None and fw:
            full_well_e_by_iso[iso_value] = min(fw)

    if not full_well_e_by_iso:
        print(f"no camera.linearity records with a known full well for device {args.device_id}; "
              "run 'calsuite camera ptc' then 'calsuite camera linearity' first")
        return 1
    # A truly ISO-invariant sensor's full well in electrons doesn't depend on
    # ISO; fall back to the single value we have for any ISO camera.ptc
    # covers but camera.linearity doesn't.
    default_full_well = next(iter(full_well_e_by_iso.values()))
    full_well_e_by_iso = {iso_value: full_well_e_by_iso.get(iso_value, default_full_well) for iso_value in read_noise_e_by_iso}

    analysis = iso.analyze_iso_invariance(read_noise_e_by_iso, full_well_e_by_iso)
    device_ref = devicesmod.DeviceRef(kind="camera", model=ptc_records[0].device.get("model", ""), id=args.device_id)
    return _save(
        st,
        kind="camera.iso",
        device_ref=device_ref,
        analysis=analysis,
        method_name="camera.iso",
        conditions={},
        inputs=[],
    )


# -- report -------------------------------------------------------------------


def _cmd_report(args) -> int:
    st = _store()
    kinds = ("camera.bias", "camera.ptc", "camera.linearity", "camera.darks", "camera.iso", "camera.shutter")
    latest = {kind: st.latest(kind, args.device_id) for kind in kinds}
    device = None
    for r in latest.values():
        if r is not None:
            device = r.device
            break
    if device is None:
        print(f"no records found for device {args.device_id}")
        return 1

    html = report.render_sensor_report(
        device=device,
        bias_record=latest["camera.bias"],
        ptc_record=latest["camera.ptc"],
        linearity_record=latest["camera.linearity"],
        darks_record=latest["camera.darks"],
        iso_record=latest["camera.iso"],
        shutter_record=latest["camera.shutter"],
    )
    if args.out:
        Path(args.out).write_text(html)
        print(f"wrote {args.out}")
    else:
        print(html)
    return 0


# -- wiring -------------------------------------------------------------------


def _add_common_capture_args(p: argparse.ArgumentParser, subcommand_name: str) -> None:
    p.add_argument("--from", dest="from_dir", default=None, help="folder of already-captured raw files")
    p.add_argument("--capture", action="store_true", help="guided tethered capture via gphoto2 first")
    p.add_argument("--count", type=int, default=10, help="frames to capture when --capture is given")
    p.add_argument("--iso", dest="iso_range", default=None, help="ISO range for guided capture, e.g. 100..6400")
    p.set_defaults(subcommand_name=subcommand_name)


def register(subparsers) -> None:
    parser = subparsers.add_parser("camera", help="camera sensor + color calibration")
    camera_subparsers = parser.add_subparsers(dest="camera_command")
    parser.set_defaults(func=lambda args: parser.print_help() or 1)

    p = camera_subparsers.add_parser("bias", help="black level + read noise from bias frames")
    _add_common_capture_args(p, "bias")
    p.set_defaults(func=_cmd_bias)

    p = camera_subparsers.add_parser("ptc", help="gain + read noise from a photon transfer curve")
    _add_common_capture_args(p, "ptc")
    p.set_defaults(func=_cmd_ptc)

    p = camera_subparsers.add_parser("linearity", help="full well + linearity from a flat exposure series")
    _add_common_capture_args(p, "linearity")
    p.set_defaults(func=_cmd_linearity)

    p = camera_subparsers.add_parser("darks", help="dark current, hot pixels, star-eater check")
    _add_common_capture_args(p, "darks")
    p.set_defaults(func=_cmd_darks)

    p = camera_subparsers.add_parser("shutter", help="shutter speed accuracy")
    _add_common_capture_args(p, "shutter")
    p.set_defaults(func=_cmd_shutter)

    p = camera_subparsers.add_parser("bulb", help="bulb-mode intervalometer timing (latency + scale error)")
    _add_common_capture_args(p, "bulb")
    p.add_argument("--commanded", default=None, help="comma-separated commanded bulb lengths in seconds")
    p.set_defaults(func=_cmd_bulb)

    p = camera_subparsers.add_parser("iso", help="ISO invariance + engineering dynamic range")
    p.add_argument("--device-id", required=True)
    p.set_defaults(func=_cmd_iso)

    p = camera_subparsers.add_parser("report", help="render the sensor HTML report")
    p.add_argument("--device-id", required=True)
    p.add_argument("--out", default=None, help="write to this path instead of stdout")
    p.set_defaults(func=_cmd_report)

    color_commands.register(camera_subparsers)

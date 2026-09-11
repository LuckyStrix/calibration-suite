"""``calsuite display`` subcommands (docs/design.md §5, replacing Wave 2C's
stub): ``nominal`` (EDID -> a ``display.nominal`` record), ``measure``
(drives a backend over the full patch set -> ``display.measurement``),
``profile`` (-> ``display.profile``), ``install`` (colormgr/dispwin, Linux
or Windows), ``validate`` (-> ``display.validation``), and ``report`` (HTML).

This is the one module in ``display/`` that touches files, the store, and
capture/backends together (house rule 5: analysis stays pure in
``analysis.py``/``patches.py``/``synth/display.py``; everything here is
orchestration).
"""

from __future__ import annotations

import sys
from pathlib import Path

from calsuite import __version__, config, devices as devicesmod, store as storemod
from calsuite.display import analysis as analysismod
from calsuite.display import constants as dc
from calsuite.display import install_linux, install_windows, osstate, patches as patchesmod, profile as profilemod
from calsuite.display import report as reportmod
from calsuite.display import validate as validatemod
from calsuite.display import window as windowmod
from calsuite.display.backends.argyll import ArgyllBackend
from calsuite.display.backends.synthetic import SyntheticBackend
from calsuite.fit import Analysis, Refusal

_METHOD_NAME = "calsuite.display.commands"


def register(subparsers) -> None:
    parser = subparsers.add_parser("display", help="display measurement, profiling, install and validation")
    display_sub = parser.add_subparsers(dest="display_command")

    nominal_p = display_sub.add_parser("nominal", help="write a display.nominal record from EDID")
    nominal_p.add_argument("--connector", default=None, help="Linux sysfs connector name, e.g. card0-eDP-1 (default: first found)")
    nominal_p.set_defaults(func=_cmd_nominal)

    measure_p = display_sub.add_parser("measure", help="measure the display's tone response, additivity, primaries, uniformity")
    measure_p.add_argument("--backend", choices=("argyll", "camera", "spectro", "synthetic"), required=True)
    measure_p.add_argument("--device-id", default=None, help="display device id (default: this laptop's EDID panel)")
    measure_p.add_argument("--steps", type=int, default=dc.RAMP_STEPS_DEFAULT)
    measure_p.add_argument("--width", type=int, default=None)
    measure_p.add_argument("--height", type=int, default=None)
    measure_p.add_argument("--windowed", action="store_true", help="don't go fullscreen (debugging)")
    measure_p.add_argument("--confirm-hdr-off", action="store_true", help="required if HDR state can't be determined")
    measure_p.add_argument("--osd-brightness", default=None, help="OSD brightness setting, recorded as part of display identity")
    measure_p.add_argument("--osd-mode", default=None, help="OSD picture-mode setting, recorded as part of display identity")
    measure_p.set_defaults(func=_cmd_measure)

    profile_p = display_sub.add_parser("profile", help="build an ICC profile from the latest measurement")
    profile_p.add_argument("--device-id", default=None)
    profile_p.add_argument("--out", default=None, help="output .icc path (default: <records>/<device>/display-profile.icc)")
    profile_p.set_defaults(func=_cmd_profile)

    install_p = display_sub.add_parser("install", help="install the latest profile")
    install_p.add_argument("--device-id", default=None)
    install_p.add_argument("--write-autostart", action="store_true", help="also write an Openbox autostart loader line")
    install_p.set_defaults(func=_cmd_install)

    validate_p = display_sub.add_parser("validate", help="validate the installed profile against the real display")
    validate_p.add_argument("--backend", choices=("argyll", "camera", "spectro", "synthetic"), required=True)
    validate_p.add_argument("--device-id", default=None)
    validate_p.add_argument("--width", type=int, default=None)
    validate_p.add_argument("--height", type=int, default=None)
    validate_p.add_argument("--windowed", action="store_true")
    validate_p.set_defaults(func=_cmd_validate)

    report_p = display_sub.add_parser("report", help="write the HTML report")
    report_p.add_argument("--device-id", default=None)
    report_p.add_argument("--out", default=None, help="output .html path (default: <records>/<device>/display-report.html)")
    report_p.set_defaults(func=_cmd_report)

    parser.set_defaults(func=_cmd_help)


def _cmd_help(args) -> int:
    print("calsuite display: use one of nominal/measure/profile/install/validate/report (see --help).")
    return 1


def _store() -> storemod.Store:
    return storemod.Store(config.records_dir())


# ---------------------------------------------------------------------------
# nominal
# ---------------------------------------------------------------------------


def _cmd_nominal(args) -> int:
    blobs = devicesmod.list_linux_edids()
    if not blobs:
        print("no EDID found under /sys/class/drm")
        return 1
    connector = args.connector or sorted(blobs)[0]
    if connector not in blobs:
        print(f"connector {connector!r} not found; available: {sorted(blobs)}")
        return 1
    info = devicesmod.parse_edid(blobs[connector])
    ref = devicesmod.display_ref(info)

    analysis = Analysis(
        result={
            "chromaticity": {k: list(v) for k, v in info.chromaticity.items()},
            "physical_size_mm": list(info.physical_size_mm),
            "gamma": info.gamma,
            "name": info.name,
            "edid_hash": devicesmod.edid_hash(blobs[connector]),
        }
    )
    record = storemod.Record.from_analysis(
        kind="display.nominal",
        device=ref.to_dict(),
        analysis=analysis,
        provenance="nominal",
        method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"connector": connector}},
    )
    path = _store().save(record)
    print(f"wrote {path}")
    print(f"  {ref.model!r} (id {ref.id}): R{info.chromaticity['r']} G{info.chromaticity['g']} "
          f"B{info.chromaticity['b']} W{info.chromaticity['w']}, gamma {info.gamma:.2f}")
    return 0


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


def _resolve_device(device_id: str | None) -> dict:
    """The display device this session is measuring. Without an explicit
    `--device-id`, reads this laptop's own eDP EDID (the only display
    known to be attached, docs/design.md §0) -- a synthetic run with no
    real panel passes `--device-id` explicitly instead."""
    if device_id is not None:
        return {"kind": "display", "model": device_id, "id": device_id, "firmware": ""}
    blobs = devicesmod.list_linux_edids()
    if not blobs:
        raise RuntimeError("no EDID found and no --device-id given -- pass --device-id explicitly (e.g. for a synthetic run)")
    info = devicesmod.parse_edid(blobs[sorted(blobs)[0]])
    return devicesmod.display_ref(info).to_dict()


def _build_backend(name: str, *, store: storemod.Store, device_id: str):
    if name == "synthetic":
        return SyntheticBackend()
    if name == "argyll":
        return ArgyllBackend()
    if name == "spectro":
        raise RuntimeError("the spectro backend needs a spectra_for_patch callable; wire it up in a script, not this CLI stub")
    if name == "camera":
        raise RuntimeError("the camera backend needs a frame_for_patch callable; wire it up in a script, not this CLI stub")
    raise ValueError(f"unknown backend {name!r}")


def _all_measurement_patches(steps: int) -> list:
    flat = []
    flat += patchesmod.gray_ramp(steps)
    for ch in ("r", "g", "b"):
        flat += patchesmod.channel_ramp(ch, steps)
    flat += patchesmod.additivity_set()
    flat += patchesmod.primaries_secondaries()
    grid = patchesmod.uniformity_grid()
    flat_grid = [p for row in grid for p in row]
    return flat, grid, flat_grid


def _measure_patches(backend, patches: list, *, width, height, windowed, sleep=None):
    """Measure `patches` with `backend`. The synthetic backend needs no
    physical window (there's no real screen to show anything on); every
    other backend opens the patch window and drives it via
    ``window.run_patch_sequence`` (using the backend's own
    `measure_via_window` where available, e.g. ``ArgyllBackend``)."""
    if isinstance(backend, SyntheticBackend):
        return backend.measure(patches)
    screen = windowmod.open_window(width, height, fullscreen=not windowed)
    try:
        if hasattr(backend, "measure_via_window"):
            return backend.measure_via_window(screen, patches, sleep=sleep)
        # No backend currently reaches this branch without a
        # `measure_via_window` (camera/spectro need capture orchestration
        # this CLI stub doesn't wire up -- see `_build_backend`); kept as
        # a plain fallback for a future backend that measures without one.
        return backend.measure(patches)
    finally:
        windowmod.close_window()


def _cmd_measure(args) -> int:
    store = _store()
    device = _resolve_device(args.device_id)

    state = osstate.gather(confirm_hdr_off=args.confirm_hdr_off, osd={"brightness": args.osd_brightness, "mode": args.osd_mode})
    hdr_refusal = osstate.refuse_if_hdr_on(state)
    if hdr_refusal is not None:
        analysis = Analysis(refusals=[hdr_refusal])
        record = storemod.Record.from_analysis(
            kind="display.measurement",
            device=device,
            analysis=analysis,
            provenance="measured",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend}},
            conditions={"osstate": state.to_dict()},
        )
        path = store.save(record)
        print(f"refused (HDR): wrote {path}")
        return 1

    backend = _build_backend(args.backend, store=store, device_id=device["id"])
    ramp_patches, grid_patches, flat_grid_patches = _all_measurement_patches(args.steps)
    all_patches = ramp_patches + flat_grid_patches
    measurements = _measure_patches(backend, all_patches, width=args.width, height=args.height, windowed=args.windowed)
    by_label = {p.label: m.xyz for p, m in zip(all_patches, measurements, strict=True)}

    channel_index = {"r": 0, "g": 1, "b": 2}

    def _ramp(channel: str) -> dict:
        prefix = f"{channel}-ramp-"
        levels, xyzs = [], []
        for p, m in zip(all_patches, measurements, strict=True):
            if p.label.startswith(prefix):
                levels.append(p.rgb[channel_index[channel]])
                xyzs.append(list(m.xyz))
        return {"levels": levels, "xyz": xyzs}

    primaries_measured = {
        "r": by_label["additivity-r"], "g": by_label["additivity-g"], "b": by_label["additivity-b"],
        "w": by_label["additivity-w"], "k": by_label["additivity-k"],
    }
    trc_analysis = analysismod.trc_fit(
        {"r": _ramp("r"), "g": _ramp("g"), "b": _ramp("b")}, black_y=float(primaries_measured["k"][1])
    )
    additivity_analysis = analysismod.additivity(
        primaries_measured["k"], primaries_measured["r"], primaries_measured["g"], primaries_measured["b"], primaries_measured["w"]
    )
    black_contrast_analysis = analysismod.black_and_contrast(primaries_measured["k"], primaries_measured["w"])

    nominal_record = store.latest("display.nominal", device["id"])
    primaries_analysis = None
    if nominal_record is not None and nominal_record.status == "ok":
        primaries_analysis = analysismod.primaries_vs_edid(primaries_measured, nominal_record.result["chromaticity"])

    grid_xyz = [[list(by_label[p.label]) for p in row] for row in grid_patches]
    uniformity_analysis = analysismod.uniformity(grid_xyz)

    combined = Analysis(
        result={
            "trc": trc_analysis.result,
            "additivity": additivity_analysis.result,
            "black_contrast": black_contrast_analysis.result,
            "uniformity": uniformity_analysis.result,
            "primaries_measured": {k: list(v) for k, v in primaries_measured.items()},
            "samples": {"rgb": [list(p.rgb) for p in all_patches], "xyz": [list(m.xyz) for m in measurements]},
            "backend_accuracy": backend.accuracy().to_dict(),
        },
        refusals=list(trc_analysis.refusals) + list(black_contrast_analysis.refusals) + list(uniformity_analysis.refusals),
    )
    if primaries_analysis is not None:
        combined.result["primaries"] = primaries_analysis.result

    record = storemod.Record.from_analysis(
        kind="display.measurement",
        device=device,
        analysis=combined,
        provenance="measured",
        method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend, "steps": args.steps}},
        conditions={"osstate": state.to_dict()},
    )
    path = store.save(record)
    print(f"wrote {path} (status={record.status})")
    return 0 if record.status == "ok" else 1


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def _cmd_profile(args) -> int:
    store = _store()
    device = _resolve_device(args.device_id)
    measurement = store.latest("display.measurement", device["id"])
    if measurement is None or measurement.status != "ok":
        print(f"no passing display.measurement record for device {device['id']!r}")
        return 1

    out_path = Path(args.out) if args.out else config.records_dir() / device["id"] / "display-profile.icc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = measurement.result["samples"]
    additivity_analysis = Analysis(result=measurement.result["additivity"])
    trc_analysis = Analysis(result=measurement.result["trc"])
    primaries_measured = measurement.result["primaries_measured"]

    try:
        result = profilemod.build_profile(
            out_path,
            rgb_list=[tuple(v) for v in samples["rgb"]],
            xyz_list=[tuple(v) for v in samples["xyz"]],
            additivity_analysis=additivity_analysis,
            trc_analysis=trc_analysis,
            primaries_measured=primaries_measured,
        )
    except profilemod.ProfileRefused as exc:
        analysis = Analysis(refusals=[Refusal("profile_refused", str(exc))])
        record = storemod.Record.from_analysis(
            kind="display.profile", device=device, analysis=analysis, provenance="derived",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {}},
            derived_from=[measurement.id],
        )
        store.save(record)
        print(f"refused: {exc}")
        return 1

    analysis = Analysis(result={"path": str(result.path), "method": result.method, "profcheck_ok": result.profcheck_ok})
    record = storemod.Record.from_analysis(
        kind="display.profile", device=device, analysis=analysis, provenance="derived",
        method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {}},
        derived_from=[measurement.id],
    )
    path = store.save(record)
    print(f"wrote {path} -- profile at {result.path} ({result.method})")
    return 0


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _cmd_install(args) -> int:
    store = _store()
    device = _resolve_device(args.device_id)
    profile_record = store.latest("display.profile", device["id"])
    if profile_record is None or profile_record.status != "ok":
        print(f"no passing display.profile record for device {device['id']!r}")
        return 1
    icc_path = profile_record.result["path"]

    if sys.platform == "win32":
        report = install_windows.install(icc_path)
    else:
        report = install_linux.install(icc_path, write_autostart=args.write_autostart)

    for step in report.steps:
        print(f"  [{'ok' if step['ok'] else 'FAIL'}] {step['step']}: {step['detail']}")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _cmd_validate(args) -> int:
    store = _store()
    device = _resolve_device(args.device_id)
    profile_record = store.latest("display.profile", device["id"])
    if profile_record is None or profile_record.status != "ok":
        print(f"no passing display.profile record for device {device['id']!r}")
        return 1
    measurement = store.latest("display.measurement", device["id"])

    backend = _build_backend(args.backend, store=store, device_id=device["id"])
    lab_patches = patchesmod.validation_set()
    lab_targets = [p.lab_target for p in lab_patches]
    rgb_values = validatemod.lab_to_rgb_via_profile(profile_record.result["path"], lab_targets)
    patches_to_show = [
        patchesmod.Patch(rgb=rgb, label=p.label, lab_target=p.lab_target) for p, rgb in zip(lab_patches, rgb_values, strict=True)
    ]
    measurements = _measure_patches(backend, patches_to_show, width=args.width, height=args.height, windowed=args.windowed)
    measured_xyz = [m.xyz for m in measurements]

    white_xyz = measurement.result["primaries_measured"]["w"] if measurement is not None else measured_xyz[0]
    analysis = validatemod.validate(measured_xyz, lab_targets, white_xyz, backend.accuracy().de00_estimate)

    record = storemod.Record.from_analysis(
        kind="display.validation", device=device, analysis=analysis, provenance="measured",
        method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend}},
        derived_from=[profile_record.id],
    )
    path = store.save(record)
    print(f"wrote {path} (status={record.status}): mean ΔE00={analysis.result.get('de00_mean'):.3f}")
    return 0 if record.status == "ok" else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _cmd_report(args) -> int:
    store = _store()
    device = _resolve_device(args.device_id)
    measurement = store.latest("display.measurement", device["id"])
    if measurement is None:
        print(f"no display.measurement record for device {device['id']!r}")
        return 1
    validation = store.latest("display.validation", device["id"])

    kwargs = {
        "trc_analysis": Analysis(result=measurement.result.get("trc", {})),
        "additivity_analysis": Analysis(result=measurement.result.get("additivity", {})),
        "uniformity_analysis": Analysis(result=measurement.result.get("uniformity", {})),
        "backend_accuracy": measurement.result.get("backend_accuracy"),
    }
    if "primaries" in measurement.result:
        kwargs["primaries_analysis"] = Analysis(result=measurement.result["primaries"])
    if validation is not None:
        kwargs["validation_analysis"] = Analysis(result=validation.result)

    html = reportmod.render(
        title="Display report",
        device=device,
        provenance=measurement.provenance,
        status=measurement.status,
        refusals=measurement.refusals,
        conditions=measurement.conditions,
        **kwargs,
    )
    out_path = Path(args.out) if args.out else config.records_dir() / device["id"] / "display-report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"wrote {out_path}")
    return 0

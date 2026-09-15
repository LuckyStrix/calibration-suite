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
from calsuite.display import constants as dc
from calsuite.display import install_linux, install_windows, osstate
from calsuite.display import report as reportmod
from calsuite.display import validate as validatemod
from calsuite.fit import Analysis, Refusal

# NOTE on imports: `analysis.py`, `patches.py` and `profile.py` pull in
# `colour` (colour-science) at their own module level, and `window.py`
# pulls in `pygame` -- both slow to import and, for pygame, noisy (its
# "Hello from the pygame community" banner). Importing any of them here,
# unconditionally, would make `calsuite --help`/`calsuite doctor` (which
# never touch a display backend at all) pay that cost just to build the
# argparse tree. So they're imported locally, inside the specific
# functions that use them, not at this module's top level -- the same
# rule applies across every area's `commands.py`.

_METHOD_NAME = "calsuite.display.commands"


def _add_camera_and_spectro_backend_args(p) -> None:
    """Flags only ``--backend camera``/``--backend spectro`` read (shared by
    ``measure`` and ``validate``, which dispatch through the same
    ``_measure_via_backend``) -- harmless no-ops for ``argyll``/``synthetic``."""
    p.add_argument(
        "--camera-device-id", default=None,
        help="(--backend camera) the camera whose exportable camera.color record to use",
    )
    p.add_argument(
        "--from", dest="from_dir", default=None,
        help="(--backend camera) folder of already-captured frames, filename-sorted to pair 1:1 with the patch sequence",
    )
    p.add_argument(
        "--capture", action="store_true",
        help="(--backend camera) drive tethered gphoto2 capture through the patch window instead of --from",
    )
    p.add_argument(
        "--capture-dir", default=None,
        help="(--backend camera, with --capture) where captured frames are saved (default: <captures>/display-measure)",
    )
    p.add_argument(
        "--absolute-scale", type=float, default=None,
        help="(--backend camera) cd/m^2 per unit, once cross-checked against another backend (design §5.1)",
    )
    p.add_argument(
        "--spectra", dest="spectra_dir", default=None,
        help="(--backend spectro) folder of one '<patch-label>.csv' spectrum file per patch",
    )
    p.add_argument(
        "--luminance-scale", type=float, default=1.0,
        help="(--backend spectro) cd/m^2 per integrated-Y unit, from the spectrophotometer's own radiometric calibration",
    )
    p.add_argument(
        "--cross-checked-against", default=None,
        help="(--backend camera/spectro) name of the backend this one was cross-checked against, for the accuracy statement",
    )


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
    _add_camera_and_spectro_backend_args(measure_p)
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
    _add_camera_and_spectro_backend_args(validate_p)
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


def build_nominal_record(connector: str, edid_bytes: bytes) -> storemod.Record:
    """A ``display.nominal`` record (provenance ``"nominal"``, house rule 2:
    an EDID reading is a claim, not a measurement) from one connector's raw
    EDID bytes. Pulled out of ``_cmd_nominal`` so ``calsuite devices``
    (cli.py) can write one of these per *detected* display too, not just
    the single one ``display nominal`` targets by ``--connector``."""
    info = devicesmod.parse_edid(edid_bytes)
    ref = devicesmod.display_ref(info)
    analysis = Analysis(
        result={
            "chromaticity": {k: list(v) for k, v in info.chromaticity.items()},
            "physical_size_mm": list(info.physical_size_mm),
            "gamma": info.gamma,
            "name": info.name,
            "edid_hash": devicesmod.edid_hash(edid_bytes),
        }
    )
    return storemod.Record.from_analysis(
        kind="display.nominal",
        device=ref.to_dict(),
        analysis=analysis,
        provenance="nominal",
        method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"connector": connector}},
    )


def _cmd_nominal(args) -> int:
    blobs = devicesmod.list_linux_edids()
    if not blobs:
        print("no EDID found under /sys/class/drm")
        return 1
    connector = args.connector or sorted(blobs)[0]
    if connector not in blobs:
        print(f"connector {connector!r} not found; available: {sorted(blobs)}")
        return 1
    record = build_nominal_record(connector, blobs[connector])
    path = _store().save(record)
    info = devicesmod.parse_edid(blobs[connector])
    print(f"wrote {path}")
    print(f"  {record.device['model']!r} (id {record.device['id']}): R{info.chromaticity['r']} G{info.chromaticity['g']} "
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


def _all_measurement_patches(steps: int) -> list:
    from calsuite.display import patches as patchesmod

    flat = []
    flat += patchesmod.gray_ramp(steps)
    for ch in ("r", "g", "b"):
        flat += patchesmod.channel_ramp(ch, steps)
    flat += patchesmod.additivity_set()
    flat += patchesmod.primaries_secondaries()
    grid = patchesmod.uniformity_grid()
    flat_grid = [p for row in grid for p in row]
    return flat, grid, flat_grid


def _reads_live_screen(args) -> bool:
    """Whether this run measures what is on the screen *right now* -- the
    only case in which the OS-state pre-flight (linear video-card LUT, no
    other VCGT loader) describes the conditions the numbers were taken
    under. ``synthetic`` stands in for the whole display; ``spectro`` and
    ``camera --from DIR`` read files captured in an earlier session.
    """
    if args.backend in ("synthetic", "spectro"):
        return False
    if args.backend == "camera":
        return bool(getattr(args, "capture", False))
    return True


def _measure_via_backend(args, patches: list, *, store: storemod.Store):
    """Get one ``Measurement`` per patch (same order as ``patches``) from
    whichever backend ``args.backend`` names, plus that backend's
    ``Accuracy`` -- the one place ``measure`` and ``validate`` both
    dispatch through, since they need the exact same four backends wired
    up the same way.

    Whether a live patch window opens depends on the backend, not on a
    uniform rule: ``synthetic`` never needs one (there's no real screen --
    ``synth/display.py`` stands in for the whole display); ``argyll``
    always needs one (a colorimeter reads whatever is currently on
    screen); ``camera`` needs one only for ``--capture`` (a tethered
    gphoto2 session has to show each patch before photographing it) --
    given ``--from DIR`` instead, the frames were already captured (by a
    previous ``--capture`` run, or by hand), so nothing needs to be shown
    now; ``spectro`` never needs one at all -- the spectrophotometer reads
    its own already-captured spectra files, independent of this process.
    """
    name = args.backend
    if name == "synthetic":
        from calsuite.display.backends.synthetic import SyntheticBackend

        backend = SyntheticBackend()
        return backend.measure(patches), backend.accuracy()

    if name == "argyll":
        from calsuite.display import window as windowmod
        from calsuite.display.backends.argyll import ArgyllBackend

        screen = windowmod.open_window(args.width, args.height, fullscreen=not args.windowed)
        try:
            backend = ArgyllBackend()
            return backend.measure_via_window(screen, patches, sleep=None), backend.accuracy()
        finally:
            windowmod.close_window()

    if name == "camera":
        return _measure_via_camera(args, patches, store=store)

    if name == "spectro":
        return _measure_via_spectro(args, patches)

    raise ValueError(f"unknown backend {name!r}")


def _measure_via_camera(args, patches: list, *, store: storemod.Store):
    from calsuite.display.backends.camera import (
        CameraBackend,
        NoCameraColorRecord,
        capture_via_gphoto2,
        frames_from_manual_folder,
    )

    camera_device_id = getattr(args, "camera_device_id", None)
    if not camera_device_id:
        raise SystemExit(
            "--backend camera needs --camera-device-id (the camera whose exportable camera.color record to use)"
        )

    if getattr(args, "capture", False):
        from calsuite.display import window as windowmod

        capture_dir = Path(args.capture_dir) if args.capture_dir else config.captures_dir() / "display-measure"
        screen = windowmod.open_window(args.width, args.height, fullscreen=not args.windowed)
        try:
            frames_by_label = capture_via_gphoto2(screen, patches, capture_dir)
        finally:
            windowmod.close_window()
    elif getattr(args, "from_dir", None):
        # Pairing rule (display.backends.camera.frames_from_manual_folder):
        # every raw/.npz file directly under the folder, in filename-sorted
        # order, pairs 1:1 with `patches` in exactly the order given here --
        # the same order this command's own patch set is built in (ramps,
        # then additivity, then primaries/secondaries, then the uniformity
        # grid, for `measure`; the validation set's own order for
        # `validate`). A capture script (or a prior `--capture` run) must
        # save frames in that same sequence, e.g. `0000.npz, 0001.npz, ...`.
        # A count mismatch refuses with a specific message naming both
        # counts, rather than a `zip(..., strict=True)` ValueError three
        # frames down or a silent truncation.
        try:
            frames_by_label = frames_from_manual_folder(Path(args.from_dir), patches)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        raise SystemExit(
            "--backend camera needs --from DIR (already-captured frames, filename-sorted to match the patch "
            "sequence) or --capture (tethered gphoto2, driven live through the patch window)"
        )

    try:
        backend = CameraBackend(
            device_id=camera_device_id,
            store=store,
            frame_for_patch=lambda p: frames_by_label[p.label],
            absolute_scale_cdm2_per_unit=getattr(args, "absolute_scale", None),
            cross_checked_against=getattr(args, "cross_checked_against", None),
        )
    except NoCameraColorRecord as exc:
        # The clear, specific error this wave's contract calls for (see
        # CameraBackend.__post_init__'s own docstring) -- not a bare
        # KeyError/AttributeError several frames further in.
        raise SystemExit(str(exc)) from exc
    return backend.measure(patches), backend.accuracy()


def _measure_via_spectro(args, patches: list):
    from calsuite.display.backends.spectro import SpectroBackend

    spectra_dir = getattr(args, "spectra_dir", None)
    if not spectra_dir:
        raise SystemExit(
            "--backend spectro needs --spectra DIR -- one '<patch-label>.csv' (wavelength_nm,value) file per "
            "patch, e.g. gray-ramp-0.5000.csv, additivity-w.csv (see display.backends.spectro.read_spectrum_csv "
            "and Patch.label for exactly which labels a given --steps/patch set needs)"
        )
    spectra_dir = Path(spectra_dir)
    missing = [p.label for p in patches if not (spectra_dir / f"{p.label}.csv").exists()]
    if missing:
        raise SystemExit(
            f"{spectra_dir}: missing spectrum file(s) for {len(missing)}/{len(patches)} patch(es), e.g. "
            f"{missing[:5]} -- expected one '<patch-label>.csv' per patch"
        )
    backend = SpectroBackend(
        spectra_for_patch=lambda p: spectra_dir / f"{p.label}.csv",
        luminance_scale=getattr(args, "luminance_scale", 1.0),
        cross_checked_against=getattr(args, "cross_checked_against", None),
    )
    return backend.measure(patches), backend.accuracy()


def _cmd_measure(args) -> int:
    from calsuite.display import analysis as analysismod

    store = _store()
    device = _resolve_device(args.device_id)

    state = osstate.gather(confirm_hdr_off=args.confirm_hdr_off, osd={"brightness": args.osd_brightness, "mode": args.osd_mode})
    preflight = [r for r in (osstate.refuse_if_hdr_on(state),) if r is not None]
    if _reads_live_screen(args):
        # Design §5.3's other two pre-flights only mean anything for a
        # backend that reads the screen *now*: a linear video-card LUT and
        # "nothing else is loading a VCGT" are statements about the machine
        # at measurement time. `synthetic` never touches a real screen, and
        # `camera --from`/`spectro` read frames captured in an earlier
        # session whose OS state this process can no longer speak for.
        preflight += [
            r
            for r in (osstate.refuse_if_gamma_not_reset(state), osstate.refuse_if_profile_loader_active(state))
            if r is not None
        ]
    if preflight:
        analysis = Analysis(refusals=preflight)
        record = storemod.Record.from_analysis(
            kind="display.measurement",
            device=device,
            analysis=analysis,
            provenance="measured",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend}},
            conditions={"osstate": state.to_dict()},
        )
        path = store.save(record)
        print(f"refused ({', '.join(r.check for r in preflight)}): wrote {path}")
        return 1

    ramp_patches, grid_patches, flat_grid_patches = _all_measurement_patches(args.steps)
    all_patches = ramp_patches + flat_grid_patches
    measurements, accuracy = _measure_via_backend(args, all_patches, store=store)
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
            # `samples` is the profile builder's training set (_cmd_profile
            # hands it straight to profile.write_ti3 -> colprof), so it holds
            # the full-screen centered colorimetric patches *only*. The
            # uniformity grid is 25 more patches that are all device RGB
            # (1,1,1) but measured off-center at UNIFORMITY_PATCH_SIZE_FRAC:
            # feeding them in gave colprof 26 contradictory readings for
            # white (a 25% luminance spread on a panel with real
            # non-uniformity), pulling the white/shaper normalization toward
            # the screen corners. Their measurements are already reported,
            # in the form that makes sense for them, under "uniformity".
            "samples": {
                "rgb": [list(p.rgb) for p in ramp_patches],
                "xyz": [list(m.xyz) for m in measurements[: len(ramp_patches)]],
            },
            "backend_accuracy": accuracy.to_dict(),
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
    from calsuite.display import profile as profilemod

    store = _store()
    device = _resolve_device(args.device_id)
    measurement = store.latest("display.measurement", device["id"])
    if measurement is None or measurement.status != "ok":
        # A missing prerequisite is a finding, not silence (house rule 3):
        # this used to print and return with no display.profile record
        # saved at all, which is exactly the shape camera.commands._cmd_iso
        # had for its own missing-prerequisite path -- name the real cause
        # and save a refused record, don't vanish.
        cause = (
            f"no display.measurement record exists for device {device['id']!r}"
            if measurement is None
            else f"the latest display.measurement record ({measurement.id}) is refused"
        )
        analysis = Analysis(refusals=[Refusal("no_passing_measurement", f"can't build a profile: {cause}")])
        record = storemod.Record.from_analysis(
            kind="display.profile", device=device, analysis=analysis, provenance="derived",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {}},
            derived_from=[measurement.id] if measurement is not None else [],
        )
        store.save(record)
        print(f"refused: {cause}")
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
    from calsuite.display import patches as patchesmod

    store = _store()
    device = _resolve_device(args.device_id)
    profile_record = store.latest("display.profile", device["id"])
    if profile_record is None or profile_record.status != "ok":
        # Same missing-prerequisite shape as _cmd_profile above: don't skip
        # the real backend measurement AND skip saving a record -- name the
        # cause and save a refused display.validation record (house rule
        # 3). No fresh measurement is attempted here (there's nothing valid
        # to validate against), same as _cmd_measure's own HDR pre-flight
        # refusal already stamps "measured" before any patch is shown.
        cause = (
            f"no display.profile record exists for device {device['id']!r}"
            if profile_record is None
            else f"the latest display.profile record ({profile_record.id}) is refused"
        )
        analysis = Analysis(refusals=[Refusal("no_passing_profile", f"can't validate: {cause}")])
        record = storemod.Record.from_analysis(
            kind="display.validation", device=device, analysis=analysis, provenance="measured",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend}},
            derived_from=[profile_record.id] if profile_record is not None else [],
        )
        store.save(record)
        print(f"refused: {cause}")
        return 1
    measurement = store.latest("display.measurement", device["id"])
    if measurement is None or measurement.status != "ok" or "w" not in measurement.result.get("primaries_measured", {}):
        # Every ΔE00 below is computed against this display's own measured
        # white. Falling back to "the first validation patch" (which is
        # cc24-dark skin, L*~37 and strongly chromatic) rather than saying
        # so turns the whole record into garbage that reads as a
        # measurement -- name the cause and refuse instead (house rule 3).
        cause = (
            f"no display.measurement record exists for device {device['id']!r}"
            if measurement is None
            else f"the latest display.measurement record ({measurement.id}) is refused or has no measured white"
        )
        analysis = Analysis(refusals=[Refusal("no_measured_white", f"can't validate: {cause}")])
        record = storemod.Record.from_analysis(
            kind="display.validation", device=device, analysis=analysis, provenance="measured",
            method={"name": _METHOD_NAME, "calsuite_version": __version__, "params": {"backend": args.backend}},
            derived_from=[profile_record.id],
        )
        store.save(record)
        print(f"refused: {cause}")
        return 1

    lab_patches = patchesmod.validation_set()
    lab_targets = [p.lab_target for p in lab_patches]
    rgb_values = validatemod.lab_to_rgb_via_profile(profile_record.result["path"], lab_targets)
    patches_to_show = [
        patchesmod.Patch(rgb=rgb, label=p.label, lab_target=p.lab_target) for p, rgb in zip(lab_patches, rgb_values, strict=True)
    ]
    measurements, accuracy = _measure_via_backend(args, patches_to_show, store=store)
    measured_xyz = [m.xyz for m in measurements]

    white_xyz = measurement.result["primaries_measured"]["w"]
    analysis = validatemod.validate(measured_xyz, lab_targets, white_xyz, accuracy.de00_estimate)

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
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0

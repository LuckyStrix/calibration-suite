"""``calsuite`` command-line entry point.

Each area (camera, lens, display) registers its own subparsers via
``<area>.commands.register(subparsers)`` (docs/implementation-plan.md: "each
area registered via calsuite/<area>/commands.py: register(subparsers)"), so
this module never grows area-specific analysis logic -- it only wires the
areas together and implements the one command that's cross-cutting enough
to belong here: ``devices``.
"""

from __future__ import annotations

import argparse
import sys

from calsuite import __version__, config, devices as devicesmod, store
from calsuite.camera import color_commands, commands as camera_commands
from calsuite.capture import gphoto2
from calsuite.display import commands as display_commands
from calsuite.lens import commands as lens_commands


def _cmd_devices(args) -> int:
    """List what's connected, what's known -- the one command
    implementation-plan.md's Wave 1 row asks to actually work now, ahead of
    the rest of the CLI. No hardware or files are required: every source
    here degrades to "unavailable" rather than raising when a tool or a
    sysfs node isn't present (this is meant to run usefully on a fresh
    checkout, on any OS, before anything has been measured).
    """
    print("Camera (gphoto2):")
    try:
        connected = gphoto2.is_camera_connected()
    except Exception as exc:  # pragma: no cover -- defensive; gphoto2 quirks vary by system
        print(f"  could not check: {exc}")
    else:
        if connected:
            print(gphoto2.detect().rstrip())
        else:
            print("  none detected (or gphoto2 is not installed)")

    print("\nDisplays (EDID):")
    written = _print_linux_edids_and_write_nominal_records()
    if sys.platform == "win32":
        _print_windows_edids()
    if written:
        print("\nWrote display.nominal records (provenance=nominal, from EDID):")
        for ref, path in written:
            print(f"  {ref.model!r} (id {ref.id}) -> {path}")

    print("\nKnown records:")
    _print_known_records()
    return 0


def _print_linux_edids_and_write_nominal_records() -> list:
    """Print every Linux EDID found under ``/sys/class/drm``, and also
    write a ``display.nominal`` record for each (docs/implementation-plan.md
    Wave 3 Build item: "calsuite devices extended to write the
    display.nominal record for each detected display, provenance
    nominal, from EDID"). Returns the ``(DeviceRef, path)`` pairs written,
    so the caller can report them; never raises -- a display whose EDID
    doesn't parse is reported and skipped, not fatal to the rest of
    ``devices``."""
    try:
        blobs = devicesmod.list_linux_edids()
    except Exception as exc:  # pragma: no cover -- no /sys/class/drm on non-Linux
        print(f"  could not read /sys/class/drm: {exc}")
        return []
    if not blobs:
        print("  none found under /sys/class/drm")
        return []
    st = store.Store(config.records_dir())
    written = []
    for connector, data in sorted(blobs.items()):
        try:
            info = devicesmod.parse_edid(data)
        except ValueError as exc:
            print(f"  {connector}: unreadable EDID ({exc})")
            continue
        ref = devicesmod.display_ref(info)
        r, g, b, w = (info.chromaticity[k] for k in "rgbw")
        print(
            f"  {connector}: {ref.model!r} (id {ref.id}), "
            f"{info.physical_size_mm[0]:.0f}x{info.physical_size_mm[1]:.0f}mm, "
            f"gamma {info.gamma:.2f}, "
            f"R{r} G{g} B{b} W{w}"
        )
        record = display_commands.build_nominal_record(connector, data)
        path = st.save(record)
        written.append((ref, path))
    return written


def _print_windows_edids() -> None:  # pragma: no cover -- exercised only on Windows
    try:
        blobs = devicesmod.read_edid_windows()
    except Exception as exc:
        print(f"  could not read the registry: {exc}")
        return
    for instance, data in sorted(blobs.items()):
        try:
            info = devicesmod.parse_edid(data)
        except ValueError as exc:
            print(f"  {instance}: unreadable EDID ({exc})")
            continue
        print(f"  {instance}: {devicesmod.display_ref(info).model!r}")


def _print_known_records() -> None:
    root = config.records_dir()
    st = store.Store(root)
    records = list(st.all())
    if not records:
        print(f"  none under {root}")
        return
    by_device: dict[str, list] = {}
    for r in records:
        by_device.setdefault(r.device.get("id", "?"), []).append(r)
    for device_id, recs in sorted(by_device.items()):
        model = recs[0].device.get("model", "")
        print(f"  {device_id} ({model}): {len(recs)} record(s)")
        for r in sorted(recs, key=lambda x: x.created):
            stale = " [stale]" if st.is_stale(r) else ""
            print(f"    {r.kind:<20} {r.created}  {r.provenance:<9} {r.status}{stale}")


def _cmd_doctor(args) -> int:
    from calsuite import doctor as doctormod

    report = doctormod.run()
    print(doctormod.format_report(report))
    return 0 if report.ok else 1


def _cmd_demo(args) -> int:
    from calsuite import demo as demomod

    result = demomod.run_demo(args.out)
    print(f"wrote {result['index']}")
    for name, path in result["reports"].items():
        print(f"  {name}: {path if path else '(not produced)'}")
    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    return 0 if any(v is not None for v in result["reports"].values()) else 1


# ---------------------------------------------------------------------------
# export lensfun|dcp|icc -- thin aliases to the area exporters (design §7:
# "calsuite export lensfun"), so an export doesn't require remembering which
# area owns which format. Each just builds the argparse.Namespace the real
# command function (lens_commands._cmd_export / color_commands._cmd_export)
# already expects and calls straight through -- no export logic lives here.
# ---------------------------------------------------------------------------


def _cmd_export_lensfun(args) -> int:
    ns = argparse.Namespace(device_id=args.device_id, lens_model=args.lens_model, out=args.out)
    return lens_commands._cmd_export(ns)


def _cmd_export_dcp(args) -> int:
    ns = argparse.Namespace(record=args.record, record2=args.record2, dcp=args.out, icc=None)
    return color_commands._cmd_export(ns)


def _cmd_export_icc(args) -> int:
    ns = argparse.Namespace(record=args.record, record2=args.record2, dcp=None, icc=args.out)
    return color_commands._cmd_export(ns)


def _cmd_export_help(args) -> int:
    print("calsuite export: pass a subcommand (lensfun, dcp, icc)")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calsuite",
        description="Camera sensor, lens and display calibration with provenance.",
    )
    parser.add_argument("--version", action="version", version=f"calsuite {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    devices_parser = subparsers.add_parser("devices", help="list connected/known camera, lens and display devices")
    devices_parser.set_defaults(func=_cmd_devices)

    camera_commands.register(subparsers)
    lens_commands.register(subparsers)
    display_commands.register(subparsers)

    export_parser = subparsers.add_parser("export", help="export records to lensfun/DCP/ICC")
    export_sub = export_parser.add_subparsers(dest="export_command")
    export_parser.set_defaults(func=_cmd_export_help)

    lensfun_p = export_sub.add_parser("lensfun", help="write lensfun XML for a lens device's latest ok distortion/tca records")
    lensfun_p.add_argument("--device-id", required=True)
    lensfun_p.add_argument("--lens-model", required=True)
    lensfun_p.add_argument("--out", default=None)
    lensfun_p.set_defaults(func=_cmd_export_lensfun)

    dcp_p = export_sub.add_parser("dcp", help="write a DCP camera profile from a camera.color record")
    dcp_p.add_argument("--record", required=True, help="path to a saved camera.color record JSON")
    dcp_p.add_argument("--record2", default=None, help="a second record (e.g. tungsten) for a dual-illuminant DCP")
    dcp_p.add_argument("--out", required=True)
    dcp_p.set_defaults(func=_cmd_export_dcp)

    icc_p = export_sub.add_parser("icc", help="write a camera input ICC profile from a camera.color record")
    icc_p.add_argument("--record", required=True, help="path to a saved camera.color record JSON")
    icc_p.add_argument("--record2", default=None)
    icc_p.add_argument("--out", required=True)
    icc_p.set_defaults(func=_cmd_export_icc)

    doctor_parser = subparsers.add_parser("doctor", help="check tool availability and record staleness")
    doctor_parser.set_defaults(func=_cmd_doctor)

    demo_parser = subparsers.add_parser("demo", help="run every analysis on synthetic data")
    demo_parser.add_argument("--out", required=True, help="directory to write the temporary store and HTML reports into")
    demo_parser.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list | None = None) -> int:
    # Several reports print a line containing 'Delta' as the actual Greek
    # letter (e.g. "mean ΔE00=..."). On Windows, console/redirected
    # stdout defaults to the system codepage (cp1252 and friends), which
    # cannot encode it -- `print()` would raise UnicodeEncodeError before a
    # single character reaches the screen or a captured-output log.
    # `reconfigure` (TextIOWrapper, Python 3.7+) is a no-op-safe way to
    # force UTF-8 for this process's own stdout/stderr; scoped to Windows
    # so Linux/macOS (already UTF-8 almost everywhere) are untouched.
    if sys.platform == "win32":  # pragma: no cover -- exercised only on Windows CI
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

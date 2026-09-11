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
from calsuite.camera import commands as camera_commands
from calsuite.capture import gphoto2
from calsuite.display import commands as display_commands
from calsuite.lens import commands as lens_commands


def _not_built(name: str):
    def _run(args) -> int:
        print(f"calsuite {name}: not built yet.")
        return 1

    return _run


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
    _print_linux_edids()
    if sys.platform == "win32":
        _print_windows_edids()

    print("\nKnown records:")
    _print_known_records()
    return 0


def _print_linux_edids() -> None:
    try:
        blobs = devicesmod.list_linux_edids()
    except Exception as exc:  # pragma: no cover -- no /sys/class/drm on non-Linux
        print(f"  could not read /sys/class/drm: {exc}")
        return
    if not blobs:
        print("  none found under /sys/class/drm")
        return
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

    for name, help_text in (
        ("export", "export records to lensfun/DCP/ICC"),
        ("doctor", "check tool availability and record staleness"),
        ("demo", "run every analysis on synthetic data"),
    ):
        p = subparsers.add_parser(name, help=f"{help_text} (not built yet)")
        p.set_defaults(func=_not_built(name))

    return parser


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

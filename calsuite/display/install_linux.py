"""Install a display profile on Linux (docs/design.md §5.5): `colormgr`
import-profile + device-add-profile + device-make-profile-default, then
`dispwin -I` if present, then (ONLY behind `--write-autostart`) an Openbox
autostart loader line -- this laptop's Openbox/X11 session has no
colord-aware settings daemon (design §0), so nothing else reloads a
profile's VCGT curves at login without one.

`colormgr`'s exact real-world output for `import-profile`/
`get-devices-by-kind` is unverified on this build machine: colormgr itself
is installed (design §0), but talks to a colord D-Bus daemon this Openbox
session doesn't run, so every real invocation here fails with "no
daemon" rather than producing output to parse against. Tests exercise this
module against a fake `colormgr` script that prints the real client's
documented `--help` field names (`colormgr --help`, confirmed on this
machine: "Profile ID", "Device ID" are not literal field names it prints,
but `Object Path` is the identifier every colormgr subcommand accepts back
in -- so lookups key on that). `_parse_id_line` is deliberately tolerant
(matches several plausible label spellings) for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from calsuite import tools

AUTOSTART_PATH = Path.home() / ".config" / "openbox" / "autostart"
AUTOSTART_MARKER = "# calsuite: load display profile VCGT"


@dataclass
class InstallReport:
    """An ordered list of what was attempted and whether it worked --
    design's own requirement: "report exactly what was done"."""

    steps: list = field(default_factory=list)

    def add(self, step: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"step": step, "ok": ok, "detail": detail})

    @property
    def ok(self) -> bool:
        return all(s["ok"] for s in self.steps) if self.steps else False

    def to_dict(self) -> dict:
        return {"steps": self.steps, "ok": self.ok}


def _parse_id_line(text: str, keys: tuple) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        for key in keys:
            prefix = f"{key}:"
            if stripped.lower().startswith(prefix.lower()):
                return stripped.split(":", 1)[1].strip()
    return None


def import_profile(icc_path: Path) -> tuple:
    """`colormgr import-profile <path>`. Returns `(object_path_or_None,
    raw_stdout)`; falls back to `find-profile-by-filename` if the id
    couldn't be parsed out of `import-profile`'s own output."""
    result = tools.run(["colormgr", "import-profile", str(icc_path)])
    object_path = _parse_id_line(result.stdout, ("Object Path", "Profile ID", "Profile"))
    if object_path is None:
        lookup = tools.run(["colormgr", "find-profile-by-filename", str(icc_path)], check=False)
        object_path = _parse_id_line(lookup.stdout, ("Object Path", "Profile ID", "Profile"))
    return object_path, result.stdout


def list_display_devices() -> tuple:
    """`colormgr get-devices-by-kind display` -- **every** display device's
    object path, in the order colord lists them, plus the raw stdout."""
    result = tools.run(["colormgr", "get-devices-by-kind", "display"], check=False)
    paths = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        for key in ("Object Path", "Device ID", "Device"):
            if stripped.lower().startswith(f"{key}:".lower()):
                value = stripped.split(":", 1)[1].strip()
                if value and value not in paths:
                    paths.append(value)
                break
    return paths, result.stdout


def find_display_device() -> tuple:
    """`colormgr get-devices-by-kind display` -- the first display
    device's object path, or `(None, stdout)` if none was found/parsed.

    "The first" is only ever the right answer on a single-display machine;
    ``install_colormgr`` refuses to guess when there is more than one. See
    ``list_display_devices``."""
    paths, stdout = list_display_devices()
    return (paths[0] if paths else None), stdout


def install_colormgr(icc_path: Path, *, device_path: str | None = None) -> InstallReport:
    """Import the profile into colord and make it this display's default.

    ``device_path`` names *which* colord display device to attach it to. On
    a multi-monitor machine this is not optional and cannot be guessed: the
    profile was measured for one specific panel, and colord's enumeration
    order says nothing about which one that is -- attaching it to the wrong
    display silently colour-manages the wrong screen with it. With more than
    one display device present and no ``device_path`` given, this reports a
    failed step listing the candidates rather than picking one.
    """
    report = InstallReport()
    if tools.which("colormgr") is None:
        report.add("colormgr import-profile", False, "colormgr not found on PATH")
        return report

    try:
        profile_id, stdout = import_profile(icc_path)
    except tools.ToolError as exc:
        report.add("colormgr import-profile", False, str(exc))
        return report
    report.add("colormgr import-profile", profile_id is not None, f"profile id: {profile_id!r}")
    if profile_id is None:
        return report

    try:
        candidates, _ = list_display_devices()
    except tools.ToolError as exc:
        report.add("colormgr get-devices-by-kind display", False, str(exc))
        return report

    if device_path is not None:
        device_id = device_path
        if candidates and device_path not in candidates:
            report.add(
                "colormgr get-devices-by-kind display",
                False,
                f"{device_path!r} is not among colord's display devices: {candidates}",
            )
            return report
        report.add("colormgr get-devices-by-kind display", True, f"device id: {device_id!r} (given)")
    elif len(candidates) == 1:
        device_id = candidates[0]
        report.add("colormgr get-devices-by-kind display", True, f"device id: {device_id!r}")
    elif not candidates:
        report.add("colormgr get-devices-by-kind display", False, "no display device found")
        return report
    else:
        report.add(
            "colormgr get-devices-by-kind display",
            False,
            f"{len(candidates)} display devices present ({candidates}) -- pass --colord-device to say which one "
            "this profile is for; colord's enumeration order doesn't identify the panel that was measured",
        )
        return report

    try:
        tools.run(["colormgr", "device-add-profile", device_id, profile_id])
        report.add("colormgr device-add-profile", True)
    except tools.ToolError as exc:
        report.add("colormgr device-add-profile", False, str(exc))
        return report

    try:
        tools.run(["colormgr", "device-make-profile-default", device_id, profile_id])
        report.add("colormgr device-make-profile-default", True)
    except tools.ToolError as exc:
        report.add("colormgr device-make-profile-default", False, str(exc))

    return report


def install_dispwin(icc_path: Path, *, display: int | None = None) -> InstallReport:
    """`dispwin -I <path>`: sets the `_ICC_PROFILE` X atom and loads the
    VCGT curves for the current session (design §5.5). ``display`` becomes
    ``dispwin -d <n>`` -- without it dispwin loads the curves into display
    1, which is only the right screen on a single-display machine."""
    report = InstallReport()
    if tools.which("dispwin") is None:
        report.add("dispwin -I", False, "dispwin not found on PATH")
        return report
    argv = ["dispwin"]
    if display is not None:
        argv += ["-d", str(display)]
    argv += ["-I", str(icc_path)]
    try:
        tools.run(argv)
        report.add("dispwin -I", True, f"display {display}" if display is not None else "")
    except tools.ToolError as exc:
        report.add("dispwin -I", False, str(exc))
    return report


def write_autostart_entry(icc_path: Path, *, autostart_path: Path = AUTOSTART_PATH) -> InstallReport:
    """Add a `dispwin -I <profile>` loader line to Openbox's autostart
    (design §5.5) -- ONLY called when the caller explicitly asked
    (`--write-autostart`); never as a side effect of any other install
    step. Idempotent: a previous calsuite-written marker+line pair is
    replaced in place, not duplicated, on a second run.
    """
    report = InstallReport()
    autostart_path = Path(autostart_path)
    autostart_path.parent.mkdir(parents=True, exist_ok=True)
    existing = autostart_path.read_text(encoding="utf-8") if autostart_path.exists() else ""
    lines = existing.splitlines()

    out_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == AUTOSTART_MARKER:
            i += 2  # drop the marker and the loader line right after it
            continue
        out_lines.append(lines[i])
        i += 1

    new_line = f'dispwin -I "{icc_path}" &'
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    out_lines += ["", AUTOSTART_MARKER, new_line]
    autostart_path.write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")
    report.add("openbox autostart", True, f"wrote {autostart_path}")
    return report


def install(
    icc_path: Path,
    *,
    write_autostart: bool = False,
    autostart_path: Path = AUTOSTART_PATH,
    colord_device: str | None = None,
    dispwin_display: int | None = None,
) -> InstallReport:
    """Full Linux install path: colormgr, then `dispwin -I`, then (only if
    `write_autostart`) the Openbox autostart entry. One combined report so
    `display/commands.py` can print exactly what was done, in order.
    ``colord_device``/``dispwin_display`` name which screen on a
    multi-monitor machine -- see ``install_colormgr``."""
    report = InstallReport()
    report.steps += install_colormgr(icc_path, device_path=colord_device).steps
    report.steps += install_dispwin(icc_path, display=dispwin_display).steps
    if write_autostart:
        report.steps += write_autostart_entry(icc_path, autostart_path=autostart_path).steps
    return report

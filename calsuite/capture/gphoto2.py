"""gphoto2 subprocess wrapper for tethered capture on Linux (docs/design.md
§3.3: "gphoto2 ... already works with the R100"). Every call goes through
``tools.run``, so it's testable against a small fake ``gphoto2`` script on
PATH (see ``tests/test_capture_gphoto2.py``) with no camera attached.
"""

from __future__ import annotations

from pathlib import Path

from calsuite import tools


def detect() -> str:
    """Raw text of ``gphoto2 --auto-detect`` (a Model/Port table, or a
    message with no camera found). Parsing is left to callers that need
    structure -- ``calsuite devices`` only needs to display it."""
    return tools.run(["gphoto2", "--auto-detect"]).stdout


def is_camera_connected() -> bool:
    """True if ``gphoto2 --auto-detect`` lists at least one camera. Never
    raises -- ``calsuite devices`` wants "no" when gphoto2 is missing or a
    camera isn't attached, not a crash."""
    if tools.which("gphoto2") is None:
        return False
    try:
        text = detect()
    except tools.ToolError:
        return False
    # gphoto2 always prints a two-line header (column names + a rule of
    # dashes) even with nothing attached; a real camera adds a third line.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) > 2


def set_config(**settings) -> None:
    """Set one or more camera config values, e.g.
    ``set_config(iso="800", shutterspeed="1/100", aperture="4")``. Each
    becomes one ``gphoto2 --set-config`` call -- gphoto2's config names
    vary by camera model, so these are passed through rather than
    validated against the R100's specific config tree."""
    for key, value in settings.items():
        tools.run(["gphoto2", f"--set-config={key}={value}"])


def capture_and_download(out_dir: Path | str, *, filename: str | None = None) -> Path:
    """``gphoto2 --capture-image-and-download``, saved under ``out_dir``.

    Returns the path gphoto2 reports it wrote, parsed from its stdout
    (``Saving file as ...``) -- gphoto2 itself chooses the exact name
    (``capt0000.cr3``, ``capt0001.cr3``, ...) unless ``filename`` is given.
    A longer timeout than most tool calls: a real shutter + download can
    take several seconds, especially for a large raw file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["gphoto2", "--capture-image-and-download"]
    args.append(f"--filename={out_dir / (filename if filename else '%f.%C')}")
    result = tools.run(args, timeout=60)
    for line in result.stdout.splitlines():
        if "Saving file as" in line:
            return Path(line.split("Saving file as", 1)[1].strip())
    raise tools.ToolError(f"gphoto2 did not report a saved filename:\n{result.stdout}")

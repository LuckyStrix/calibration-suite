"""Argyll CGATS ``.ti3`` reader/writer, for display RGB->XYZ measurement
sets (docs/design.md §5.4: every display measurement backend -- argyll,
camera, spectro, synthetic -- converges on this format, which ``colprof``
then turns into an ICC profile).

``.ti3`` is Argyll's own dialect of the ANSI CGATS.5-1993 text format: a
``CTI3``-tagged header of ``KEYWORD "value"`` / ``KEYWORD number`` lines, a
``BEGIN_DATA_FORMAT``/``END_DATA_FORMAT`` field-name block, then a
``BEGIN_DATA``/``END_DATA`` block of whitespace-separated numeric rows.

Format confirmed against ArgyllCMS's own documentation, fetched 2026-09-11:
https://www.argyllcms.com/doc/File_Formats.html (.ti3 overview: "ASCII
text, CGATS, Argyll specific format, used to hold device value and
CIE/Spectral value pairs") and https://www.argyllcms.com/doc/ti3_format.html
(field/keyword detail) -- key facts taken from there and used below:
``DEVICE_CLASS`` is one of OUTPUT/DISPLAY/INPUT/EMISINPUT, ``COLOR_REP`` is
``"RGB_XYZ"`` for a display measurement set, device (RGB) values are
written as **percentages 0-100**, and XYZ values are **normalized to
Y=100**. This module's own Python-facing API stays in the more numpy-
friendly [0, 1] range for both (Y=1) and converts at the file boundary, so
callers never have to remember which function wants which scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# The field order .ti3 files from Argyll's own tools (dispread, etc.) use.
# SAMPLE_ID is 1-based and required by convention even though nothing here
# reads it back by value -- Argyll's own tools expect to find it.
FIELDS = ("SAMPLE_ID", "RGB_R", "RGB_G", "RGB_B", "XYZ_X", "XYZ_Y", "XYZ_Z")


@dataclass(frozen=True)
class TI3:
    device_class: str
    descriptor: str
    samples: list = field(default_factory=list)  # [{"rgb": (r,g,b) in [0,1], "xyz": (x,y,z), Y in [0,1]}, ...]


def write_ti3(
    path: Path | str,
    samples: list,
    *,
    device_class: str = "DISPLAY",
    descriptor: str = "calsuite display measurement",
    originator: str = "calsuite",
) -> None:
    """Write a ``.ti3``. ``samples`` is a list of
    ``{"rgb": (r, g, b), "xyz": (x, y, z)}`` with every value in ``[0, 1]``
    (RGB drive level; XYZ with Y=1 for a perfect white) -- converted here to
    the format's own 0-100 scale.
    """
    if device_class not in ("OUTPUT", "DISPLAY", "INPUT", "EMISINPUT"):
        raise ValueError(f"device_class must be OUTPUT/DISPLAY/INPUT/EMISINPUT, got {device_class!r}")

    lines = [
        "CTI3",
        f'DESCRIPTOR "{descriptor}"',
        f'ORIGINATOR "{originator}"',
        f'CREATED "{datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")}"',
        f'DEVICE_CLASS "{device_class}"',
        'COLOR_REP "RGB_XYZ"',
        f"NUMBER_OF_FIELDS {len(FIELDS)}",
        "BEGIN_DATA_FORMAT",
        " ".join(FIELDS),
        "END_DATA_FORMAT",
        f"NUMBER_OF_SETS {len(samples)}",
        "BEGIN_DATA",
    ]
    for i, s in enumerate(samples, start=1):
        r, g, b = s["rgb"]
        x, y, z = s["xyz"]
        lines.append(f"{i} {r * 100:.6f} {g * 100:.6f} {b * 100:.6f} {x * 100:.6f} {y * 100:.6f} {z * 100:.6f}")
    lines.append("END_DATA")
    Path(path).write_text("\n".join(lines) + "\n")


def _split_keyword_line(line: str) -> tuple[str, str] | None:
    parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    key, value = parts
    return key, value.strip().strip('"')


def read_ti3(path: Path | str) -> TI3:
    """Read a ``.ti3`` back. Tolerant of extra header keywords and of a
    data-format field order that differs from ``FIELDS`` (it looks fields
    up by name), but requires ``RGB_R``/``RGB_G``/``RGB_B`` and
    ``XYZ_X``/``XYZ_Y``/``XYZ_Z`` to be present -- anything without those
    isn't an RGB->XYZ display measurement set, which is all this module
    reads.
    """
    text = Path(path).read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    device_class, descriptor = "", ""
    data_format: list[str] = []
    data_rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "BEGIN_DATA_FORMAT":
            i += 1
            data_format = lines[i].split()
            i += 1
            assert lines[i] == "END_DATA_FORMAT"
        elif line == "BEGIN_DATA":
            i += 1
            while lines[i] != "END_DATA":
                data_rows.append(lines[i].split())
                i += 1
        else:
            kv = _split_keyword_line(line)
            if kv:
                key, value = kv
                if key == "DEVICE_CLASS":
                    device_class = value
                elif key == "DESCRIPTOR":
                    descriptor = value
        i += 1

    if not data_format:
        raise ValueError(f"{path}: no BEGIN_DATA_FORMAT/END_DATA_FORMAT block found")
    required = ("RGB_R", "RGB_G", "RGB_B", "XYZ_X", "XYZ_Y", "XYZ_Z")
    missing = [f for f in required if f not in data_format]
    if missing:
        raise ValueError(f"{path}: missing required field(s) {missing} -- not an RGB_XYZ display .ti3")
    idx = {name: data_format.index(name) for name in required}

    samples = []
    for row in data_rows:
        rgb = tuple(float(row[idx[f]]) / 100.0 for f in ("RGB_R", "RGB_G", "RGB_B"))
        xyz = tuple(float(row[idx[f]]) / 100.0 for f in ("XYZ_X", "XYZ_Y", "XYZ_Z"))
        samples.append({"rgb": rgb, "xyz": xyz})

    return TI3(device_class=device_class, descriptor=descriptor, samples=samples)

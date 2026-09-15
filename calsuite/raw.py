"""Raw frame loading: rawpy for pixels, exiftool (or dcraw as a fallback) for
metadata. House rule 1 (docs/design.md): every analysis works on the raw CFA
planes, never a demosaiced image -- ``RawFrame`` never holds anything but
the untouched Bayer mosaic, and ``planes()``/``optical_black()`` are the only
sanctioned ways to slice it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from calsuite import tools

# A pixel or two right at the visible-array boundary and the physical sensor
# edge shows transition artifacts (charge bleed, a slightly different bias)
# that aren't representative of either the visible pixels or a clean
# optical-black reference. Skipping this many columns on each side of the
# masked left-margin band is a small, cheap safety margin against that,
# not a value derived from a specific sensor's datasheet.
OPTICAL_BLACK_EDGE_SKIP_PX = 4


@dataclass(frozen=True)
class FrameMeta:
    """Everything about *how* a frame was taken that a downstream analysis
    might need to know, or might need to refuse on (docs/design.md
    "settings that silently change raw data"). Every field defaults to an
    empty/None value because the two metadata sources (exiftool, dcraw) do
    not offer the same information -- dcraw in particular gives none of
    serial/firmware/lens/sensor_temp_c, so those come back empty on the
    fallback path rather than raising.
    """

    model: str = ""
    serial: str = ""
    firmware: str = ""
    lens: str = ""
    focal: float | None = None  # mm
    aperture: float | None = None  # f-number
    exposure_s: float | None = None
    iso: int | None = None
    timestamp: str = ""  # the camera's own clock -- NOT guaranteed UTC; no EXIF field reliably gives a zone
    sensor_temp_c: float | None = None
    settings: dict = field(default_factory=dict)  # e.g. long_exposure_nr, high_iso_nr, highlight_tone_priority, shutter_mode


@dataclass(frozen=True)
class RawFrame:
    """The full raw CFA array (visible pixels *and* the masked margins used
    for optical-black), never demosaiced.

    ``visible`` is a ``(row_slice, col_slice)`` pair into ``cfa`` that
    selects the visible sensor area; ``pattern`` (e.g. ``"RGGB"``) names the
    2x2 Bayer tile *as seen starting at* ``visible``'s origin -- because the
    margins can shift the tile's phase, the pattern at (0, 0) of the full
    ``cfa`` array is not necessarily the same string.
    """

    cfa: np.ndarray
    pattern: str
    visible: tuple
    black_level: tuple
    white_level: float
    meta: FrameMeta
    path: str
    sha256: str


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _pattern_at(raw_pattern: np.ndarray, color_desc: bytes, row_offset: int, col_offset: int) -> str:
    """rawpy's ``raw_pattern`` tiles starting at raw array (0, 0); shift its
    phase by ``row_offset``/``col_offset`` (mod tile size) to get the 2x2
    (or larger) tile as it appears starting at some other origin, then
    render it as a string via ``color_desc`` (e.g. ``b"RGBG"``)."""
    ph, pw = raw_pattern.shape
    rows = []
    for r in range(ph):
        row = []
        for c in range(pw):
            idx = raw_pattern[(row_offset + r) % ph, (col_offset + c) % pw]
            row.append(chr(color_desc[idx]))
        rows.append("".join(row))
    return "".join(rows)


def _plane_positions(pattern: str) -> dict[tuple, str]:
    """Map each of the 4 positions in a 2x2 tile to a canonical plane name.
    The first 'G' encountered in raster order (top-left, top-right,
    bottom-left, bottom-right) is G1, the second is G2 -- an arbitrary but
    fixed convention so ``planes()`` and ``optical_black()`` agree with each
    other and with every caller."""
    if len(pattern) != 4:
        raise ValueError(f"expected a 2x2 (4-char) CFA pattern, got {pattern!r}")
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    names = {}
    g_count = 0
    for pos, ch in zip(positions, pattern, strict=True):
        if ch == "G":
            g_count += 1
            names[pos] = f"G{g_count}"
        else:
            names[pos] = ch
    return names


def _channel_at(pattern: str, visible: tuple, row: int, col: int) -> str:
    """Which plane name (R/G1/G2/B) sits at absolute ``cfa`` coordinate
    (row, col), given that ``pattern`` is defined at ``visible``'s origin."""
    top, left = visible[0].start, visible[1].start
    r = (row - top) % 2
    c = (col - left) % 2
    return _plane_positions(pattern)[(r, c)]


def planes(frame: RawFrame, area: str = "visible") -> dict:
    """``{"R": arr, "G1": arr, "G2": arr, "B": arr}``, each a float64 array
    at quarter resolution (one sample per 2x2 tile). This is the *only*
    sanctioned way analysis code reads pixel values out of a ``RawFrame`` --
    it never demosaics, it just deals the four interleaved sub-images back
    out (house rule 1).

    ``area="visible"`` (default) restricts to the visible sensor area;
    ``area="full"`` returns the whole ``cfa`` array including margins (used
    by things that specifically want, e.g., the full top-margin band rather
    than ``optical_black``'s left-margin slice).
    """
    if area == "visible":
        sub = frame.cfa[frame.visible[0], frame.visible[1]]
        row0, col0 = frame.visible[0].start, frame.visible[1].start
    elif area == "full":
        sub = frame.cfa
        row0, col0 = 0, 0
    else:
        raise ValueError(f"unknown area {area!r}, expected 'visible' or 'full'")

    # Walk the 2x2 phase (dr, dc) *within `sub`*, translate each phase's
    # first pixel to an absolute (cfa) coordinate, and ask `_channel_at`
    # (the same helper optical_black() uses) which plane name lives there --
    # one source of truth for "which channel is at this coordinate" no
    # matter which sub-region of the full array is being sliced.
    out = {}
    for dr in (0, 1):
        for dc in (0, 1):
            name = _channel_at(frame.pattern, frame.visible, row0 + dr, col0 + dc)
            out[name] = sub[dr::2, dc::2].astype(np.float64)
    return out


def black_level_by_channel(frame: RawFrame) -> dict:
    """Map ``frame.black_level``'s 4 values -- documented on ``RawFrame`` as
    "4 values from metadata" with no channel names attached -- to the same
    channel names ``planes()``/``optical_black()`` use ("R", "G1", "G2",
    "B"). Several callers used to work around the missing mapping with a
    plain ``mean(black_level)`` scalar, which is silently wrong on any
    sensor with a genuinely different black level per channel (real on
    some CMOS designs, where the two green amplifier chains -- G1 and G2 --
    can read a few DN apart); this is the one place that mapping is done,
    so ``camera/bias.py`` and ``camera/chart.py`` (and anything else that
    needs a per-channel black) can do it correctly.

    ``black_level``'s 4 entries are **indexed by LibRaw color index, not by
    raster position**: ``cblack[0..3]`` is one value per color-filter index
    of ``color_desc`` (``b"RGBG"`` for a Bayer sensor), i.e. the order
    "R, first-G, B, second-G". Verified directly against this project's own
    reference camera: for ``capt0000.cr3`` (Canon R100) rawpy reports
    ``color_desc = b"RGBG"`` and ``raw_pattern = [[0, 1], [3, 2]]``, so the
    *second* green is index 3 and sits at raster position (1, 0) while B is
    index 2 at (1, 1). Zipping the four values onto raster positions
    ``(0,0), (0,1), (1,0), (1,1)`` -- which this function used to do,
    against its own docstring -- therefore swaps **B and G2** on every
    standard Bayer sensor. That is silently wrong in exactly the case the
    function exists for: a sensor whose green chains read a different black
    level from blue.

    ``frame.pattern``, by contrast, is named relative to the *visible*
    origin (``frame.visible``), which is a different tile whenever either
    margin (``frame.visible[0].start`` / ``frame.visible[1].start``) is
    odd -- the visible tile is then the absolute tile's phase shifted by
    one row and/or column. Only the two greens are affected by that shift
    (R is R and B is B wherever they sit), and this function undoes it
    before naming them, so it's correct in general, not just on sensors
    (the R100 included: both its margins are even) where the two origins
    happen to coincide.
    """
    row_shift = frame.visible[0].start % 2
    col_shift = frame.visible[1].start % 2
    visible_names = _plane_positions(frame.pattern)
    # The visible-origin plane name at each position of the *absolute*
    # origin's tile, in that tile's raster order -- so "first G"/"second G"
    # below are the greens as LibRaw's color indices 1 and 3 number them.
    names_at_absolute = [
        visible_names[((r + row_shift) % 2, (c + col_shift) % 2)] for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    greens = [name for name in names_at_absolute if name.startswith("G")]
    if len(greens) != 2 or "R" not in names_at_absolute or "B" not in names_at_absolute:
        raise ValueError(f"expected an RGGB-family 2x2 CFA, got pattern {frame.pattern!r}")
    color_index_order = ("R", greens[0], "B", greens[1])  # LibRaw cblack[0..3]
    return dict(zip(color_index_order, frame.black_level, strict=True))


def optical_black(frame: RawFrame) -> dict:
    """Per-CFA-channel float64 arrays from the masked **left-margin**
    columns (the extra columns outside the visible width, e.g. 288 of them
    on the R100 -- docs/design.md §3.1), skipping ``OPTICAL_BLACK_EDGE_SKIP_PX``
    columns on each side of that band. These pixels see no light and no
    lens; they're the closest thing to a pure bias+dark-current reference
    the sensor itself provides, used by camera/bias.py and camera/darks.py.
    """
    left = frame.visible[1].start
    lo, hi = OPTICAL_BLACK_EDGE_SKIP_PX, left - OPTICAL_BLACK_EDGE_SKIP_PX
    if hi <= lo:
        raise ValueError(
            f"left margin ({left}px) too narrow for a {OPTICAL_BLACK_EDGE_SKIP_PX}px edge skip on each side"
        )
    region = frame.cfa[:, lo:hi]
    out = {}
    for dr in (0, 1):
        for dc in (0, 1):
            name = _channel_at(frame.pattern, frame.visible, dr, lo + dc)
            out[name] = region[dr::2, dc::2].astype(np.float64)
    return out


NPZ_EXTENSIONS = {".npz"}
# A synthetic ``RawFrame`` (synth.sensor.frame(), synth.color.render_chart(),
# ...) has no real raw file behind it, so it could never be written to disk
# and read back before this format existed -- which is exactly what stopped
# several areas from writing a CLI-level test (a command that takes
# ``--from DIR`` needs *files* in that directory, not an in-memory object).
# ``.npz`` round-trips every field a real raw file's ``load()`` produces, so
# a synthetic frame can be saved once and then handled identically to a real
# raw everywhere a folder of captures is accepted (docs/design.md §3.3:
# "manual import is first-class").


def load(path: Path | str) -> RawFrame:
    """Load a raw file's pixels via rawpy and its metadata via exiftool
    (preferred) or dcraw (fallback -- exiftool is a system package the user
    installs; dcraw is more commonly already present, per docs/design.md
    §0). A ``.npz`` path (see ``save_npz``) is loaded via ``load_npz``
    instead -- the two formats are interchangeable everywhere a ``RawFrame``
    is accepted."""
    path = Path(path)
    # Case-insensitive, like the real raw extensions elsewhere: a
    # `frame.NPZ` handed to rawpy dies with LibRawFileUnsupportedError.
    if path.suffix.lower() in NPZ_EXTENSIONS:
        return load_npz(path)

    import rawpy  # imported lazily so devices.py/store.py etc. stay importable without it

    with rawpy.imread(str(path)) as raw:
        cfa = raw.raw_image.copy()  # full array, margins included -- never demosaiced
        sizes = raw.sizes
        top, left = sizes.top_margin, sizes.left_margin
        visible = (slice(top, top + sizes.height), slice(left, left + sizes.width))
        pattern = _pattern_at(raw.raw_pattern, raw.color_desc, top, left)
        black_level = tuple(float(v) for v in raw.black_level_per_channel)
        white_level = float(raw.white_level)

    meta = _read_metadata(path)
    return RawFrame(
        cfa=cfa,
        pattern=pattern,
        visible=visible,
        black_level=black_level,
        white_level=white_level,
        meta=meta,
        path=str(path),
        sha256=sha256_file(path),
    )


def save_npz(frame: RawFrame, path: Path | str) -> Path:
    """Write ``frame`` to ``path`` (must end in ``.npz``) so it round-trips
    through ``load()``/``load_npz()`` -- every ``RawFrame`` field, including
    every ``FrameMeta`` field (``settings`` included), not just the pixels.

    Strings are stored as 0-d numpy arrays (``np.array("RGGB")`` etc.) and
    ``FrameMeta`` as one JSON string in the same form, rather than via
    ``allow_pickle=True`` -- an ``.npz`` written by this suite should be
    loadable by anything reading numpy's format, not just by a Python
    process willing to unpickle an unknown file.
    """
    path = Path(path)
    if path.suffix not in NPZ_EXTENSIONS:
        raise ValueError(f"save_npz path must end in .npz, got {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cfa=frame.cfa,
        pattern=np.array(frame.pattern),
        visible_row_start=np.array(frame.visible[0].start),
        visible_row_stop=np.array(frame.visible[0].stop),
        visible_col_start=np.array(frame.visible[1].start),
        visible_col_stop=np.array(frame.visible[1].stop),
        black_level=np.array(frame.black_level, dtype=np.float64),
        white_level=np.array(frame.white_level, dtype=np.float64),
        meta_json=np.array(json.dumps(asdict(frame.meta))),
        sha256=np.array(frame.sha256),
    )
    return path


def load_npz(path: Path | str) -> RawFrame:
    """Load a ``RawFrame`` written by ``save_npz`` -- the synthetic-frame
    round trip that lets ``capture.manual.scan_folder`` and every area's
    ``--from DIR`` treat a synthetic session exactly like a folder of real
    raw files (docs/implementation-plan.md Wave 3, fix list item 4).

    ``path`` (not the embedded value) becomes the returned frame's
    ``.path``, and its SHA-256 is recomputed from the file on disk rather
    than trusted from inside it -- same rule ``load()`` follows for a real
    raw file: the hash always describes the actual bytes being read, which
    is what a record's ``inputs[].sha256`` is for.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        cfa = npz["cfa"]
        pattern = str(npz["pattern"])
        visible = (
            slice(int(npz["visible_row_start"]), int(npz["visible_row_stop"])),
            slice(int(npz["visible_col_start"]), int(npz["visible_col_stop"])),
        )
        black_level = tuple(float(v) for v in npz["black_level"])
        white_level = float(npz["white_level"])
        meta = FrameMeta(**json.loads(str(npz["meta_json"])))
    return RawFrame(
        cfa=cfa,
        pattern=pattern,
        visible=visible,
        black_level=black_level,
        white_level=white_level,
        meta=meta,
        path=str(path),
        sha256=sha256_file(path),
    )


# ---------------------------------------------------------------------------
# metadata: exiftool -j -n, else dcraw -i -v
# ---------------------------------------------------------------------------


def _read_metadata(path: Path) -> FrameMeta:
    if tools.which("exiftool"):
        return _metadata_from_exiftool(path)
    if tools.which("dcraw"):
        return _metadata_from_dcraw(path)
    return FrameMeta()  # neither tool present -- pixels still loaded, metadata just empty


def _metadata_from_exiftool(path: Path) -> FrameMeta:
    # `-n` gives numeric values (a bare mm/seconds/f-number) instead of
    # exiftool's formatted strings ("1/83", "f/1.8"), which is what lets the
    # fields below be parsed with a plain float()/int() rather than a second
    # regex layer on top of exiftool's own formatting.
    #
    # NOTE: the exact tag names below (SerialNumber, HighISONoiseReduction,
    # HighlightTonePriority, ShutterMode, CameraTemperature...) are taken
    # from exiftool's documented Canon tag list, but exiftool isn't
    # installed on this build machine (docs/design.md §0's "missing" list),
    # so this path is unverified against a real Canon file and is covered
    # only by a fake-exiftool test fixture. Treat mismatches as a tag-name
    # fix, not an architecture problem.
    result = tools.run(["exiftool", "-j", "-n", str(path)])
    records = json.loads(result.stdout)
    d = records[0] if records else {}

    def _num(*keys):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    continue
        return None

    focal = _num("FocalLength")
    aperture = _num("FNumber", "ApertureValue")
    exposure_s = _num("ExposureTime")
    iso_val = _num("ISO")
    sensor_temp = _num("CameraTemperature", "SensorTemperature")

    return FrameMeta(
        model=str(d.get("Model", "")),
        serial=str(d.get("SerialNumber", d.get("InternalSerialNumber", ""))),
        firmware=str(d.get("FirmwareVersion", "")),
        lens=str(d.get("LensModel", d.get("LensType", d.get("Lens", "")))),
        focal=focal,
        aperture=aperture,
        exposure_s=exposure_s,
        iso=int(iso_val) if iso_val is not None else None,
        timestamp=str(d.get("DateTimeOriginal", d.get("CreateDate", ""))),
        sensor_temp_c=sensor_temp,
        settings={
            "long_exposure_nr": d.get("LongExposureNoiseReduction"),
            "high_iso_nr": d.get("HighISONoiseReduction"),
            "highlight_tone_priority": d.get("HighlightTonePriority"),
            "shutter_mode": d.get("ShutterMode", d.get("ElectronicFrontCurtainShutter")),
        },
    )


# dcraw -i -v output, one "Key: value" line per field, e.g.:
#   Timestamp: Thu May 21 09:23:18 2026
#   Camera: Canon EOS R100
#   ISO speed: 800
#   Shutter: 1/83.0 sec
#   Aperture: f/1.8
#   Focal length: 50.0 mm
# Confirmed against this build machine's real ~/capt0000.cr3 (docs/design.md
# §0). It gives no serial, firmware, lens name, or sensor temperature at
# all -- there's simply no line for them -- so those fields stay empty on
# this path; that's a real, permanent limitation of the fallback, not a
# parsing gap.
_DCRAW_PATTERNS = {
    "timestamp": re.compile(r"^Timestamp:\s*(.+)$", re.M),
    "camera": re.compile(r"^Camera:\s*(.+)$", re.M),
    "iso": re.compile(r"^ISO speed:\s*(\d+)", re.M),
    "shutter": re.compile(r"^Shutter:\s*(.+?)\s*sec\s*$", re.M),
    "aperture": re.compile(r"^Aperture:\s*f/([\d.]+)", re.M),
    "focal": re.compile(r"^Focal length:\s*([\d.]+)\s*mm", re.M),
}


def _parse_shutter(text: str) -> float:
    text = text.strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def _metadata_from_dcraw(path: Path) -> FrameMeta:
    result = tools.run(["dcraw", "-i", "-v", str(path)])
    text = result.stdout

    def _match(key):
        m = _DCRAW_PATTERNS[key].search(text)
        return m.group(1).strip() if m else None

    iso_str = _match("iso")
    shutter_str = _match("shutter")
    aperture_str = _match("aperture")
    focal_str = _match("focal")

    return FrameMeta(
        model=_match("camera") or "",
        timestamp=_match("timestamp") or "",
        iso=int(iso_str) if iso_str else None,
        exposure_s=_parse_shutter(shutter_str) if shutter_str else None,
        aperture=float(aperture_str) if aperture_str else None,
        focal=float(focal_str) if focal_str else None,
        # serial / firmware / lens / sensor_temp_c / settings: not exposed
        # by `dcraw -i -v`, so left at their FrameMeta defaults.
    )

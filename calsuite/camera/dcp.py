"""DNG Camera Profile (DCP) reader/writer, and the ICC-colorant helper for
``formats/icc.py`` (docs/design.md sec 3.2: "write a DCP directly ... Also
write a DCP READER ... ICC input profile ... via formats/icc.py").

**File format, verified 2026-09-11 (web search, primary DNG spec PDF was not
directly fetchable -- see citations below):**

A DCP is TIFF-structured but is not a TIFF: the byte-order mark ("II" for
little-endian, matching what this module writes) is followed not by the
usual TIFF magic number 42, but by the bytes ``0x52 0x43`` -- i.e. the file
starts with the four bytes ``b"IIRC"`` (0x49 0x49 0x52 0x43). Read as a
little-endian uint16, ``0x52 0x43`` *is* ``0x4352`` -- both this module's
docstring and the instructions describe the same fact two equivalent ways.
Source: fileformats.archiveteam.org's "DNG camera profile" page, which
states the DCP file signature as "'M' 'M' 0x43 0x52, or 'I' 'I' 0x52 0x43"
(the big/little-endian byte-order pair), corroborated by a GitHub tifffile
issue (cgohlke/tifffile#306) describing the same "16-bit byte order mark
... followed by a 16-bit 'magic' number equal to 0x4352 ('CR')". After that
4-byte header, the rest of the file is one ordinary baseline-TIFF IFD (2-byte
entry count, 12-byte entries, 4-byte next-IFD offset) holding the profile
tags below -- no image data, no sub-IFDs, matching how a real ``.dcp``
sidecar (as opposed to a DNG file with an *embedded* profile) is just that
one IFD.

**Tag IDs**, verified 2026-09-11 against exiftool.org's EXIF tag table
(https://exiftool.org/TagNames/EXIF.html, the "DNG" tag group -- exiftool's
tag tables are themselves built from Adobe's published DNG specification and
are the standard cross-reference the raw-processing community cites for
exact numeric tag IDs):

=====================  ======  =========================================
tag                     id      TIFF type
=====================  ======  =========================================
UniqueCameraModel      0xC614  ASCII
ColorMatrix1           0xC621  SRATIONAL[9]
ColorMatrix2           0xC622  SRATIONAL[9]
CalibrationIlluminant1 0xC65A  SHORT[1]
CalibrationIlluminant2 0xC65B  SHORT[1]
ForwardMatrix1         0xC714  SRATIONAL[9]
ForwardMatrix2         0xC715  SRATIONAL[9]
ProfileName            0xC6F8  ASCII
ProfileEmbedPolicy     0xC6FD  LONG[1]
ProfileCopyright       0xC6FE  ASCII
=====================  ======  =========================================

**CalibrationIlluminant codes** (the same enumeration as the EXIF
``LightSource`` tag, which DNG's ``CalibrationIlluminant1/2`` reuses):
17 = Standard Light A (tungsten-like), 20 = D55, 21 = D65, 22 = D75,
23 = D50 -- verified 2026-09-11 by web search (multiple independent
secondary sources -- Pillow's ``ExifTags`` docs, exif.readthedocs.io -- list
the same numbering); matches the instructions' own "StdA=17, D65=21".

**Matrix normalization** (docs/design.md / instructions: "ForwardMatrix maps
camera neutral (1,1,1) to D50 XYZ (0.9642, 1, 0.8249)"): see
``build_forward_and_color_matrices``'s docstring for the derivation, which
makes that property hold *exactly* (not just approximately) by construction.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calsuite.formats.icc import ICC_PCS_ILLUMINANT_D50

DCP_BYTE_ORDER = b"II"
DCP_MAGIC = 0x4352  # "CR" -- see module docstring for the citation trail

TIFF_ASCII = 2
TIFF_SHORT = 3
TIFF_LONG = 4
TIFF_SRATIONAL = 10
_TYPE_SIZE = {TIFF_ASCII: 1, TIFF_SHORT: 2, TIFF_LONG: 4, TIFF_SRATIONAL: 8}

TAG_UNIQUE_CAMERA_MODEL = 0xC614
TAG_COLOR_MATRIX1 = 0xC621
TAG_COLOR_MATRIX2 = 0xC622
TAG_CALIBRATION_ILLUMINANT1 = 0xC65A
TAG_CALIBRATION_ILLUMINANT2 = 0xC65B
TAG_FORWARD_MATRIX1 = 0xC714
TAG_FORWARD_MATRIX2 = 0xC715
TAG_PROFILE_NAME = 0xC6F8
TAG_PROFILE_EMBED_POLICY = 0xC6FD
TAG_PROFILE_COPYRIGHT = 0xC6FE

# Standard Light A / D55 / D65 / D75 / D50, per the EXIF LightSource
# enumeration DNG's CalibrationIlluminant tags reuse (module docstring).
ILLUMINANT_CODES = {"A": 17, "D55": 20, "D65": 21, "D75": 22, "D50": 23}

_SRATIONAL_DENOM = 1_000_000
# A fixed denominator rather than Fraction.limit_denominator's variable one:
# every matrix entry is representable to 1e-6, comfortably finer than any
# real camera's photon-shot-noise-limited color-matrix precision, and a
# fixed denominator makes the encoder/decoder trivially symmetric.

DEFAULT_PROFILE_EMBED_POLICY = 0
# "Allow Copying" per the DNG spec's ProfileEmbedPolicy enumeration (0-3);
# the most permissive of the four, appropriate for a profile this suite
# generates and expects to hand to darktable/RawTherapee/Lightroom freely.


# ---------------------------------------------------------------------------
# generic minimal TIFF-IFD encode/decode -- just enough for a flat DCP IFD
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    tag: int
    type: int
    count: int
    data: bytes


def _encode_ascii(s: str) -> bytes:
    return s.encode("ascii") + b"\x00"


def _encode_short(values) -> bytes:
    return b"".join(struct.pack("<H", int(v)) for v in values)


def _encode_long(values) -> bytes:
    return b"".join(struct.pack("<I", int(v)) for v in values)


def _encode_srational(values, denom: int = _SRATIONAL_DENOM) -> bytes:
    out = bytearray()
    for v in values:
        out += struct.pack("<ii", int(round(v * denom)), denom)
    return bytes(out)


def _decode_srational(data: bytes, count: int) -> list:
    out = []
    for i in range(count):
        num, den = struct.unpack_from("<ii", data, i * 8)
        out.append(num / den if den else 0.0)
    return out


def _build_ifd(entries: list) -> bytes:
    """Encode one flat IFD: 2-byte count, ``count`` 12-byte entries (value
    data inline if it fits in 4 bytes, else an offset into the value area
    right after the fixed-size entry table), 4-byte next-IFD offset (always
    0 -- a DCP has exactly one IFD), then the value area itself. Per TIFF
    6.0, entries must be sorted by ascending tag id."""
    entries = sorted(entries, key=lambda e: e.tag)
    n = len(entries)
    header_and_table_size = 2 + 12 * n + 4
    ifd_start = 8  # right after this module's 8-byte DCP header
    value_area_offset = ifd_start + header_and_table_size

    table = bytearray(struct.pack("<H", n))
    value_area = bytearray()
    cursor = value_area_offset
    for e in entries:
        table += struct.pack("<HHI", e.tag, e.type, e.count)
        if len(e.data) <= 4:
            table += e.data + b"\x00" * (4 - len(e.data))
        else:
            table += struct.pack("<I", cursor)
            padded = e.data + (b"\x00" if len(e.data) % 2 else b"")  # TIFF: value-area entries are word-aligned
            value_area += padded
            cursor += len(padded)
    table += struct.pack("<I", 0)  # next IFD offset: none
    return bytes(table) + bytes(value_area)


def _parse_ifd(data: bytes, ifd_offset: int) -> dict:
    (n,) = struct.unpack_from("<H", data, ifd_offset)
    tags = {}
    for i in range(n):
        entry_off = ifd_offset + 2 + 12 * i
        tag, type_, count = struct.unpack_from("<HHI", data, entry_off)
        size = _TYPE_SIZE.get(type_, 1) * count
        value_field = data[entry_off + 8 : entry_off + 12]
        if size <= 4:
            raw = value_field[:size]
        else:
            (offset,) = struct.unpack_from("<I", value_field)
            raw = data[offset : offset + size]
        tags[tag] = (type_, count, raw)
    return tags


# ---------------------------------------------------------------------------
# matrix normalization (DNG-spec convention)
# ---------------------------------------------------------------------------


def build_forward_and_color_matrices(matrix_raw_to_xyz: np.ndarray, raw_white: np.ndarray) -> tuple:
    """Derive DNG-convention ``(ColorMatrix, ForwardMatrix)`` from a Tier
    A/B ``matrix_raw_to_xyz`` (raw->XYZ under the shooting illuminant,
    color.py's Y=1-diffuser convention) and ``raw_white`` (the black-
    subtracted, G-averaged raw RGB of the chart's white/reference patch --
    the same one the white-preserving fit option targets).

    **ForwardMatrix** (white-balanced camera RGB -> D50 XYZ): let
    ``xyz_w = matrix_raw_to_xyz @ raw_white`` be this fit's own prediction
    for the white patch. A Bradford von Kries chromatic-adaptation matrix
    ``CAT`` built from ``(xyz_w, D50_white)`` satisfies ``CAT @ xyz_w ==
    D50_white`` *exactly*, by the definition of a von-Kries CAT (it is a
    cone-response rescale that sends its own source white to its own target
    white with no residual, not an approximation). Composing
    ``M_d50 = CAT @ matrix_raw_to_xyz`` and then
    ``ForwardMatrix = M_d50 @ diag(raw_white)`` (so its input is
    white-balanced raw, i.e. raw/raw_white) gives
    ``ForwardMatrix @ [1, 1, 1] == M_d50 @ raw_white == CAT @ xyz_w ==
    D50_white`` exactly -- the property the instructions ask to verify
    (``ForwardMatrix . [1,1,1] ~= D50``), which this construction satisfies
    to floating-point precision rather than only approximately. Adapting
    from this fit's *own* predicted white (rather than the nominal
    illuminant's textbook chromaticity) is deliberate: a real chart's
    "white" patch is never a perfectly neutral, perfectly-illuminant-colored
    reflector, and this way the identity holds regardless.

    **ColorMatrix** (D50 XYZ -> camera-native raw, the inverse direction,
    used elsewhere in the DNG render pipeline to solve for white balance):
    this suite doesn't model DNG's separate CameraCalibration/AnalogBalance
    tags (left at their spec-default identity by omission), so ColorMatrix
    is built as the direct inverse convention:
    ``ColorMatrix = diag(1 / raw_white) @ inverse(M_d50)``, which satisfies
    ``ColorMatrix @ D50_white == [1, 1, 1]`` -- the reference white maps to
    a unit-neutral raw response, the spec's stated normalization goal for
    ColorMatrix (a perfect reflector under the calibration illuminant maps
    as closely as possible to neutral).
    """
    import colour

    M = np.asarray(matrix_raw_to_xyz, dtype=np.float64)
    raw_w = np.asarray(raw_white, dtype=np.float64)
    if np.any(raw_w == 0):
        raise ValueError("raw_white has a zero channel -- cannot build a white-balance-normalized DCP matrix")

    xyz_w = M @ raw_w
    d50 = np.array(ICC_PCS_ILLUMINANT_D50, dtype=np.float64)
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(xyz_w, d50, transform="Bradford")
    M_d50 = cat @ M

    forward_matrix = M_d50 * raw_w[np.newaxis, :]  # M_d50 @ diag(raw_w)
    color_matrix = np.linalg.inv(M_d50) / raw_w[:, np.newaxis]  # diag(1/raw_w) @ inverse(M_d50)
    return color_matrix, forward_matrix


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def write_dcp(
    path: Path | str | None,
    *,
    unique_camera_model: str,
    profile_name: str,
    color_matrix1: np.ndarray,
    calibration_illuminant1: int,
    forward_matrix1: np.ndarray,
    color_matrix2: np.ndarray | None = None,
    calibration_illuminant2: int | None = None,
    forward_matrix2: np.ndarray | None = None,
    profile_copyright: str = "Copyright calsuite -- no rights reserved on this measurement",
    profile_embed_policy: int = DEFAULT_PROFILE_EMBED_POLICY,
) -> bytes:
    """Write a DCP. Single-illuminant when only the ``...1`` matrices are
    given; dual-illuminant (docs/design.md: "Shoot under two illuminants
    ... for a dual-illuminant DCP") when ``color_matrix2`` /
    ``calibration_illuminant2`` / ``forward_matrix2`` are also given -- all
    three or none, matching the DNG spec's own all-or-nothing pairing of
    the "2" tags. Returns the raw bytes (and writes them to ``path`` unless
    it's ``None``, same convenience ``formats/icc.py`` offers)."""
    has_second = color_matrix2 is not None or calibration_illuminant2 is not None or forward_matrix2 is not None
    if has_second and (color_matrix2 is None or calibration_illuminant2 is None or forward_matrix2 is None):
        raise ValueError("color_matrix2/calibration_illuminant2/forward_matrix2 must be given together or not at all")

    def _matrix9(m):
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError(f"expected a 3x3 matrix, got shape {m.shape}")
        return m.flatten().tolist()  # row-major, matching dcraw/LibRaw's DNG ColorMatrix layout

    entries = [
        _Entry(TAG_UNIQUE_CAMERA_MODEL, TIFF_ASCII, len(unique_camera_model) + 1, _encode_ascii(unique_camera_model)),
        _Entry(TAG_PROFILE_NAME, TIFF_ASCII, len(profile_name) + 1, _encode_ascii(profile_name)),
        _Entry(TAG_PROFILE_COPYRIGHT, TIFF_ASCII, len(profile_copyright) + 1, _encode_ascii(profile_copyright)),
        _Entry(TAG_PROFILE_EMBED_POLICY, TIFF_LONG, 1, _encode_long([profile_embed_policy])),
        _Entry(TAG_CALIBRATION_ILLUMINANT1, TIFF_SHORT, 1, _encode_short([calibration_illuminant1])),
        _Entry(TAG_COLOR_MATRIX1, TIFF_SRATIONAL, 9, _encode_srational(_matrix9(color_matrix1))),
        _Entry(TAG_FORWARD_MATRIX1, TIFF_SRATIONAL, 9, _encode_srational(_matrix9(forward_matrix1))),
    ]
    if has_second:
        entries += [
            _Entry(TAG_CALIBRATION_ILLUMINANT2, TIFF_SHORT, 1, _encode_short([calibration_illuminant2])),
            _Entry(TAG_COLOR_MATRIX2, TIFF_SRATIONAL, 9, _encode_srational(_matrix9(color_matrix2))),
            _Entry(TAG_FORWARD_MATRIX2, TIFF_SRATIONAL, 9, _encode_srational(_matrix9(forward_matrix2))),
        ]

    header = DCP_BYTE_ORDER + struct.pack("<H", DCP_MAGIC) + struct.pack("<I", 8)
    data = header + _build_ifd(entries)
    if path is not None:
        Path(path).write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DCPProfile:
    unique_camera_model: str
    profile_name: str
    profile_copyright: str
    profile_embed_policy: int
    calibration_illuminant1: int
    color_matrix1: np.ndarray
    forward_matrix1: np.ndarray
    calibration_illuminant2: int | None
    color_matrix2: np.ndarray | None
    forward_matrix2: np.ndarray | None


def _read_ascii(tags, tag) -> str:
    _, _, raw = tags[tag]
    return raw.rstrip(b"\x00").decode("ascii")


def _read_short(tags, tag) -> int:
    _, _, raw = tags[tag]
    return struct.unpack_from("<H", raw)[0]


def _read_long(tags, tag) -> int:
    _, _, raw = tags[tag]
    return struct.unpack_from("<I", raw)[0]


def _read_matrix3x3(tags, tag) -> np.ndarray:
    _, count, raw = tags[tag]
    values = _decode_srational(raw, count)
    return np.array(values, dtype=np.float64).reshape(3, 3)


def read_dcp(path: Path | str | bytes) -> DCPProfile:
    """Parse a DCP written by ``write_dcp`` (or any spec-conforming
    single-flat-IFD DCP with these same tags) back into a ``DCPProfile`` --
    the reader half of the round trip the instructions call for."""
    data = path if isinstance(path, (bytes, bytearray)) else Path(path).read_bytes()
    if data[:2] != DCP_BYTE_ORDER:
        raise ValueError(f"not a little-endian DCP: byte-order mark is {data[:2]!r}, expected {DCP_BYTE_ORDER!r}")
    (magic,) = struct.unpack_from("<H", data, 2)
    if magic != DCP_MAGIC:
        raise ValueError(f"not a DCP: magic number is 0x{magic:04X}, expected 0x{DCP_MAGIC:04X}")
    (ifd_offset,) = struct.unpack_from("<I", data, 4)
    tags = _parse_ifd(data, ifd_offset)

    has_second = TAG_CALIBRATION_ILLUMINANT2 in tags
    return DCPProfile(
        unique_camera_model=_read_ascii(tags, TAG_UNIQUE_CAMERA_MODEL),
        profile_name=_read_ascii(tags, TAG_PROFILE_NAME),
        profile_copyright=_read_ascii(tags, TAG_PROFILE_COPYRIGHT),
        profile_embed_policy=_read_long(tags, TAG_PROFILE_EMBED_POLICY),
        calibration_illuminant1=_read_short(tags, TAG_CALIBRATION_ILLUMINANT1),
        color_matrix1=_read_matrix3x3(tags, TAG_COLOR_MATRIX1),
        forward_matrix1=_read_matrix3x3(tags, TAG_FORWARD_MATRIX1),
        calibration_illuminant2=_read_short(tags, TAG_CALIBRATION_ILLUMINANT2) if has_second else None,
        color_matrix2=_read_matrix3x3(tags, TAG_COLOR_MATRIX2) if has_second else None,
        forward_matrix2=_read_matrix3x3(tags, TAG_FORWARD_MATRIX2) if has_second else None,
    )


# ---------------------------------------------------------------------------
# ICC colorant helper (docs/design.md: "ICC input profile ... colorants =
# D50-adapted (Bradford via colour-science) camera->XYZ columns ... via
# formats/icc.py")
# ---------------------------------------------------------------------------


def icc_colorant_matrix(matrix_raw_to_xyz: np.ndarray, illuminant_xy: tuple) -> np.ndarray:
    """Bradford-adapt ``matrix_raw_to_xyz``'s columns (raw->XYZ under the
    shooting illuminant) to D50 -- the PCS illuminant every ICC profile's
    tags are defined against (``formats/icc.py``'s own docstring, ICC.1:2010
    sec 7.2.16) -- so the result can be passed straight to
    ``formats.icc.write_profile(..., matrix=...)``. Uses the *nominal*
    illuminant chromaticity (not a fitted white, unlike
    ``build_forward_and_color_matrices``) because an ICC matrix/TRC profile
    has no per-white-patch escape hatch the way DCP's ForwardMatrix
    construction does -- it simply declares the shooting illuminant's own
    white point adapted to D50."""
    import colour

    M = np.asarray(matrix_raw_to_xyz, dtype=np.float64)
    xyz_w_shoot = np.array(colour.xy_to_XYZ(illuminant_xy), dtype=np.float64)
    d50 = np.array(ICC_PCS_ILLUMINANT_D50, dtype=np.float64)
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(xyz_w_shoot, d50, transform="Bradford")
    return cat @ M

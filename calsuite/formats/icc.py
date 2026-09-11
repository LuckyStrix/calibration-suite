"""Minimal ICC v4 matrix/TRC profile writer and reader.

Covers exactly the profile shape camera input profiles and simple display
profiles need (docs/design.md §3.2, §5.4): a 3x3 RGB->XYZ matrix plus a
per-channel tone curve (gamma or a sampled table), device class ``scnr``
(input, e.g. a camera) or ``mntr`` (display). This is NOT a general ICC
library -- no LUT-based (``mAB `` / ``mBA `` / A2B/B2A) profiles, no
multi-illuminant tags, no v2 ``textDescriptionType``. It writes and reads
only what it writes, and its correctness is checked two ways: a round trip
through this module's own reader, and (in tests) opening the file with
Pillow's ``PIL.ImageCms`` (littleCMS) as an independent parser -- the two
disagreeing would mean the bytes are wrong, not just that our reader
matches our writer's bugs.

Byte layout follows the ICC.1:2010 (v4.3) specification: a 128-byte header,
a tag table, then tag data. Reference:
https://www.color.org/specification/ICC1v43_2010-12.pdf
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The ICC Profile Connection Space illuminant is always D50, regardless of
# a profile's own device white point (ICC.1:2010 §7.2.16) -- these are the
# spec's own published values (Table 22 / Annex D), not a measurement.
ICC_PCS_ILLUMINANT_D50 = (0.9642, 1.0000, 0.8249)

DEVICE_CLASSES = {
    "scnr": "input device profile (camera, scanner)",
    "mntr": "display device profile (monitor)",
}


def _s15f16(x: float) -> bytes:
    return struct.pack(">i", round(x * 65536))


def _read_s15f16(data: bytes) -> float:
    return struct.unpack(">i", data)[0] / 65536.0


def _u8f8(x: float) -> bytes:
    return struct.pack(">H", max(0, min(0xFFFF, round(x * 256))))


def _read_u8f8(data: bytes) -> float:
    return struct.unpack(">H", data)[0] / 256.0


def _mluc(text: str, lang: bytes = b"en", country: bytes = b"US") -> bytes:
    """multiLocalizedUnicodeType (ICC.1:2010 §10.13) -- the v4 tag type for
    ``desc``/``cprt``. One record, UTF-16BE text."""
    s = text.encode("utf-16-be")
    n_records, record_size = 1, 12
    string_offset = 16 + n_records * record_size
    out = b"mluc" + b"\x00" * 4 + struct.pack(">II", n_records, record_size)
    out += lang + country + struct.pack(">II", len(s), string_offset)
    out += s
    return out


def _read_mluc(data: bytes) -> str:
    n_records, record_size = struct.unpack(">II", data[8:16])
    if n_records == 0:
        return ""
    rec = data[16 : 16 + record_size]
    length, offset = struct.unpack(">II", rec[4:12])
    return data[offset : offset + length].decode("utf-16-be")


def _xyz_tag(*triplets: tuple) -> bytes:
    """XYZType (§10.18): one or more s15Fixed16 XYZ triplets. Profiles here
    always write exactly one (the tag's single value)."""
    out = b"XYZ " + b"\x00" * 4
    for x, y, z in triplets:
        out += _s15f16(x) + _s15f16(y) + _s15f16(z)
    return out


def _read_xyz_tag(data: bytes) -> tuple:
    x, y, z = (_read_s15f16(data[8 + 4 * i : 12 + 4 * i]) for i in range(3))
    return (x, y, z)


def _curv_tag(value) -> bytes:
    """curveType (§10.5). A plain float writes a single-gamma curve
    (count=1, u8Fixed8); an array writes a sampled table (count=N,
    uInt16Number, domain/range both [0, 1])."""
    if np.isscalar(value):
        return b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + _u8f8(float(value))
    values = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    body = struct.pack(">I", len(values))
    for v in values:
        body += struct.pack(">H", round(v * 65535))
    return b"curv" + b"\x00" * 4 + body


def _read_curv_tag(data: bytes):
    (count,) = struct.unpack(">I", data[8:12])
    if count == 0:
        return 1.0  # identity curve
    if count == 1:
        return _read_u8f8(data[12:14])
    values = np.frombuffer(data[12 : 12 + 2 * count], dtype=">u2").astype(np.float64) / 65535.0
    return values


def _pack_header(*, profile_size: int, device_class: str, version_bytes: bytes, created: datetime) -> bytearray:
    h = bytearray(128)
    struct.pack_into(">I", h, 0, profile_size)
    h[8:12] = version_bytes
    h[12:16] = device_class.encode("ascii")
    h[16:20] = b"RGB "
    h[20:24] = b"XYZ "
    struct.pack_into(
        ">HHHHHH", h, 24, created.year, created.month, created.day, created.hour, created.minute, created.second
    )
    h[36:40] = b"acsp"
    struct.pack_into(">I", h, 64, 1)  # rendering intent: 1 = media-relative colorimetric
    x, y, z = ICC_PCS_ILLUMINANT_D50
    h[68:72], h[72:76], h[76:80] = _s15f16(x), _s15f16(y), _s15f16(z)
    h[80:84] = b"CSUI"  # profile creator signature: calsuite
    return h


def _normalize_trc(trc) -> dict:
    """Accepts a single float/array (applied to all three channels) or a
    ``{"r": ..., "g": ..., "b": ...}`` dict (per-channel)."""
    if isinstance(trc, dict):
        missing = {"r", "g", "b"} - set(trc)
        if missing:
            raise ValueError(f"trc dict missing channel(s): {sorted(missing)}")
        return {k: trc[k] for k in ("r", "g", "b")}
    return {"r": trc, "g": trc, "b": trc}


def write_profile(
    path: Path | str | None,
    *,
    device_class: str,
    description: str,
    matrix,
    trc,
    white_xyz: tuple | None = None,
    copyright_text: str = "Copyright calsuite -- no rights reserved on this measurement",
    version: tuple = (4, 3, 0),
) -> bytes:
    """Write a matrix/TRC ICC profile. Returns the raw bytes (and writes
    them to ``path`` unless it's ``None``, which is handy for the "does
    Pillow accept these bytes" test without a temp file).

    ``matrix`` is 3x3, D50-adapted XYZ, **columns** R/G/B (so
    ``matrix[:, 0]`` is the red primary's XYZ) -- the layout the ``rXYZ``/
    ``gXYZ``/``bXYZ`` tags store directly. ``white_xyz`` defaults to the
    columns' sum, which is the matrix/TRC model's own definition of the
    profile's white point (ICC.1:2010 §6.3.4.2, Annex F): D50-adapted RGB
    (1,1,1) must land on the declared white.
    """
    if device_class not in DEVICE_CLASSES:
        raise ValueError(f"device_class must be one of {tuple(DEVICE_CLASSES)}, got {device_class!r}")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3 (rows XYZ, columns R,G,B), got shape {matrix.shape}")
    if white_xyz is None:
        white_xyz = tuple(matrix.sum(axis=1))

    curves = _normalize_trc(trc)
    version_bytes = bytes([version[0], (version[1] << 4) | version[2], 0, 0])

    tags = {
        "desc": _mluc(description),
        "cprt": _mluc(copyright_text),
        "wtpt": _xyz_tag(white_xyz),
        "rXYZ": _xyz_tag(tuple(matrix[:, 0])),
        "gXYZ": _xyz_tag(tuple(matrix[:, 1])),
        "bXYZ": _xyz_tag(tuple(matrix[:, 2])),
        "rTRC": _curv_tag(curves["r"]),
        "gTRC": _curv_tag(curves["g"]),
        "bTRC": _curv_tag(curves["b"]),
    }

    n = len(tags)
    header_size, table_size = 128, 4 + 12 * n
    offset = header_size + table_size
    entries, blob = [], bytearray()
    for sig, data in tags.items():
        padded = data + b"\x00" * ((4 - len(data) % 4) % 4)
        entries.append((sig, offset, len(data)))
        blob += padded
        offset += len(padded)

    header = _pack_header(
        profile_size=header_size + table_size + len(blob),
        device_class=device_class,
        version_bytes=version_bytes,
        created=datetime.now(timezone.utc),
    )
    table = struct.pack(">I", n)
    for sig, off, size in entries:
        table += sig.encode("ascii") + struct.pack(">II", off, size)

    data = bytes(header) + table + bytes(blob)
    if path is not None:
        Path(path).write_bytes(data)
    return data


@dataclass(frozen=True)
class ICCProfile:
    device_class: str
    data_colour_space: str
    pcs: str
    version: tuple
    description: str
    copyright: str
    white_xyz: tuple
    matrix: np.ndarray  # 3x3, columns R, G, B
    trc: dict  # {"r": float-or-ndarray, "g": ..., "b": ...}


def read_profile(path: Path | str | bytes) -> ICCProfile:
    """Read back a profile written by ``write_profile``. Not a general ICC
    reader -- it looks up exactly the tags this module writes and raises
    ``KeyError`` if one is missing."""
    data = path if isinstance(path, (bytes, bytearray)) else Path(path).read_bytes()
    device_class = data[12:16].decode("ascii").strip()
    data_colour_space = data[16:20].decode("ascii").strip()
    pcs = data[20:24].decode("ascii").strip()
    version = (data[8], data[9] >> 4, data[9] & 0x0F)

    (n,) = struct.unpack_from(">I", data, 128)
    tags = {}
    for i in range(n):
        entry_off = 132 + 12 * i
        sig = data[entry_off : entry_off + 4].decode("ascii")
        off, size = struct.unpack_from(">II", data, entry_off + 4)
        tags[sig] = data[off : off + size]

    matrix = np.column_stack([_read_xyz_tag(tags["rXYZ"]), _read_xyz_tag(tags["gXYZ"]), _read_xyz_tag(tags["bXYZ"])])
    return ICCProfile(
        device_class=device_class,
        data_colour_space=data_colour_space,
        pcs=pcs,
        version=version,
        description=_read_mluc(tags["desc"]),
        copyright=_read_mluc(tags["cprt"]),
        white_xyz=_read_xyz_tag(tags["wtpt"]),
        matrix=matrix,
        trc={
            "r": _read_curv_tag(tags["rTRC"]),
            "g": _read_curv_tag(tags["gTRC"]),
            "b": _read_curv_tag(tags["bTRC"]),
        },
    )

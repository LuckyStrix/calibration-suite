"""Device identity: camera bodies, lenses and displays.

Two things live here on purpose:

1. A privacy-preserving device id (``device_id``) used as the directory name
   under ``records/`` for every kind of device. Raw serial numbers are never
   written to a record -- this repo is public (implementation-plan.md
   "Privacy") -- only a truncated one-way hash of one.
2. A **pure** EDID parser (``parse_edid``): bytes in, an ``EDIDInfo`` out, no
   file I/O. The OS-specific readers that supply those bytes
   (``read_edid_linux`` / ``read_edid_windows``) are kept separate so the
   parser itself -- the part with a precision requirement (±0.001 on
   chromaticity, exact physical size) -- can be tested against a byte
   fixture with no display attached, on any OS, in CI.
"""

from __future__ import annotations

import hashlib
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from calsuite.constants import (
    DEVICE_ID_SERIAL_HASH_LEN,
    EDID_BASE_BLOCK_LENGTH,
    EDID_CHROMATICITY_STEPS,
    EDID_HEADER_MAGIC,
    EDID_YEAR_OFFSET,
)

# ---------------------------------------------------------------------------
# device id / DeviceRef
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lowercase, hyphen-separated, filesystem- and URL-safe. ``"Canon EOS
    R100"`` -> ``"canon-eos-r100"``."""
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return s or "unknown"


def device_id(model: str, serial: str | None) -> str:
    """``slug(model)-sha256(serial)[:8]``, or ``slug(model)-unknown`` when no
    serial is available (a lens with no reported serial, a display whose
    EDID carries no serial descriptor, etc).

    One-way by construction: nothing here lets an id be turned back into a
    serial number, which is what makes it safe to publish device ids (as
    directory names, in a public git repo) even though the serials that fed
    them must never appear anywhere in this project.
    """
    base = slug(model)
    if not serial:
        return f"{base}-unknown"
    digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:DEVICE_ID_SERIAL_HASH_LEN]
    return f"{base}-{digest}"


@dataclass(frozen=True)
class DeviceRef:
    """A device identity as it appears in a Record -- never a raw serial."""

    kind: str  # "camera" | "lens" | "display"
    model: str
    id: str
    firmware: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "model": self.model, "id": self.id, "firmware": self.firmware}


def camera_ref(meta) -> DeviceRef:
    """Build a camera ``DeviceRef`` from a ``raw.FrameMeta``. Accepts
    anything with ``.model``/``.serial``/``.firmware`` attributes (duck
    typed) rather than importing ``raw.FrameMeta`` by name, to keep this
    module importable without rawpy present (devices.py has no raw
    dependency of its own)."""
    return DeviceRef(kind="camera", model=meta.model, id=device_id(meta.model, meta.serial), firmware=meta.firmware)


def lens_ref(meta) -> DeviceRef:
    """Build a lens ``DeviceRef`` from a ``raw.FrameMeta``. Lenses rarely
    report a serial through EXIF, so this is usually a ``-unknown`` id --
    lens identity in that case comes from the model string alone, which is
    good enough to separate "RF 50mm" from "RF-S 18-45mm" but not two
    copies of the same lens model."""
    return DeviceRef(kind="lens", model=meta.lens, id=device_id(meta.lens, None), firmware="")


# ---------------------------------------------------------------------------
# EDID -- pure parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EDIDInfo:
    manufacturer_pnp_id: str  # 3-letter PNP ID, e.g. "CSO"
    product_code: int
    serial_number: int  # raw 32-bit serial field; 0 if unused. NEVER persisted to a record.
    week: int  # 1-54, 0 = unspecified
    year: int  # calendar year of manufacture (or model year, if week == 255)
    edid_version: int
    edid_revision: int
    digital: bool
    physical_size_mm: tuple[float, float]  # (width, height)
    gamma: float
    chromaticity: dict  # {"r": (x, y), "g": (x, y), "b": (x, y), "w": (x, y)}
    name: str = ""  # from the 0xFC "display name" descriptor, if present
    serial_string: str = ""  # from the 0xFF "serial" text descriptor, if present. NEVER persisted.


def _checksum_ok(block: bytes) -> bool:
    return (sum(block) % 256) == 0


def verify_edid(data: bytes) -> None:
    """Raise ValueError with a specific reason if ``data`` isn't a valid
    EDID base block. Kept separate from ``parse_edid`` so a caller (or a
    test) can check validity without parsing."""
    if len(data) < EDID_BASE_BLOCK_LENGTH:
        raise ValueError(f"EDID block too short: {len(data)} bytes, need {EDID_BASE_BLOCK_LENGTH}")
    block = data[:EDID_BASE_BLOCK_LENGTH]
    if block[:8] != EDID_HEADER_MAGIC:
        raise ValueError(f"missing EDID header magic (got {block[:8].hex()})")
    if not _checksum_ok(block):
        raise ValueError("EDID checksum does not sum to 0 mod 256 -- corrupt or truncated read")


def _decode_pnp_id(hi: int, lo: int) -> str:
    """Bytes 8-9, big-endian: bit 15 reserved (0), then three 5-bit
    letter codes, 1=A .. 26=Z (VESA E-EDID 1.4 §3.4.1)."""
    value = (hi << 8) | lo
    letters = []
    for shift in (10, 5, 0):
        code = (value >> shift) & 0x1F
        letters.append(chr(code + ord("A") - 1))
    return "".join(letters)


def _decode_chromaticity_10bit(hi8: int, lo2: int) -> float:
    """Combine an 8-bit high byte and a 2-bit low fragment into the 10-bit
    fixed-point fraction of 1.0 the format actually stores (§3.10.2)."""
    return ((hi8 << 2) | lo2) / EDID_CHROMATICITY_STEPS


def _parse_detailed_timing_image_size(desc: bytes) -> tuple[float, float] | None:
    """18-byte detailed timing descriptor -> (width_mm, height_mm), or None
    if this descriptor isn't a detailed timing (pixel clock == 0 means it's
    a monitor descriptor instead, handled separately)."""
    pixel_clock = desc[0] | (desc[1] << 8)
    if pixel_clock == 0:
        return None
    h_mm = desc[12] | ((desc[14] & 0xF0) << 4)
    v_mm = desc[13] | ((desc[14] & 0x0F) << 8)
    return (float(h_mm), float(v_mm))


_MONITOR_DESCRIPTOR_NAME = 0xFC  # "Display Product Name"
_MONITOR_DESCRIPTOR_UNSPECIFIED_TEXT = 0xFE  # "Alphanumeric Data String" -- often used for the
# name/model when a panel skips 0xFC entirely, which is common on laptop panels (this laptop's
# eDP-1 included: its first 0xFE holds "CSOT T3", the marketing name; its second holds the raw
# part number "MNG007JA1-2"). Neither is a per-unit serial -- it's identical across every panel
# of that model -- so it's treated as `name`, not zeroed for privacy the way a real serial is.
_MONITOR_DESCRIPTOR_SERIAL = 0xFF  # "Display Product Serial Number" -- an actual per-unit serial


def _parse_monitor_descriptor_text(desc: bytes) -> tuple[int, str] | None:
    """18-byte monitor descriptor (pixel clock bytes == 0) -> (tag, text),
    or None if it's a type this parser doesn't need (range limits, etc).
    Text descriptors are ASCII, terminated by 0x0A and padded with 0x20."""
    if desc[0] != 0 or desc[1] != 0:
        return None  # it's a detailed timing, not a monitor descriptor
    tag = desc[3]
    if tag not in (_MONITOR_DESCRIPTOR_NAME, _MONITOR_DESCRIPTOR_UNSPECIFIED_TEXT, _MONITOR_DESCRIPTOR_SERIAL):
        return None
    raw_text = desc[5:18]
    end = raw_text.find(b"\x0a")
    text_bytes = raw_text[:end] if end != -1 else raw_text
    return tag, text_bytes.decode("ascii", errors="replace").rstrip()


def parse_edid(data: bytes) -> EDIDInfo:
    """Pure: bytes -> EDIDInfo. No file or OS access. Precision matches the
    format's own resolution (chromaticity to 1/1024, physical size to 1mm),
    not a chosen tolerance -- see ``EDID_CHROMATICITY_STEPS``.
    """
    verify_edid(data)
    b = data[:EDID_BASE_BLOCK_LENGTH]

    manufacturer_pnp_id = _decode_pnp_id(b[8], b[9])
    (product_code,) = struct.unpack_from("<H", b, 10)
    (serial_number,) = struct.unpack_from("<I", b, 12)
    week, year_byte = b[16], b[17]
    year = year_byte + EDID_YEAR_OFFSET
    edid_version, edid_revision = b[18], b[19]
    digital = bool(b[20] & 0x80)

    gamma_byte = b[23]
    # 0xFF means "gamma is defined in an extension block"; we don't read
    # those (see EDID_BASE_BLOCK_LENGTH's docstring), so surface it as NaN
    # rather than silently reporting a wrong number.
    gamma = float("nan") if gamma_byte == 0xFF else (gamma_byte + 100) / 100.0

    rg, bw = b[25], b[26]
    rx_hi, ry_hi, gx_hi, gy_hi, bx_hi, by_hi, wx_hi, wy_hi = b[27:35]
    chromaticity = {
        "r": (
            _decode_chromaticity_10bit(rx_hi, (rg >> 6) & 0x03),
            _decode_chromaticity_10bit(ry_hi, (rg >> 4) & 0x03),
        ),
        "g": (
            _decode_chromaticity_10bit(gx_hi, (rg >> 2) & 0x03),
            _decode_chromaticity_10bit(gy_hi, rg & 0x03),
        ),
        "b": (
            _decode_chromaticity_10bit(bx_hi, (bw >> 6) & 0x03),
            _decode_chromaticity_10bit(by_hi, (bw >> 4) & 0x03),
        ),
        "w": (
            _decode_chromaticity_10bit(wx_hi, (bw >> 2) & 0x03),
            _decode_chromaticity_10bit(wy_hi, bw & 0x03),
        ),
    }

    # Physical size: prefer the detailed timing descriptor's mm fields
    # (1mm resolution, up to 4095mm) over the coarse max-image-size bytes
    # 21-22 (1cm resolution) -- the design doc's 344x215mm figure for this
    # laptop panel only comes out exact from the finer field.
    coarse_size_mm = (float(b[21]) * 10.0, float(b[22]) * 10.0)
    physical_size_mm = coarse_size_mm
    name = ""
    serial_string = ""
    for offset in (54, 72, 90, 108):
        desc = b[offset : offset + 18]
        detailed = _parse_detailed_timing_image_size(desc)
        if detailed is not None:
            # First (= preferred) detailed timing wins. This used to
            # overwrite on every descriptor, so the *last* timing's size was
            # reported -- harmless while they agree, wrong when they don't,
            # and EDID's own rule is that descriptor 1 is the preferred
            # timing.
            if detailed != (0.0, 0.0) and physical_size_mm == coarse_size_mm:
                physical_size_mm = detailed
            continue
        text = _parse_monitor_descriptor_text(desc)
        if text is None:
            continue
        tag, value = text
        if tag == _MONITOR_DESCRIPTOR_NAME and not name:
            name = value
        elif tag == _MONITOR_DESCRIPTOR_UNSPECIFIED_TEXT and not name:
            # First unspecified-text descriptor only: a second one (this
            # panel's is a raw part number) is not the display's "name".
            name = value
        elif tag == _MONITOR_DESCRIPTOR_SERIAL:
            serial_string = value

    return EDIDInfo(
        manufacturer_pnp_id=manufacturer_pnp_id,
        product_code=product_code,
        serial_number=serial_number,
        week=week,
        year=year,
        edid_version=edid_version,
        edid_revision=edid_revision,
        digital=digital,
        physical_size_mm=physical_size_mm,
        gamma=gamma,
        chromaticity=chromaticity,
        name=name,
        serial_string=serial_string,
    )


def edid_hash(raw: bytes) -> str:
    """sha256 of the raw EDID block -- used by doctor.py to notice a panel
    swap (the hash changing between runs on the same output). This hashes
    the whole block, serial bytes included; unlike ``device_id`` it is
    never stored next to a human-readable serial and its only job is "same
    blob as last time?", so there is nothing to redact."""
    return hashlib.sha256(raw).hexdigest()


def display_ref(edid: EDIDInfo) -> DeviceRef:
    """Build a display ``DeviceRef`` from parsed EDID. Model is the name
    descriptor when present, else ``"<PNPID> <product_code>"``; the "serial"
    fed to ``device_id`` is the EDID serial number/string if either is
    present (and immediately hashed -- see ``device_id``), never stored raw.
    """
    model = edid.name or f"{edid.manufacturer_pnp_id} {edid.product_code:04x}"
    serial = edid.serial_string or (str(edid.serial_number) if edid.serial_number else None)
    return DeviceRef(kind="display", model=model, id=device_id(model, serial), firmware="")


# ---------------------------------------------------------------------------
# EDID -- OS-specific byte sources (not pure; kept separate from the parser)
# ---------------------------------------------------------------------------


def list_linux_edids(drm_dir: Path | str = "/sys/class/drm") -> dict[str, bytes]:
    """``{"card0-eDP-1": <128+ bytes>, ...}`` for every connector whose sysfs
    EDID node is non-empty. A disconnected output's ``edid`` file reads back
    zero bytes (not an error, not missing) -- those are skipped rather than
    surfaced as a parse failure, since "nothing plugged in here" is the
    normal case for most outputs on most machines.
    """
    root = Path(drm_dir)
    out = {}
    for card_dir in sorted(root.glob("*/edid")):
        try:
            data = card_dir.read_bytes()
        except OSError:
            continue
        if len(data) >= EDID_BASE_BLOCK_LENGTH:
            out[card_dir.parent.name] = data
    return out


def read_edid_windows() -> dict[str, bytes]:
    """``{instance_path: edid_bytes}`` read from the Windows registry.

    UNVERIFIED: written from the documented registry layout
    (``HKLM\\SYSTEM\\CurrentControlSet\\Enum\\DISPLAY\\<PNPid>\\<instance>\\
    Device Parameters\\EDID``, docs/design.md §6) and has not been run
    against a real Windows machine as part of this build -- there wasn't
    one available. Treat its output with suspicion until confirmed there;
    the registry layout below is standard but instance naming varies enough
    across driver versions that it's worth double-checking on first use.
    """
    if sys.platform != "win32":
        raise RuntimeError("read_edid_windows() only works on Windows")
    import winreg  # noqa: PLC0415 -- platform-guarded import, only reachable on win32

    out = {}
    display_key = r"SYSTEM\CurrentControlSet\Enum\DISPLAY"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, display_key) as root:
        for i in range(_winreg_subkey_count(winreg, root)):
            pnp_id = winreg.EnumKey(root, i)
            with winreg.OpenKey(root, pnp_id) as pnp_key:
                for j in range(_winreg_subkey_count(winreg, pnp_key)):
                    instance = winreg.EnumKey(pnp_key, j)
                    param_path = f"{instance}\\Device Parameters"
                    try:
                        with winreg.OpenKey(pnp_key, param_path) as params:
                            value, _ = winreg.QueryValueEx(params, "EDID")
                            out[f"{pnp_id}\\{instance}"] = bytes(value)
                    except OSError:
                        continue
    return out


def _winreg_subkey_count(winreg_module, key) -> int:
    return winreg_module.QueryInfoKey(key)[0]

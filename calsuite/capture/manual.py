"""Manual import: point the suite at a folder of already-captured raw
files and it sorts them into roles by a cheap signal check plus exposure
time -- no tethering required. "Manual import is first-class" (docs/
design.md §3.3): every measurement must work from a folder of raws shot on
either OS, tethering is only ever a convenience layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calsuite import raw as rawmod

# A mean signal (above black) below this, in DN, marks a frame as a
# bias/dark candidate rather than a flat or target. Set well below a
# typical flat's mid-gray target (commonly 30-50% of full well) and well
# above read noise alone, so the split doesn't need a per-camera lookup.
NEAR_BLACK_DN = 200.0

# At or below this exposure time, a near-black frame is a bias (dominated
# by read noise, negligible dark current); above it, it's a dark. Bias
# frames are conventionally shot at the camera's shortest available
# exposure, and 1/1000s is comfortably shorter than any realistic
# bias-vs-dark boundary needs to be exact about.
BIAS_MAX_EXPOSURE_S = 1.0 / 1000

# Stride for the cheap classification mean -- every Nth row/column keeps
# classification fast on a full-size frame without needing a full-frame
# reduction just to decide "near black or not".
SUBSAMPLE_STRIDE = 8

# A flat sits in this fraction of the white level: bright enough that shot
# noise dominates read noise (the point of a flat), dim enough to leave
# headroom before clipping.
FLAT_LOW_FRACTION = 0.10
FLAT_HIGH_FRACTION = 0.90


@dataclass
class ManifestEntry:
    path: str
    role: str  # "bias" | "dark" | "flat" | "target"
    mean_dn: float
    exposure_s: float | None
    iso: int | None


@dataclass
class Manifest:
    entries: list = field(default_factory=list)

    def by_role(self, role: str) -> list:
        return [e for e in self.entries if e.role == role]


def _cheap_mean(frame) -> float:
    """A fast, subsampled mean over the visible area -- enough to tell
    "near black" from "not"; not a photometric measurement (camera/bias.py
    etc. read the full frame when the number actually matters)."""
    row_slice, col_slice = frame.visible
    sub = frame.cfa[row_slice, col_slice][::SUBSAMPLE_STRIDE, ::SUBSAMPLE_STRIDE]
    return float(np.mean(sub))


def classify(frame) -> str:
    """bias / dark / flat / target, from a cheap mean signal and the
    frame's exposure time."""
    mean_dn = _cheap_mean(frame)
    black = float(np.mean(frame.black_level)) if frame.black_level else 0.0
    signal = mean_dn - black

    if signal < NEAR_BLACK_DN:
        exposure = frame.meta.exposure_s
        if exposure is not None and exposure <= BIAS_MAX_EXPOSURE_S:
            return "bias"
        return "dark"

    # `signal` is already relative to black, so the flat band has to be a
    # fraction of the DN *range* (white - black), not of white itself --
    # comparing a black-relative number against an absolute fraction of
    # white would shift the whole band down by exactly `black`.
    white = frame.white_level or 65535.0
    span = max(white - black, 1.0)
    if FLAT_LOW_FRACTION * span <= signal <= FLAT_HIGH_FRACTION * span:
        return "flat"
    return "target"


_RAW_EXTENSIONS = {".cr3", ".CR3", ".dng", ".DNG", ".nef", ".NEF"}


def scan_folder(folder: Path | str) -> Manifest:
    """Load every raw file directly under ``folder`` (non-recursive -- one
    capture session, one folder), classify it, and return a ``Manifest``.
    A one-shot batch operation: decoding every frame isn't free, but
    manual import runs once per session, not per pixel.
    """
    folder = Path(folder)
    manifest = Manifest()
    for path in sorted(folder.iterdir()):
        if path.suffix not in _RAW_EXTENSIONS:
            continue
        frame = rawmod.load(path)
        manifest.entries.append(
            ManifestEntry(
                path=str(path),
                role=classify(frame),
                mean_dn=_cheap_mean(frame),
                exposure_s=frame.meta.exposure_s,
                iso=frame.meta.iso,
            )
        )
    return manifest

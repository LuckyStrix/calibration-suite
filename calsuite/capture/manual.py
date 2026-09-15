"""Manual import: point the suite at a folder of already-captured raw
files and it sorts them into roles by a cheap signal check plus exposure
time -- no tethering required. "Manual import is first-class" (docs/
design.md §3.3): every measurement must work from a folder of raws shot on
either OS, tethering is only ever a convenience layered on top.

**Classification rule** (bias / dark / near_black / flat / target): first,
a mean signal (above black) below ``NEAR_BLACK_DN`` splits off bias/dark by
exposure time (``BIAS_MAX_EXPOSURE_S``) -- or, with no exposure time on the
frame at all (``raw._read_metadata`` returns an empty ``FrameMeta`` when
neither exiftool nor dcraw is installed, a supported configuration), by
declining to split at all: the role is ``near_black``, since a bias and a
dark are the same picture without an exposure time to tell them apart.
Above that, a frame in the flat's
DN band (``FLAT_LOW_FRACTION``..``FLAT_HIGH_FRACTION`` of the DN range) is
still only called "flat" if it also looks spatially uniform
(``FLAT_MAX_CV``, a coefficient-of-variation cap) -- a slanted-edge or
ChArUco target shot at a exposure that happens to average to ~50% signal
lands in the exact same DN band as a real flat, but is bimodal (large
near-black and near-white regions) rather than uniform, so its CV is far
higher than any real flat's PRNU-plus-shot-noise scatter. Anything in the
signal band that fails the uniformity check, or above the flat band
entirely, is called "target". A caller that already knows every frame in a
folder is one role (lens sessions, most of all -- see lens/commands.py's
own docstring on why it bypasses this heuristic entirely for that reason)
can skip the heuristic altogether via ``expected_role``.
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

FLAT_MAX_CV = 0.15
# A uniformly-lit flat's spread across the frame is shot noise plus a few
# percent PRNU (synth.sensor.SensorModel's own default prnu_std is 1%) --
# comfortably under 15% coefficient of variation even with a noticeably
# non-uniform light source. A two-tone target (a slanted edge, a ChArUco
# board) sitting at a similar *mean* signal is a much starker outlier: half
# its area near black and half near white gives a CV close to 1.0, nowhere
# near this threshold, so 0.15 separates the two cases with a lot of margin
# on both sides rather than being finely tuned.


@dataclass
class ManifestEntry:
    path: str
    role: str  # "bias" | "dark" | "flat" | "target"
    mean_dn: float
    exposure_s: float | None
    iso: int | None
    frame: object = None
    # The already-decoded RawFrame for this entry, cached at scan_folder()
    # time so a caller (camera/commands.py, camera/color_commands.py) never
    # has to call raw.load() on the same path a second time -- decoding a
    # real raw file isn't free, and scan_folder() already paid for it once
    # per file to classify it.


@dataclass
class Manifest:
    entries: list = field(default_factory=list)

    def by_role(self, role: str) -> list:
        return [e for e in self.entries if e.role == role]


def _cheap_subsample(frame) -> np.ndarray:
    """The same fast, subsampled slice of the visible area every cheap
    classification statistic below is computed from -- one subsample, two
    statistics (mean, then CV), rather than two separate reductions over
    the full frame."""
    row_slice, col_slice = frame.visible
    return frame.cfa[row_slice, col_slice][::SUBSAMPLE_STRIDE, ::SUBSAMPLE_STRIDE]


def _cheap_mean(frame) -> float:
    """A fast, subsampled mean over the visible area -- enough to tell
    "near black" from "not"; not a photometric measurement (camera/bias.py
    etc. read the full frame when the number actually matters)."""
    return float(np.mean(_cheap_subsample(frame)))


def _cheap_cv(frame) -> float:
    """Coefficient of variation (std/mean) of the same cheap subsample --
    near-zero for a uniformly-lit flat, close to 1 for a two-tone target
    (a slanted edge, a ChArUco board) at a similar mean signal. See the
    module docstring's "classification rule" and ``FLAT_MAX_CV``."""
    sub = _cheap_subsample(frame).astype(np.float64)
    mean = float(np.mean(sub))
    if mean <= 0:
        return 0.0
    return float(np.std(sub)) / mean


def classify(frame, expected_role: str | None = None) -> str:
    """bias / dark / near_black / flat / target, from a cheap mean signal, the frame's
    exposure time, and (to tell a flat from a same-signal-band target) its
    spatial uniformity -- see the module docstring's "classification
    rule". ``expected_role``, when given, is returned unconditionally
    (documented escape hatch for a caller that already knows every frame
    in a folder is one role, e.g. because it isn't a bias/dark/flat/target
    session at all)."""
    if expected_role is not None:
        return expected_role

    mean_dn = _cheap_mean(frame)
    black = float(np.mean(frame.black_level)) if frame.black_level else 0.0
    signal = mean_dn - black

    if signal < NEAR_BLACK_DN:
        exposure = frame.meta.exposure_s
        if exposure is None:
            # Bias and dark are the *same* picture without an exposure time
            # to tell them apart, and `raw._read_metadata` returns an empty
            # FrameMeta when neither exiftool nor dcraw is installed (a
            # supported configuration: "pixels still loaded, metadata just
            # empty"). Calling it "dark" then made `camera bias --from DIR`
            # report "no bias frames found" on a folder of perfectly good
            # bias frames, and counted those same frames as darks. Say
            # undecided instead, and let the command name the cause.
            return "near_black"
        if exposure <= BIAS_MAX_EXPOSURE_S:
            return "bias"
        return "dark"

    # `signal` is already relative to black, so the flat band has to be a
    # fraction of the DN *range* (white - black), not of white itself --
    # comparing a black-relative number against an absolute fraction of
    # white would shift the whole band down by exactly `black`.
    white = frame.white_level or 65535.0
    span = max(white - black, 1.0)
    if FLAT_LOW_FRACTION * span <= signal <= FLAT_HIGH_FRACTION * span:
        if _cheap_cv(frame) <= FLAT_MAX_CV:
            return "flat"
        # Same DN band as a flat, but too spatially non-uniform to be one
        # -- a slanted-edge/ChArUco target that happens to average to a
        # mid-level signal, not a uniformly-lit flat field.
        return "target"
    return "target"


_RAW_EXTENSIONS = {".cr3", ".CR3", ".dng", ".DNG", ".nef", ".NEF"} | rawmod.NPZ_EXTENSIONS | {
    ext.upper() for ext in rawmod.NPZ_EXTENSIONS
}
# ``.npz`` (raw.save_npz's format) alongside the real raw extensions --
# a synthetic RawFrame written that way is otherwise indistinguishable
# from a real capture to everything downstream of this scan (demo.py's
# whole premise, and every area's CLI test that needs a --from DIR of
# frames with no camera attached).


def scan_folder(folder: Path | str, expected_role: str | None = None) -> Manifest:
    """Load every raw file directly under ``folder`` (non-recursive -- one
    capture session, one folder), classify it, and return a ``Manifest``.
    A one-shot batch operation: decoding every frame isn't free, but
    manual import runs once per session, not per pixel -- and each
    ``ManifestEntry.frame`` caches exactly that decode, so a caller
    (camera/commands.py, camera/color_commands.py) never re-decodes the
    same file just to get the frame classification already needed.

    ``expected_role``, when given, is passed straight through to
    ``classify()`` for every frame -- a caller that already knows every
    frame in this folder is one role skips the bias/dark/flat/target
    heuristic entirely rather than fighting it.
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
                role=classify(frame, expected_role=expected_role),
                mean_dn=_cheap_mean(frame),
                exposure_s=frame.meta.exposure_s,
                iso=frame.meta.iso,
                frame=frame,
            )
        )
    return manifest

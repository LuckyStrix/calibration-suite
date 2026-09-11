"""The characterized camera as a colorimeter (docs/design.md §5.1: "free,
ΔE00 2-5 on sRGB-ish panels, worse on narrow-band wide-gamut" -- and "honest
only if it's cross-checked once against one of the other two").

Depends on a ``camera.color`` record produced by the color agent's own
analysis (``camera/color.py``, a different area of this same wave): a
raw->XYZ 3x3 matrix. Per this wave's contract, that code is never imported
here -- only ``record.result["matrix_raw_to_xyz"]`` (that exact key) is
read, as plain data, from the most recent exportable ``camera.color``
record for the camera device. If no such record exists, this fails with a
clear, specific message (``NoCameraColorRecord``) rather than a KeyError
three frames down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calsuite import raw as rawmod
from calsuite import store as storemod
from calsuite.capture import gphoto2 as gphoto2mod
from calsuite.capture import manual as manualmod
from calsuite.display import constants as dc
from calsuite.display import window as windowmod
from calsuite.display.backends.base import Accuracy, Measurement

ROI_FRACTION = 0.10
# Mean over the central 10% (by each linear dimension, ~1% of area) of
# each CFA plane -- away from whatever lens is mounted's own edge
# vignetting/distortion (design §4.3), which would otherwise bias a
# colorimetric reading independently of anything about the display being
# measured. A generous margin, not tuned to a specific lens.


class NoCameraColorRecord(RuntimeError):
    """No exportable ``camera.color`` record exists for the given camera
    device -- this wave's explicit contract: "fail with a clear message"
    rather than propagate a bare KeyError/AttributeError from deep inside
    the matrix application."""


def load_matrix_raw_to_xyz(store: storemod.Store, device_id: str) -> np.ndarray:
    """Read the raw->XYZ 3x3 matrix from the most recent exportable
    ``camera.color`` record for `device_id`. Reads it purely as data
    (``record.result["matrix_raw_to_xyz"]``, the exact key this wave's
    contract specifies) -- never imports the color agent's own code."""
    record = store.latest("camera.color", device_id)
    if record is None:
        raise NoCameraColorRecord(f"no camera.color record found for camera device {device_id!r}")
    try:
        storemod.require_exportable(record)
    except storemod.ExportRefused as exc:
        raise NoCameraColorRecord(
            f"the most recent camera.color record ({record.id}) for {device_id!r} is not exportable: {exc}"
        ) from exc
    if "matrix_raw_to_xyz" not in record.result:
        raise NoCameraColorRecord(
            f"camera.color record {record.id} has no 'matrix_raw_to_xyz' key in its result"
        )
    matrix = np.asarray(record.result["matrix_raw_to_xyz"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise NoCameraColorRecord(
            f"camera.color record {record.id}'s matrix_raw_to_xyz has shape {matrix.shape}, expected (3, 3)"
        )
    return matrix


def _central_roi_mean(plane: np.ndarray, roi_fraction: float = ROI_FRACTION) -> float:
    rows, cols = plane.shape
    r0, r1 = int(rows * (0.5 - roi_fraction / 2)), int(rows * (0.5 + roi_fraction / 2))
    c0, c1 = int(cols * (0.5 - roi_fraction / 2)), int(cols * (0.5 + roi_fraction / 2))
    return float(np.mean(plane[max(r0, 0) : max(r1, r0 + 1), max(c0, 0) : max(c1, c0 + 1)]))


def frame_to_raw_rgb(frame) -> tuple:
    """Central-ROI, black-subtracted, exposure-normalized (r, g, b) from
    one raw frame of a displayed patch -- CFA planes only, never
    demosaiced (house rule 1); `g` is the mean of the G1/G2 sub-planes
    (the standard way to collapse two green samples into one scalar
    per-patch signal -- not a demosaic, since no spatial interpolation is
    involved).

    The black level subtracted is ``frame.black_level`` (the ``RawFrame``'s
    own per-metadata black, per this wave's contract: "black level taken
    from raw"), averaged to one scalar since ``RawFrame.black_level``'s 4
    values are indexed by the raw file's own CFA tile phase, not
    necessarily the R/G1/G2/B plane names ``raw.planes()`` returns -- most
    sensors (including ``synth.sensor.SensorModel``'s) report one uniform
    black level across all 4 positions, so a mean is exact for the common
    case and a reasonable approximation otherwise.

    Dividing by exposure time gives *relative* luminance across patches
    shot at different shutter speeds (design §5.2: "camera exposure
    bracketing is fine for relative luminance") -- it does not by itself
    produce absolute cd/m^2, which is exactly why this backend's accuracy
    statement calls out needing a cross-check against another backend.
    """
    planes = rawmod.planes(frame)
    black = float(np.mean(frame.black_level)) if frame.black_level else 0.0
    exposure_s = frame.meta.exposure_s or 1.0
    r = (_central_roi_mean(planes["R"]) - black) / exposure_s
    g = (_central_roi_mean(planes["G1"]) + _central_roi_mean(planes["G2"])) / 2.0 - black
    g = g / exposure_s
    b = (_central_roi_mean(planes["B"]) - black) / exposure_s
    return (r, g, b)


@dataclass
class CameraBackend:
    """`frame_for_patch`: ``Patch -> RawFrame``, however the caller wants to
    supply one raw frame per patch (``capture_via_gphoto2``/
    ``frames_from_manual_folder`` below build one of these; a test can pass
    a plain dict lookup)."""

    device_id: str
    store: storemod.Store
    frame_for_patch: object  # Callable[[Patch], RawFrame]
    absolute_scale_cdm2_per_unit: float | None = None  # set once cross-checked against another backend
    cross_checked_against: str | None = None
    name: str = "camera"

    def __post_init__(self):
        self._matrix = load_matrix_raw_to_xyz(self.store, self.device_id)

    def accuracy(self) -> Accuracy:
        from calsuite.constants import ESTIMATED_ACCURACY  # foundation constant, not display/-specific

        basis = "camera as colorimeter (design §5.1/§10 estimate)"
        if self.cross_checked_against is None:
            basis += " -- NOT YET cross-checked against another backend; design §5.1 requires this before trusting its numbers"
        else:
            basis += f" -- cross-checked against {self.cross_checked_against!r}"
        return Accuracy(
            de00_estimate=ESTIMATED_ACCURACY["display_camera_de00"],
            basis=basis,
            cross_checked_against=self.cross_checked_against,
        )

    def measure(self, patches: list) -> list:
        out = []
        for patch in patches:
            frame = self.frame_for_patch(patch)
            raw_rgb = np.asarray(frame_to_raw_rgb(frame), dtype=np.float64)
            xyz = self._matrix @ raw_rgb
            if self.absolute_scale_cdm2_per_unit is not None:
                xyz = xyz * self.absolute_scale_cdm2_per_unit
            out.append(Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)))
        return out


# ---------------------------------------------------------------------------
# capture orchestration (I/O -- this is a backend module, house rule 5's
# "I/O ... only in the backends/window/install/commands modules" allows it)
# ---------------------------------------------------------------------------


def capture_via_gphoto2(screen, patches: list, out_dir: Path, *, settle_s: float = dc.SETTLE_TIME_S, sleep=None) -> dict:
    """Show each patch and photograph it via tethered gphoto2 capture,
    returning ``{patch.label: RawFrame}``."""
    out_dir = Path(out_dir)

    def _capture_one(patch):
        path = gphoto2mod.capture_and_download(out_dir, filename=f"{patch.label}.%C")
        return rawmod.load(path)

    frames = windowmod.run_patch_sequence(screen, patches, _capture_one, settle_s=settle_s, sleep=sleep)
    return {patch.label: frame for patch, frame in zip(patches, frames, strict=True)}


def frames_from_manual_folder(folder: Path, patches: list) -> dict:
    """Pair a folder of already-captured raw files (``capture.manual``,
    design §3.3: "manual import is first-class") with `patches` in
    filename-sorted order -- the same order a guided capture session shot
    them in, since manual import has no other way to know which frame
    belongs to which patch. Returns ``{patch.label: RawFrame}``.
    """
    manifest = manualmod.scan_folder(folder)
    if len(manifest.entries) != len(patches):
        raise ValueError(
            f"{folder} has {len(manifest.entries)} raw file(s) but {len(patches)} patches were requested -- "
            "manual capture must shoot exactly one frame per patch, in order"
        )
    frames = {}
    for patch, entry in zip(patches, manifest.entries, strict=True):
        frames[patch.label] = rawmod.load(entry.path)
    return frames

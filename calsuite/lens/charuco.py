"""ChArUco board definition and detection.

House rule 1 (docs/design.md): nothing here ever demosaics. Detection runs
on the **green plane** -- the average of G1/G2, normalized to 8-bit for
OpenCV's detector only (the normalization is display-only; every downstream
number is still built from the raw-scale corner positions) -- and,
separately, on each of the four CFA planes individually for lens/tca.py's
per-channel comparison.

Every plane array from ``raw.planes()`` is at quarter resolution (one
sample per 2x2 Bayer tile): plane pixel ``(py, px)`` sits at full-sensor
pixel ``(visible_row0 + 2*py + dr, visible_col0 + 2*px + dc)``, where
``(dr, dc)`` is that plane's phase within the 2x2 tile (raw.py's own
docstring: "the first 'G' encountered in raster order... is G1, the second
is G2"). ``plane_to_sensor`` re-derives that phase locally rather than
reaching into ``raw.py``'s private helpers, since Wave 2B doesn't own
``raw.py`` and the mapping is a documented, stable convention rather than
an implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from calsuite import raw as rawmod
from calsuite.lens import constants as C

if TYPE_CHECKING:
    import cv2.aruco as aruco

PLANE_NAMES = ("R", "G1", "G2", "B")


@dataclass(frozen=True)
class Detection:
    """One plane's ChArUco detection, corners already mapped to full-sensor
    pixel coordinates."""

    plane: str
    corners: np.ndarray  # (N, 2) float32, sensor px
    ids: np.ndarray  # (N,) int32


def build_board(square_length: float = 1.0) -> aruco.CharucoBoard:
    """The shared board definition (targets/generate_targets.py rasterizes
    it, this module detects it, synth/lens.py renders a synthetic version
    of it). ``square_length`` is in whatever real-world unit the caller
    wants object points in (mm, typically) -- it only scales the board's
    object points, not its layout, so detection code that doesn't care
    about physical units can leave it at the default 1.0.
    """
    import cv2.aruco as aruco

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, C.BOARD_DICT_NAME))
    marker_length = square_length * C.BOARD_MARKER_RATIO
    return aruco.CharucoBoard(
        (C.BOARD_SQUARES_X, C.BOARD_SQUARES_Y), square_length, marker_length, dictionary
    )


def _plane_offsets(pattern: str) -> dict:
    """Reimplements raw.py's documented plane-naming convention locally
    (see module docstring) -- (row, col) phase within the 2x2 tile for each
    of R/G1/G2/B, given the 4-char CFA pattern string at the visible area's
    origin."""
    if len(pattern) != 4:
        raise ValueError(f"expected a 2x2 (4-char) CFA pattern, got {pattern!r}")
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    offsets = {}
    g_count = 0
    for pos, ch in zip(positions, pattern, strict=True):
        if ch == "G":
            g_count += 1
            offsets[f"G{g_count}"] = pos
        else:
            offsets[ch] = pos
    return offsets


def plane_to_sensor(frame, corners_xy: np.ndarray, plane_name: str) -> np.ndarray:
    """Map ``(N, 2)`` plane-pixel ``(x, y)`` corners to sensor pixel
    coordinates *within the visible area*: ``plane_px * 2 + cfa_offset``
    (Wave 2B task prompt), where the offset is the plane's own phase inside
    the 2x2 tile.

    Visible-area, not absolute-CFA: ``raw.planes`` slices the visible area,
    and every consumer of these coordinates measures them against
    ``image_size_from_frame``, which is the visible size. Adding the visible
    origin here (as this used to) left the corners in full-CFA coordinates
    while the image size stayed visible-sized, so ``tca.fit_tca`` and
    ``distortion.coverage_grid`` both took the optical center to be
    ``((w-1)/2, (h-1)/2)`` of the *visible* frame while reading corners
    offset by the margins -- 288 px left and 56 px above the real center on
    this project's own synthetic sensor. On noise-free data that turned a
    pure radial TCA scale of 1.000400 into a measured 1.000390 with 0.09 px
    of "residual", and skewed the coverage histogram the distortion
    refusal reads (387 vs 790 corners in opposite sectors under uniform
    coverage). No test caught it because both analyses' own tests build
    corners directly in visible-relative coordinates and never call this.
    """
    dr, dc = _plane_offsets(frame.pattern)[plane_name]
    out = np.empty_like(corners_xy, dtype=np.float64)
    out[:, 0] = corners_xy[:, 0] * 2.0 + dc  # x <-> column
    out[:, 1] = corners_xy[:, 1] * 2.0 + dr  # y <-> row
    return out


def to_8bit(plane: np.ndarray) -> np.ndarray:
    """Normalize a float plane to 8-bit for OpenCV's detector -- detection
    only, per house rule 1's discipline of never letting a demosaiced or
    tone-mapped image feed a *measurement* (only the detector's own corner
    search, which produces coordinates read back against the raw-scale
    plane)."""
    lo, hi = np.percentile(plane, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (plane.astype(np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _detect(img8: np.ndarray, board: aruco.CharucoBoard):
    import cv2.aruco as aruco

    detector = aruco.CharucoDetector(board)
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(img8)
    if corners is None or ids is None or len(ids) == 0:
        return None, None
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2), np.asarray(ids, dtype=np.int32).reshape(-1)


def detect_plane(frame, plane_name: str, board: aruco.CharucoBoard | None = None) -> Detection | None:
    """Detect the board on a single CFA plane (``"R"``, ``"G1"``, ``"G2"``
    or ``"B"``), for lens/tca.py's per-channel comparison. Returns ``None``
    if nothing was detected."""
    board = board or build_board()
    planes = rawmod.planes(frame)
    corners, ids = _detect(to_8bit(planes[plane_name]), board)
    if corners is None:
        return None
    return Detection(plane=plane_name, corners=plane_to_sensor(frame, corners, plane_name), ids=ids)


def detect_green(frame, board: aruco.CharucoBoard | None = None) -> Detection | None:
    """Detect the board on the **green plane** -- the average of G1/G2,
    which is what distortion.py fits against (green has 2x the sample
    density of R/B in a Bayer mosaic, so it's the sharpest, most reliable
    plane for corner-finding). Corner coordinates are still mapped back to
    full-sensor pixels -- the choice of G1's phase offset for that mapping
    is arbitrary between G1/G2 (both map to the same *sensor* grid up to a
    sub-pixel phase the averaging already blurred), so G1's offset is used.
    """
    board = board or build_board()
    planes = rawmod.planes(frame)
    green = 0.5 * (planes["G1"] + planes["G2"])
    corners, ids = _detect(to_8bit(green), board)
    if corners is None:
        return None
    return Detection(plane="G", corners=plane_to_sensor(frame, corners, "G1"), ids=ids)


def detect_all_planes(frame, board: aruco.CharucoBoard | None = None) -> dict:
    """``{"R": Detection|None, "G1": ..., "G2": ..., "B": ...}`` -- the
    per-channel detections lens/tca.py matches against each other by id."""
    board = board or build_board()
    return {name: detect_plane(frame, name, board) for name in PLANE_NAMES}


def object_points_for(board: aruco.CharucoBoard, ids: np.ndarray) -> np.ndarray:
    """Board-frame object points (N, 3) for the given charuco ids, in the
    board's own square-length units -- ``board.getChessboardCorners()`` is
    indexed by id (confirmed against a live board: row i's corner has id
    i), so a plain fancy-index is exact and avoids going back through
    OpenCV's ``matchImagePoints`` (which exists for the same purpose but
    additionally re-derives the image points, unneeded here since callers
    already have sensor-mapped corners)."""
    all_corners = board.getChessboardCorners()
    return all_corners[ids].astype(np.float64)


def image_size_from_frame(frame) -> tuple:
    """``(width, height)`` in **sensor** pixels of the frame's visible
    area -- the convention cv2.calibrateCamera and lensfun's Hugin-radius
    normalization both use (width first)."""
    rows = frame.visible[0].stop - frame.visible[0].start
    cols = frame.visible[1].stop - frame.visible[1].start
    return cols, rows


def render_board_image(board: aruco.CharucoBoard, size_px: tuple, margin_px: int = 0) -> np.ndarray:
    """8-bit board image at ``size_px = (width, height)`` -- used by
    targets/generate_targets.py to rasterize the printable/displayable
    target, and by synth/lens.py as the texture a synthetic camera renders.
    Thin wrapper so both call sites agree on the (width, height) argument
    order OpenCV's ``generateImage`` expects."""
    return board.generateImage(size_px, marginSize=margin_px)

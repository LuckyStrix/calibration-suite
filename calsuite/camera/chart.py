"""Chart patch sampling and reference data (docs/design.md sec 3.2 Tier A).

Two independent things live here:

1. **Reference data**: what a chart's patches *should* be colorimetrically
   (X, Y, Z under some known illuminant), from colour-science's built-in
   ColorChecker datasets or a custom CSV.
2. **Sampling**: given a ``raw.RawFrame`` and 4 user-clicked corner pixel
   coordinates, recover a per-patch, per-CFA-plane mean/std/clipped-fraction
   -- never demosaicing (house rule 1), and never touching a file (house
   rule 5: this module is pure array-in, dataclass-out; ``color_commands.py``
   is the only place that loads a frame from disk).

Corner convention: the 4 corners are the **pixel centers of the four corner
patches** (top-left, top-right, bottom-right, bottom-left, in that order),
in full-sensor pixel (x=column, y=row) coordinates -- the same convention
dcamprof/Argyll's chart readers use, and the one that lets a straight
perspective homography place every other patch center by simple bilinear
interpolation in grid-index space, with no separate estimate of the
physical card's outer border.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calsuite import raw as rawmod
from calsuite.camera import color_constants as cc
from calsuite.fit import Refusal

CHANNELS = ("R", "G1", "G2", "B")

# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePatch:
    name: str
    XYZ: tuple  # (X, Y, Z); Y=1.0 for a perfect reflecting diffuser (see color.py's module docstring)
    is_neutral: bool


@dataclass(frozen=True)
class ReferenceChart:
    name: str
    rows: int
    cols: int
    patches: list  # list[ReferencePatch], length rows*cols, row-major (matches sample_chart's order)
    illuminant_xy: tuple  # (x, y) chromaticity this chart's XYZ values were computed under

    def neutral_indices(self) -> list:
        return [i for i, p in enumerate(self.patches) if p.is_neutral]

    def brightest_neutral_index(self) -> int:
        """The patch used as "the illuminant white" for the white-preserving
        constraint -- the neutral patch with the highest reference Y."""
        idx = self.neutral_indices()
        if not idx:
            raise ValueError(f"reference chart {self.name!r} has no neutral patches")
        return max(idx, key=lambda i: self.patches[i].XYZ[1])


def _xyY_to_XYZ(x: float, y: float, Y: float) -> tuple:
    if y == 0:
        return (0.0, 0.0, 0.0)
    return (x * Y / y, Y, (1 - x - y) * Y / y)


def reference_colorchecker(name: str = "ColorChecker24 - After November 2014") -> ReferenceChart:
    """Reference XYZ for a colour-science built-in ColorChecker, e.g. the
    default "ColorChecker24 - After November 2014" (the current X-Rite
    ColorChecker Classic dye set -- docs/design.md sec 3.2). ``colour``'s
    chromaticity dataset already expresses Y on the perfect-diffuser=1
    scale ``color.py`` expects (patch Y ranges ~0.03 black to ~0.88 white),
    so no rescaling happens here.
    """
    import colour

    cc_data = colour.CCS_COLOURCHECKERS[name]
    patches = []
    for patch_name, xyY in cc_data.data.items():
        x, y, Y = (float(v) for v in xyY)
        XYZ = _xyY_to_XYZ(x, y, Y)
        # The bottom row of every colour-science ColorChecker24 dataset is
        # the 6-step neutral gray ramp (white 9.5 .. black 2); every other
        # patch name in the dataset is a saturated/skin/foliage color. There
        # is no chroma-free way to ask colour-science "is this neutral", so
        # this greps the dataset's own naming convention rather than
        # thresholding a computed a*/b* (which would need an assumed PCS
        # white unrelated to what's being measured).
        is_neutral = "neutral" in patch_name.lower() or patch_name.lower().startswith(("white", "black"))
        patches.append(ReferencePatch(name=patch_name, XYZ=XYZ, is_neutral=is_neutral))

    return ReferenceChart(
        name=name,
        rows=int(cc_data.rows),
        cols=int(cc_data.columns),
        patches=patches,
        illuminant_xy=tuple(float(v) for v in cc_data.illuminant),
    )


_NEUTRAL_CHROMA_MAX = 4.0
# For a custom CSV chart with no naming convention to lean on, a patch is
# called "neutral" if its CIE 1976 a*b* chroma (computed against the chart's
# own illuminant as the Lab reference white) is below this -- a*b* chroma of
# 4 is a commonly used rule-of-thumb "visually neutral" cutoff (well below a
# just-noticeable difference in *chroma* for most observers at this
# lightness range), chosen so a chart with a genuine light-gray patch still
# gets flagged neutral even if it isn't a mathematically perfect gray.


def reference_from_csv(path: Path | str, *, illuminant_xy: tuple | None = None) -> ReferenceChart:
    """Custom chart reference from a CSV with columns ``name,X,Y,Z`` or
    ``name,L,a,b`` (auto-detected from the header), plus rows/cols and the
    illuminant it was measured under.

    Layout and illuminant come from optional leading comment lines
    (``# rows=N``, ``# cols=M``, ``# illuminant=D65`` or
    ``# illuminant_xy=x,y``) -- the instructions call for "a custom CSV
    (name, X, Y, Z or L*a*b*, with its illuminant)"; this is the plainest
    way to carry that illuminant alongside the same flat file rather than
    inventing a second sidecar format. ``illuminant_xy`` overrides whatever
    the file says, for a caller that already knows it.
    """
    import colour

    path = Path(path)
    rows_hint = cols_hint = None
    file_illuminant_xy = None
    data_lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if "=" in body:
                key, _, value = body.partition("=")
                key = key.strip().lower()
                if key == "rows":
                    rows_hint = int(value)
                elif key == "cols":
                    cols_hint = int(value)
                elif key == "illuminant":
                    ill2 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]
                    file_illuminant_xy = tuple(float(v) for v in ill2[value.strip()])
                elif key == "illuminant_xy":
                    x_str, y_str = value.split(",")
                    file_illuminant_xy = (float(x_str), float(y_str))
            continue
        data_lines.append(line)

    reader = csv.DictReader(data_lines)
    fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]
    if {"x", "y", "z"} <= set(fieldnames):
        mode = "xyz"
    elif {"l", "a", "b"} <= set(fieldnames):
        mode = "lab"
    else:
        raise ValueError(f"{path}: header must have columns name,X,Y,Z or name,L,a,b -- got {fieldnames}")

    resolved_illuminant_xy = illuminant_xy or file_illuminant_xy
    if resolved_illuminant_xy is None:
        raise ValueError(
            f"{path}: no illuminant given -- add a '# illuminant=D65' header line or pass illuminant_xy="
        )

    patches = []
    for row in reader:
        row = {k.strip().lower(): v for k, v in row.items()}
        name = row.get("name", f"patch{len(patches)}")
        if mode == "xyz":
            XYZ = (float(row["x"]), float(row["y"]), float(row["z"]))
        else:
            Lab = np.array([float(row["l"]), float(row["a"]), float(row["b"])])
            XYZ = tuple(float(v) for v in colour.Lab_to_XYZ(Lab, illuminant=resolved_illuminant_xy))
        Lab_check = colour.XYZ_to_Lab(np.array(XYZ), illuminant=resolved_illuminant_xy)
        chroma = float(np.hypot(Lab_check[1], Lab_check[2]))
        patches.append(ReferencePatch(name=name, XYZ=XYZ, is_neutral=chroma < _NEUTRAL_CHROMA_MAX))

    n = len(patches)
    rows = rows_hint or cc.DEFAULT_CHART_ROWS
    cols = cols_hint or (n // rows if rows else cc.DEFAULT_CHART_COLS)
    if rows * cols != n:
        raise ValueError(f"{path}: {n} patches does not fit a {rows}x{cols} grid -- set '# rows=' / '# cols='")

    return ReferenceChart(
        name=path.stem, rows=rows, cols=cols, patches=patches, illuminant_xy=resolved_illuminant_xy
    )


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchSample:
    name: str
    row: int
    col: int
    mean: dict  # {"R": float, "G1": float, "G2": float, "B": float} -- raw DN, NOT black-subtracted
    std: dict
    clipped_fraction: dict
    n_px: dict


@dataclass(frozen=True)
class ChartSample:
    rows: int
    cols: int
    patches: list  # list[PatchSample], row-major -- index i matches ReferenceChart.patches[i]
    corners: list  # the 4 (x, y) corners as given
    black_level: tuple  # frame.black_level, carried through so color.py can black-subtract
    white_level: float
    source_path: str = ""


def homography_from_corners(corners: list, rows: int, cols: int) -> np.ndarray:
    """4 (x, y) pixel corners (TL, TR, BR, BL patch *centers*) -> the 3x3
    homography mapping grid index (col, row) in [0, cols-1] x [0, rows-1] to
    that pixel space."""
    src = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]], dtype=np.float32)
    dst = np.array(corners, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``points``: (N, 2) grid coords -> (N, 2) pixel coords."""
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(pts, h)
    return out.reshape(-1, 2)


def _sample_region(plane: np.ndarray, quad_xy: np.ndarray, white_level: float) -> tuple:
    """Rasterize the convex quad ``quad_xy`` (pixel coords, this plane's own
    quarter-resolution frame) into ``plane`` and return
    ``(mean, std, clipped_fraction, n_px)``. Returns zeros/NaN-safe values
    (0 px) if the quad falls entirely outside the plane."""
    x0 = int(np.floor(quad_xy[:, 0].min()))
    x1 = int(np.ceil(quad_xy[:, 0].max())) + 1
    y0 = int(np.floor(quad_xy[:, 1].min()))
    y1 = int(np.ceil(quad_xy[:, 1].max())) + 1
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, plane.shape[1]), min(y1, plane.shape[0])
    if x1c <= x0c or y1c <= y0c:
        return 0.0, 0.0, 0.0, 0

    local_quad = (quad_xy - [x0c, y0c]).astype(np.int32)
    mask = np.zeros((y1c - y0c, x1c - x0c), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_quad, 1)
    mask_bool = mask.astype(bool)
    n_px = int(mask_bool.sum())
    if n_px == 0:
        return 0.0, 0.0, 0.0, 0

    values = plane[y0c:y1c, x0c:x1c][mask_bool]
    mean = float(values.mean())
    std = float(values.std())
    clipped = float(np.mean(values >= (white_level - 1.0)))
    return mean, std, clipped, n_px


def sample_chart(
    frame: rawmod.RawFrame,
    corners: list,
    *,
    rows: int = cc.DEFAULT_CHART_ROWS,
    cols: int = cc.DEFAULT_CHART_COLS,
    central_fraction: float = cc.PATCH_CENTRAL_FRACTION,
) -> ChartSample:
    """Sample every patch of an ``rows`` x ``cols`` chart from ``frame``,
    per CFA plane, using 4 pixel corners (see module docstring for the
    corner convention).
    """
    if len(corners) != 4:
        raise ValueError(f"expected 4 corners, got {len(corners)}")
    h = homography_from_corners(corners, rows, cols)
    planes = rawmod.planes(frame)
    row0, col0 = frame.visible[0].start, frame.visible[1].start
    half = central_fraction / 2.0

    grid_r, grid_c = np.mgrid[0:rows, 0:cols]
    centers_grid = np.stack([grid_c.ravel(), grid_r.ravel()], axis=1).astype(np.float64)

    patches = []
    for (gc, gr) in centers_grid:
        # Central-fraction box corners in grid-index space, mapped through
        # the same homography as patch centers -- this naturally accounts
        # for perspective (the box shrinks/rotates exactly like the patch
        # it sits inside).
        box_grid = np.array(
            [[gc - half, gr - half], [gc + half, gr - half], [gc + half, gr + half], [gc - half, gr + half]]
        )
        box_px = apply_homography(h, box_grid)  # full-sensor pixel (x, y)
        # Convert to each CFA plane's own quarter-resolution frame: shift by
        # the visible-area origin and halve. This ignores each plane's 0/1
        # phase offset within its 2x2 tile (a <1 quarter-pixel difference
        # between R/G1/G2/B, negligible next to a multi-pixel patch).
        box_plane = np.column_stack([(box_px[:, 0] - col0) / 2.0, (box_px[:, 1] - row0) / 2.0])

        mean, std, clipped, n_px = {}, {}, {}, {}
        for ch in CHANNELS:
            m, s, c, n = _sample_region(planes[ch], box_plane, frame.white_level)
            mean[ch], std[ch], clipped[ch], n_px[ch] = m, s, c, n

        patches.append(
            PatchSample(name="", row=int(gr), col=int(gc), mean=mean, std=std, clipped_fraction=clipped, n_px=n_px)
        )

    return ChartSample(
        rows=rows,
        cols=cols,
        patches=patches,
        corners=[tuple(float(v) for v in c) for c in corners],
        black_level=tuple(frame.black_level),
        white_level=float(frame.white_level),
        source_path=frame.path,
    )


def label_patches(chart_sample: ChartSample, reference: ReferenceChart) -> ChartSample:
    """Attach each ``ReferenceChart`` patch's name to the matching (by
    row-major index) ``ChartSample`` patch -- kept as a separate step from
    ``sample_chart`` because sampling needs no reference at all, only the
    grid geometry."""
    if reference.rows != chart_sample.rows or reference.cols != chart_sample.cols:
        raise ValueError(
            f"reference chart is {reference.rows}x{reference.cols}, sample is "
            f"{chart_sample.rows}x{chart_sample.cols}"
        )
    named = [
        PatchSample(
            name=reference.patches[i].name,
            row=p.row,
            col=p.col,
            mean=p.mean,
            std=p.std,
            clipped_fraction=p.clipped_fraction,
            n_px=p.n_px,
        )
        for i, p in enumerate(chart_sample.patches)
    ]
    return ChartSample(
        rows=chart_sample.rows,
        cols=chart_sample.cols,
        patches=named,
        corners=chart_sample.corners,
        black_level=chart_sample.black_level,
        white_level=chart_sample.white_level,
        source_path=chart_sample.source_path,
    )


# ---------------------------------------------------------------------------
# refusals (docs/design.md sec 3.2)
# ---------------------------------------------------------------------------


def refusals_for_fit(chart_sample: ChartSample, reference: ReferenceChart, model: str) -> list:
    """The four chart-quality refusal checks: glare, uneven lighting,
    clipping, and too few patches for the requested model order. Returns a
    list of ``fit.Refusal`` (empty if the chart passes all four)."""
    refusals = []
    n_patches = len(chart_sample.patches)
    needed = cc.min_patches(model)
    if n_patches < needed:
        refusals.append(
            Refusal(
                "too_few_patches",
                f"{n_patches} patches is below the minimum for model {model!r} "
                f"({cc.MIN_PATCHES_PER_TERM}x{cc.MODEL_TERMS[model]} terms)",
                value=n_patches,
                threshold=needed,
            )
        )

    neutral_idx = reference.neutral_indices()

    # -- glare: within-patch spatial non-uniformity on neutral patches -----
    worst_cv, worst_name = 0.0, None
    for i in neutral_idx:
        p = chart_sample.patches[i]
        for ch in CHANNELS:
            if p.n_px[ch] == 0 or p.mean[ch] <= 0:
                continue
            cv_ = p.std[ch] / p.mean[ch]
            if cv_ > worst_cv:
                worst_cv, worst_name = cv_, p.name or f"({p.row},{p.col})"
    if worst_cv > cc.GLARE_NEUTRAL_CV_MAX:
        refusals.append(
            Refusal(
                "glare",
                f"neutral patch {worst_name!r} has within-patch coefficient of variation "
                f"{worst_cv:.4f}, above the glare threshold",
                value=worst_cv,
                threshold=cc.GLARE_NEUTRAL_CV_MAX,
            )
        )

    # -- uneven lighting: neutral-patch response vs. reference vs. position -
    if len(neutral_idx) >= 3:
        # frame.black_level's 4 entries are positional (one per raw-pattern
        # tile slot), not named "R"/"G1"/"G2"/"B" -- raw.py exposes no
        # per-name mapping for it, so a plain mean across all 4 is used as a
        # single scalar black estimate here. Good enough for a *ratio*-based
        # gradient check (a few DN of black-level error is negligible next
        # to a mid-gray patch's signal), not precise enough for the actual
        # matrix fit, which black-subtracts per plane properly (color.py).
        black = float(np.mean(chart_sample.black_level))
        rows_, cols_, ratios = [], [], []
        for i in neutral_idx:
            p = chart_sample.patches[i]
            ref_y = reference.patches[i].XYZ[1]
            g_mean = 0.5 * (p.mean["G1"] + p.mean["G2"]) - black
            if ref_y <= 0 or g_mean <= 0:
                continue
            rows_.append(p.row)
            cols_.append(p.col)
            ratios.append(g_mean / ref_y)
        if len(ratios) >= 3:
            A = np.column_stack([rows_, cols_, np.ones(len(ratios))])
            coeffs, *_ = np.linalg.lstsq(A, ratios, rcond=None)
            predicted = A @ coeffs
            mean_ratio = float(np.mean(ratios))
            gradient = float((predicted.max() - predicted.min()) / mean_ratio) if mean_ratio > 0 else 0.0
            if gradient > cc.UNEVEN_LIGHTING_GRADIENT_MAX:
                refusals.append(
                    Refusal(
                        "uneven_lighting",
                        f"neutral-patch response vs. reference varies {gradient:.3f} "
                        "(peak-to-peak / mean) across the chart's fitted position trend",
                        value=gradient,
                        threshold=cc.UNEVEN_LIGHTING_GRADIENT_MAX,
                    )
                )

    # -- clipping: any channel of any patch -------------------------------
    worst_clip, worst_clip_name = 0.0, None
    for p in chart_sample.patches:
        for ch in CHANNELS:
            if p.clipped_fraction[ch] > worst_clip:
                worst_clip, worst_clip_name = p.clipped_fraction[ch], p.name or f"({p.row},{p.col})"
    if worst_clip > cc.CLIPPED_FRACTION_MAX:
        refusals.append(
            Refusal(
                "clipping",
                f"patch {worst_clip_name!r} has {worst_clip:.4%} of pixels at or near white level",
                value=worst_clip,
                threshold=cc.CLIPPED_FRACTION_MAX,
            )
        )

    return refusals

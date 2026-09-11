"""Synthetic chart renderer: paints a ``chart.ReferenceChart``'s patches into
a synthetic Bayer ``raw.RawFrame``, from either a known raw->XYZ matrix or
known SSFs + an illuminant + spectral reflectances -- the round-trip fixture
every ``camera/{chart,color,ssf}.py`` test builds on (house rule 4).

Reuses ``synth/sensor.py``'s ``SensorModel``/``frame`` for all of the actual
noise physics (shot noise, read noise, PRNU/DSNU, hot pixels, clipping) --
this module's only job is to turn "which patch is at which pixel" plus a
target raw RGB per patch into the single per-pixel ``flux_e_per_s`` array
``sensor.frame`` expects, then call it once. Knobs for glare, an uneven-
lighting gradient, forced clipping and chart perspective (via 4 corners)
let every refusal in ``chart.refusals_for_fit`` be triggered on demand from
otherwise-clean synthetic data (house rule 4's "assert it comes back", and
its mirror image -- assert a bad one is refused).
"""

from __future__ import annotations

import cv2
import numpy as np

from calsuite.camera import chart
from calsuite.raw import RawFrame
from calsuite.synth import sensor as synth_sensor


def plane_positions(pattern: str) -> dict:
    """Which (row-phase, col-phase) in a CFA pattern's 2x2 tile is R / G1 /
    G2 / B. A small local copy of ``raw._plane_positions``'s convention
    (first 'G' in raster order is G1, second is G2) rather than importing
    that underscore-prefixed foundation helper -- this is the one place in
    the color-wave code that needs to go the *opposite* direction from
    ``raw.planes()`` (channel name -> array), so it needs the same mapping,
    but re-deriving 6 lines locally beats reaching into another module's
    private API for it."""
    if len(pattern) != 4:
        raise ValueError(f"expected a 2x2 (4-char) CFA pattern, got {pattern!r}")
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    names, g_count = {}, 0
    for pos, ch in zip(positions, pattern, strict=True):
        if ch == "G":
            g_count += 1
            names[pos] = f"G{g_count}"
        else:
            names[pos] = ch
    return names


def _paint_patch_canvases(
    rows: int,
    cols: int,
    corners: list,
    canvas_shape: tuple,
    patch_rgb: np.ndarray,
    *,
    glare_indices: set,
    glare_extra_fraction: float,
    lighting_gradient_fraction: float,
    clip_indices: set,
    clip_flux: float,
) -> tuple:
    """Fill 3 ``canvas_shape``-shaped canvases (R, G, B target flux, e-/s)
    by rasterizing each patch's full grid cell (grid offsets +/- 0.5 around its
    center, so adjacent patches tile the card edge-to-edge with no gutter --
    simpler than modeling a gutter, and irrelevant to what's being tested:
    the *central* 50% ``chart.sample_chart`` actually reads stays well
    inside a full cell regardless)."""
    h = chart.homography_from_corners(corners, rows, cols)
    canvases = [np.zeros(canvas_shape, dtype=np.float64) for _ in range(3)]

    grid_r, grid_c = np.mgrid[0:rows, 0:cols]
    idx = 0
    for gr, gc in zip(grid_r.ravel(), grid_c.ravel(), strict=True):
        cell_grid = np.array(
            [[gc - 0.5, gr - 0.5], [gc + 0.5, gr - 0.5], [gc + 0.5, gr + 0.5], [gc - 0.5, gr + 0.5]]
        )
        cell_px = chart.apply_homography(h, cell_grid)  # (x, y)

        rgb = patch_rgb[idx].copy()
        if lighting_gradient_fraction:
            # A linear gradient across chart *columns*, peak-to-peak equal
            # to the given fraction of the mean signal -- "uneven lighting"
            # per docs/design.md, applied as a per-patch flux multiplier so
            # it shows up in the sampled means exactly the way a real
            # off-axis light source would.
            t = (gc / max(cols - 1, 1)) - 0.5  # -0.5 .. 0.5 across the chart
            rgb = rgb * (1.0 + lighting_gradient_fraction * t)
        if idx in clip_indices:
            rgb = np.full_like(rgb, clip_flux)

        for c in range(3):
            _fill_poly(canvases[c], cell_px, float(rgb[c]))

        if idx in glare_indices:
            # A small bright hot spot in the patch's center -- a specular
            # reflection covers only part of a real patch, so this paints a
            # smaller (half linear size = 1/4 area) sub-cell at
            # (1 + glare_extra_fraction)x the patch's own flux on top of the
            # already-painted cell, which is exactly the kind of localized
            # spatial non-uniformity chart.refusals_for_fit's glare check
            # (within-patch coefficient of variation) is built to catch.
            hotspot_grid = np.array(
                [[gc - 0.25, gr - 0.25], [gc + 0.25, gr - 0.25], [gc + 0.25, gr + 0.25], [gc - 0.25, gr + 0.25]]
            )
            hotspot_px = chart.apply_homography(h, hotspot_grid)
            for c in range(3):
                _fill_poly(canvases[c], hotspot_px, float(rgb[c]) * (1.0 + glare_extra_fraction))
        idx += 1

    return tuple(canvases)


def _fill_poly(canvas: np.ndarray, quad_xy: np.ndarray, value: float) -> None:
    poly = np.round(quad_xy).astype(np.int32)
    cv2.fillConvexPoly(canvas, poly, float(value))


def _merge_channels(model: synth_sensor.SensorModel, canvas_r, canvas_g, canvas_b) -> np.ndarray:
    """Combine the 3 per-channel flux canvases (already sliced down to the
    *visible* region, matching ``model.shape``) into the single per-pixel
    ``flux_e_per_s`` array ``synth.sensor.frame`` expects, using
    ``plane_positions`` to pick, at each of the 4 phases in the CFA tile,
    the canvas matching that phase's actual channel (both G phases read
    from the same green canvas -- this generator has no separate G1/G2
    scene content, matching ``color.black_subtracted_rgb``'s G1==G2
    assumption)."""
    flux = np.empty(model.shape, dtype=np.float64)
    for (dr, dc), name in plane_positions(model.pattern).items():
        source = canvas_g if name.startswith("G") else (canvas_r if name == "R" else canvas_b)
        flux[dr::2, dc::2] = source[dr::2, dc::2]
    return flux


def _patch_rgb_from_matrix(reference: chart.ReferenceChart, matrix_raw_to_xyz: np.ndarray, signal_scale: float) -> np.ndarray:
    inv_m = np.linalg.inv(np.asarray(matrix_raw_to_xyz, dtype=np.float64))
    rgb = np.array([inv_m @ np.asarray(p.XYZ, dtype=np.float64) for p in reference.patches])
    return np.clip(rgb, 0.0, None) * signal_scale


def _patch_rgb_from_ssf(reference: chart.ReferenceChart, reflectances: dict, ssf_obj, illuminant_sd, signal_scale: float) -> np.ndarray:
    from calsuite.camera.ssf import camera_response

    rgb = np.array([camera_response(ssf_obj, illuminant_sd, reflectances[p.name]) for p in reference.patches])
    return np.clip(rgb, 0.0, None) * signal_scale


def _render(
    reference: chart.ReferenceChart,
    patch_rgb: np.ndarray,
    corners: list,
    *,
    sensor: synth_sensor.SensorModel | None,
    exposure_s: float,
    temp_c: float,
    rng: np.random.Generator,
    glare_patch_names,
    glare_extra_fraction: float,
    lighting_gradient_fraction: float,
    clip_patch_names,
) -> RawFrame:
    model = sensor or synth_sensor.SensorModel(shape=(480, 720))
    name_to_idx = {p.name: i for i, p in enumerate(reference.patches)}
    glare_indices = {name_to_idx[n] for n in (glare_patch_names or []) if n in name_to_idx}
    clip_indices = {name_to_idx[n] for n in (clip_patch_names or []) if n in name_to_idx}
    clip_flux = model.white_level * model.gain_e_per_dn * 50.0  # comfortably beyond full well at any sane exposure

    # ``corners`` are full-sensor px (chart.py's documented convention, so
    # the exact same corners list can be handed to both this renderer and
    # chart.sample_chart on the resulting frame) -- paint at the *full*
    # cfa's resolution (visible area + margins) so the homography built
    # from those corners lines up, then slice down to the visible window
    # before handing flux to synth.sensor.frame, which expects an array
    # shaped exactly like the visible area alone.
    top, left = model.top_margin, model.left_margin
    full_shape = (model.shape[0] + top, model.shape[1] + left)
    canvas_r, canvas_g, canvas_b = _paint_patch_canvases(
        reference.rows,
        reference.cols,
        corners,
        full_shape,
        patch_rgb,
        glare_indices=glare_indices,
        glare_extra_fraction=glare_extra_fraction,
        lighting_gradient_fraction=lighting_gradient_fraction,
        clip_indices=clip_indices,
        clip_flux=clip_flux,
    )
    visible = (slice(top, top + model.shape[0]), slice(left, left + model.shape[1]))
    flux = _merge_channels(model, canvas_r[visible], canvas_g[visible], canvas_b[visible])
    return synth_sensor.frame(model, exposure_s, flux, temp_c, rng)


def render_chart(
    reference: chart.ReferenceChart,
    matrix_raw_to_xyz: np.ndarray,
    corners: list,
    *,
    sensor: synth_sensor.SensorModel | None = None,
    exposure_s: float = 1.0,
    temp_c: float = 20.0,
    rng: np.random.Generator,
    signal_scale: float = 1.0,
    glare_patch_names: list | None = None,
    glare_extra_fraction: float = 0.15,
    lighting_gradient_fraction: float = 0.0,
    clip_patch_names: list | None = None,
) -> RawFrame:
    """Render ``reference`` from a **known** raw->XYZ matrix (the Tier A
    round trip: paint a chart with matrix M, refit, recover M). ``corners``
    are 4 full-sensor-px patch-center coordinates, the same convention
    ``chart.sample_chart`` documents -- pass the identical ``corners`` list
    to both this function and ``chart.sample_chart`` on the frame it
    returns. ``signal_scale`` converts the matrix's raw units into electrons/second;
    pick it so the brightest patch sits safely under the sensor's full well
    unless ``clip_patch_names`` deliberately pushes some patches over it.
    """
    rgb = _patch_rgb_from_matrix(reference, matrix_raw_to_xyz, signal_scale)
    return _render(
        reference,
        rgb,
        corners,
        sensor=sensor,
        exposure_s=exposure_s,
        temp_c=temp_c,
        rng=rng,
        glare_patch_names=glare_patch_names,
        glare_extra_fraction=glare_extra_fraction,
        lighting_gradient_fraction=lighting_gradient_fraction,
        clip_patch_names=clip_patch_names,
    )


def render_chart_from_ssf(
    reference: chart.ReferenceChart,
    reflectances: dict,
    ssf_obj,
    illuminant_sd,
    corners: list,
    *,
    sensor: synth_sensor.SensorModel | None = None,
    exposure_s: float = 1.0,
    temp_c: float = 20.0,
    rng: np.random.Generator,
    signal_scale: float = 1.0,
    glare_patch_names: list | None = None,
    glare_extra_fraction: float = 0.15,
    lighting_gradient_fraction: float = 0.0,
    clip_patch_names: list | None = None,
) -> RawFrame:
    """Render ``reference`` from known SSFs + illuminant + spectral
    reflectances (Tier B's round trip: no assumed matrix at all, just the
    same spectral integration ``camera/ssf.py`` uses to build its own
    training data). ``reflectances`` maps each reference patch's ``name`` to
    a ``colour.SpectralDistribution`` -- e.g. ``colour.SDS_COLOURCHECKERS
    ["ISO 17321-1"]``, matched against ``reference``'s own patch names.
    """
    rgb = _patch_rgb_from_ssf(reference, reflectances, ssf_obj, illuminant_sd, signal_scale)
    return _render(
        reference,
        rgb,
        corners,
        sensor=sensor,
        exposure_s=exposure_s,
        temp_c=temp_c,
        rng=rng,
        glare_patch_names=glare_patch_names,
        glare_extra_fraction=glare_extra_fraction,
        lighting_gradient_fraction=lighting_gradient_fraction,
        clip_patch_names=clip_patch_names,
    )

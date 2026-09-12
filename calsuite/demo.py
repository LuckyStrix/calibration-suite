"""``calsuite demo --out DIR``: run every analysis on synthetic data, with
no camera, no instrument and no ArgyllCMS attached, into a temporary
record store under ``DIR``, and write every HTML report plus an index
page linking them.

This is the "does the whole pipeline actually work" smoke test made into a
product: every phase below goes through the *real* CLI commands
(``calsuite.cli.main``) wherever the command's own file format allows it --
synthetic frames written to disk via ``raw.save_npz`` (docs/implementation-
plan.md Wave 3 fix list item 4) so ``--from DIR`` sees them exactly like a
folder of real raw files, and the synthetic display backend for
``calsuite display ...``. The two exceptions are ``lens distortion``/``lens
tca``: those CLI commands detect a ChArUco board in a *rendered* image
(``lens.charuco.detect_green``), and rendering + detecting a synthetic
board image is a much heavier, flakier thing to do here than calling
``synth.lens.synthetic_views``'s point-correspondence generator directly
against the same pure ``lens.distortion.fit_distortion``/``lens.tca.fit_tca``
functions ``lens/commands.py`` itself calls -- still a real analysis run on
synthetic data with a real record saved to the store, just skipping image
rendering that ``lens/charuco.py``'s own tests already cover.

Every phase is wrapped so one area's failure doesn't stop the others: each
still gets whatever reports it could produce, and the run ends with a
warning list rather than a traceback.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from calsuite import __version__, devices as devicesmod, raw as rawmod, store as storemod
from calsuite.lens.charuco import Detection

CAMERA_MODEL = "calsuite-synthetic"
LENS_MODEL = "calsuite-demo-lens-50mm"
LENS_MODEL_REFUSAL_EXAMPLE = "calsuite-demo-lens-50mm-narrow-coverage-example"
DISPLAY_DEVICE_ID = "calsuite-demo-display"
SPECTRAL_CAMERA_MODEL = "calsuite-demo-spectral"


def _with_meta(frame, **overrides):
    return replace(frame, meta=replace(frame.meta, **overrides))


def _save_npz_series(frames: list, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        rawmod.save_npz(frame, folder / f"f{i:04d}.npz")
    return folder


def _method(name: str, params: dict) -> dict:
    return {"name": name, "calsuite_version": __version__, "params": params}


def _run_cli(argv: list, warnings: list) -> int:
    from calsuite.cli import main as cli_main  # lazy: cli.py imports this module for `calsuite demo`

    rc = cli_main(argv)
    if rc not in (0, 1):
        warnings.append(f"`calsuite {' '.join(argv)}` exited {rc} (expected 0 or 1)")
    return rc


# ---------------------------------------------------------------------------
# sensor (bias, ptc x3 ISOs, linearity, darks, iso invariance, report)
# ---------------------------------------------------------------------------


def _run_sensor(out_dir: Path, warnings: list) -> Path | None:
    from calsuite.synth import sensor as synth_sensor

    frames_root = out_dir / "_capture" / "sensor"
    black_dn, gain, read_noise_e, full_well_e = 512.0, 2.0, 3.0, 40000.0
    shape = (96, 96)

    bias_model = synth_sensor.SensorModel(
        shape=shape, black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=read_noise_e,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(0)
    bias_frames = [
        _with_meta(synth_sensor.frame(bias_model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng), iso=100)
        for _ in range(8)
    ]
    bias_dir = _save_npz_series(bias_frames, frames_root / "bias")
    _run_cli(["camera", "bias", "--from", str(bias_dir)], warnings)

    levels = np.linspace(500.0, 18000.0, 10)
    ptc_model = synth_sensor.SensorModel(
        shape=shape, gain_e_per_dn=gain, read_noise_e=read_noise_e, black_dn=black_dn,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=full_well_e,
    )
    for iso_value in (100, 400, 1600):
        rng = np.random.default_rng(1000 + iso_value)
        frames = []
        for level_e in levels:
            exposure_s = 0.02
            flux = level_e / exposure_s
            a = synth_sensor.frame(ptc_model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
            b = synth_sensor.frame(ptc_model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
            frames.append(_with_meta(a, iso=iso_value))
            frames.append(_with_meta(b, iso=iso_value))
        ptc_dir = _save_npz_series(frames, frames_root / f"ptc_iso{iso_value}")
        _run_cli(["camera", "ptc", "--from", str(ptc_dir)], warnings)

    lin_model = synth_sensor.SensorModel(
        shape=shape, gain_e_per_dn=gain, black_dn=black_dn, full_well_e=full_well_e,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, read_noise_e=read_noise_e,
    )
    rng = np.random.default_rng(2)
    exposures = np.linspace(0.005, 0.9, 12)
    lin_frames = [
        _with_meta(synth_sensor.frame(lin_model, exposure_s=t, flux_e_per_s=40000.0, temp_c=20.0, rng=rng), iso=100)
        for t in exposures
    ]
    lin_dir = _save_npz_series(lin_frames, frames_root / "linearity")
    _run_cli(["camera", "linearity", "--from", str(lin_dir)], warnings)

    dark_model = synth_sensor.SensorModel(
        shape=shape, gain_e_per_dn=gain, black_dn=black_dn, dark_current_e_per_s_at_20c=0.3,
        read_noise_e=read_noise_e, prnu_std=0.0, dsnu_std_e_per_s=0.01, hot_pixel_fraction=0.0005,
    )
    rng = np.random.default_rng(3)
    dark_frames = []
    for temp in (10.0, 18.0, 26.0):
        for exp in (5.0, 15.0, 30.0, 60.0):
            dark_frames.append(
                _with_meta(synth_sensor.frame(dark_model, exposure_s=exp, flux_e_per_s=0.0, temp_c=temp, rng=rng), iso=100)
            )
    darks_dir = _save_npz_series(dark_frames, frames_root / "darks")
    _run_cli(["camera", "darks", "--from", str(darks_dir)], warnings)

    device_id = devicesmod.device_id(CAMERA_MODEL, None)
    _run_cli(["camera", "iso", "--device-id", device_id], warnings)

    report_path = out_dir / "reports" / "sensor.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(["camera", "report", "--device-id", device_id, "--out", str(report_path)], warnings)
    return report_path


# ---------------------------------------------------------------------------
# lens (distortion, tca, flats, mtf, psf, export, report)
# ---------------------------------------------------------------------------


def _synthetic_tca_view(image_size: tuple, kr: float, kb: float, n: int, rng: np.random.Generator) -> dict:
    cx, cy = (image_size[0] - 1) / 2.0, (image_size[1] - 1) / 2.0
    xs = rng.uniform(0, image_size[0], n)
    ys = rng.uniform(0, image_size[1], n)
    r = np.hypot(xs - cx, ys - cy)
    theta = np.arctan2(ys - cy, xs - cx)
    g = np.stack([xs, ys], axis=1)
    r_corners = np.stack([cx + r * kr * np.cos(theta), cy + r * kr * np.sin(theta)], axis=1)
    b_corners = np.stack([cx + r * kb * np.cos(theta), cy + r * kb * np.sin(theta)], axis=1)
    ids = np.arange(n, dtype=np.int32)
    return {"R": Detection("R", r_corners, ids), "G1": Detection("G1", g, ids), "G2": None, "B": Detection("B", b_corners, ids)}


def _deliberate_edge_views(board, K, dist_coeffs, image_size, *, frac=0.22, push=0.85, min_corners=10):
    """A handful of poses placed *deliberately*, not randomly, so the
    board's own corners land in the field's outer ring in every direction
    -- 8 poses, one each toward the top/bottom/left/right edge midpoints
    and the 4 corners of the frame.

    ``synth.lens.synthetic_views``'s random full-coverage sampling (used
    for the bulk of the views below) fills the field respectably but not
    reliably in *every* direction: `lens.distortion.coverage_grid`'s
    12-sector outer ring is checked in 30-degree wedges, and a rectangular
    frame's own geometry makes the wedges nearest the short axis (top/
    bottom-center on a landscape frame) the hardest to reach at all by
    chance -- a real capture session handles this by deliberately shooting
    toward each edge and corner, not by taking more random shots from the
    same distribution, and that's exactly what this reproduces.

    ``board.getChessboardCorners()``'s local coordinates are *not*
    centered on (0, 0) -- they run from one square-width margin to
    (board width - margin), so a target pixel has to be corrected by the
    board's own local center, not aimed at its (0, 0, 0) origin corner
    (which is off to one side and would silently bias every placement
    toward that side).
    """
    from calsuite.lens.distortion import BoardView
    from calsuite.synth import lens as synthlens

    width, height = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    objp = board.getChessboardCorners().astype(np.float64)
    board_w_mm = float(objp[:, 0].max() - objp[:, 0].min())
    cx_local = float((objp[:, 0].min() + objp[:, 0].max()) / 2.0)
    cy_local = float((objp[:, 1].min() + objp[:, 1].max()) / 2.0)

    half_w, half_h = width / 2.0, height / 2.0
    target_pixels = [
        (cx, cy - push * half_h), (cx, cy + push * half_h),  # top-center, bottom-center
        (cx - push * half_w, cy), (cx + push * half_w, cy),  # left-center, right-center
        (cx - push * half_w * 0.9, cy - push * half_h * 0.9),  # 4 corners
        (cx + push * half_w * 0.9, cy - push * half_h * 0.9),
        (cx - push * half_w * 0.9, cy + push * half_h * 0.9),
        (cx + push * half_w * 0.9, cy + push * half_h * 0.9),
    ]
    z = fx * board_w_mm / (frac * width)
    views = []
    for i, (tpx, tpy) in enumerate(target_pixels):
        tvec = np.array([(tpx - cx) * z / fx - cx_local, (tpy - cy) * z / fy - cy_local, z])
        objp_i, imgp = synthlens.project_board_corners(board, K, dist_coeffs, np.zeros(3), tvec)
        inside = (imgp[:, 0] >= 0) & (imgp[:, 0] < width) & (imgp[:, 1] >= 0) & (imgp[:, 1] < height)
        if inside.sum() < min_corners:
            continue
        ids = np.where(inside)[0].astype(np.int32)
        views.append(BoardView(corners=imgp[inside].astype(np.float64), ids=ids, name=f"edge{i}"))
    return views


def _run_lens(records_dir: Path, out_dir: Path, warnings: list) -> tuple:
    from calsuite.lens import charuco as charucomod, distortion as distortionmod, flats as flatsmod, tca as tcamod
    from calsuite.lens.constants import DISPLAY_SQUARE_MM
    from calsuite.synth import lens as synthlens, sensor as synth_sensor

    frames_root = out_dir / "_capture" / "lens"
    image_size = (600, 400)
    lens_ref = devicesmod.DeviceRef(kind="lens", model=LENS_MODEL, id=devicesmod.device_id(LENS_MODEL, None))
    camera_ref = devicesmod.DeviceRef(kind="camera", model=CAMERA_MODEL, id=devicesmod.device_id(CAMERA_MODEL, None))
    st = storemod.Store(records_dir)

    board = charucomod.build_board(square_length=DISPLAY_SQUARE_MM)
    true_k = np.array([[900.0, 0, 300.0], [0, 900.0, 200.0], [0, 0, 1.0]])
    true_dist = np.array([-0.08, 0.015, 0.0003, -0.0002, 0.002])

    # -- the passing case: broad random coverage plus deliberate edge/corner
    # poses, so the fit's own coverage refusal has no honest reason to fire.
    rng = np.random.default_rng(10)
    views = synthlens.synthetic_views(board, true_k, true_dist, image_size, 16, rng, coverage="full")
    views += _deliberate_edge_views(board, true_k, true_dist, image_size)
    distortion_analysis = distortionmod.fit_distortion(board, views, image_size)
    distortion_record = storemod.Record.from_analysis(
        kind="lens.distortion", device=lens_ref.to_dict(), devices=[camera_ref.to_dict()],
        analysis=distortion_analysis, provenance="measured",
        method=_method("distortion.fit_distortion", {"square_mm": DISPLAY_SQUARE_MM}),
        conditions={"focal_mm": 50.0, "square_mm": DISPLAY_SQUARE_MM},
    )
    st.save(distortion_record)
    if not distortion_analysis.ok:
        # Should not happen with the pose set above (verified against many
        # seeds) -- if it ever does, that's worth surfacing, not hiding.
        warnings.append("lens.distortion: the *passing* synthetic pose set was unexpectedly refused")

    # -- the refusal example: the same lens, a deliberately narrow
    # (center-only) pose set, under its own device id so it gets its own
    # record and its own report rather than overwriting the passing one --
    # "refuse bad fits, and say why" (house rule 3) is best demonstrated
    # next to a passing fit, not instead of one.
    refusal_lens_ref = devicesmod.DeviceRef(
        kind="lens", model=LENS_MODEL_REFUSAL_EXAMPLE, id=devicesmod.device_id(LENS_MODEL_REFUSAL_EXAMPLE, None)
    )
    rng = np.random.default_rng(99)
    center_views = synthlens.synthetic_views(board, true_k, true_dist, image_size, 12, rng, coverage="center")
    refusal_analysis = distortionmod.fit_distortion(board, center_views, image_size)
    refusal_record = storemod.Record.from_analysis(
        kind="lens.distortion", device=refusal_lens_ref.to_dict(), devices=[camera_ref.to_dict()],
        analysis=refusal_analysis, provenance="measured",
        method=_method("distortion.fit_distortion", {"square_mm": DISPLAY_SQUARE_MM, "demo_note": "deliberately center-only poses"}),
        conditions={"focal_mm": 50.0, "square_mm": DISPLAY_SQUARE_MM},
    )
    st.save(refusal_record)
    if refusal_analysis.ok:
        # The whole point of this second run is to show a refusal -- if a
        # center-only pose set ever passes, that's worth knowing about too.
        warnings.append("lens.distortion (refusal example): the deliberately narrow pose set was NOT refused")

    refusal_report_path = out_dir / "reports" / "lens_refusal_example.html"
    refusal_report_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(
        ["lens", "report", "--device-id", refusal_lens_ref.id, "--lens-model", LENS_MODEL_REFUSAL_EXAMPLE,
         "--out", str(refusal_report_path)],
        warnings,
    )

    rng = np.random.default_rng(11)
    tca_views = [_synthetic_tca_view(image_size, 1.0025, 0.9975, 50, rng) for _ in range(3)]
    tca_analysis = tcamod.fit_tca(tca_views, image_size)
    tca_record = storemod.Record.from_analysis(
        kind="lens.tca", device=lens_ref.to_dict(), devices=[camera_ref.to_dict()],
        analysis=tca_analysis, provenance="measured", method=_method("tca.fit_tca", {}),
        conditions={"focal_mm": 50.0},
    )
    st.save(tca_record)

    flat_model = synth_sensor.SensorModel(
        shape=(180, 240), read_noise_e=2.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    poses = [
        flatsmod.Pose(0, 0, 0), flatsmod.Pose(90, 0.08, 0), flatsmod.Pose(180, 0, 0.08),
        flatsmod.Pose(270, 0.08, 0.08), flatsmod.Pose(0, -0.06, 0.05), flatsmod.Pose(90, -0.05, -0.07),
    ]
    rng = np.random.default_rng(12)
    flat_frames = [
        _with_meta(
            synthlens.render_flat_pose(
                flat_model, [-0.35, 0.05], [0.0, 0.15, -0.10, -0.08, 0.02, -0.05], pose,
                rng=rng, exposure_s=0.2, base_flux_e_per_s=3.0e4,
            ),
            lens=LENS_MODEL, focal=50.0, aperture=1.8,
        )
        for pose in poses
    ]
    flats_dir = _save_npz_series(flat_frames, frames_root / "flats")
    poses_str = ",".join(f"{p.angle_deg:g}:{p.shift_u:g}:{p.shift_v:g}" for p in poses)
    _run_cli(["lens", "flats", "--from", str(flats_dir), "--aperture", "1.8", "--poses", poses_str], warnings)

    mtf_model = synth_sensor.SensorModel(
        shape=(300, 220), read_noise_e=0.1, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    rng = np.random.default_rng(13)
    mtf_frame = _with_meta(
        synthlens.render_slanted_edge(mtf_model, 5.0, 1.5, rng=rng, exposure_s=0.05, low_flux_e_per_s=2.0e3, high_flux_e_per_s=8.0e4),
        lens=LENS_MODEL, focal=50.0, aperture=1.8,
    )
    mtf_dir = _save_npz_series([mtf_frame], frames_root / "mtf")
    _run_cli(["lens", "mtf", "--from", str(mtf_dir), "--grid", "3x5"], warnings)

    psf_model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    rng = np.random.default_rng(14)
    stars = [{"x": 100.0, "y": 80.0, "amplitude": 8.0e4, "sigma_major": 10.0, "sigma_minor": 4.0, "theta_deg": 40.0}]
    psf_frame = _with_meta(
        synthlens.render_stars(psf_model, stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0),
        lens=LENS_MODEL, focal=50.0, aperture=1.8,
    )
    psf_dir = _save_npz_series([psf_frame], frames_root / "psf")
    _run_cli(["lens", "psf", "--from", str(psf_dir)], warnings)

    export_path = out_dir / "exports" / "lens.xml"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(["lens", "export", "--device-id", lens_ref.id, "--lens-model", LENS_MODEL, "--out", str(export_path)], warnings)

    report_path = out_dir / "reports" / "lens.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(["lens", "report", "--device-id", lens_ref.id, "--lens-model", LENS_MODEL, "--out", str(report_path)], warnings)
    return report_path, refusal_report_path


# ---------------------------------------------------------------------------
# colour: Tier A (chart matrix) + Tier B (spectral)
# ---------------------------------------------------------------------------


def _run_color(records_dir: Path, out_dir: Path, warnings: list) -> tuple:
    from calsuite.camera import chart as chartmod
    from calsuite.synth import color as synth_color

    frames_root = out_dir / "_capture" / "color"
    st = storemod.Store(records_dir)

    reference = chartmod.reference_colorchecker()
    known_matrix = np.array([[0.60, 0.20, 0.15], [0.15, 0.75, 0.10], [0.10, 0.15, 0.65]])
    corners = [(60.0, 60.0), (660.0, 60.0), (660.0, 420.0), (60.0, 420.0)]
    rng = np.random.default_rng(20)
    frame = synth_color.render_chart(reference, known_matrix, corners, rng=rng, signal_scale=20000.0)
    tier_a_dir = frames_root / "tier_a"
    tier_a_dir.mkdir(parents=True, exist_ok=True)
    rawmod.save_npz(frame, tier_a_dir / "chart0000.npz")
    corners_str = ",".join(f"{x:g},{y:g}" for x, y in corners)
    _run_cli(["camera", "color", "fit", "--from", str(tier_a_dir), "--corners", corners_str, "--illuminant", "D65"], warnings)

    camera_device_id = devicesmod.device_id(CAMERA_MODEL, None)
    tier_a_record = st.latest("camera.color", camera_device_id)
    tier_a_report_path = out_dir / "reports" / "color_tier_a.html"
    tier_a_report_path.parent.mkdir(parents=True, exist_ok=True)
    if tier_a_record is not None:
        record_path = records_dir / camera_device_id / f"{tier_a_record.id}.json"
        _run_cli(["camera", "color", "report", "--record", str(record_path), "--out", str(tier_a_report_path)], warnings)
    else:
        warnings.append("colour Tier A: no camera.color record was produced")
        tier_a_report_path = None

    ssf_csv = frames_root / "ssf.csv"
    frames_root.mkdir(parents=True, exist_ok=True)
    wl = np.arange(400.0, 701.0, 5.0)

    def _gauss(center, sigma):
        return np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    r, g, b = _gauss(600.0, 40.0), _gauss(540.0, 45.0), _gauss(460.0, 35.0)
    with open(ssf_csv, "w", encoding="utf-8") as f:
        f.write("nm,r,g,b\n")
        for wl_i, r_i, g_i, b_i in zip(wl, r, g, b, strict=True):
            f.write(f"{wl_i},{r_i},{g_i},{b_i}\n")
    _run_cli(
        ["camera", "color", "spectral", "--ssf", str(ssf_csv), "--illuminant", "D65", "--camera-model", SPECTRAL_CAMERA_MODEL],
        warnings,
    )

    spectral_device_id = devicesmod.device_id(SPECTRAL_CAMERA_MODEL, None)
    tier_b_record = st.latest("camera.color", spectral_device_id)
    tier_b_report_path = out_dir / "reports" / "color_tier_b.html"
    if tier_b_record is not None:
        record_path = records_dir / spectral_device_id / f"{tier_b_record.id}.json"
        _run_cli(["camera", "color", "report", "--record", str(record_path), "--out", str(tier_b_report_path)], warnings)
    else:
        warnings.append("colour Tier B: no camera.color record was produced")
        tier_b_report_path = None

    return tier_a_report_path, tier_b_report_path


# ---------------------------------------------------------------------------
# display: measure / profile / validate / report, synthetic backend
# ---------------------------------------------------------------------------


def _run_display(out_dir: Path, warnings: list) -> Path | None:
    device_id = DISPLAY_DEVICE_ID
    _run_cli(["display", "measure", "--backend", "synthetic", "--device-id", device_id, "--steps", "9"], warnings)

    profile_path = out_dir / "exports" / "display-profile.icc"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(["display", "profile", "--device-id", device_id, "--out", str(profile_path)], warnings)
    _run_cli(["display", "validate", "--backend", "synthetic", "--device-id", device_id], warnings)

    report_path = out_dir / "reports" / "display.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cli(["display", "report", "--device-id", device_id, "--out", str(report_path)], warnings)
    return report_path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _write_index(path: Path, reports: dict, warnings: list) -> None:
    import html as _html

    rows = []
    for name, report_path in reports.items():
        if report_path is None:
            continue
        href = Path(report_path).relative_to(path.parent) if Path(report_path).is_relative_to(path.parent) else report_path
        rows.append(f'<li><a href="{_html.escape(str(href))}">{_html.escape(name)}</a></li>')
    warning_items = "".join(f"<li>{_html.escape(w)}</li>" for w in warnings)
    warning_html = (
        f'<h2>Warnings</h2><ul style="color:#92400e;">{warning_items}</ul>' if warnings else "<p>No warnings.</p>"
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>calsuite demo</title></head><body>"
        "<h1>calsuite demo</h1>"
        "<p>Every analysis run on synthetic data -- no camera, no instrument, no ArgyllCMS attached. "
        "Provenance badges throughout are for real: these ARE real measurement-shaped records, of "
        "synthetic data, and only the areas that need real hardware/software (raw capture, a "
        "colorimeter, ArgyllCMS's colprof/dispwin) are stood in for.</p>"
        f"<h2>Reports</h2><ul>{''.join(rows)}</ul>"
        f"{warning_html}"
        "</body></html>",
        encoding="utf-8",
    )


def run_demo(out_dir: Path | str) -> dict:
    """Run every analysis on synthetic data into a temporary store under
    ``out_dir``, write every HTML report plus an index page, and return
    ``{"reports": {name: path_or_None}, "warnings": [...], "index": path}``.
    """
    out_dir = Path(out_dir)
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    prior_records_env = os.environ.get("CALSUITE_RECORDS")
    os.environ["CALSUITE_RECORDS"] = str(records_dir)
    warnings: list = []
    reports: dict = {}
    try:
        try:
            reports["Sensor"] = _run_sensor(out_dir, warnings)
        except Exception as exc:  # noqa: BLE001 -- one area's bug must not blank the other reports
            warnings.append(f"sensor demo failed: {exc!r}")
        try:
            lens_report, lens_refusal_report = _run_lens(records_dir, out_dir, warnings)
            reports["Lens"] = lens_report
            reports["Lens (refusal example: narrow pose coverage)"] = lens_refusal_report
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"lens demo failed: {exc!r}")
        try:
            tier_a, tier_b = _run_color(records_dir, out_dir, warnings)
            reports["Colour (Tier A -- chart matrix)"] = tier_a
            reports["Colour (Tier B -- spectral)"] = tier_b
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"colour demo failed: {exc!r}")
        try:
            reports["Display"] = _run_display(out_dir, warnings)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"display demo failed: {exc!r}")
    finally:
        if prior_records_env is None:
            os.environ.pop("CALSUITE_RECORDS", None)
        else:
            os.environ["CALSUITE_RECORDS"] = prior_records_env

    index_path = out_dir / "index.html"
    _write_index(index_path, reports, warnings)
    return {"reports": reports, "warnings": warnings, "index": index_path}

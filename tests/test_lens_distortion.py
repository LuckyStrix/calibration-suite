import numpy as np

from calsuite.lens import charuco, distortion as D
from calsuite.lens.distortion import BoardView
from calsuite.synth import lens as synthlens

TRUE_K = np.array([[900.0, 0, 300.0], [0, 900.0, 200.0], [0, 0, 1.0]])
TRUE_DIST = np.array([-0.08, 0.015, 0.0003, -0.0002, 0.002])
IMAGE_SIZE = (600, 400)  # (width, height)


def _board():
    return charuco.build_board(square_length=30.0)


def _true_ru_rd(hugin_scale_px: float, f_px: float, ru_hugin: np.ndarray) -> np.ndarray:
    """The *ground-truth* Rd-Ru curve, straight from TRUE_DIST -- what
    distortion.refit_ptlens_poly3 is supposed to reproduce, independent of
    however calibrateCamera's own fit came out."""
    k1, k2, p1, p2, k3 = TRUE_DIST
    ru_norm = ru_hugin * hugin_scale_px / f_px
    rd_norm = ru_norm * (1.0 + k1 * ru_norm**2 + k2 * ru_norm**4 + k3 * ru_norm**6)
    return rd_norm * f_px / hugin_scale_px - ru_hugin


def test_distortion_recovered_within_half_tenth_px():
    """docs/design.md: distortion recovered to < 0.05 px error in the
    distortion map. Uses synth.lens's point-projection generator (not a
    rendered image -- that's charuco.py's own detector-noise concern),
    which is exactly the round trip distortion.fit_distortion promises:
    known Brown-Conrady coefficients in, an equivalent ptlens curve out."""
    board = _board()
    rng = np.random.default_rng(0)
    views = synthlens.synthetic_views(board, TRUE_K, TRUE_DIST, IMAGE_SIZE, 20, rng, coverage="full")
    assert len(views) >= D.DISTORTION_MIN_VIEWS

    analysis = D.fit_distortion(board, views, IMAGE_SIZE)
    ptlens = analysis.result["ptlens"]
    f_px = analysis.result["camera_matrix"][0][0]
    hugin_scale_px = min(IMAGE_SIZE) / 2.0

    ru = np.linspace(0.0, 1.2, 200)
    true_curve = _true_ru_rd(hugin_scale_px, f_px, ru)
    fitted_curve = ru * (ptlens["a"] * ru**3 + ptlens["b"] * ru**2 + ptlens["c"] * ru + 1.0) - ru
    # both curves are in Hugin-normalized units (1.0 = half the shorter
    # image dimension) -- multiply back to px to compare against the
    # design doc's own px tolerance.
    px_error = np.abs(fitted_curve - true_curve) * hugin_scale_px
    assert px_error.max() < 0.05


def test_coverage_refusal_fires_when_only_center_is_covered():
    board = _board()
    rng = np.random.default_rng(1)
    views = synthlens.synthetic_views(board, TRUE_K, TRUE_DIST, IMAGE_SIZE, 12, rng, coverage="center")
    analysis = D.fit_distortion(board, views, IMAGE_SIZE)
    assert not analysis.ok
    assert any(r.check == "coverage" for r in analysis.refusals)


def test_coverage_refusal_does_not_fire_for_a_deliberately_full_grid():
    """Sanity check that the coverage check isn't vacuously always-true:
    corners placed directly (bypassing perspective geometry) across every
    radial/angular bin should pass."""
    n_angular = 24
    thetas = np.linspace(-np.pi, np.pi, n_angular, endpoint=False)
    cx, cy = (IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0
    half_diag = np.hypot(*IMAGE_SIZE) / 2.0
    r = 0.95 * half_diag
    corners = np.stack([cx + r * np.cos(thetas), cy + r * np.sin(thetas)], axis=1)
    views = [BoardView(corners=corners, ids=np.arange(n_angular, dtype=np.int32), name="ring")]
    grid = D.coverage_grid(views, IMAGE_SIZE)
    assert D._coverage_refusal(grid) is None


def test_outlier_view_is_dropped_and_result_stays_close_to_ground_truth():
    board = _board()
    rng = np.random.default_rng(2)
    views = synthlens.synthetic_views(board, TRUE_K, TRUE_DIST, IMAGE_SIZE, 20, rng, coverage="full")

    # corrupt one view's corners with a large, obviously-wrong shift
    bad = views[0]
    corrupted = BoardView(corners=bad.corners + 25.0, ids=bad.ids, name="corrupted")
    views_with_outlier = [corrupted, *views[1:]]

    analysis = D.fit_distortion(board, views_with_outlier, IMAGE_SIZE)
    assert analysis.result["n_views_dropped"] >= 1
    assert "corrupted" in analysis.result["dropped_views"]
    # the refit distortion coefficients should still be close to ground truth
    fitted_dist = np.array(analysis.result["dist_coeffs"])
    assert np.allclose(fitted_dist[:2], TRUE_DIST[:2], atol=0.01)


def test_min_views_refusal():
    board = _board()
    rng = np.random.default_rng(3)
    views = synthlens.synthetic_views(board, TRUE_K, TRUE_DIST, IMAGE_SIZE, 2, rng, coverage="full")
    analysis = D.fit_distortion(board, views, IMAGE_SIZE)
    assert not analysis.ok
    assert any(r.check == "min_views" for r in analysis.refusals)


def test_refit_ptlens_matches_poly3_when_true_distortion_is_pure_cubic():
    """poly3 is ptlens with a=0, c=0, b=k1 (lens/constants.py's citation of
    mod-coord.cpp) -- fit both against a pure-k1 Brown-Conrady curve and
    check the ptlens refit lands near a=c=0."""
    K = TRUE_K
    dist_k1_only = np.array([0.05, 0.0, 0.0, 0.0, 0.0])
    refit = D.refit_ptlens_poly3(dist_k1_only, K, IMAGE_SIZE)
    assert abs(refit["ptlens"]["a"]) < 1e-3
    assert abs(refit["ptlens"]["c"]) < 1e-3
    assert abs(refit["poly3"]["k1"] - refit["ptlens"]["b"]) < 1e-6

import numpy as np
import pytest

from calsuite.lens import tca as T
from calsuite.lens.charuco import Detection

IMAGE_SIZE = (600, 400)


def _synthetic_view(kr: float, kb: float, n: int, rng: np.random.Generator, *, g2: bool = False) -> dict:
    cx, cy = (IMAGE_SIZE[0] - 1) / 2.0, (IMAGE_SIZE[1] - 1) / 2.0
    xs = rng.uniform(0, IMAGE_SIZE[0], n)
    ys = rng.uniform(0, IMAGE_SIZE[1], n)
    r_g = np.hypot(xs - cx, ys - cy)
    theta = np.arctan2(ys - cy, xs - cx)
    g_corners = np.stack([xs, ys], axis=1)
    r_corners = np.stack([cx + r_g * kr * np.cos(theta), cy + r_g * kr * np.sin(theta)], axis=1)
    b_corners = np.stack([cx + r_g * kb * np.cos(theta), cy + r_g * kb * np.sin(theta)], axis=1)
    ids = np.arange(n, dtype=np.int32)
    green_plane = "G2" if g2 else "G1"
    other_green = "G1" if g2 else "G2"
    return {
        "R": Detection("R", r_corners, ids),
        green_plane: Detection(green_plane, g_corners, ids),
        other_green: None,
        "B": Detection("B", b_corners, ids),
    }


def test_tca_scale_recovered():
    rng = np.random.default_rng(0)
    kr_true, kb_true = 1.0025, 0.9975
    views = [_synthetic_view(kr_true, kb_true, 60, rng) for _ in range(3)]
    analysis = T.fit_tca(views, IMAGE_SIZE)
    assert analysis.ok
    assert analysis.result["kr"] == pytest.approx(kr_true, abs=1e-3)
    assert analysis.result["kb"] == pytest.approx(kb_true, abs=1e-3)
    # mirrored under the poly3 element, matching mil-canon.xml's own convention
    assert analysis.result["vr"] == analysis.result["kr"]
    assert analysis.result["vb"] == analysis.result["kb"]


def test_tca_works_with_only_g2_present():
    rng = np.random.default_rng(1)
    views = [_synthetic_view(1.001, 0.998, 40, rng, g2=True)]
    analysis = T.fit_tca(views, IMAGE_SIZE)
    assert analysis.ok


def test_tca_refuses_with_no_matched_corners():
    ids = np.array([1, 2, 3], dtype=np.int32)
    corners = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    view = {"R": Detection("R", corners, ids), "G1": None, "G2": None, "B": Detection("B", corners, ids)}
    analysis = T.fit_tca([view], IMAGE_SIZE)
    assert not analysis.ok
    assert analysis.refusals[0].check == "no_matched_corners"


def test_tca_refuses_with_too_few_matched_corners():
    rng = np.random.default_rng(2)
    views = [_synthetic_view(1.001, 0.998, 3, rng)]
    analysis = T.fit_tca(views, IMAGE_SIZE)
    assert not analysis.ok
    assert analysis.refusals[0].check == "too_few_matched_corners"

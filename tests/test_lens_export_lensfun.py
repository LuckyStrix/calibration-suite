from pathlib import Path

import numpy as np
import pytest

from calsuite import store
from calsuite.fit import Analysis
from calsuite.lens import export_lensfun as E

try:
    import lensfunpy

    HAVE_LENSFUNPY = True
except ImportError:  # pragma: no cover -- exercised only where lensfunpy is absent
    HAVE_LENSFUNPY = False


def _device():
    return {"kind": "lens", "model": "Test Lens 50mm", "id": "test-lens-50mm-unknown", "firmware": ""}


def _ok_record(result: dict) -> store.Record:
    return store.Record.from_analysis(
        kind="lens.distortion",
        device=_device(),
        analysis=Analysis(result=result),
        provenance="measured",
        method={"name": "distortion.fit_distortion"},
        conditions={"focal_mm": 50.0},
    )


def _refused_record() -> store.Record:
    a = Analysis()
    a.refuse("coverage", "outer field not covered")
    return store.Record.from_analysis(
        kind="lens.distortion", device=_device(), analysis=a, provenance="measured", method={"name": "x"}
    )


def test_build_xml_shape():
    xml_text = E.build_xml(
        lens_model="Test Lens 50mm",
        distortion={"model": "ptlens", "focal": 50.0, "params": {"a": 0.001, "b": -0.004, "c": 0.002}},
        tca={"focal": 50.0, "vr": 1.0002, "vb": 0.9998},
        vignetting=[{"focal": 50.0, "aperture": 1.8, "distance": 10.0, "k1": -0.2, "k2": 0.1, "k3": -0.05}],
    )
    assert "<lensdatabase" in xml_text
    assert "Canon EOS R100" in xml_text  # default camera block, per the Wave 2B brief
    assert 'model="ptlens"' in xml_text
    assert 'model="poly3"' in xml_text  # tca
    assert 'model="pa"' in xml_text  # vignetting


def test_export_records_refuses_a_refused_record():
    with pytest.raises(store.ExportRefused):
        E.export_records(lens_model="Test Lens 50mm", distortion_record=_refused_record())


def test_export_records_refuses_weak_provenance():
    a = Analysis(result={"ptlens": {"a": 0.0, "b": 0.0, "c": 0.0}})
    record = store.Record.from_analysis(
        kind="lens.distortion", device=_device(), analysis=a, provenance="nominal", method={"name": "x"}
    )
    with pytest.raises(store.ExportRefused):
        E.export_records(lens_model="Test Lens 50mm", distortion_record=record)


def test_export_records_builds_xml_for_an_ok_record():
    record = _ok_record({"ptlens": {"a": 0.0012, "b": -0.0043, "c": 0.0021}})
    xml_text = E.export_records(lens_model="Test Lens 50mm", distortion_record=record)
    assert "0.001200" in xml_text


def test_write_lensfun_to_explicit_out_dir(tmp_path):
    xml_text = E.build_xml(lens_model="Test Lens 50mm")
    path = E.write_lensfun(xml_text, out=tmp_path, filename="test.xml")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == xml_text


def test_write_lensfun_to_explicit_xml_path(tmp_path):
    xml_text = E.build_xml(lens_model="Test Lens 50mm")
    target = tmp_path / "sub" / "out.xml"
    path = E.write_lensfun(xml_text, out=target)
    assert path == target
    assert path.exists()


def test_compare_with_vendor_reads_mil_canon_if_present():
    """Uses the real, locally-installed lensfun database
    (docs/implementation-plan.md's "Facts checked": lensfun 0.3.3 is
    installed and mil-canon.xml already has the RF 50mm F1.8 STM entry) --
    skips cleanly if it's not on this machine."""
    system_path = Path("/usr/share/lensfun/version_1/mil-canon.xml")
    if not system_path.exists():
        pytest.skip("system lensfun database not installed on this machine")
    comparison = E.compare_with_vendor({"a": 0.002, "b": -0.009, "c": 0.014}, system_xml_path=system_path)
    assert comparison["available"] is True
    assert comparison["provenance"] == "vendor"
    # the Wave 2B task prompt's own "facts checked": a=0.002 b=-0.009 c=0.014
    assert comparison["distortion_diff"]["a"] == pytest.approx(0.0, abs=1e-6)
    assert comparison["distortion_diff"]["b"] == pytest.approx(0.0, abs=1e-6)
    assert comparison["distortion_diff"]["c"] == pytest.approx(0.0, abs=1e-6)


def test_compare_with_vendor_reports_unavailable_for_missing_file(tmp_path):
    comparison = E.compare_with_vendor({"a": 0, "b": 0, "c": 0}, system_xml_path=tmp_path / "nope.xml")
    assert comparison["available"] is False


@pytest.mark.skipif(not HAVE_LENSFUNPY, reason="lensfunpy not importable")
def test_lensfunpy_round_trip_reads_back_our_own_coefficients(tmp_path):
    """Load our exported XML with lensfunpy's own ``Database`` (an
    independent C++ parser, not our ``ElementTree`` reader) and confirm it
    reports back the *exact* ptlens/tca coefficients we wrote -- the
    concrete, well-defined half of "exported XML loads in lensfunpy and
    reproduces [our] ground truth". The other half -- does applying the
    correction actually undo a known distortion, per-pixel -- used to be
    flagged here as an abandoned gap; it's now closed by
    ``test_lensfun_geometry_correction_matches_independently_derived_ground_truth``
    below, which works out lensfun's real per-pixel coordinate convention
    (verified against source, not assumed) instead of stopping at this
    coefficient-readback check.
    """
    xml_text = E.build_xml(
        lens_model="Test Lens 50mm",
        lens_mount="Canon RF",
        distortion={"model": "ptlens", "focal": 50.0, "params": {"a": 0.0012, "b": -0.0043, "c": 0.0021}},
        tca={"focal": 50.0, "vr": 1.00021, "vb": 0.99987},
    )
    path = E.write_lensfun(xml_text, out=tmp_path, filename="round_trip.xml")

    db = lensfunpy.Database(paths=[str(path)], load_common=False, load_bundled=False)
    lenses = [lens for lens in db.lenses if lens.model == "Test Lens 50mm"]
    assert len(lenses) == 1
    lens = lenses[0]

    calib = lens.interpolate_distortion(50.0)
    assert calib.model == lensfunpy.DistortionModel.PTLENS
    assert calib.terms[0] == pytest.approx(0.0012, abs=1e-6)
    assert calib.terms[1] == pytest.approx(-0.0043, abs=1e-6)
    assert calib.terms[2] == pytest.approx(0.0021, abs=1e-6)

    tca_calib = lens.interpolate_tca(50.0)
    assert tca_calib.terms[0] == pytest.approx(1.00021, abs=1e-5)
    assert tca_calib.terms[1] == pytest.approx(0.99987, abs=1e-5)

    # Modifier smoke test: it should at least run and produce a real,
    # non-identity remap grid (the qualitative half of the round trip).
    cams = db.cameras
    crop = cams[0].crop_factor if cams else 1.613
    modifier = lensfunpy.Modifier(lens, crop, 600, 400)
    modifier.initialize(50.0, 2.8)
    coords = modifier.apply_geometry_distortion()
    assert coords.shape == (400, 600, 2)
    _, identity_x = np.mgrid[0:400, 0:600]
    assert not (coords[:, :, 0] == identity_x).all()


@pytest.mark.skipif(not HAVE_LENSFUNPY, reason="lensfunpy not importable")
def test_lensfun_geometry_correction_matches_independently_derived_ground_truth(tmp_path):
    """The highest-value gap the bug hunt left open: does applying OUR
    exported lensfun XML actually undo OUR known distortion the way our own
    fit says it should -- not just read our coefficients back unchanged
    (the previous test's scope)?

    Two facts, verified against real source rather than assumed (see
    lens/constants.py's "CORRECTION" note and distortion.refit_ptlens_poly3's
    docstring for the full derivation and citations):

    - lensfunpy 1.18.0 bundles its own liblensfun 0.3.4
      (``lensfunpy.lensfun_version() == (0, 3, 4, 0)``), not this machine's
      system liblensfun 0.3.3 -- but ``libs/lensfun/{modifier,mod-coord}.cpp``
      are byte-identical between the ``v0.3.3`` and ``v0.3.4`` GitHub tags
      for every function this test depends on, so this exercises exactly
      what darktable/system tooling runs here too.
    - lensfun's actual ptlens/poly3 formula is
      ``Rd = Ru*(a*Ru^3+b*Ru^2+c*Ru+d)`` with ``d = 1-a-b-c``, not
      ``... + 1``. This structurally pins ``Rd(Hugin r=1) == Ru(Hugin r=1)``
      for *any* (a,b,c) -- lensfun's own correction therefore differs from
      a real lens's true curve (which has no reason to satisfy that) by a
      bounded amount of ``~|a+b+c|`` in Hugin-normalized units. This is a
      real limitation of the export format, not a bug in our fit (fitting
      to lensfun's "+d" family instead was checked and makes the
      approximation of the true curve *worse*, not better).

    So this test proves two different things at two different tolerances,
    rather than picking one loose tolerance that would hide the difference
    between them:

    (1) lensfunpy's actual per-pixel output matches a closed-form
        reproduction of lensfun's own documented "+d" formula to within
        float32/Newton-iteration noise -- i.e. our understanding of
        lensfun's real coordinate convention (center, radius
        normalization, direction, the "+d not +1" constant) is exactly
        right; no other bug (wrong axis, sign flip, stray crop/aspect
        scale) is hiding in the difference.
    (2) lensfun's real output differs from an *independently* re-derived
        ground truth (``cv2.projectPoints``, OpenCV's own forward
        Brown-Conrady, not our fit run again) by no more than the
        explained ``|a+b+c|``-scaled bound above, plus a small safety
        margin -- proving the *size* of the known, documented gap is what
        we say it is, not silently absorbing a real bug into a loose
        tolerance.

    Then (3) renders an actual synthetic image through the same known
    distortion, corrects it with lensfun's real per-pixel map, and checks
    the corrected pixels against the true undistorted pattern -- the
    literal "does this un-distort a photo" check -- with an intensity
    tolerance derived from (2) via the pattern's own gradient bound, not
    picked independently.
    """
    import cv2

    from calsuite.lens import distortion as distortionmod

    width, height = 600, 400  # aspect 1.5: lensfun's own default AspectRatio
    # for a lens with no <aspect-ratio> element (ours), so the
    # calibration-vs-image aspect ratio mismatch -- a separate, tiny (<0.1px)
    # effect -- stays at zero instead of mixing into what this test measures.
    fx = fy = 500.0
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    # A moderate, realistic barrel distortion -- purely radial (k1, k2; no
    # p1/p2 tangential terms), because ptlens/poly3 have no tangential term
    # at all, so nonzero p1/p2 here would measure a model-family gap that
    # has nothing to do with lensfun's *normalization*, which is the point.
    dist_coeffs = np.array([-0.03, 0.006, 0.0, 0.0, 0.0])

    ptlens = distortionmod.refit_ptlens_poly3(dist_coeffs, K, (width, height))["ptlens"]
    a, b, c = ptlens["a"], ptlens["b"], ptlens["c"]

    record = store.Record.from_analysis(
        kind="lens.distortion",
        device=_device(),
        analysis=Analysis(result={"ptlens": {"a": a, "b": b, "c": c}}),
        provenance="measured",
        method={"name": "distortion.fit_distortion"},
        conditions={"focal_mm": 50.0},
    )
    xml_text = E.export_records(lens_model="Test Lens 50mm", distortion_record=record, lens_cropfactor=1.0)
    path = E.write_lensfun(xml_text, out=tmp_path, filename="geometry_roundtrip.xml")

    db = lensfunpy.Database(paths=[str(path)], load_common=False, load_bundled=False)
    lens = next(lens for lens in db.lenses if lens.model == "Test Lens 50mm")
    modifier = lensfunpy.Modifier(lens, 1.0, width, height)
    # scale=1.0 explicitly disables lensfunpy's own auto-scale
    # (Modifier.GetAutoScale) -- the one auto-scale factor the task brief
    # anticipated as a possible fallback -- so there is nothing left that
    # needs explaining away by loosening a tolerance; the default (0.0)
    # would additionally zoom the image to hide empty corners, unrelated to
    # what this test checks.
    modifier.initialize(50.0, 2.8, scale=1.0)
    coords = modifier.apply_geometry_distortion()  # coords[y,x] = (xd, yd): where to sample the raw image

    ys, xs = np.mgrid[0:height, 0:width]

    # --- ground truth: an INDEPENDENT re-derivation of the same known
    # distortion, via cv2.projectPoints (OpenCV's own forward Brown-Conrady
    # formula), not our own fit run a second time.
    xn, yn = (xs - cx) / fx, (ys - cy) / fy
    object_points = np.stack([xn, yn, np.ones_like(xn)], axis=-1).reshape(-1, 1, 3)
    ground_truth, _ = cv2.projectPoints(object_points, np.zeros(3), np.zeros(3), K, dist_coeffs)
    ground_truth = ground_truth.reshape(height, width, 2)

    # --- lensfun's own, real, documented convention: Rd = Ru*(a*Ru^3+b*Ru^2
    # +c*Ru+d), d = 1-a-b-c, in Hugin-normalized radius (r=1 at half the
    # shorter pixel dimension, centered the same way lens/distortion.py
    # itself centers it).
    hugin_scale_px = min(width, height) / 2.0
    xu, yu = (xs - cx) / hugin_scale_px, (ys - cy) / hugin_scale_px
    ru_hugin = np.hypot(xu, yu)
    d = 1.0 - a - b - c
    factor = a * ru_hugin**3 + b * ru_hugin**2 + c * ru_hugin + d
    predicted = np.stack([xu * factor * hugin_scale_px + cx, yu * factor * hugin_scale_px + cy], axis=-1)

    # (1) PROVEN, tight.
    tight_tolerance_px = 0.2
    per_pixel_diff = np.linalg.norm(coords - predicted, axis=-1)
    assert np.nanmax(per_pixel_diff) < tight_tolerance_px, (
        f"lensfun's actual output does not match its own documented formula "
        f"(max diff {np.nanmax(per_pixel_diff):.4f}px) -- the formula citation in "
        f"lens/constants.py may be stale again, or lensfunpy changed conventions."
    )

    # (2) EXPLAINED, bounded -- not silently loosened.
    corner_ru_hugin = float(np.max(ru_hugin))
    explained_bound_px = abs(a + b + c) * hugin_scale_px * corner_ru_hugin
    safety_margin_px = 0.3  # covers the smaller a*Ru^3+b*Ru^2+c*Ru cross terms + float32 noise
    ground_truth_diff = np.linalg.norm(coords - ground_truth, axis=-1)
    assert np.nanmax(ground_truth_diff) < explained_bound_px + safety_margin_px, (
        f"lensfun's correction differs from the true distortion by more than the "
        f"explained ptlens '+d' bound ({np.nanmax(ground_truth_diff):.4f}px vs "
        f"{explained_bound_px + safety_margin_px:.4f}px) -- a real bug may be hiding "
        f"behind what should be an already-explained gap."
    )
    # The explained gap is itself non-trivial here -- this isn't "explain
    # away a rounding error", the raw (unaccounted-for) comparison in (2)
    # would have failed outright without the "+d" understanding.
    assert explained_bound_px > 0.3

    # --- (3) the literal round trip: render a synthetic image through the
    # SAME known distortion, correct it with lensfun's real per-pixel map,
    # and compare to the true undistorted pattern.
    period_px = 240.0  # low frequency: keeps the pattern's own gradient small
    # relative to the position errors bounded above, so an intensity
    # mismatch here reports position error, not pattern aliasing.

    def pattern(x, y):
        return 0.5 + 0.5 * np.sin(2 * np.pi * x / period_px) * np.sin(2 * np.pi * y / period_px)

    # The raw (distorted) synthetic capture: for every raw pixel, the
    # scene's ideal (undistorted) appearance at the location that distorts
    # TO that raw pixel -- the inverse of the forward map above, computed
    # independently via cv2.undistortPoints (distorted -> ideal, with P=K
    # to get pixel-unit output directly).
    raw_points = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    undistorted_px = cv2.undistortPoints(raw_points, K, dist_coeffs, P=K).reshape(height, width, 2)
    raw_image = pattern(undistorted_px[..., 0], undistorted_px[..., 1]).astype(np.float32)

    corrected = cv2.remap(
        raw_image,
        coords[..., 0].astype(np.float32),
        coords[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    ground_truth_image = pattern(xs, ys).astype(np.float32)

    valid = np.isfinite(corrected)
    assert valid.mean() > 0.9  # lensfun leaves only a thin border of holes near the frame edge
    intensity_diff = np.abs(corrected[valid] - ground_truth_image[valid])
    # Gradient bound: |d(pattern)/d(position)| <= 2*pi/period_px, so the
    # position-error bound from (2) translates directly into an intensity
    # tolerance -- not a number picked independently of that proof.
    max_gradient = 2 * np.pi / period_px
    intensity_tolerance = max_gradient * (explained_bound_px + safety_margin_px) + 0.02
    assert np.nanmax(intensity_diff) < intensity_tolerance, (
        f"corrected image does not match the undistorted ground truth pattern "
        f"(max diff {np.nanmax(intensity_diff):.4f} vs tolerance {intensity_tolerance:.4f})"
    )

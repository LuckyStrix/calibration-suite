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
    reproduces [our] ground truth". A per-pixel comparison through
    ``lensfunpy.Modifier`` was attempted but not included: Modifier applies
    the stored Hugin-normalized coefficients inside its own internal
    "natural" (focal-length-normalized) coordinate system, via
    ``rescale_polynomial_coefficients`` (real-focal, crop factor, aspect
    ratio, and an auto-scale all feed into that conversion) -- reproducing
    that exact conversion independently, just to re-check numbers this
    ``interpolate_distortion`` check already confirms are stored correctly,
    was judged not worth the risk of a fragile test within this wave's
    scope. Flagged in the final report as a gap, not silently dropped.
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

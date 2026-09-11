import hashlib
import os

import numpy as np
import pytest

from calsuite import raw
from calsuite.synth import sensor as synth_sensor


def _synthetic_frame(**overrides):
    model = synth_sensor.SensorModel(shape=(64, 96), top_margin=8, left_margin=32, **overrides)
    rng = np.random.default_rng(0)
    return synth_sensor.frame(model, exposure_s=0.01, flux_e_per_s=5000.0, temp_c=20.0, rng=rng)


def test_planes_shapes_and_names():
    frame = _synthetic_frame()
    p = raw.planes(frame)
    assert set(p) == {"R", "G1", "G2", "B"}
    vis_rows = frame.visible[0].stop - frame.visible[0].start
    vis_cols = frame.visible[1].stop - frame.visible[1].start
    for arr in p.values():
        assert arr.shape == (vis_rows // 2, vis_cols // 2)
        assert arr.dtype == np.float64


def test_planes_full_area_is_larger_than_visible():
    frame = _synthetic_frame()
    visible_planes = raw.planes(frame, area="visible")
    full_planes = raw.planes(frame, area="full")
    assert full_planes["R"].size > visible_planes["R"].size


def test_planes_unknown_area_raises():
    frame = _synthetic_frame()
    with pytest.raises(ValueError):
        raw.planes(frame, area="bogus")


def test_optical_black_is_near_bias_level_even_when_visible_saturates():
    frame = _synthetic_frame(full_well_e=5000.0)
    model = synth_sensor.SensorModel(shape=(64, 96), top_margin=8, left_margin=32, full_well_e=5000.0)
    rng = np.random.default_rng(1)
    frame = synth_sensor.frame(model, exposure_s=1.0, flux_e_per_s=1e6, temp_c=20.0, rng=rng)

    ob = raw.optical_black(frame)
    assert set(ob) == {"R", "G1", "G2", "B"}
    for arr in ob.values():
        assert arr.mean() == pytest.approx(model.black_dn, abs=30.0)

    visible_mean = raw.planes(frame)["R"].mean()
    assert visible_mean >= frame.white_level - 5  # visible area clipped near white, in contrast


def test_optical_black_too_narrow_margin_raises():
    model = synth_sensor.SensorModel(shape=(32, 32), top_margin=4, left_margin=4)
    rng = np.random.default_rng(2)
    frame = synth_sensor.frame(model, exposure_s=0.01, flux_e_per_s=1000.0, temp_c=20.0, rng=rng)
    with pytest.raises(ValueError):
        raw.optical_black(frame)


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert raw.sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_parse_shutter_fraction_and_decimal():
    assert raw._parse_shutter("1/83.0") == pytest.approx(1 / 83.0)
    assert raw._parse_shutter("4") == pytest.approx(4.0)


def test_metadata_from_dcraw_fallback(tmp_path, monkeypatch):
    fake_dcraw = tmp_path / "dcraw"
    fake_dcraw.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "Filename: x.cr3\n"
        "Timestamp: Thu May 21 09:23:18 2026\n"
        "Camera: Canon EOS R100\n"
        "ISO speed: 800\n"
        "Shutter: 1/83.0 sec\n"
        "Aperture: f/1.8\n"
        "Focal length: 50.0 mm\n"
        "EOF\n"
    )
    fake_dcraw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    meta = raw._metadata_from_dcraw(tmp_path / "whatever.cr3")
    assert meta.model == "Canon EOS R100"
    assert meta.iso == 800
    assert meta.exposure_s == pytest.approx(1 / 83.0)
    assert meta.aperture == pytest.approx(1.8)
    assert meta.focal == pytest.approx(50.0)
    # Not exposed by `dcraw -i -v` at all -- see raw.py's docstring.
    assert meta.serial == ""
    assert meta.lens == ""


def test_metadata_from_exiftool_fake(tmp_path, monkeypatch):
    fake_exiftool = tmp_path / "exiftool"
    fake_exiftool.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        '[{"Model":"Canon EOS R100","SerialNumber":"12345","FocalLength":50.0,'
        '"FNumber":1.8,"ExposureTime":0.012,"ISO":800,'
        '"DateTimeOriginal":"2026:05:21 09:23:18"}]\n'
        "EOF\n"
    )
    fake_exiftool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    meta = raw._metadata_from_exiftool(tmp_path / "whatever.cr3")
    assert meta.model == "Canon EOS R100"
    assert meta.serial == "12345"
    assert meta.iso == 800
    assert meta.aperture == pytest.approx(1.8)
    assert meta.focal == pytest.approx(50.0)
    assert meta.exposure_s == pytest.approx(0.012)


def test_read_metadata_falls_back_to_empty_when_no_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # neither exiftool nor dcraw present
    meta = raw._read_metadata(tmp_path / "whatever.cr3")
    assert meta.model == ""
    assert meta.iso is None

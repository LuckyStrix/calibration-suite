import pytest

from calsuite.formats import cgats


def test_write_read_round_trip(tmp_path):
    samples = [
        {"rgb": (0.0, 0.0, 0.0), "xyz": (0.02, 0.02, 0.02)},
        {"rgb": (1.0, 0.0, 0.0), "xyz": (0.41, 0.21, 0.02)},
        {"rgb": (1.0, 1.0, 1.0), "xyz": (0.9505, 1.0, 1.0890)},
    ]
    path = tmp_path / "test.ti3"
    cgats.write_ti3(path, samples, device_class="DISPLAY", descriptor="unit test")

    result = cgats.read_ti3(path)
    assert result.device_class == "DISPLAY"
    assert result.descriptor == "unit test"
    assert len(result.samples) == 3
    for expected, got in zip(samples, result.samples, strict=True):
        for a, b in zip(expected["rgb"], got["rgb"], strict=True):
            assert a == pytest.approx(b, abs=1e-6)
        for a, b in zip(expected["xyz"], got["xyz"], strict=True):
            assert a == pytest.approx(b, abs=1e-6)


def test_write_rejects_bad_device_class(tmp_path):
    with pytest.raises(ValueError):
        cgats.write_ti3(tmp_path / "x.ti3", [], device_class="BOGUS")


def test_file_has_expected_cgats_structure(tmp_path):
    path = tmp_path / "test.ti3"
    cgats.write_ti3(path, [{"rgb": (0.5, 0.5, 0.5), "xyz": (0.4, 0.4, 0.4)}])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("CTI3")
    assert "BEGIN_DATA_FORMAT" in text and "END_DATA_FORMAT" in text
    assert "RGB_R" in text and "XYZ_X" in text
    assert "NUMBER_OF_SETS 1" in text


def test_read_rejects_missing_xyz_fields(tmp_path):
    path = tmp_path / "bad.ti3"
    path.write_text(
        "CTI3\n"
        'DEVICE_CLASS "DISPLAY"\n'
        "NUMBER_OF_FIELDS 2\n"
        "BEGIN_DATA_FORMAT\n"
        "RGB_R RGB_G\n"
        "END_DATA_FORMAT\n"
        "NUMBER_OF_SETS 1\n"
        "BEGIN_DATA\n"
        "0.0 0.0\n"
        "END_DATA\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        cgats.read_ti3(path)

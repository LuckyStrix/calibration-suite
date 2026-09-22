import pytest

from calsuite.formats import cgats

# A real Argyll-produced .ti3 (from a chart scan, DEVICE_CLASS INPUT --
# unrelated to this suite's own DISPLAY writer), copied verbatim (field
# order, whitespace, alphanumeric SAMPLE_IDs, extra STDEV_* fields, "inf"
# stdev values and all) from another project on this machine
# (RIT/EFIP/.../colorCalibrationTesting/2D/backend/calibration/color/
# allLight.ti3) rather than referenced by path, so this test doesn't depend
# on a file outside this repo existing on CI. This is Argyll's own tooling's
# actual output, not anything this suite (or this test) wrote -- the point
# is to check `read_ti3` against a genuinely independent producer of the
# format, not just against our own `write_ti3`.
_REAL_ARGYLL_TI3 = """CTI3

DESCRIPTOR "Argyll Calibration Target chart information 3"
ORIGINATOR "Argyll target"
CREATED "Fri Aug  7 15:22:28 2026"
DEVICE_CLASS "INPUT"
COLOR_REP "XYZ_RGB"

NUMBER_OF_FIELDS 10
BEGIN_DATA_FORMAT
SAMPLE_ID XYZ_X XYZ_Y XYZ_Z RGB_R RGB_G RGB_B STDEV_R STDEV_G STDEV_B
END_DATA_FORMAT

NUMBER_OF_SETS 4
BEGIN_DATA
A01 11.52185 10.08245 5.088943 44.80741 37.76115 27.96813 18.80055 15.29311 10.16162
A02 39.17590 34.95036 19.22045 44.75692 37.71466 28.05436 18.69146 15.10400 9.869057
B01 40.71693 31.18084 4.991907 6.776532 4.875257 3.878843 inf inf inf
D01 87.81319 91.31598 73.94960 6.776532 4.875257 3.878843 inf inf inf
END_DATA
"""


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


def test_read_ti3_parses_a_real_argyll_produced_file(tmp_path):
    """Cross-check against an independent producer of the format (task
    item 4), not our own writer: real Argyll tooling wrote this file, with
    field-order/whitespace/SAMPLE_ID conventions (alphanumeric IDs, extra
    STDEV_* columns, literal "inf" stdev values, COLOR_REP "XYZ_RGB" with
    the name's field order reversed from ours) that `write_ti3` never
    produces and a round trip through only our own writer could never
    exercise. `read_ti3` is documented as tolerant of exactly this
    (extra header keywords, any data-format field order) provided the six
    RGB_*/XYZ_* fields it needs are present -- this is what actually
    proves that, and checks the numeric percentage-to-[0,1] conversion
    (the Argyll spec's own stated field semantics, ``formats/cgats.py``'s
    module docstring) against real, non-round, human-unfriendly numbers
    that a hand-picked test fixture would be unlikely to expose a /100
    vs /255 or similar scale bug in.
    """
    path = tmp_path / "real_argyll.ti3"
    path.write_text(_REAL_ARGYLL_TI3, encoding="utf-8")

    result = cgats.read_ti3(path)
    assert result.device_class == "INPUT"
    assert len(result.samples) == 4

    # A01, straight from the file above, /100 per the Argyll spec's
    # "device values as percentages 0-100" / "XYZ normalized to Y=100"
    # convention this module's docstring cites.
    a01 = result.samples[0]
    assert a01["rgb"] == pytest.approx((0.4480741, 0.3776115, 0.2796813))
    assert a01["xyz"] == pytest.approx((0.1152185, 0.1008245, 0.05088943))

    # D01 is the brightest patch in the file (Y=91.316, i.e. this chart's
    # own near-white, not padded/rounded to exactly 100) -- confirms the
    # /100 scaling is applied uniformly, not just for tidy round numbers.
    d01 = result.samples[3]
    assert d01["xyz"] == pytest.approx((0.8781319, 0.9131598, 0.7394960))


def test_write_ti3_matches_the_documented_percentage_scale_in_the_raw_text(tmp_path):
    """Not a round trip through our own reader -- a direct check of the
    written *text* against the Argyll spec's stated field semantics
    (module docstring: "device (RGB) values are written as percentages
    0-100 ... XYZ values are normalized to Y=100"), so a bug that scaled
    by the wrong factor but still round-tripped through a matching bug in
    `read_ti3` wouldn't hide here."""
    path = tmp_path / "scale.ti3"
    cgats.write_ti3(path, [{"rgb": (0.5, 0.25, 0.0), "xyz": (0.4, 1.0, 0.1)}])
    text = path.read_text(encoding="utf-8")
    data_line = text.splitlines()[text.splitlines().index("BEGIN_DATA") + 1]
    fields = data_line.split()
    assert fields[0] == "1"  # 1-based SAMPLE_ID
    assert [float(v) for v in fields[1:]] == pytest.approx([50.0, 25.0, 0.0, 40.0, 100.0, 10.0])


# A real .cal that ArgyllCMS's own `dispwin -s` wrote back from this
# laptop's Video LUT (2026-09-22), truncated to its header + first few rows
# -- checked against for the same reason as `_REAL_ARGYLL_TI3` above: this
# is dispwin's own output, not anything this suite wrote, so it pins
# write_cal's header/field layout to what the real tool actually expects.
_REAL_ARGYLL_CAL_HEADER = """CAL

DESCRIPTOR "Argyll Device Calibration Curves"
ORIGINATOR "Argyll synthcal"
CREATED "Tue Sep 22 16:01:48 2026"
DEVICE_CLASS "DISPLAY"
COLOR_REP "RGB"

NUMBER_OF_FIELDS 4
BEGIN_DATA_FORMAT
RGB_I RGB_R RGB_G RGB_B
END_DATA_FORMAT
"""


def test_write_cal_matches_real_argyll_header_layout(tmp_path):
    path = tmp_path / "vcgt.cal"
    curves = {"r": [0.0, 0.5, 1.0], "g": [0.0, 0.4, 1.0], "b": [0.0, 0.6, 1.0]}
    cgats.write_cal(path, curves)
    text = path.read_text(encoding="utf-8")

    # Same keyword lines, in the same order, as a real dispwin-written .cal
    # (only DESCRIPTOR/ORIGINATOR/CREATED's *values* legitimately differ).
    real_keywords = [ln.split(maxsplit=1)[0] for ln in _REAL_ARGYLL_CAL_HEADER.splitlines() if ln.strip()]
    written_keywords = [
        ln.split(maxsplit=1)[0] for ln in text.split("NUMBER_OF_SETS")[0].splitlines() if ln.strip()
    ]
    assert written_keywords == real_keywords
    for line in ('DEVICE_CLASS "DISPLAY"', 'COLOR_REP "RGB"', "NUMBER_OF_FIELDS 4",
                 "BEGIN_DATA_FORMAT", "RGB_I RGB_R RGB_G RGB_B", "END_DATA_FORMAT"):
        assert line in text

    lines = text.splitlines()
    assert "NUMBER_OF_SETS 3" in lines
    data = lines[lines.index("BEGIN_DATA") + 1 : lines.index("END_DATA")]
    assert len(data) == 3
    first, last = data[0].split(), data[-1].split()
    assert [float(v) for v in first] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert [float(v) for v in last] == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_write_cal_rejects_mismatched_channel_lengths(tmp_path):
    with pytest.raises(ValueError):
        cgats.write_cal(tmp_path / "bad.cal", {"r": [0.0, 1.0], "g": [0.0, 0.5, 1.0], "b": [0.0, 1.0]})


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

import hashlib
import json

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
        # Each channel's optical-black region here averages >400 pixels of
        # read noise (3e- RMS / gain 2 = 1.5 DN/px) plus negligible dark
        # current (0.05 e/s * 1s) -- the mean's own std is ~1.5/sqrt(432)
        # ~= 0.07 DN, and 30 random seeds never exceeded 0.22 DN of
        # deviation from black_dn. abs=30.0 (nearly 140x that worst case)
        # would pass even if optical_black() silently mixed in a chunk of
        # the visible (near-saturated) region, or dropped the bias
        # baseline entirely and read raw electron counts instead of DN --
        # abs=2.0 leaves a comfortable ~9x margin over the worst observed
        # seed while still catching either of those.
        assert arr.mean() == pytest.approx(model.black_dn, abs=2.0)

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


def test_metadata_from_dcraw_fallback(tmp_path, fake_bin):
    fake_bin(
        "dcraw",
        "print('''Filename: x.cr3\n"
        "Timestamp: Thu May 21 09:23:18 2026\n"
        "Camera: Canon EOS R100\n"
        "ISO speed: 800\n"
        "Shutter: 1/83.0 sec\n"
        "Aperture: f/1.8\n"
        "Focal length: 50.0 mm''')\n",
    )

    meta = raw._metadata_from_dcraw(tmp_path / "whatever.cr3")
    assert meta.model == "Canon EOS R100"
    assert meta.iso == 800
    assert meta.exposure_s == pytest.approx(1 / 83.0)
    assert meta.aperture == pytest.approx(1.8)
    assert meta.focal == pytest.approx(50.0)
    # Not exposed by `dcraw -i -v` at all -- see raw.py's docstring.
    assert meta.serial == ""
    assert meta.lens == ""


def test_metadata_from_exiftool_fake(tmp_path, fake_bin):
    payload = json.dumps(
        [
            {
                "Model": "Canon EOS R100",
                "SerialNumber": "12345",
                "FocalLength": 50.0,
                "FNumber": 1.8,
                "ExposureTime": 0.012,
                "ISO": 800,
                "DateTimeOriginal": "2026:05:21 09:23:18",
            }
        ]
    )
    fake_bin("exiftool", f"print({payload!r})\n")

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


def test_black_level_by_channel_matches_planes_names():
    frame = _synthetic_frame()
    by_channel = raw.black_level_by_channel(frame)
    assert set(by_channel) == {"R", "G1", "G2", "B"}
    # SensorModel gives one uniform black level across all 4 slots by
    # construction (synth/sensor.py), so every channel should read it back
    # identically regardless of margin phase.
    for value in by_channel.values():
        assert value == pytest.approx(frame.black_level[0])


def test_black_level_by_channel_handles_odd_margin_phase_shift():
    # top_margin=7 (odd) puts the visible tile's phase one row away from
    # the absolute-origin tile black_level is indexed against -- this is
    # exactly the case a plain "trust frame.pattern's raster order" mapping
    # gets wrong; here every position gets a distinct value so a wrong
    # mapping is very likely to be caught by strict equality below.
    model = synth_sensor.SensorModel(shape=(32, 48), top_margin=7, left_margin=16)
    rng = np.random.default_rng(3)
    frame = synth_sensor.frame(model, exposure_s=0.01, flux_e_per_s=1000.0, temp_c=20.0, rng=rng)
    frame = raw.RawFrame(
        cfa=frame.cfa,
        pattern=frame.pattern,
        visible=frame.visible,
        black_level=(500.0, 510.0, 520.0, 530.0),
        white_level=frame.white_level,
        meta=frame.meta,
        path=frame.path,
        sha256=frame.sha256,
    )
    by_channel = raw.black_level_by_channel(frame)
    # Exact values, not `set(...) == {...}`: the old assertion passed for
    # *any* permutation of the four, which is how a B/G2 swap lived here
    # undetected. top_margin=7 makes the absolute-origin tile "GBRG", so
    # LibRaw's cblack order (R, first-G, B, second-G) at that origin is
    # (R, G2, B, G1) in this frame's visible-origin names.
    assert by_channel == {"R": 500.0, "G2": 510.0, "B": 520.0, "G1": 530.0}


def test_black_level_by_channel_uses_librraw_color_index_order_not_raster_order():
    """``black_level`` is LibRaw's ``cblack[0..3]``: one value per *color
    index* of ``color_desc`` (b"RGBG"), i.e. R, first-G, B, second-G --
    not the four raster positions of the tile. Confirmed against this
    project's reference camera: for capt0000.cr3 (Canon R100) rawpy reports
    ``raw_pattern = [[0, 1], [3, 2]]``, so index 2 (B) is at raster
    position (1, 1) and index 3 (the second green) at (1, 0). Mapping by
    raster position swaps B and G2 on every standard Bayer sensor -- the
    exact case this function exists to get right.
    """
    frame = raw.RawFrame(
        cfa=np.zeros((8, 8), dtype=np.uint16),
        pattern="RGGB",  # even margins: the visible tile is the absolute tile
        visible=(slice(0, 8), slice(0, 8)),
        black_level=(100.0, 200.0, 300.0, 400.0),
        white_level=16383.0,
        meta=raw.FrameMeta(),
        path="synthetic",
        sha256="",
    )
    assert raw.black_level_by_channel(frame) == {"R": 100.0, "G1": 200.0, "B": 300.0, "G2": 400.0}


def test_save_npz_and_load_npz_round_trip(tmp_path):
    frame = _synthetic_frame()
    path = raw.save_npz(frame, tmp_path / "frame0.npz")
    assert path.exists()

    loaded = raw.load_npz(path)
    assert np.array_equal(loaded.cfa, frame.cfa)
    assert loaded.pattern == frame.pattern
    assert loaded.visible == frame.visible
    assert loaded.black_level == frame.black_level
    assert loaded.white_level == frame.white_level
    assert loaded.meta.model == frame.meta.model
    assert loaded.meta.exposure_s == pytest.approx(frame.meta.exposure_s)
    assert loaded.meta.sensor_temp_c == pytest.approx(frame.meta.sensor_temp_c)
    assert loaded.path == str(path)
    assert loaded.sha256 == raw.sha256_file(path)

    # raw.load() dispatches to load_npz for a .npz path -- the whole point
    # is that every ``--from DIR`` consumer can treat this like a real raw.
    via_load = raw.load(path)
    assert np.array_equal(via_load.cfa, frame.cfa)


def test_save_npz_round_trips_frame_meta_settings(tmp_path):
    meta = raw.FrameMeta(
        model="calsuite-synthetic",
        serial="abc123",
        firmware="1.2.3",
        lens="RF 50mm",
        focal=50.0,
        aperture=1.8,
        exposure_s=0.01,
        iso=800,
        timestamp="2026:05:21 09:23:18",
        sensor_temp_c=21.5,
        settings={"long_exposure_nr": "Off", "high_iso_nr": None},
    )
    frame = raw.RawFrame(
        cfa=np.zeros((8, 8), dtype=np.uint16),
        pattern="RGGB",
        visible=(slice(0, 8), slice(0, 8)),
        black_level=(512.0, 512.0, 512.0, 512.0),
        white_level=16383.0,
        meta=meta,
        path="<synthetic>",
        sha256="",
    )
    path = raw.save_npz(frame, tmp_path / "meta.npz")
    loaded = raw.load_npz(path)
    assert loaded.meta == meta


def test_save_npz_requires_npz_suffix(tmp_path):
    frame = _synthetic_frame()
    with pytest.raises(ValueError):
        raw.save_npz(frame, tmp_path / "frame0.raw")


def test_load_dispatches_on_a_case_insensitive_npz_suffix(tmp_path):
    """`.NPZ` is not `.npz`: the dispatch used to compare the suffix
    literally, so an uppercase path went to rawpy and died with
    LibRawFileUnsupportedError. The real raw extensions are listed in both
    cases everywhere else."""
    frame = _synthetic_frame()
    path = raw.save_npz(frame, tmp_path / "frame0.npz")
    upper = path.with_name("FRAME1.NPZ")
    upper.write_bytes(path.read_bytes())

    loaded = raw.load(upper)
    assert np.array_equal(loaded.cfa, frame.cfa)
    assert loaded.pattern == frame.pattern

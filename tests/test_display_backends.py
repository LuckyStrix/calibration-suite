import numpy as np
import pytest

from calsuite import store as storemod
from calsuite.display import patches as patchesmod
from calsuite.display.backends import argyll as argyllmod
from calsuite.display.backends import camera as cameramod
from calsuite.display.backends import spectro as spectromod
from calsuite.display.backends.synthetic import SyntheticBackend
from calsuite.synth.display import DisplayModel

# -- synthetic ---------------------------------------------------------------


def test_synthetic_backend_measures_patches():
    backend = SyntheticBackend(model=DisplayModel())
    result = backend.measure(patchesmod.primaries_secondaries())
    assert len(result) == 8
    assert backend.accuracy().de00_estimate > 0


# -- argyll --------------------------------------------------------------


def test_argyll_parse_xyz_various_separators():
    assert argyllmod.parse_xyz("Result is XYZ: 41.246343, 21.977356, 2.573102, D50 Lab: ...") == pytest.approx(
        (41.246343, 21.977356, 2.573102)
    )
    assert argyllmod.parse_xyz("XYZ: 1.0 2.0 3.0") == pytest.approx((1.0, 2.0, 3.0))


def test_argyll_parse_xyz_raises_when_absent():
    with pytest.raises(ValueError):
        argyllmod.parse_xyz("no reading here")


def test_argyll_backend_measure_with_fake_spotread(fake_bin):
    fake_bin("spotread", "print('Result is XYZ: 95.05, 100.00, 108.90, D50 Lab: 100 0 0')\n")
    backend = argyllmod.ArgyllBackend()
    results = backend.measure(patchesmod.primaries_secondaries()[:2])
    assert len(results) == 2
    assert results[0].xyz == pytest.approx(np.array([95.05, 100.00, 108.90]))


def test_argyll_backend_missing_spotread_raises(monkeypatch):
    from calsuite import tools

    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    backend = argyllmod.ArgyllBackend()
    with pytest.raises(tools.ToolError):
        backend.measure_one()


def test_argyll_backend_accuracy_reports_estimate():
    backend = argyllmod.ArgyllBackend(cross_checked_against="camera")
    acc = backend.accuracy()
    assert acc.de00_estimate > 0
    assert acc.cross_checked_against == "camera"


# -- spectro -------------------------------------------------------------


def test_spectrum_to_xyz_of_d65_like_flat_spectrum_is_near_white():
    wavelengths = np.arange(380, 731, 5)
    values = np.ones_like(wavelengths, dtype=np.float64)
    xyz = spectromod.spectrum_to_xyz(wavelengths, values)
    x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
    # An equal-energy spectrum's chromaticity is a fixed, well-known point
    # (roughly (0.333, 0.333) with the 1931 2-degree CMFs) -- a useful
    # sanity check that the integration isn't wildly wrong, independent of
    # any particular display's primaries.
    assert x == pytest.approx(1 / 3, abs=0.03)
    assert y == pytest.approx(1 / 3, abs=0.03)


def test_spectrum_to_xyz_is_order_independent_of_wavelength_sorting():
    # Some spectrophotometer export tools write wavelength columns in
    # descending order (830 -> 360). numpy.interp silently assumes its xp
    # argument is ascending; fed a descending array it produces garbage
    # (verified: all-zero XYZ for a real, non-zero spectrum) rather than
    # raising, which would otherwise either quietly zero out a real
    # reading or (for a partially-sorted/jittered wavelength column)
    # produce a numerically wrong-but-plausible XYZ.
    wavelengths = np.arange(380, 731, 5)
    values = np.ones_like(wavelengths, dtype=np.float64)
    xyz_ascending = spectromod.spectrum_to_xyz(wavelengths, values)
    xyz_descending = spectromod.spectrum_to_xyz(wavelengths[::-1], values[::-1])
    assert xyz_descending == pytest.approx(xyz_ascending)
    assert xyz_ascending[1] > 0


def test_read_spectrum_csv_round_trip(tmp_path):
    path = tmp_path / "spec.csv"
    path.write_text("wavelength,value\n400,0.1\n500,0.5\n600,0.9\n", encoding="utf-8")
    wavelengths, values = spectromod.read_spectrum_csv(path)
    assert list(wavelengths) == [400.0, 500.0, 600.0]
    assert list(values) == [0.1, 0.5, 0.9]


def test_spectro_backend_measure(tmp_path):
    path = tmp_path / "white.csv"
    path.write_text("400,1\n500,1\n600,1\n700,1\n", encoding="utf-8")
    backend = spectromod.SpectroBackend(spectra_for_patch=lambda patch: path, luminance_scale=100.0)
    patch = patchesmod.Patch(rgb=(1.0, 1.0, 1.0), label="w")
    result = backend.measure([patch])
    assert len(result) == 1
    assert result[0].xyz[1] > 0


# -- camera ----------------------------------------------------------------


def test_camera_backend_raises_without_camera_color_record(tmp_path):
    store = storemod.Store(tmp_path)
    with pytest.raises(cameramod.NoCameraColorRecord):
        cameramod.load_matrix_raw_to_xyz(store, "canon-eos-r100-unknown")


def test_camera_backend_reads_matrix_from_exportable_record(tmp_path):
    store = storemod.Store(tmp_path)
    from calsuite.fit import Analysis

    analysis = Analysis(result={"matrix_raw_to_xyz": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    record = storemod.Record.from_analysis(
        kind="camera.color",
        device={"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-unknown"},
        analysis=analysis,
        provenance="measured",
        method={"name": "test", "calsuite_version": "0", "params": {}},
    )
    store.save(record)
    matrix = cameramod.load_matrix_raw_to_xyz(store, "canon-eos-r100-unknown")
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, np.eye(3))


def test_camera_backend_refuses_nonexportable_record(tmp_path):
    store = storemod.Store(tmp_path)
    from calsuite.fit import Analysis, Refusal

    analysis = Analysis(refusals=[Refusal("x", "bad chart")])
    record = storemod.Record.from_analysis(
        kind="camera.color",
        device={"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-unknown"},
        analysis=analysis,
        provenance="measured",
        method={"name": "test", "calsuite_version": "0", "params": {}},
    )
    store.save(record)
    with pytest.raises(cameramod.NoCameraColorRecord):
        cameramod.load_matrix_raw_to_xyz(store, "canon-eos-r100-unknown")


def test_frame_to_raw_rgb_black_subtracted_and_exposure_normalized():
    from calsuite.raw import FrameMeta, RawFrame

    # A tiny synthetic RGGB frame: uniform per-channel DN values, exposure
    # 2s, black level 100 -- frame_to_raw_rgb should subtract black and
    # divide by exposure_s.
    cfa = np.zeros((8, 8), dtype=np.uint16)
    cfa[0::2, 0::2] = 300  # R
    cfa[0::2, 1::2] = 200  # G1
    cfa[1::2, 0::2] = 220  # G2
    cfa[1::2, 1::2] = 500  # B
    frame = RawFrame(
        cfa=cfa,
        pattern="RGGB",
        visible=(slice(0, 8), slice(0, 8)),
        black_level=(100.0, 100.0, 100.0, 100.0),
        white_level=1023.0,
        meta=FrameMeta(exposure_s=2.0),
        path="<test>",
        sha256="",
    )
    r, g, b = cameramod.frame_to_raw_rgb(frame)
    assert r == pytest.approx((300 - 100) / 2.0)
    assert g == pytest.approx(((200 - 100) + (220 - 100)) / 2.0 / 2.0)
    assert b == pytest.approx((500 - 100) / 2.0)


def test_frame_to_raw_rgb_uses_per_channel_black_not_a_flat_mean():
    """raw.py's own convention (and CLAUDE.md's documented fix in
    camera/bias.py, camera/chart.py, camera/color.py) is that
    ``RawFrame.black_level``'s 4 values must be mapped to R/G1/G2/B via
    ``raw.black_level_by_channel`` before use -- a plain ``mean(black_level)``
    silently biases every channel whenever the sensor's black level isn't
    identical across all 4 tile positions (real on some CMOS designs, per
    that function's own docstring). ``frame_to_raw_rgb`` was the one
    remaining call site still doing the flat-mean version, which biases
    every camera-backend display measurement -- the exact "plausible-
    looking wrong number" this suite exists to prevent, since the result
    is still a normal-looking float, just quietly off by the difference
    between a channel's own black level and the 4-way average.
    """
    from calsuite.raw import FrameMeta, RawFrame, black_level_by_channel

    cfa = np.zeros((8, 8), dtype=np.uint16)
    cfa[0::2, 0::2] = 300  # R
    cfa[0::2, 1::2] = 250  # G1
    cfa[1::2, 0::2] = 250  # G2
    cfa[1::2, 1::2] = 600  # B
    # Distinct per-channel black levels, in LibRaw's own cblack[0..3] order
    # -- by *color index* (R, first-G, B, second-G), not by raster position;
    # visible starts at (0, 0), so the visible tile is the absolute tile.
    frame = RawFrame(
        cfa=cfa,
        pattern="RGGB",
        visible=(slice(0, 8), slice(0, 8)),
        black_level=(100.0, 120.0, 90.0, 400.0),
        white_level=1023.0,
        meta=FrameMeta(exposure_s=2.0),
        path="<test>",
        sha256="",
    )
    black = black_level_by_channel(frame)
    assert black == {"R": 100.0, "G1": 120.0, "B": 90.0, "G2": 400.0}

    r, g, b = cameramod.frame_to_raw_rgb(frame)
    assert r == pytest.approx((300 - black["R"]) / 2.0)
    assert g == pytest.approx(((250 - black["G1"]) + (250 - black["G2"])) / 2.0 / 2.0)
    assert b == pytest.approx((600 - black["B"]) / 2.0)

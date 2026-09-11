import numpy as np
import pytest

from calsuite import provenance as prov
from calsuite import store
from calsuite.camera import dcp
from calsuite.fit import Analysis
from calsuite.formats import icc as iccmod
from calsuite.formats.icc import ICC_PCS_ILLUMINANT_D50

MATRIX = np.array(
    [
        [0.55, 0.22, 0.18],
        [0.18, 0.70, 0.12],
        [0.12, 0.18, 0.60],
    ]
)
RAW_WHITE = np.array([3200.0, 4500.0, 2800.0])


def test_build_forward_and_color_matrices_dng_normalization():
    color_matrix, forward_matrix = dcp.build_forward_and_color_matrices(MATRIX, RAW_WHITE)
    predicted_white = forward_matrix @ np.array([1.0, 1.0, 1.0])
    assert predicted_white == pytest.approx(ICC_PCS_ILLUMINANT_D50, abs=1e-9)

    # ColorMatrix's own stated normalization: the D50 white maps back to a
    # unit-neutral raw response.
    neutral = color_matrix @ np.array(ICC_PCS_ILLUMINANT_D50)
    assert neutral == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)


def test_dcp_round_trip_single_illuminant(tmp_path):
    color_matrix, forward_matrix = dcp.build_forward_and_color_matrices(MATRIX, RAW_WHITE)
    path = tmp_path / "test.dcp"
    dcp.write_dcp(
        path,
        unique_camera_model="calsuite-test-camera",
        profile_name="calsuite test profile",
        color_matrix1=color_matrix,
        calibration_illuminant1=dcp.ILLUMINANT_CODES["D65"],
        forward_matrix1=forward_matrix,
    )
    profile = dcp.read_dcp(path)
    assert profile.unique_camera_model == "calsuite-test-camera"
    assert profile.profile_name == "calsuite test profile"
    assert profile.calibration_illuminant1 == 21
    assert profile.color_matrix1 == pytest.approx(color_matrix, abs=1e-5)
    assert profile.forward_matrix1 == pytest.approx(forward_matrix, abs=1e-5)
    assert profile.calibration_illuminant2 is None
    assert profile.color_matrix2 is None


def test_dcp_round_trip_dual_illuminant():
    color_matrix1, forward_matrix1 = dcp.build_forward_and_color_matrices(MATRIX, RAW_WHITE)
    matrix_a = MATRIX * 1.1  # a distinct "tungsten" fit, just needs to be a different invertible matrix
    color_matrix2, forward_matrix2 = dcp.build_forward_and_color_matrices(matrix_a, RAW_WHITE * 0.8)

    data = dcp.write_dcp(
        None,
        unique_camera_model="calsuite-test-camera",
        profile_name="dual illuminant",
        color_matrix1=color_matrix1,
        calibration_illuminant1=dcp.ILLUMINANT_CODES["D65"],
        forward_matrix1=forward_matrix1,
        color_matrix2=color_matrix2,
        calibration_illuminant2=dcp.ILLUMINANT_CODES["A"],
        forward_matrix2=forward_matrix2,
    )
    profile = dcp.read_dcp(data)
    assert profile.calibration_illuminant1 == 21
    assert profile.calibration_illuminant2 == 17
    assert profile.color_matrix2 == pytest.approx(color_matrix2, abs=1e-5)
    assert profile.forward_matrix2 == pytest.approx(forward_matrix2, abs=1e-5)


def test_dcp_magic_bytes():
    data = dcp.write_dcp(
        None,
        unique_camera_model="x",
        profile_name="y",
        color_matrix1=np.eye(3),
        calibration_illuminant1=21,
        forward_matrix1=np.eye(3),
    )
    assert data[:4] == b"IIRC"


def test_dcp_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.dcp"
    path.write_bytes(b"MM\x00\x2a" + b"\x00" * 20)
    with pytest.raises(ValueError):
        dcp.read_dcp(path)


def test_icc_colorant_matrix_matches_pillow_prediction(tmp_path):
    ImageCms = pytest.importorskip("PIL.ImageCms")
    from PIL import Image

    illuminant_xy = (0.3127, 0.3290)  # D65
    colorant_matrix = dcp.icc_colorant_matrix(MATRIX, illuminant_xy)

    path = tmp_path / "camera.icc"
    iccmod.write_profile(
        path, device_class="scnr", description="calsuite test camera", matrix=colorant_matrix, trc=1.0
    )
    profile = iccmod.read_profile(path)
    assert profile.matrix == pytest.approx(colorant_matrix, abs=2e-5)

    # Independent-parser check (littleCMS via Pillow): a raw RGB run through
    # this profile's own PCS transform should land close to what our matrix
    # predicts directly (XYZ = colorant_matrix @ raw_rgb, then to Lab) --
    # no extra chromatic adaptation needed on our side, because
    # ``icc_colorant_matrix`` already targets the ICC PCS's own D50 white,
    # matching test_formats_icc.py's style of cross-checking against Pillow
    # rather than only this project's own reader.
    src_profile = ImageCms.ImageCmsProfile(str(path))
    lab_profile = ImageCms.createProfile("LAB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        src_profile, lab_profile, "RGB", "LAB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
    )
    raw_rgb = np.array([0.5, 0.3, 0.2])
    swatch = Image.new("RGB", (1, 1), tuple(int(round(v * 255)) for v in raw_rgb))
    out = ImageCms.applyTransform(swatch, transform)
    # Pillow/littleCMS's 8-bit "LAB" mode encodes L in [0, 255] -> [0, 100]
    # and a*/b* in [0, 255] -> [-128, 127] (byte - 128) -- verified
    # empirically against colour.sRGB_to_XYZ+Bradford-D50 on a known sRGB
    # swatch while writing this test (agreement to ~0.2 Lab units).
    l_byte, a_byte, b_byte = out.getpixel((0, 0))
    lab_pillow = np.array([l_byte * 100.0 / 255.0, a_byte - 128.0, b_byte - 128.0])

    import colour

    d50_xy = (0.3457, 0.3585)  # CIE 1931 2-degree D50 chromaticity, matches ICC_PCS_ILLUMINANT_D50's XYZ
    xyz_predicted = colorant_matrix @ raw_rgb
    lab_predicted = colour.XYZ_to_Lab(xyz_predicted, illuminant=d50_xy)
    # Loose tolerance: littleCMS's 8-bit LAB quantization plus its own
    # rendering-intent/gamut handling are not bit-identical to a bare
    # matrix multiply, only close for an in-gamut color.
    assert lab_pillow == pytest.approx(lab_predicted, abs=3.0)


def test_export_refuses_a_refused_record(tmp_path):
    analysis = Analysis()
    analysis.refuse("glare", "synthetic refusal for this test")
    analysis.result = {"matrix_raw_to_xyz": MATRIX.tolist(), "white_patch_raw_rgb": RAW_WHITE.tolist()}
    record = store.Record.from_analysis(
        kind="camera.color",
        device={"kind": "camera", "model": "x", "id": "x-unknown"},
        analysis=analysis,
        provenance="derived",
        method={"name": "camera.color.fit", "calsuite_version": "0.1.0", "params": {}},
    )
    assert record.status == "refused"
    with pytest.raises(store.ExportRefused):
        store.require_exportable(record)


def test_export_refuses_weak_provenance():
    analysis = Analysis()
    analysis.result = {"matrix_raw_to_xyz": MATRIX.tolist(), "white_patch_raw_rgb": RAW_WHITE.tolist()}
    record = store.Record.from_analysis(
        kind="camera.color",
        device={"kind": "camera", "model": "x", "id": "x-unknown"},
        analysis=analysis,
        provenance="nominal",
        method={"name": "camera.color.fit", "calsuite_version": "0.1.0", "params": {}},
    )
    assert not prov.is_exportable(record.provenance)
    with pytest.raises(store.ExportRefused):
        store.require_exportable(record)

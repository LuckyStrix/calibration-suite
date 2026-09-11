import numpy as np
import pytest

from calsuite.formats import icc

# Standard Bradford D65->D50 chromatic adaptation matrix, as embedded in
# common sRGB ICC profiles and documented in the ICC spec (Annex D). Used
# only in this test, to build a D50-adapted sRGB matrix to round-trip
# against Pillow's own built-in sRGB profile -- formats/icc.py itself is
# generic and takes an already-D50-adapted matrix from its caller.
BRADFORD_D65_TO_D50 = np.array(
    [
        [1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092345, 0.0150436, 0.7521316],
    ]
)

# sRGB primaries + D65 white point (IEC 61966-2-1).
_SRGB_XY = {"r": (0.6400, 0.3300), "g": (0.3000, 0.6000), "b": (0.1500, 0.0600), "w": (0.3127, 0.3290)}


def _xy_to_xyz_unit_y(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _srgb_matrix_d50():
    """RGB(linear)->XYZ(D50) for sRGB primaries: primary xy -> unit-Y XYZ,
    scale each column so R+G+B sums to the white point, then Bradford-adapt
    D65->D50 -- the standard construction for a matrix-based ICC profile."""
    r_xyz = _xy_to_xyz_unit_y(*_SRGB_XY["r"])
    g_xyz = _xy_to_xyz_unit_y(*_SRGB_XY["g"])
    b_xyz = _xy_to_xyz_unit_y(*_SRGB_XY["b"])
    w_xyz = _xy_to_xyz_unit_y(*_SRGB_XY["w"])
    primaries = np.column_stack([r_xyz, g_xyz, b_xyz])
    scale = np.linalg.solve(primaries, w_xyz)
    matrix_d65 = primaries * scale
    return BRADFORD_D65_TO_D50 @ matrix_d65


def test_write_read_round_trip_matrix_and_gamma(tmp_path):
    matrix = _srgb_matrix_d50()
    path = tmp_path / "test.icc"
    icc.write_profile(path, device_class="scnr", description="calsuite test profile", matrix=matrix, trc=2.2)

    profile = icc.read_profile(path)
    assert profile.device_class == "scnr"
    assert profile.data_colour_space == "RGB"
    assert profile.pcs == "XYZ"
    assert profile.description == "calsuite test profile"
    assert profile.matrix == pytest.approx(matrix, abs=2e-5)
    assert profile.trc["r"] == pytest.approx(2.2, abs=0.01)
    assert profile.trc["g"] == pytest.approx(2.2, abs=0.01)
    assert profile.trc["b"] == pytest.approx(2.2, abs=0.01)
    assert profile.white_xyz == pytest.approx(tuple(matrix.sum(axis=1)), abs=2e-5)


def test_write_read_round_trip_curve_table(tmp_path):
    matrix = _srgb_matrix_d50()
    curve = np.linspace(0, 1, 32) ** 2.4  # a sampled TRC, not a plain gamma value
    path = tmp_path / "test_table.icc"
    icc.write_profile(path, device_class="mntr", description="table trc", matrix=matrix, trc=curve)

    profile = icc.read_profile(path)
    assert profile.device_class == "mntr"
    assert isinstance(profile.trc["r"], np.ndarray)
    assert len(profile.trc["r"]) == 32
    assert profile.trc["r"] == pytest.approx(curve, abs=1e-4)


def test_per_channel_trc_dict(tmp_path):
    matrix = _srgb_matrix_d50()
    path = tmp_path / "per_channel.icc"
    icc.write_profile(path, device_class="scnr", description="x", matrix=matrix, trc={"r": 2.0, "g": 2.2, "b": 2.4})
    profile = icc.read_profile(path)
    assert profile.trc["r"] == pytest.approx(2.0, abs=0.01)
    assert profile.trc["g"] == pytest.approx(2.2, abs=0.01)
    assert profile.trc["b"] == pytest.approx(2.4, abs=0.01)


def test_rejects_bad_device_class():
    with pytest.raises(ValueError):
        icc.write_profile(None, device_class="bogus", description="x", matrix=np.eye(3), trc=2.2)


def test_rejects_non_3x3_matrix():
    with pytest.raises(ValueError):
        icc.write_profile(None, device_class="scnr", description="x", matrix=np.eye(4), trc=2.2)


def test_read_profile_accepts_raw_bytes():
    matrix = _srgb_matrix_d50()
    data = icc.write_profile(None, device_class="scnr", description="bytes-in-memory", matrix=matrix, trc=2.2)
    profile = icc.read_profile(data)
    assert profile.description == "bytes-in-memory"


def test_pillow_opens_profile_and_transform_matches_matrix(tmp_path):
    ImageCms = pytest.importorskip("PIL.ImageCms")
    from PIL import Image

    matrix = _srgb_matrix_d50()
    path = tmp_path / "srgb_like.icc"
    icc.write_profile(path, device_class="scnr", description="srgb-like", matrix=matrix, trc=2.2)

    # Pillow (via littleCMS) accepting the file at all is the point of this
    # test -- an independent parser succeeding, not just our own reader.
    src_profile = ImageCms.ImageCmsProfile(str(path))
    srgb_profile = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransformFromOpenProfiles(
        src_profile,
        srgb_profile,
        "RGB",
        "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    # A profile built from sRGB's own primaries/gamma, converted through
    # itself to Pillow's *built-in* sRGB profile under relative colorimetric
    # intent, should land close to identity -- if the matrix or gamma were
    # wrong, this would visibly drift by more than 8-bit round-trip error.
    swatch = Image.new("RGB", (2, 2), (200, 100, 50))
    out = ImageCms.applyTransform(swatch, transform)
    got = np.array(out)[0, 0].astype(np.int64)
    assert got == pytest.approx(np.array([200, 100, 50]), abs=6)

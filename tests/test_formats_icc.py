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


def test_pillow_lab_transform_matches_our_own_matrix_and_gamma_maths(tmp_path):
    """A second, more targeted independent-implementation cross-check
    (task item 4): the round trip above uses sRGB's own primaries, so a
    bug that swaps rows/columns of the matrix or double-applies a
    chromatic adaptation could still land close to identity by symmetry
    and go unnoticed. This uses a deliberately asymmetric matrix instead
    (not derived from any real primary set): compute our own predicted
    Lab for a set of RGB values from the *exact* matrix/gamma numbers
    written into the file (plain numpy + ``colour.XYZ_to_Lab``), then ask
    littleCMS (via Pillow) to map that Lab back through the *same* file to
    RGB, and check we get back the RGB we started with -- if
    ``write_profile``'s rXYZ/gXYZ/bXYZ/TRC tag bytes don't encode exactly
    what was asked, this round trip won't close.

    Deliberately goes LAB->RGB, not RGB->LAB: ``display/validate.py``'s own
    ``lab_to_rgb_via_profile`` already establishes (and documents, having
    verified it empirically) Pillow's "LAB" mode *input* packing convention
    (L byte = L*/100*255, a/b byte = value+128) for building a LAB image to
    feed into a transform. The *output* byte packing of an RGB->LAB
    transform turned out, while writing this test, to not follow that same
    "+128" convention consistently (some paths behave like a signed byte
    instead) -- rather than ship a test built on a guessed convention for
    a code path this suite doesn't otherwise use, this reuses the one
    input convention that's actually verified.
    """
    ImageCms = pytest.importorskip("PIL.ImageCms")
    from PIL import Image

    import colour

    # Not derived from any real primary set -- deliberately asymmetric so a
    # row/column-order bug shows up as a large error, not a rounding one.
    matrix_d50 = np.array(
        [
            [0.55, 0.18, 0.20],
            [0.25, 0.70, 0.06],
            [0.02, 0.10, 0.92],
        ]
    )
    gamma = 2.4
    path = tmp_path / "asymmetric.icc"
    icc.write_profile(path, device_class="scnr", description="asymmetric-test", matrix=matrix_d50, trc=gamma)

    # Excludes near-black: gamma=2.4 makes low RGB values steep enough that
    # 8-bit Lab quantization on the way in amplifies into several counts of
    # RGB error on the way out -- a real (explainable) rounding effect, not
    # what this test is checking for, so kept out of the asserted set.
    rgb_values = np.array([(0.8, 0.3, 0.1), (0.2, 0.9, 0.4), (0.5, 0.5, 0.9), (0.7, 0.7, 0.7)])
    xyz_d50 = rgb_values**gamma @ matrix_d50.T  # the exact matrix/TRC model formats/icc.py's docstring describes
    d50_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    our_lab = np.array([colour.XYZ_to_Lab(xyz, illuminant=d50_xy) for xyz in xyz_d50])

    lab_profile = ImageCms.createProfile("LAB", colorTemp=5000)  # D50-referenced, matching the ICC PCS
    dst_profile = ImageCms.ImageCmsProfile(str(path))
    transform = ImageCms.buildTransformFromOpenProfiles(
        lab_profile, dst_profile, "LAB", "RGB", renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC
    )
    recovered_rgb = []
    for l_star, a_star, b_star in our_lab:
        px = (
            max(0, min(255, round(l_star / 100.0 * 255))),
            max(0, min(255, round(a_star + 128))),
            max(0, min(255, round(b_star + 128))),
        )
        swatch = Image.new("LAB", (1, 1), px)
        recovered_rgb.append(ImageCms.applyTransform(swatch, transform).getpixel((0, 0)))
    recovered_rgb = np.array(recovered_rgb, dtype=np.float64)

    expected_rgb = np.round(rgb_values * 255.0)
    # Tolerance covers 8-bit quantization at each of the two hops (RGB->Lab
    # by our own maths, Lab->RGB by littleCMS reading the file) -- a real
    # encoding bug (wrong axis, missing/duplicated adaptation, wrong scale)
    # showed up as tens-to-hundreds of RGB levels during development.
    assert recovered_rgb == pytest.approx(expected_rgb, abs=2.0)

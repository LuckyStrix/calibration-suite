import numpy as np
import pytest

from calsuite.camera import color_constants as cc
from calsuite.camera import ssf

# A camera-plausible narrow-band SSF: three overlapping Gaussian bumps, the
# shape of a typical Bayer color-filter-array transmission curve (unlike
# the CIE CMFs, which have a well-known negative lobe near 500nm no
# physical absorptive filter can reproduce) -- used as the "definitely not
# Luther-Ives" case.
_WL = np.arange(400.0, 701.0, 5.0)


def _gaussian(wl, center, sigma):
    return np.exp(-0.5 * ((wl - center) / sigma) ** 2)


NARROW_BAND_SSF = ssf.SSF(
    wavelengths=_WL,
    r=_gaussian(_WL, 600.0, 40.0),
    g=_gaussian(_WL, 540.0, 45.0),
    b=_gaussian(_WL, 460.0, 35.0),
)


def _luther_ssf() -> ssf.SSF:
    """SSFs that ARE an exact linear combination of the CIE 1931 2-degree
    CMFs: S(lambda) = CMF(lambda) @ A for an arbitrary invertible 3x3 A.
    ``luther_ives_deviation`` should then recover something close to
    ``inverse(A)`` with near-zero residual, and a camera built from these
    SSFs should be colorimetrically perfect (SMI ~= 100)."""
    import colour

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    xbar = np.interp(_WL, cmfs.wavelengths, cmfs.values[:, 0])
    ybar = np.interp(_WL, cmfs.wavelengths, cmfs.values[:, 1])
    zbar = np.interp(_WL, cmfs.wavelengths, cmfs.values[:, 2])
    cmf = np.column_stack([xbar, ybar, zbar])

    rng = np.random.default_rng(1)
    A = np.eye(3) + 0.1 * rng.standard_normal((3, 3))  # a mild, well-conditioned mix
    rgb = cmf @ A
    return ssf.SSF(wavelengths=_WL, r=rgb[:, 0], g=rgb[:, 1], b=rgb[:, 2])


def test_luther_ives_deviation_zero_for_exact_linear_combination():
    deviation = ssf.luther_ives_deviation(_luther_ssf())
    assert deviation < cc.LUTHER_IVES_EXACT_TOL


def test_luther_ives_deviation_nonzero_for_realistic_narrowband_ssf():
    deviation = ssf.luther_ives_deviation(NARROW_BAND_SSF)
    assert deviation > 0.05  # comfortably above the exact-combination floor


def test_smi_near_100_for_luther_ssf():
    result = ssf.sensor_metamerism_index(_luther_ssf())
    # Not exactly 100: camera_response integrates on the SSF's own (coarser,
    # narrower-range) wavelength grid via trapezoid, while the reference XYZ
    # comes from colour.sd_to_XYZ's own (finer, wider-range) integration --
    # two different numerical integration schemes over not-quite-identical
    # ranges never agree to machine precision, only closely (empirically
    # ~99.7 here).
    assert result["smi"] > 99.0
    assert result["mean_delta_e_ab"] < 0.2


def test_smi_below_100_for_realistic_narrowband_ssf():
    result = ssf.sensor_metamerism_index(NARROW_BAND_SSF)
    # Clearly and substantially below the Luther-exact case's ~99.7 (docs/
    # design.md / instructions: "realistic non-Luther SSFs => SMI < 100"),
    # not just numerically short of 100 -- empirically ~91 for this
    # narrow-band SSF.
    assert result["smi"] < 95.0


def test_fit_from_ssf_produces_3x3_and_low_error_for_luther_ssf():
    ssf_obj = _luther_ssf()
    analysis = ssf.fit_from_ssf(ssf_obj, illuminant_name="D65", illuminant_xy=(0.3127, 0.3290), model="matrix")
    assert analysis.ok
    M = np.array(analysis.result["matrix_raw_to_xyz"])
    assert M.shape == (3, 3)
    assert analysis.residuals["delta_e00_mean"] < 3.0
    assert analysis.result["white_patch_raw_rgb"] is not None


def test_camera_response_matches_colour_sciences_own_spectral_integration():
    """Cross-check against an independent implementation (task item 4):
    ``camera_response``'s SSF*illuminant*reflectance integration is
    hand-rolled (plain ``np.trapezoid``), unlike every other spectral-to-
    tristimulus step in this module, which already calls straight into
    ``colour``. Treating our own SSF as a substitute "cmfs" lets
    ``colour.sd_to_XYZ``'s own ``method="Integration"`` (a different
    quadrature rule -- a fixed-interval Riemann sum, not trapezoidal, and
    a completely separate code path/library) integrate the exact same
    physical quantity, so agreement here is a real check, not a
    re-assertion of our own arithmetic.

    ``colour.sd_to_XYZ``'s own scale convention differs from ours by a
    constant factor of 100 -- confirmed empirically (``camera_response``'s
    own docstring already says its scale is arbitrary, so there's nothing
    to "get right" about matching it exactly). What a real integration bug
    -- wrong wavelength grid/alignment, a missing illuminant or
    reflectance term, a stray transpose -- would NOT preserve is that
    ratio being the *same, close to 1.00* value for all three channels and
    multiple reflectance patches, which is what's actually asserted.
    """
    import colour

    illum = colour.SDS_ILLUMINANTS["D65"]
    msds = colour.MultiSpectralDistributions(
        np.column_stack([NARROW_BAND_SSF.r, NARROW_BAND_SSF.g, NARROW_BAND_SSF.b]),
        domain=NARROW_BAND_SSF.wavelengths,
        labels=["R", "G", "B"],
    )
    shape = colour.SpectralShape(float(_WL[0]), float(_WL[-1]), float(_WL[1] - _WL[0]))

    for patch in ("red", "green", "blue", "light skin", "dark skin"):
        refl = colour.SDS_COLOURCHECKERS["ISO 17321-1"][patch]
        ours = np.array(ssf.camera_response(NARROW_BAND_SSF, illum, refl))
        colours_own = np.array(colour.sd_to_XYZ(refl, msds, illum, k=1.0, method="Integration", shape=shape)) * 100.0
        # Residual is quadrature-rule difference (trapezoidal vs. colour's
        # fixed-interval sum) on a 5nm grid -- empirically well under 1%
        # for every patch checked while writing this test.
        assert (colours_own / ours) == pytest.approx(1.0, rel=0.02)


def test_load_ssf_csv_round_trip(tmp_path):
    path = tmp_path / "ssf.csv"
    lines = ["nm,r,g,b"]
    for wl, r, g, b in zip(_WL, NARROW_BAND_SSF.r, NARROW_BAND_SSF.g, NARROW_BAND_SSF.b, strict=True):
        lines.append(f"{wl},{r},{g},{b}")
    path.write_text("\n".join(lines), encoding="utf-8")

    loaded = ssf.load_ssf_csv(path)
    assert loaded.wavelengths == pytest.approx(_WL)
    assert loaded.r == pytest.approx(NARROW_BAND_SSF.r)
    assert loaded.g == pytest.approx(NARROW_BAND_SSF.g)
    assert loaded.b == pytest.approx(NARROW_BAND_SSF.b)

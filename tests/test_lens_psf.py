import numpy as np
import pytest

from calsuite import raw as rawmod
from calsuite.lens import psf as P
from calsuite.synth import lens as synthlens
from calsuite.synth import sensor as synth_sensor


def _model():
    return synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )


def test_ellipticity_and_orientation_recovered():
    """Stars are rendered at *sensor*-px sigma, but psf.py measures the
    green plane (quarter resolution, one sample per 2x2 tile) -- so the
    recovered sigma is half the sensor-px value the generator was given.
    """
    rng = np.random.default_rng(9)
    true_sigma_major, true_sigma_minor, true_theta = 10.0, 4.0, 40.0
    stars = [
        {"x": 100.0, "y": 80.0, "amplitude": 8.0e4, "sigma_major": true_sigma_major,
         "sigma_minor": true_sigma_minor, "theta_deg": true_theta},
    ]
    frame = synthlens.render_stars(_model(), stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0)
    plane = rawmod.planes(frame)["G1"]

    analysis = P.psf_field(plane)
    assert analysis.ok
    star = analysis.result["stars"][0]

    assert star["sigma_major"] == pytest.approx(true_sigma_major / 2.0, rel=0.1)
    assert star["sigma_minor"] == pytest.approx(true_sigma_minor / 2.0, rel=0.15)
    true_ellipticity = 1.0 - true_sigma_minor / true_sigma_major
    assert star["ellipticity"] == pytest.approx(true_ellipticity, abs=0.1)

    # orientation has a 180-degree axis ambiguity
    orientation_mod180 = star["orientation_deg"] % 180.0
    true_mod180 = true_theta % 180.0
    diff = min(abs(orientation_mod180 - true_mod180), 180.0 - abs(orientation_mod180 - true_mod180))
    assert diff < 10.0


def test_circular_star_has_near_zero_ellipticity_and_is_unclassified_meaningfully():
    rng = np.random.default_rng(10)
    stars = [{"x": 160.0, "y": 120.0, "amplitude": 6.0e4, "sigma_major": 5.0, "sigma_minor": 5.0, "theta_deg": 0.0}]
    frame = synthlens.render_stars(_model(), stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0)
    plane = rawmod.planes(frame)["G1"]
    analysis = P.psf_field(plane)
    assert analysis.ok
    assert analysis.result["stars"][0]["ellipticity"] < 0.1


def test_no_blobs_refusal():
    plane = np.full((100, 120), 500.0) + np.random.default_rng(0).normal(0, 1.0, (100, 120))
    analysis = P.psf_field(plane)
    assert not analysis.ok
    assert analysis.refusals[0].check == "no_blobs_detected"


def test_saturated_star_is_refused_instead_of_reporting_a_biased_fwhm():
    """A fully clipped star's second-moment FWHM/ellipticity are biased
    (measured wide, not just noisy) because the intensity-weighted moment
    spreads over the whole flat-topped clipped core -- ``saturation_dn``
    (the frame's real ceiling) must exclude it, refusing outright since it's
    the only star."""
    model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=40000.0,
    )
    rng = np.random.default_rng(9)
    stars = [{"x": 100.0, "y": 80.0, "amplitude": 5.0e6, "sigma_major": 10.0, "sigma_minor": 4.0, "theta_deg": 40.0}]
    frame = synthlens.render_stars(model, stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0)
    plane = rawmod.planes(frame)["G1"]
    assert plane.max() >= model.white_level  # sanity: this star really is clipped

    unaware = P.psf_field(plane)
    assert unaware.ok  # documents the pre-fix blind spot: no saturation_dn, no refusal
    assert unaware.result["stars"][0]["sigma_major"] > 10.0 / 2.0 * 1.3  # biased >30% wide vs. the true PSF

    aware = P.psf_field(plane, saturation_dn=model.white_level)
    assert not aware.ok
    assert aware.refusals[0].check == "all_blobs_saturated"


def test_partial_saturation_excludes_only_the_saturated_star():
    model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=40000.0,
    )
    rng = np.random.default_rng(11)
    stars = [
        {"x": 80.0, "y": 60.0, "amplitude": 1.5e5, "sigma_major": 5.0, "sigma_minor": 5.0, "theta_deg": 0.0},
        {"x": 220.0, "y": 160.0, "amplitude": 6.0e4, "sigma_major": 5.0, "sigma_minor": 5.0, "theta_deg": 0.0},
    ]
    frame = synthlens.render_stars(model, stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0)
    plane = rawmod.planes(frame)["G1"]

    analysis = P.psf_field(plane, saturation_dn=model.white_level)
    assert analysis.ok
    assert analysis.result["n_saturated_excluded"] == 1
    assert analysis.result["n_stars"] == 1
    assert analysis.result["stars"][0]["x"] == pytest.approx(220.0 / 2.0, abs=2.0)


def test_sagittal_vs_meridional_classification():
    # major axis pointing straight along the radial direction from center -> sagittal
    center = (100.0, 100.0)
    out = P.classify_orientation(orientation_deg=0.0, x=150.0, y=100.0, center=center)
    assert out["orientation"] == "sagittal"
    # major axis perpendicular to the radial direction -> meridional
    out2 = P.classify_orientation(orientation_deg=90.0, x=150.0, y=100.0, center=center)
    assert out2["orientation"] == "meridional"


def test_moments_are_not_biased_by_a_window_much_wider_than_the_blob():
    """The moment sums are r^2-weighted, so a fixed 15 px window around a
    2 px blob gives noise in its far corners ~56x the leverage it has at
    the core, and the window's own median over-estimates the background.
    A truth-2.0/1.0 plane-px star measured 2.05/1.13 -- 13% too round, and
    outside this module's own 15% tolerance on sigma_minor. The window is
    now iterated down to the blob's own size with an annulus background.
    """
    model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    rng = np.random.default_rng(3)
    # sigma in *sensor* px; one plane px spans two of them.
    frame = synthlens.render_stars(
        model,
        [{"x": 160.0, "y": 120.0, "amplitude": 8.0e4, "sigma_major": 4.0, "sigma_minor": 2.0, "theta_deg": 40.0}],
        rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0,
    )
    plane = rawmod.planes(frame)["G1"]

    m = P.moments(plane, 80.0, 60.0)
    assert m["sigma_major"] == pytest.approx(2.0, rel=0.03)
    assert m["sigma_minor"] == pytest.approx(1.0, rel=0.03)
    assert m["ellipticity"] == pytest.approx(0.5, abs=0.03)
    # The window it actually used is reported, and it is far tighter than
    # the PSF_WINDOW_RADIUS_PX default.
    assert m["window_px"] < 15
    assert m["window_truncated"] is False


def test_moments_flag_a_blob_wider_than_the_window():
    """A blob whose wings run past even the full window has sigmas biased
    *low* by the truncation. That is reported rather than left for a reader
    to discover."""
    model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    rng = np.random.default_rng(3)
    frame = synthlens.render_stars(
        model,
        [{"x": 160.0, "y": 120.0, "amplitude": 8.0e4, "sigma_major": 20.0, "sigma_minor": 8.0, "theta_deg": 40.0}],
        rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0,
    )
    m = P.moments(rawmod.planes(frame)["G1"], 80.0, 60.0)
    assert m["window_truncated"] is True
    assert m["sigma_major"] < 10.0  # truth is 10 plane px; truncation reads low

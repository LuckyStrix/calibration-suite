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


def test_sagittal_vs_meridional_classification():
    # major axis pointing straight along the radial direction from center -> sagittal
    center = (100.0, 100.0)
    out = P.classify_orientation(orientation_deg=0.0, x=150.0, y=100.0, center=center)
    assert out["orientation"] == "sagittal"
    # major axis perpendicular to the radial direction -> meridional
    out2 = P.classify_orientation(orientation_deg=90.0, x=150.0, y=100.0, center=center)
    assert out2["orientation"] == "meridional"

import math

import numpy as np
import pytest

from calsuite import raw as rawmod
from calsuite.lens import mtf as M
from calsuite.lens.constants import (
    MTF_EDGE_MAX_ANGLE_DEG,
    MTF_EDGE_MIN_ANGLE_DEG,
    MTF_MIN_ROWS,
    MTF_SATURATION_FRACTION,
)
from calsuite.synth import lens as synthlens
from calsuite.synth import sensor as synth_sensor


def _model(shape=(300, 220)):
    return synth_sensor.SensorModel(
        shape=shape, read_noise_e=0.1, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )


def _plane(angle_deg, sigma_px, rng, **kwargs):
    model = _model()
    frame = synthlens.render_slanted_edge(model, angle_deg, sigma_px, rng=rng, exposure_s=0.05, **kwargs)
    return rawmod.planes(frame)["G1"]


def test_mtf50_within_5pct_of_analytic():
    rng = np.random.default_rng(0)
    sigma_px = 1.5  # sensor px
    plane = _plane(5.0, sigma_px, rng, low_flux_e_per_s=2.0e3, high_flux_e_per_s=8.0e4)
    analysis = M.edge_sfr(plane)
    assert analysis.ok, analysis.refusals

    f_analytic = math.sqrt(math.log(2) / (2 * math.pi**2 * sigma_px**2))  # cycles/sensor-px
    achieved = analysis.result["mtf50_cycles_per_sensor_px"]
    assert achieved == pytest.approx(f_analytic, rel=0.05)


def test_mtf50_lp_per_mm_uses_pixel_pitch():
    from calsuite.lens.constants import R100_PIXEL_PITCH_MM

    rng = np.random.default_rng(1)
    plane = _plane(5.0, 1.5, rng)
    analysis = M.edge_sfr(plane)
    expected = analysis.result["mtf50_cycles_per_sensor_px"] / R100_PIXEL_PITCH_MM
    assert analysis.result["mtf50_lp_per_mm"] == pytest.approx(expected)


def test_refuses_edge_angle_too_shallow():
    rng = np.random.default_rng(2)
    plane = _plane(0.2, 1.5, rng)  # well under MTF_EDGE_MIN_ANGLE_DEG
    analysis = M.edge_sfr(plane)
    assert not analysis.ok
    assert analysis.refusals[0].check == "edge_angle"
    assert MTF_EDGE_MIN_ANGLE_DEG > 0.2


def test_refuses_edge_angle_too_steep():
    rng = np.random.default_rng(3)
    plane = _plane(25.0, 1.5, rng)  # well over MTF_EDGE_MAX_ANGLE_DEG
    analysis = M.edge_sfr(plane)
    assert not analysis.ok
    assert analysis.refusals[0].check == "edge_angle"
    assert MTF_EDGE_MAX_ANGLE_DEG < 25.0


def test_refuses_low_contrast():
    rng = np.random.default_rng(4)
    plane = _plane(5.0, 1.5, rng, low_flux_e_per_s=1.0e4, high_flux_e_per_s=1.05e4)
    analysis = M.edge_sfr(plane)
    assert not analysis.ok
    assert analysis.refusals[0].check == "low_contrast"


def test_refuses_too_few_rows():
    rng = np.random.default_rng(5)
    plane = _plane(5.0, 1.5, rng)[: MTF_MIN_ROWS - 1, :]
    analysis = M.edge_sfr(plane)
    assert not analysis.ok
    assert analysis.refusals[0].check == "too_few_rows"


def test_refuses_saturated_edge_just_over_the_ceiling():
    roi = np.full((MTF_MIN_ROWS + 5, 20), 100.0)
    saturation_dn = 1000.0
    roi[10, 5] = saturation_dn * MTF_SATURATION_FRACTION  # exactly at the refusal boundary
    analysis = M.edge_sfr(roi, saturation_dn=saturation_dn)
    assert not analysis.ok
    assert analysis.refusals[0].check == "saturated"


def test_does_not_refuse_saturated_just_under_the_ceiling():
    roi = np.full((MTF_MIN_ROWS + 5, 20), 100.0)
    saturation_dn = 1000.0
    roi[10, 5] = saturation_dn * MTF_SATURATION_FRACTION - 1.0  # just inside
    analysis = M.edge_sfr(roi, saturation_dn=saturation_dn)
    assert all(r.check != "saturated" for r in analysis.refusals)


def test_clipped_bright_plateau_is_refused_instead_of_reporting_a_biased_mtf50():
    """A slanted edge whose bright plateau (and part of the transition
    itself) is clipped against the sensor's saturation level used to come
    back `ok=True` with a badly *inflated* MTF50 -- clipping the top of the
    erf-shaped transition flattens the LSF, which looks like a sharper
    edge, not a lower-contrast one, so `low_contrast` never caught it.
    ``saturation_dn`` (the frame's real ceiling) must refuse it instead."""
    model = synth_sensor.SensorModel(
        shape=(300, 220), read_noise_e=0.1, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=1500.0,
    )
    rng = np.random.default_rng(7)
    frame = synthlens.render_slanted_edge(
        model, 5.0, 1.2, rng=rng, exposure_s=0.3, low_flux_e_per_s=2.0e3, high_flux_e_per_s=4.0e4
    )
    plane = rawmod.planes(frame)["G1"]
    assert plane.max() >= model.white_level  # sanity: this ROI really is clipped

    unaware = M.edge_sfr(plane)
    assert unaware.ok  # documents the pre-fix blind spot: no saturation_dn, no refusal

    aware = M.edge_sfr(plane, saturation_dn=model.white_level)
    assert not aware.ok
    assert aware.refusals[0].check == "saturated"


def test_field_grid_reports_map_and_per_cell_refusals():
    rng = np.random.default_rng(6)
    model = _model(shape=(300, 500))
    frame = synthlens.render_slanted_edge(model, 5.0, 1.5, rng=rng, exposure_s=0.05)
    plane = rawmod.planes(frame)["G1"]
    analysis = M.mtf_field_grid(plane, grid=(3, 5))
    assert analysis.ok
    grid = analysis.result["mtf50_cycles_per_sensor_px_map"]
    assert len(grid) == 3 and len(grid[0]) == 5
    assert analysis.result["n_cells_ok"] > 0

import numpy as np
import pytest

from calsuite import raw as rawmod
from calsuite.lens import flats as F
from calsuite.synth import lens as synthlens
from calsuite.synth import sensor as synth_sensor

V_TRUE = [-0.35, 0.05]
S_TRUE = [0.0, 0.15, -0.10, -0.08, 0.02, -0.05]  # s0's absolute value is not identifiable -- see below


def _model():
    return synth_sensor.SensorModel(
        shape=(180, 240), read_noise_e=2.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )


def _render(poses, rng):
    model = _model()
    images = []
    for pose in poses:
        frame = synthlens.render_flat_pose(model, V_TRUE, S_TRUE, pose, rng=rng, exposure_s=0.2, base_flux_e_per_s=3.0e4)
        plane = rawmod.planes(frame)["G1"] - model.black_dn
        images.append(np.clip(plane, 1.0, None))
    return images


def test_self_calibrating_flat_recovers_v_and_s_with_shifts():
    rng = np.random.default_rng(3)
    poses = [
        F.Pose(0, 0, 0),
        F.Pose(90, 0.08, 0),
        F.Pose(180, 0, 0.08),
        F.Pose(270, 0.08, 0.08),
        F.Pose(0, -0.06, 0.05),
        F.Pose(90, -0.05, -0.07),
    ]
    images = _render(poses, rng)
    analysis = F.self_calibrate_flat(images, poses)
    assert analysis.ok, analysis.refusals

    v_coeffs = analysis.result["v_coeffs"]
    assert v_coeffs == pytest.approx(V_TRUE, abs=0.03)

    # s0 (the constant term) absorbs the arbitrary overall scale of V*S
    # (base_flux_e_per_s, gain, exposure_s, ...) -- only the *shape* terms
    # (su, sv, suu, suv, svv) are identifiable, per self_calibrate_flat's
    # own docstring on the V(0)=1 normalization convention.
    s_coeffs = analysis.result["s_coeffs"]
    assert s_coeffs[1:] == pytest.approx(S_TRUE[1:], abs=0.03)


def test_degeneracy_refusal_fires_for_rotation_only_poses():
    rng = np.random.default_rng(4)
    poses = [F.Pose(angle_deg=a) for a in (0.0, 90.0, 180.0, 270.0)]
    images = _render(poses, rng)
    analysis = F.self_calibrate_flat(images, poses)
    assert not analysis.ok
    assert analysis.refusals[0].check == "degenerate_poses"


def test_degeneracy_refusal_does_not_fire_when_shifts_are_included():
    rng = np.random.default_rng(5)
    poses = [
        F.Pose(0, 0, 0),
        F.Pose(90, 0.08, 0),
        F.Pose(180, 0, 0.08),
        F.Pose(270, 0.08, 0.08),
    ]
    images = _render(poses, rng)
    analysis = F.self_calibrate_flat(images, poses)
    assert analysis.ok, analysis.refusals


def test_evaluate_v_map_is_one_at_center():
    v_map = F.evaluate_v_map(V_TRUE, (180, 240))
    cy, cx = 89, 119  # near-center pixel
    assert v_map[cy, cx] == pytest.approx(1.0, abs=1e-3)


def test_fit_pa_recovers_the_polynomial_shape():
    pa = F.fit_pa(V_TRUE)
    # V(r) = exp(v2 r^2 + v4 r^4) ~= 1 + v2 r^2 + (v4 + v2^2/2) r^4 + ... for small r
    assert pa["k1"] == pytest.approx(V_TRUE[0], abs=0.05)


def test_relative_t_stop():
    # a lens passing half the light of a known T/2.0 lens at the same
    # f-number has a higher (worse) T-stop, by sqrt(2).
    t_other = F.relative_t_stop(known_t_stop=2.0, known_signal=1.0, other_signal=0.5)
    assert t_other == pytest.approx(2.0 * (2.0**0.5))

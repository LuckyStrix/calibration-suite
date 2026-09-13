import numpy as np
import pytest

from calsuite import raw
from calsuite.synth import sensor as synth_sensor


def test_bias_frame_mean_near_black_level():
    model = synth_sensor.SensorModel(shape=(128, 128), read_noise_e=4.0, black_dn=512.0, gain_e_per_dn=2.0)
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    for arr in raw.planes(frame).values():
        assert arr.mean() == pytest.approx(model.black_dn, abs=2.0)


def test_ptc_variance_matches_shot_plus_read_noise():
    """Classic photon-transfer-curve identity (docs/design.md §3.1):
    Var(A - B) / 2 = signal_e / gain + read_noise_dn^2, for a pair of flats
    at the same mean signal. Generate two flats from a KNOWN model and
    check the identity recovers gain/read-noise within a loose tolerance
    -- the "true 0.30 fits to 0.2997" move (house rule 4)."""
    gain = 2.5
    read_noise_e = 3.0
    black_dn = 512.0
    model = synth_sensor.SensorModel(
        shape=(512, 512),
        gain_e_per_dn=gain,
        read_noise_e=read_noise_e,
        black_dn=black_dn,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(42)
    flux, exposure_s = 20000.0, 0.5
    a = synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
    b = synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)

    pa, pb = raw.planes(a)["R"], raw.planes(b)["R"]
    mean_dn = 0.5 * (pa.mean() + pb.mean())
    var_diff = (pa.astype(np.float64) - pb.astype(np.float64)).var()

    read_noise_dn = read_noise_e / gain
    predicted_var = 2.0 * ((mean_dn - black_dn) / gain + read_noise_dn**2)

    assert var_diff == pytest.approx(predicted_var, rel=0.1)


def test_dark_current_doubles_per_step():
    model = synth_sensor.SensorModel()
    base = model.dark_current(20.0)
    doubled = model.dark_current(20.0 + synth_sensor.DARK_CURRENT_DOUBLING_C)
    assert doubled == pytest.approx(2 * base, rel=1e-9)


def test_optical_black_receives_no_flux_even_when_visible_saturates():
    model = synth_sensor.SensorModel(shape=(64, 96), top_margin=8, left_margin=32, full_well_e=5000.0)
    rng = np.random.default_rng(1)
    frame = synth_sensor.frame(model, exposure_s=1.0, flux_e_per_s=1e6, temp_c=20.0, rng=rng)
    for arr in raw.optical_black(frame).values():
        assert arr.mean() < model.black_dn + 50
    assert raw.planes(frame)["R"].mean() >= frame.white_level - 5


def test_fixed_pattern_maps_reproducible_with_same_seed():
    model_a = synth_sensor.SensorModel(shape=(64, 64), fixed_pattern_seed=7, hot_pixel_fraction=0.02)
    model_b = synth_sensor.SensorModel(shape=(64, 64), fixed_pattern_seed=7, hot_pixel_fraction=0.02)
    prnu_a, dsnu_a, hot_a = synth_sensor._fixed_pattern_maps(model_a)
    prnu_b, dsnu_b, hot_b = synth_sensor._fixed_pattern_maps(model_b)
    assert np.array_equal(hot_a, hot_b)
    assert np.array_equal(prnu_a, prnu_b)
    assert hot_a.sum() > 0  # the fraction chosen should actually produce some hot pixels at this size


def test_fixed_pattern_maps_differ_across_seeds():
    model_a = synth_sensor.SensorModel(shape=(64, 64), fixed_pattern_seed=1)
    model_b = synth_sensor.SensorModel(shape=(64, 64), fixed_pattern_seed=2)
    prnu_a, _, _ = synth_sensor._fixed_pattern_maps(model_a)
    prnu_b, _, _ = synth_sensor._fixed_pattern_maps(model_b)
    assert not np.array_equal(prnu_a, prnu_b)


def test_r100_like_geometry():
    model = synth_sensor.SensorModel.r100_like(gain_e_per_dn=2.0, read_noise_e=3.0)
    assert model.shape == synth_sensor.R100_VISIBLE_SHAPE
    assert model.top_margin == synth_sensor.R100_TOP_MARGIN
    assert model.left_margin == synth_sensor.R100_LEFT_MARGIN
    assert model.gain_e_per_dn == 2.0


def test_row_banding_increases_row_to_row_variance():
    rng1 = np.random.default_rng(3)
    model_plain = synth_sensor.SensorModel(
        shape=(200, 64), row_banding_std_dn=0.0, read_noise_e=1.0, prnu_std=0, dsnu_std_e_per_s=0
    )
    frame_plain = synth_sensor.frame(model_plain, exposure_s=0.01, flux_e_per_s=0.0, temp_c=20.0, rng=rng1)

    rng2 = np.random.default_rng(3)
    model_band = synth_sensor.SensorModel(
        shape=(200, 64), row_banding_std_dn=20.0, read_noise_e=1.0, prnu_std=0, dsnu_std_e_per_s=0
    )
    frame_band = synth_sensor.frame(model_band, exposure_s=0.01, flux_e_per_s=0.0, temp_c=20.0, rng=rng2)

    row_means_plain = raw.planes(frame_plain)["R"].mean(axis=1)
    row_means_band = raw.planes(frame_band)["R"].mean(axis=1)
    assert row_means_band.std() > row_means_plain.std() * 3


def test_frame_clips_at_white_level():
    model = synth_sensor.SensorModel(shape=(32, 32), full_well_e=1000.0, gain_e_per_dn=2.0)
    rng = np.random.default_rng(5)
    frame = synth_sensor.frame(model, exposure_s=10.0, flux_e_per_s=1e9, temp_c=20.0, rng=rng)
    assert frame.cfa.max() <= frame.white_level


# ---------------------------------------------------------------------------
# per-channel black level (black_dn_by_channel)
# ---------------------------------------------------------------------------


def test_black_dn_by_channel_dict_produces_the_requested_per_channel_offsets():
    """The whole point of black_dn_by_channel: raw.planes() should read
    back each CFA channel's own requested black level, not one pooled
    scalar -- house rule 4's round trip, applied to the gap this closes."""
    per_channel = {"R": 500.0, "G1": 510.0, "G2": 520.0, "B": 530.0}
    model = synth_sensor.SensorModel(
        shape=(128, 128),
        black_dn_by_channel=per_channel,
        read_noise_e=0.5,
        prnu_std=0.0,
        dsnu_std_e_per_s=0.0,
        hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    for ch, arr in raw.planes(frame).items():
        assert arr.mean() == pytest.approx(per_channel[ch], abs=0.5)


def test_black_dn_by_channel_tuple_form_matches_dict_form():
    """A 4-tuple is accepted as a shorthand for the (R, G1, G2, B)-ordered
    dict -- both forms must produce the identical frame."""
    per_channel_dict = {"R": 500.0, "G1": 512.0, "G2": 517.0, "B": 525.0}
    per_channel_tuple = (500.0, 512.0, 517.0, 525.0)
    model_dict = synth_sensor.SensorModel(shape=(32, 32), black_dn_by_channel=per_channel_dict)
    model_tuple = synth_sensor.SensorModel(shape=(32, 32), black_dn_by_channel=per_channel_tuple)
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
    frame_a = synth_sensor.frame(model_dict, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng_a)
    frame_b = synth_sensor.frame(model_tuple, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng_b)
    assert frame_a.black_level == frame_b.black_level
    assert np.array_equal(frame_a.cfa, frame_b.cfa)


def test_black_dn_by_channel_populates_raw_black_level_in_absolute_phase_order():
    """`RawFrame.black_level` is positional in the *absolute-origin* raster
    order (raw.black_level_by_channel's docstring), which differs from the
    visible-origin channel naming whenever either margin is odd. Round-trip
    through raw.black_level_by_channel() -- the one sanctioned reader of
    that mapping -- and check it recovers exactly the requested per-channel
    values, for both an even-margin and an odd-margin sensor."""
    per_channel = {"R": 500.0, "G1": 510.0, "G2": 520.0, "B": 530.0}
    for top, left in [(8, 16), (9, 17), (8, 17), (9, 16)]:
        model = synth_sensor.SensorModel(
            shape=(64, 64), top_margin=top, left_margin=left, black_dn_by_channel=per_channel
        )
        rng = np.random.default_rng(2)
        frame = synth_sensor.frame(model, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
        assert raw.black_level_by_channel(frame) == per_channel, f"top={top} left={left}"


def test_black_dn_by_channel_applies_to_optical_black_margin_too():
    """The per-channel offset is a property of the readout amplifier chain,
    which keeps reading out through the masked margin exactly as it does
    through the visible area -- raw.optical_black() should see the same
    per-channel levels as raw.planes(), not the flat scalar."""
    per_channel = {"R": 500.0, "G1": 510.0, "G2": 520.0, "B": 530.0}
    model = synth_sensor.SensorModel(
        shape=(64, 64),
        top_margin=9,
        left_margin=17,
        black_dn_by_channel=per_channel,
        read_noise_e=0.5,
        full_well_e=5000.0,
    )
    rng = np.random.default_rng(3)
    frame = synth_sensor.frame(model, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    for ch, arr in raw.optical_black(frame).items():
        assert arr.mean() == pytest.approx(per_channel[ch], abs=1.0)


def test_black_dn_by_channel_none_is_byte_for_byte_identical_to_scalar_black_dn():
    """The existing scalar `black_dn` knob must keep working unchanged --
    the default (`black_dn_by_channel=None`) is not just approximately the
    same as before, it produces the exact same frame."""
    kwargs = dict(shape=(48, 48), gain_e_per_dn=2.2, read_noise_e=2.0, black_dn=512.0, prnu_std=0.02)
    model_a = synth_sensor.SensorModel(**kwargs)
    model_b = synth_sensor.SensorModel(**kwargs, black_dn_by_channel=None)
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    frame_a = synth_sensor.frame(model_a, exposure_s=1.0, flux_e_per_s=100.0, temp_c=20.0, rng=rng_a)
    frame_b = synth_sensor.frame(model_b, exposure_s=1.0, flux_e_per_s=100.0, temp_c=20.0, rng=rng_b)
    assert frame_a.black_level == frame_b.black_level == (512.0, 512.0, 512.0, 512.0)
    assert np.array_equal(frame_a.cfa, frame_b.cfa)


def test_black_dn_by_channel_dict_rejects_wrong_keys():
    model = synth_sensor.SensorModel(black_dn_by_channel={"R": 1.0, "G1": 2.0, "G2": 3.0})
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        synth_sensor.frame(model, exposure_s=1e-6, flux_e_per_s=0.0, temp_c=20.0, rng=rng)

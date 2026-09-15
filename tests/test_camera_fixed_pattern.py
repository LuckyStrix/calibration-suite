from dataclasses import replace

import numpy as np
import pytest

from calsuite.camera import fixed_pattern
from calsuite.synth import sensor as synth_sensor


def _frames(model, exposure_s, flux, temp_c, n, seed):
    rng = np.random.default_rng(seed)
    return [synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=temp_c, rng=rng) for _ in range(n)]


def _add_periodic_row_band(frame, period_rows: int, amplitude_dn: float):
    """Add a fixed (same every frame, unlike synth.sensor's own
    ``row_banding_std_dn`` -- see that constructor arg's docstring: it's
    deliberately a *temporal*, frame-to-frame-independent artifact, so
    averaging several frames cancels it out exactly like it should cancel
    ordinary read noise) periodic row offset directly onto a frame's CFA
    array, for a test vehicle that a per-frame FFT peak detector can
    actually find."""
    rows = np.arange(frame.cfa.shape[0])
    band = amplitude_dn * np.sin(2 * np.pi * rows / period_rows)
    cfa = frame.cfa.astype(np.float64) + band[:, None]
    return replace(frame, cfa=np.clip(cfa, 0, 65535).astype(np.uint16))


def test_prnu_std_recovered_near_injected_value():
    prnu_std = 0.03
    model = synth_sensor.SensorModel(
        shape=(128, 128), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=2.0,
        prnu_std=prnu_std, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    a = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 6, seed=0),
        biases=_frames(model, 1e-4, 0.0, 20.0, 4, seed=2),
        black_dn=512.0,
    )
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        assert abs(data["prnu_std_pct"] - prnu_std * 100.0) < prnu_std * 100.0 * 0.5


def test_row_banding_detected_when_present():
    model = synth_sensor.SensorModel(
        shape=(128, 64), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=1.0, prnu_std=0.0,
        dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    plain_biases = _frames(model, 1e-4, 0.0, 20.0, 6, seed=3)
    banded_biases = [_add_periodic_row_band(f, period_rows=16, amplitude_dn=15.0) for f in plain_biases]

    a_band = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=banded_biases,
        black_dn=512.0,
    )
    a_plain = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=plain_biases,
        black_dn=512.0,
    )
    for ch in fixed_pattern.CHANNELS:
        assert (
            a_band.result["channels"][ch]["row_banding_peak_ratio"]
            > a_plain.result["channels"][ch]["row_banding_peak_ratio"]
        )


def _add_periodic_col_band(frame, period_cols: int, amplitude_dn: float):
    """The column-axis mirror of ``_add_periodic_row_band`` above -- an
    artifact that repeats across columns (constant down each column)."""
    cols = np.arange(frame.cfa.shape[1])
    band = amplitude_dn * np.sin(2 * np.pi * cols / period_cols)
    cfa = frame.cfa.astype(np.float64) + band[None, :]
    return replace(frame, cfa=np.clip(cfa, 0, 65535).astype(np.uint16))


def test_column_banding_detected_as_column_not_row():
    """The sharpest test for row/column confusion in banding_spectrum:
    inject a periodic offset that varies column-to-column (constant down
    a column) and check it's flagged as column banding, and specifically
    NOT as row banding -- the mirror image of
    test_row_banding_detected_when_present above."""
    model = synth_sensor.SensorModel(
        shape=(64, 128), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=1.0, prnu_std=0.0,
        dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    plain_biases = _frames(model, 1e-4, 0.0, 20.0, 6, seed=4)
    banded_biases = [_add_periodic_col_band(f, period_cols=16, amplitude_dn=15.0) for f in plain_biases]

    a_band = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=banded_biases,
        black_dn=512.0,
    )
    a_plain = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 10000.0, 20.0, 4, seed=2),
        biases=plain_biases,
        black_dn=512.0,
    )
    for ch in fixed_pattern.CHANNELS:
        assert (
            a_band.result["channels"][ch]["col_banding_peak_ratio"]
            > a_plain.result["channels"][ch]["col_banding_peak_ratio"]
        )
        assert (
            a_band.result["channels"][ch]["row_banding_peak_ratio"]
            < a_band.result["channels"][ch]["col_banding_peak_ratio"]
        )


def test_prnu_std_recovered_with_per_channel_black_dn():
    """prnu_map's normalization (signal = stacked - black, then divided by
    signal's own mean) is genuinely sensitive to getting each channel's
    black level right -- unlike dsnu's std, which a constant offset can't
    move: subtracting the wrong black shifts the mean it divides by, and
    biases the reported PRNU% (verified manually -- with the per-channel
    truth here, subtracting the flat mean of the 4 values instead throws
    the R channel's recovered PRNU 20%+ off). dsnu_map/prnu_map already
    thread a dict `black_dn` through `_black_for` per channel rather than a
    flat mean, so this pins that a real per-channel black spread still
    recovers the injected PRNU correctly with a fairly tight tolerance (a
    regression guard: no bug found here, see final report). 30 flats
    (rather than the 6 other tests in this file use) average down enough of
    the per-pixel shot noise that the tolerance can actually be tight
    enough to matter."""
    prnu_std = 0.03
    black_dn_by_channel = {"R": 420.0, "G1": 480.0, "G2": 540.0, "B": 600.0}
    model = synth_sensor.SensorModel(
        shape=(256, 256), black_dn_by_channel=black_dn_by_channel, gain_e_per_dn=2.0, read_noise_e=2.0,
        prnu_std=prnu_std, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    a = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 4, seed=1),
        flats=_frames(model, 0.5, 2000.0, 20.0, 30, seed=0),
        biases=_frames(model, 1e-4, 0.0, 20.0, 4, seed=2),
        black_dn=black_dn_by_channel,
    )
    assert a.ok, a.refusals
    for _ch, data in a.result["channels"].items():
        assert abs(data["prnu_std_pct"] - prnu_std * 100.0) < prnu_std * 100.0 * 0.1


def test_refuses_with_too_few_frames():
    model = synth_sensor.SensorModel(shape=(32, 32))
    a = fixed_pattern.analyze_fixed_pattern(
        darks=_frames(model, 5.0, 0.0, 20.0, 1, seed=1),
        flats=_frames(model, 0.5, 5000.0, 20.0, 1, seed=2),
        biases=_frames(model, 1e-4, 0.0, 20.0, 1, seed=3),
        black_dn=512.0,
    )
    assert not a.ok
    assert a.refusals[0].check == "insufficient_frames"


def _darks(dsnu_std_e_per_s, n=3, seed=1):
    model = synth_sensor.SensorModel(
        shape=(128, 128), read_noise_e=3.0, gain_e_per_dn=2.0, black_dn=512.0,
        prnu_std=0.0, dsnu_std_e_per_s=dsnu_std_e_per_s, hot_pixel_fraction=0.0,
        # High enough that the generator's `clip(dark_rate, 0, None)` doesn't
        # truncate the injected spread -- this test is about the analysis,
        # not about the generator's own floor.
        dark_current_e_per_s_at_20c=20.0, full_well_e=200_000,
    )
    rng = np.random.default_rng(seed)
    return [synth_sensor.frame(model, exposure_s=5.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng) for _ in range(n)]


def test_dsnu_subtracts_the_temporal_noise_floor_stacking_leaves_behind():
    """Averaging N darks divides the per-pixel temporal noise *variance* by
    N; it doesn't remove it. The std of the stacked image is therefore
    sqrt(dsnu^2 + var_temporal/N), and reporting it directly -- as
    `dsnu_std_dn` used to -- reports the floor: at N=3 with 3 e- read noise
    and gain 2 it read ~0.89 DN whether the injected DSNU was 0.125 DN or
    exactly zero.
    """
    # 2.0 e-/s of DSNU over a 5 s exposure at gain 2 is 5.0 DN of pattern.
    stats = fixed_pattern.dsnu_stats(_darks(2.0), 512.0)["R"]
    assert stats["resolved"] is True
    assert stats["dsnu_std_dn"] == pytest.approx(5.0, rel=0.1)
    # The raw (unsubtracted) figure the old code reported is visibly higher.
    assert stats["observed_std_dn"] > stats["dsnu_std_dn"]
    assert stats["temporal_floor_dn"] > 0


def test_dsnu_refuses_to_report_a_pattern_it_cannot_see():
    """With no DSNU at all, the stacked image still has the temporal floor
    in it. The answer is "not resolved by this many frames", not a number
    that happens to equal read_noise/sqrt(N)."""
    stats = fixed_pattern.dsnu_stats(_darks(0.0), 512.0)["R"]
    assert stats["resolved"] is False
    assert stats["dsnu_std_dn"] is None
    assert stats["observed_std_dn"] == pytest.approx(stats["temporal_floor_dn"], rel=0.1)

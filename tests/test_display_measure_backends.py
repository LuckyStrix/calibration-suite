"""CLI-level tests for `calsuite display measure --backend camera` and
`--backend spectro` (docs/implementation-plan.md Wave 3 fix list item 7):
both used to be CLI stubs ("wire it up in a script, not this CLI stub").
Uses synthetic `.npz` frames (camera) and synthetic spectra CSVs (spectro)
-- no camera, no instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from calsuite import cli, raw as rawmod, store as storemod
from calsuite.display import commands as display_commands
from calsuite.synth import sensor as synth_sensor

CAMERA_DEVICE_ID = "test-cam-unknown"
DISPLAY_DEVICE_ID = "test-display-backend"
STEPS = 2  # keeps the patch set (and so the frame/spectra count) small


def _patches():
    ramp, _grid, flat_grid = display_commands._all_measurement_patches(STEPS)
    return ramp + flat_grid


def _save_camera_color_record(records_dir, matrix) -> None:
    st = storemod.Store(records_dir)
    record = storemod.Record(
        schema=1,
        id="camera.color-test-0001",
        kind="camera.color",
        device={"kind": "camera", "model": "Test Cam", "id": CAMERA_DEVICE_ID, "firmware": ""},
        provenance="measured",
        status="ok",
        result={"matrix_raw_to_xyz": matrix.tolist()},
    )
    st.save(record)


def _write_camera_frames(patches, folder, rng, *, gamma=2.2):
    """One synthetic capture per patch, with a *per-CFA-channel* flux.

    This used to apply one scalar flux per frame, from `sum(p.rgb) / 3`, so
    every CFA plane of every frame got the same signal: through the identity
    colour matrix below, a full-red patch and a full-blue patch measured
    identically, and the "camera as colorimeter" path was exercised without
    any colour in it at all. A backend that returned R=G=B for everything
    passed. The flux now follows each channel's own drive level through a
    known display gamma, so the recovered TRC can be checked against it.
    """
    model = synth_sensor.SensorModel(
        shape=(64, 64), black_dn=100.0, gain_e_per_dn=2.0, read_noise_e=1.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=200_000.0,
    )
    rows, cols = model.shape
    # The visible-area CFA phase map for this model's pattern: which of
    # r/g/b each pixel sees.
    channel_index = {"R": 0, "G": 1, "B": 2}
    phase = np.empty((2, 2), dtype=int)
    for k, letter in enumerate(model.pattern):
        phase[k // 2, k % 2] = channel_index[letter]

    folder.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches):
        # A real display: a black floor plus a power-law contribution.
        per_channel = [2_000.0 + 60_000.0 * (float(level) ** gamma) for level in p.rgb]
        flux = np.empty((rows, cols), dtype=np.float64)
        for dr in (0, 1):
            for dc in (0, 1):
                flux[dr::2, dc::2] = per_channel[phase[dr, dc]]
        frame = synth_sensor.frame(model, exposure_s=0.05, flux_e_per_s=flux, temp_c=20.0, rng=rng)
        rawmod.save_npz(frame, folder / f"{i:04d}.npz")


def _write_spectra(patches, folder):
    folder.mkdir(parents=True, exist_ok=True)
    wavelengths = range(400, 701, 50)
    for p in patches:
        mean_level = sum(p.rgb) / 3.0
        with open(folder / f"{p.label}.csv", "w", encoding="utf-8") as f:
            for wl in wavelengths:
                f.write(f"{wl},{1.0 + mean_level * 10.0}\n")


# ---------------------------------------------------------------------------
# camera backend
# ---------------------------------------------------------------------------


def test_measure_camera_backend_from_dir_writes_a_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    _save_camera_color_record(tmp_path / "records", np.eye(3) * 0.9)

    frames_dir = tmp_path / "frames"
    _write_camera_frames(_patches(), frames_dir, np.random.default_rng(0))

    rc = cli.main(
        [
            "display", "measure", "--backend", "camera",
            "--device-id", DISPLAY_DEVICE_ID, "--camera-device-id", CAMERA_DEVICE_ID,
            "--from", str(frames_dir), "--steps", str(STEPS),
        ]
    )
    assert rc in (0, 1)  # STEPS=2 is too short a ramp to fit a gamma

    st = storemod.Store(tmp_path / "records")
    record = st.latest("display.measurement", DISPLAY_DEVICE_ID)
    assert record is not None
    assert record.provenance == "measured"
    assert "camera as colorimeter" in record.result["backend_accuracy"]["basis"]


def test_camera_backend_recovers_the_synthetic_displays_gamma_and_its_colour(tmp_path, monkeypatch):
    """A real round trip through the camera backend: a synthetic display
    with a known gamma, photographed by a synthetic sensor, measured back.

    The other camera-backend tests above are wiring checks -- they assert
    `rc in (0, 1)`, which accepts every outcome the command has. This one
    asserts recovered quantities, so a backend that returned the same
    numbers for every patch (which is what the old fixture's single scalar
    flux per frame actually produced) cannot pass.
    """
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    # A realistic raw->XYZ matrix (sRGB primaries), not the identity the
    # wiring tests use: with an identity matrix Y comes from the green
    # channel alone, so a red ramp's Y never moves and there is no tone
    # curve to recover. On a real camera every primary carries luminance.
    srgb_to_xyz = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    _save_camera_color_record(tmp_path / "records", srgb_to_xyz)

    steps = 9
    ramp, _grid, flat_grid = display_commands._all_measurement_patches(steps)
    frames_dir = tmp_path / "frames"
    _write_camera_frames(ramp + flat_grid, frames_dir, np.random.default_rng(3), gamma=2.2)

    rc = cli.main(
        [
            "display", "measure", "--backend", "camera",
            "--device-id", DISPLAY_DEVICE_ID + "-gamma", "--camera-device-id", CAMERA_DEVICE_ID,
            "--from", str(frames_dir), "--steps", str(steps),
        ]
    )
    st = storemod.Store(tmp_path / "records")
    record = st.latest("display.measurement", DISPLAY_DEVICE_ID + "-gamma")
    assert record is not None
    assert rc == 0, record.refusals

    gammas = record.result["trc"]["effective_gamma"]
    assert set(gammas) == {"r", "g", "b"}
    assert gammas["r"] == pytest.approx(2.2, rel=0.1)
    assert gammas["g"] == pytest.approx(2.2, rel=0.1)
    # Blue gets a wider tolerance on purpose: it carries ~7% of Y against
    # green's ~72%, so its black-corrected luminance at the low end of the
    # ramp is a small difference of larger numbers and the recovered gamma
    # moves by ~0.2 between noise realizations. That is a real property of
    # measuring a display with a camera, not a slack assertion -- a backend
    # with no colour in it produces a *flat* blue ramp, which refuses
    # outright (`trc_b_zero_max`) rather than landing anywhere near 2.2.
    assert gammas["b"] == pytest.approx(2.2, rel=0.25)

    # ...and the measurement has colour in it at all: full red, green and
    # blue must not measure the same XYZ.
    primaries = record.result["primaries_measured"]
    assert primaries["r"] != primaries["g"] != primaries["b"]
    assert primaries["r"][0] > primaries["b"][0]  # X is larger for red than for blue
    assert primaries["b"][2] > primaries["r"][2]  # Z is larger for blue than for red


def test_measure_camera_backend_refuses_clearly_on_frame_count_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    _save_camera_color_record(tmp_path / "records", np.eye(3) * 0.9)

    patches = _patches()
    frames_dir = tmp_path / "frames"
    _write_camera_frames(patches[:-3], frames_dir, np.random.default_rng(1))  # too few frames

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "display", "measure", "--backend", "camera",
                "--device-id", DISPLAY_DEVICE_ID, "--camera-device-id", CAMERA_DEVICE_ID,
                "--from", str(frames_dir), "--steps", str(STEPS),
            ]
        )
    message = str(exc.value)
    assert "raw file" in message and str(len(patches)) in message


def test_measure_camera_backend_errors_clearly_with_no_camera_color_record(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    frames_dir = tmp_path / "frames"
    _write_camera_frames(_patches(), frames_dir, np.random.default_rng(2))

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "display", "measure", "--backend", "camera",
                "--device-id", DISPLAY_DEVICE_ID, "--camera-device-id", "no-such-camera",
                "--from", str(frames_dir), "--steps", str(STEPS),
            ]
        )
    assert "no camera.color record" in str(exc.value)


def test_measure_camera_backend_needs_from_or_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    _save_camera_color_record(tmp_path / "records", np.eye(3) * 0.9)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "display", "measure", "--backend", "camera",
                "--device-id", DISPLAY_DEVICE_ID, "--camera-device-id", CAMERA_DEVICE_ID,
                "--steps", str(STEPS),
            ]
        )
    assert "--from" in str(exc.value) and "--capture" in str(exc.value)


# ---------------------------------------------------------------------------
# spectro backend
# ---------------------------------------------------------------------------


def test_measure_spectro_backend_from_spectra_dir_writes_a_measurement(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    spectra_dir = tmp_path / "spectra"
    _write_spectra(_patches(), spectra_dir)

    rc = cli.main(
        [
            "display", "measure", "--backend", "spectro",
            "--device-id", DISPLAY_DEVICE_ID + "-spectro", "--spectra", str(spectra_dir), "--steps", str(STEPS),
        ]
    )
    assert rc in (0, 1)

    st = storemod.Store(tmp_path / "records")
    record = st.latest("display.measurement", DISPLAY_DEVICE_ID + "-spectro")
    assert record is not None
    assert "spectrophotometer" in record.result["backend_accuracy"]["basis"]


def test_measure_spectro_backend_errors_clearly_on_missing_spectrum_file(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    patches = _patches()
    spectra_dir = tmp_path / "spectra"
    _write_spectra(patches, spectra_dir)
    (spectra_dir / f"{patches[0].label}.csv").unlink()

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "display", "measure", "--backend", "spectro",
                "--device-id", DISPLAY_DEVICE_ID + "-spectro", "--spectra", str(spectra_dir), "--steps", str(STEPS),
            ]
        )
    assert "missing spectrum file" in str(exc.value)


def test_measure_spectro_backend_needs_spectra_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["display", "measure", "--backend", "spectro", "--device-id", DISPLAY_DEVICE_ID, "--steps", str(STEPS)]
        )
    assert "--spectra" in str(exc.value)

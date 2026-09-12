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


def _write_camera_frames(patches, folder, rng):
    model = synth_sensor.SensorModel(
        shape=(64, 64), black_dn=100.0, gain_e_per_dn=2.0, read_noise_e=1.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=200_000.0,
    )
    folder.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(patches):
        mean_level = sum(p.rgb) / 3.0
        flux = 5_000.0 + mean_level * 50_000.0
        frame = synth_sensor.frame(model, exposure_s=0.05, flux_e_per_s=flux, temp_c=20.0, rng=rng)
        rawmod.save_npz(frame, folder / f"{i:04d}.npz")


def _write_spectra(patches, folder):
    folder.mkdir(parents=True, exist_ok=True)
    wavelengths = range(400, 701, 50)
    for p in patches:
        mean_level = sum(p.rgb) / 3.0
        with open(folder / f"{p.label}.csv", "w") as f:
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
    assert rc in (0, 1)

    st = storemod.Store(tmp_path / "records")
    record = st.latest("display.measurement", DISPLAY_DEVICE_ID)
    assert record is not None
    assert record.provenance == "measured"
    assert "camera as colorimeter" in record.result["backend_accuracy"]["basis"]


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

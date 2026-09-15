"""Command-level wiring checks for ``lens/commands.py``.

These sit above the pure-analysis unit tests (``test_lens_flats.py`` etc.):
they exercise ``calsuite lens export``/``calsuite lens report`` through the
real ``cli.main`` + ``Store``, the same way ``test_camera_commands.py`` and
``test_display_commands.py`` do for their areas, to catch a command that
forgets to load or forward a record kind it has -- a bug the pure-function
tests (which only ever call ``export_lensfun.export_records``/
``report.render_lens_report`` directly with hand-built records) can't see.
"""

from __future__ import annotations

import numpy as np
import pytest

from calsuite import cli, raw as rawmod, store
from calsuite.fit import Analysis
from calsuite.lens import flats as flatsmod
from calsuite.synth import lens as synthlens, sensor as synth_sensor


def _device() -> dict:
    return {"kind": "lens", "model": "Test Lens 50mm", "id": "test-lens-50mm-unknown", "firmware": ""}


def _write_flats_capture(tmp_path, *, black_dn=100.0):
    """A small (but non-degenerate -- angle+shift poses, per
    lens/flats.py's condition-number check) self-calibrating-flat session,
    written as ``.npz`` so ``lens/commands.py``'s real ``--from DIR`` path
    (``_RAW_EXTENSIONS``) reads it exactly like a folder of real captures.
    Mirrors ``demo.py::_run_lens``'s own flats session."""
    model = synth_sensor.SensorModel(
        shape=(180, 240), read_noise_e=2.0, gain_e_per_dn=2.0, black_dn=black_dn,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    poses = [
        flatsmod.Pose(0, 0, 0), flatsmod.Pose(90, 0.08, 0), flatsmod.Pose(180, 0, 0.08),
        flatsmod.Pose(270, 0.08, 0.08), flatsmod.Pose(0, -0.06, 0.05), flatsmod.Pose(90, -0.05, -0.07),
    ]
    rng = np.random.default_rng(12)
    folder = tmp_path / "flats_capture"
    folder.mkdir()
    for i, pose in enumerate(poses):
        frame = synthlens.render_flat_pose(
            model, [-0.35, 0.05], [0.0, 0.15, -0.10, -0.08, 0.02, -0.05], pose,
            rng=rng, exposure_s=0.2, base_flux_e_per_s=3.0e4,
        )
        rawmod.save_npz(frame, folder / f"f{i:04d}.npz")
    poses_str = ",".join(f"{p.angle_deg:g}:{p.shift_u:g}:{p.shift_v:g}" for p in poses)
    return folder, poses_str


def test_lens_flats_records_the_given_focus_distance(tmp_path, monkeypatch):
    """`_cmd_flats` used to never capture a focus distance at all, so every
    exported `<vignetting distance="...">` silently got a fabricated 0.00
    from `export_records`'s `... or 0.0` fallback (house rule 2: no
    fabricated numbers in an exported file). With the same `--distance` flag
    `lens distortion` already has, the record's conditions carry the real
    value and the export reflects it."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    folder, poses_str = _write_flats_capture(tmp_path)

    rc = cli.main(
        ["lens", "flats", "--from", str(folder), "--aperture", "1.8", "--poses", poses_str, "--distance", "0.5"]
    )
    assert rc == 0

    st = store.Store(records_dir)
    records = list(st.all(kind="lens.flats"))
    assert len(records) == 1
    assert records[0].conditions["focus_distance_m"] == pytest.approx(0.5)
    device_id = records[0].device["id"]

    out_path = tmp_path / "out.xml"
    rc = cli.main(["lens", "export", "--device-id", device_id, "--lens-model", "Test Lens 50mm", "--out", str(out_path)])
    assert rc == 0
    xml_text = out_path.read_text(encoding="utf-8")
    assert "<vignetting" in xml_text
    assert 'distance="0.50"' in xml_text


def test_lens_flats_omits_distance_attribute_when_none_was_given(tmp_path, monkeypatch):
    """Without `--distance`, the exporter must not invent one: no
    `focus_distance_m` in the record's conditions, and no `distance`
    attribute at all on the exported `<vignetting>` element -- not a
    fabricated `distance="0.00"`. lensfun's own XML schema
    (`libs/lensfun/database.cpp`'s `<vignetting>` element handler) has no
    "required" check on `distance` (unlike `model`/`focal`/`aperture`), so
    omitting it is what the format actually allows; a value it silently
    defaults to zero internally either way is not a substitute for that
    honesty in the file we write."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    folder, poses_str = _write_flats_capture(tmp_path)

    rc = cli.main(["lens", "flats", "--from", str(folder), "--aperture", "1.8", "--poses", poses_str])
    assert rc == 0

    st = store.Store(records_dir)
    records = list(st.all(kind="lens.flats"))
    assert len(records) == 1
    assert "focus_distance_m" not in records[0].conditions
    device_id = records[0].device["id"]

    out_path = tmp_path / "out.xml"
    rc = cli.main(["lens", "export", "--device-id", device_id, "--lens-model", "Test Lens 50mm", "--out", str(out_path)])
    assert rc == 0
    xml_text = out_path.read_text(encoding="utf-8")
    assert "<vignetting" in xml_text
    assert "distance=" not in xml_text


def _save_ok_flats_record(records_dir, *, aperture: float = 1.8) -> None:
    st = store.Store(records_dir)
    analysis = Analysis(
        result={
            "v_coeffs": [-0.35, 0.05],
            "s_coeffs": [0.0, 0.15, -0.10, -0.08, 0.02, -0.05],
            "image_shape": [180, 240],
            "n_poses": 4,
            "condition_number": 12.0,
            "poses": [],
        },
        residuals={"log_domain_rms": 0.01},
    )
    record = store.Record.from_analysis(
        kind="lens.flats",
        device=_device(),
        analysis=analysis,
        provenance="measured",
        method={"name": "flats.self_calibrate_flat"},
        conditions={"focal_mm": 50.0, "aperture": aperture},
    )
    # Mirrors what `_cmd_flats` adds after a successful fit (lens/commands.py):
    # the refit-to-lensfun's-"pa"-model dict, computed from v_coeffs.
    record.result["pa"] = {"k1": -0.34, "k2": 0.06, "k3": -0.01, "residual_rms": 0.001}
    st.save(record)


def test_lens_export_includes_vignetting_from_a_saved_flats_record(tmp_path, monkeypatch):
    """`calsuite lens export` must pick up an existing `ok` `lens.flats`
    record and write it as a `<vignetting>` element -- `export_lensfun.
    export_records` already accepts `flats_records` and turns each into one,
    but `_cmd_export` never fetched or passed them, so real vignetting data
    silently never reached the exported XML even when a flats session had
    already produced it."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    _save_ok_flats_record(records_dir, aperture=1.8)

    out_path = tmp_path / "out.xml"
    rc = cli.main(
        [
            "lens",
            "export",
            "--device-id",
            "test-lens-50mm-unknown",
            "--lens-model",
            "Test Lens 50mm",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    xml_text = out_path.read_text(encoding="utf-8")
    assert "<vignetting" in xml_text
    assert 'model="pa"' in xml_text
    assert 'aperture="1.8"' in xml_text
    assert 'k1="-0.340000"' in xml_text


def test_lens_psf_command_refuses_a_saturated_star_field_end_to_end(tmp_path, monkeypatch):
    """The full ``calsuite lens psf`` path -- not just ``psf.psf_field``
    directly -- must refuse a saturated star field: non-zero exit code, a
    saved record with ``status="refused"``, and the refusal visible in the
    HTML report (house rule 3's refusal has to actually reach the user, not
    just live inside the pure-analysis return value)."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    model = synth_sensor.SensorModel(
        shape=(240, 320), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=40000.0,
    )
    rng = np.random.default_rng(9)
    stars = [{"x": 100.0, "y": 80.0, "amplitude": 5.0e6, "sigma_major": 10.0, "sigma_minor": 4.0, "theta_deg": 40.0}]
    frame = synthlens.render_stars(model, stars, rng=rng, exposure_s=0.3, background_flux_e_per_s=20.0)
    folder = tmp_path / "psf_capture"
    folder.mkdir()
    rawmod.save_npz(frame, folder / "f0000.npz")

    rc = cli.main(["lens", "psf", "--from", str(folder)])
    assert rc == 1

    st = store.Store(records_dir)
    records = list(st.all(kind="lens.psf"))
    assert len(records) == 1
    assert records[0].status == "refused"
    assert records[0].refusals[0]["check"] == "all_blobs_saturated"
    device_id = records[0].device["id"]

    out_path = tmp_path / "report.html"
    rc = cli.main(["lens", "report", "--device-id", device_id, "--lens-model", "Test Lens 50mm", "--out", str(out_path)])
    assert rc == 0
    html = out_path.read_text(encoding="utf-8")
    assert "all_blobs_saturated" in html or "refused" in html.lower()


def test_lens_report_includes_vignetting_section_for_a_saved_flats_record(tmp_path, monkeypatch):
    """Same wiring gap as the export command: `render_lens_report` already
    renders a "Vignetting" section whenever it's given
    `flats_records_by_aperture`, but `_cmd_report` never built that dict, so
    a lens report never showed vignetting even with an `ok` `lens.flats`
    record on file."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    _save_ok_flats_record(records_dir, aperture=2.8)

    out_path = tmp_path / "report.html"
    rc = cli.main(
        [
            "lens",
            "report",
            "--device-id",
            "test-lens-50mm-unknown",
            "--lens-model",
            "Test Lens 50mm",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    html = out_path.read_text(encoding="utf-8")
    assert "Vignetting" in html


def test_lens_flats_subtracts_the_black_pedestal_before_fitting(tmp_path, monkeypatch):
    """`self_calibrate_flat` fits a *multiplicative* model (it works on
    log(V . S)), so a black pedestal is not absorbed anywhere -- it flattens
    the recovered falloff. `_cmd_flats` handed it `raw.planes(f)["G1"]`
    straight, in defiance of that function's own "already black-subtracted
    and positive" contract; with a realistic 2048 DN pedestal on ~3000 DN of
    signal, a true corner V of 0.741 came back as 0.846 and the exported
    lensfun k1 was 40% low. `test_lens_flats.py` never saw it because its
    own `_render` does the subtraction the command was skipping, and the
    session here uses the synthetic sensor's default 100 DN, small enough to
    hide it.
    """
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    folder, poses_str = _write_flats_capture(tmp_path, black_dn=2048.0)

    rc = cli.main(["lens", "flats", "--from", str(folder), "--aperture", "1.8", "--poses", poses_str])
    assert rc == 0

    st = store.Store(tmp_path / "records")
    record = next(iter(st.all(kind="lens.flats")))
    v_coeffs = record.result["v_coeffs"]
    # The same ground truth _write_flats_capture renders with.
    assert v_coeffs == pytest.approx([-0.35, 0.05], abs=0.03)

    # Corner vignetting at r=1, the number that reaches the lensfun export.
    corner = 1.0 + v_coeffs[0] + v_coeffs[1]
    assert corner == pytest.approx(0.70, abs=0.03)  # un-subtracted, this read 0.85

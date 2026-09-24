"""CLI-level tests for ``calsuite camera ...``.

The foundation gives no file format for a *synthetic* ``RawFrame`` -- a real
``raw.load()`` needs an actual CR3/DNG/NEF rawpy can decode, and
``synth.sensor.frame()`` builds a ``RawFrame`` directly in memory. So, the
same way ``tests/test_capture_manual.py`` does it, these tests monkeypatch
``calsuite.raw.load`` (the one place both ``capture.manual.scan_folder`` and
``camera.commands`` call to turn a path into a ``RawFrame``) to hand back
pre-built synthetic frames keyed by filename, while real (placeholder,
content-irrelevant) files on disk let ``scan_folder`` glob/iterate normally.
This exercises the whole ``--from DIR`` path -- manifest scan, role
classification, analysis, and the record save -- with nothing but synthetic
data and no real raw decoder involved.
"""

from pathlib import Path

import numpy as np
import pytest

from calsuite import cli, raw as rawmod, store
from calsuite.synth import sensor as synth_sensor


def _iso_record(kind, *, iso_value, status="ok", result=None, device_id="dev-1", model="calsuite-test-cam"):
    """A minimal store.Record good enough for ``_cmd_iso`` to read --
    it only looks at ``conditions["iso"]``, ``status`` and
    ``result["channels"][ch]["fit"/"full_well_e"]``."""
    return store.Record(
        schema=1,
        id=f"{kind}-iso{iso_value}-{status}",
        kind=kind,
        device={"kind": "camera", "model": model, "id": device_id},
        status=status,
        provenance="measured",
        conditions={"iso": iso_value},
        result=result or {},
    )


def _bias_frame(seed, black_dn=512.0, gain=2.0, read_noise_e=3.0):
    model = synth_sensor.SensorModel(
        shape=(48, 64), black_dn=black_dn, gain_e_per_dn=gain, read_noise_e=read_noise_e,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0,
    )
    rng = np.random.default_rng(seed)
    return synth_sensor.frame(model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng)


@pytest.fixture
def synthetic_capture_dir(tmp_path, monkeypatch):
    frames_by_name = {}
    for i in range(6):
        name = f"bias{i:04d}.cr3"
        (tmp_path / name).write_bytes(b"placeholder -- content is irrelevant, load() is faked below")
        frames_by_name[name] = _bias_frame(seed=i)

    def fake_load(path):
        frame = frames_by_name[Path(path).name]
        # raw.load() also stamps a real sha256/path per file; synth frames
        # don't carry one, so fill in something per-file here instead of
        # leaving every frame's sha256 identically "".
        from dataclasses import replace

        return replace(frame, path=str(path), sha256=f"fake-sha-{Path(path).name}")

    monkeypatch.setattr(rawmod, "load", fake_load)
    return tmp_path


def test_camera_bias_from_dir_saves_ok_record(synthetic_capture_dir, tmp_path, monkeypatch):
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))

    rc = cli.main(["camera", "bias", "--from", str(synthetic_capture_dir)])
    assert rc == 0

    st = store.Store(records_dir)
    records = list(st.all(kind="camera.bias"))
    assert len(records) == 1
    assert records[0].status == "ok"
    assert records[0].provenance == "measured"
    assert records[0].inputs and all("sha256" in i for i in records[0].inputs)


def test_camera_bias_without_from_or_capture_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    with pytest.raises(SystemExit):
        cli.main(["camera", "bias"])


def test_camera_iso_missing_full_well_saves_refused_record_not_nothing(tmp_path, monkeypatch):
    """Regression for the vanishing-record bug: 3 ok camera.ptc records (so
    the read-noise sweep alone is enough to characterize the curve -- no
    conflation with ``too_few_isos``) but no camera.linearity record at
    all, so there is no electron-domain full well to pair with it. This
    used to ``print(...); return 1`` with no camera.iso record saved at
    all; it must now save one, refused, naming the real cause."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    st = store.Store(records_dir)
    for iso_value in (100, 400, 800):
        st.save(_iso_record(
            "camera.ptc", iso_value=iso_value,
            result={"channels": {"G1": {"fit": {"read_noise_e": 3.0}}}},
        ))

    rc = cli.main(["camera", "iso", "--device-id", "dev-1"])
    assert rc == 1

    records = list(store.Store(records_dir).all(kind="camera.iso", device_id="dev-1"))
    assert len(records) == 1, "the record must be saved, not silently dropped"
    assert records[0].status == "refused"
    assert records[0].provenance == "derived"
    checks = [r["check"] for r in records[0].refusals]
    assert "no_full_well" in checks
    assert "too_few_isos" not in checks  # 3 ok ISOs is plenty; this must be the ONLY refusal


def test_camera_iso_missing_full_well_names_the_refused_upstream_ptc(tmp_path, monkeypatch):
    """Same missing-full-well shape as above, but reproducing the actual
    demo failure mode (docs/design.md's ISO sweep, seed offset 7): one
    ISO's camera.ptc refused, leaving too few ok ISOs *and* no full well
    at once -- both refusals must land in the one saved record, and the
    no_full_well message should name the refused ISO rather than stay
    generic when that information is available."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    st = store.Store(records_dir)
    st.save(_iso_record("camera.ptc", iso_value=100, result={"channels": {"G1": {"fit": {"read_noise_e": 5.0}}}}))
    st.save(_iso_record("camera.ptc", iso_value=400, result={"channels": {"G1": {"fit": {"read_noise_e": 3.5}}}}))
    st.save(_iso_record("camera.ptc", iso_value=1600, status="refused"))

    rc = cli.main(["camera", "iso", "--device-id", "dev-1"])
    assert rc == 1

    records = list(store.Store(records_dir).all(kind="camera.iso", device_id="dev-1"))
    assert len(records) == 1
    assert records[0].status == "refused"
    checks = {r["check"]: r["message"] for r in records[0].refusals}
    assert "too_few_isos" in checks  # only 2 ok ISOs
    assert "no_full_well" in checks
    assert "1600" in checks["no_full_well"]


def test_camera_iso_with_no_ptc_records_at_all_saves_refused_record(tmp_path, monkeypatch):
    """The other old silent path: no camera.ptc records of any kind for
    the device. Must still save a (too_few_isos, 0 vs 3) record rather
    than just printing and exiting."""
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))

    rc = cli.main(["camera", "iso", "--device-id", "never-measured"])
    assert rc == 1

    records = list(store.Store(tmp_path / "records").all(kind="camera.iso", device_id="never-measured"))
    assert len(records) == 1
    assert records[0].status == "refused"
    assert any(r["check"] == "too_few_isos" for r in records[0].refusals)


def test_camera_color_subcommand_is_wired_in(capsys):
    # camera/commands.py must call color_commands.register(camera_subparsers)
    # so the color area's own subcommand (built independently, in the same
    # wave) stays reachable through `calsuite camera color ...` -- this only
    # checks that the wiring exists and doesn't crash, not what the color
    # area's own subcommand does with its arguments.
    rc = cli.main(["camera", "color"])
    assert rc in (0, 1)
    assert "camera color" in capsys.readouterr().out


def test_camera_fixed_pattern_end_to_end_writes_a_record(tmp_path, monkeypatch):
    """`camera/fixed_pattern.py` was implemented and unit-tested but
    reachable from nothing: no command called `analyze_fixed_pattern`, so
    the `camera.fixed_pattern` record kind that `store.SHELF_LIFE_DAYS`
    declares -- and that `doctor.check_stale_records` looks for -- could
    never exist. Its three inputs are three of the roles
    `capture.manual.classify` already sorts a folder into.
    """
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    model = synth_sensor.SensorModel(
        shape=(96, 96), black_dn=512.0, gain_e_per_dn=2.0, read_noise_e=3.0,
        prnu_std=0.01, dsnu_std_e_per_s=0.5, hot_pixel_fraction=0.0, full_well_e=40_000.0,
    )
    rng = np.random.default_rng(5)
    folder = tmp_path / "frames"
    folder.mkdir()
    frames = []
    for _ in range(4):  # biases: near black, short exposure
        frames.append(synth_sensor.frame(model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng))
    for _ in range(4):  # darks: near black, long exposure
        frames.append(synth_sensor.frame(model, exposure_s=30.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng))
    for _ in range(4):  # flats: mid-level, spatially uniform
        frames.append(synth_sensor.frame(model, exposure_s=0.1, flux_e_per_s=100_000.0, temp_c=20.0, rng=rng))
    for i, frame in enumerate(frames):
        rawmod.save_npz(frame, folder / f"{i:04d}.npz")

    rc = cli.main(["camera", "fixed-pattern", "--from", str(folder)])
    assert rc == 0

    st = store.Store(tmp_path / "records")
    records = list(st.all(kind="camera.fixed_pattern"))
    assert len(records) == 1
    channels = records[0].result["channels"]
    assert set(channels) == {"R", "G1", "G2", "B"}
    # PRNU was injected at 1%; DSNU is reported only if it clears the
    # temporal-noise floor, and either way the floor itself is recorded.
    assert channels["R"]["prnu_std_pct"] == pytest.approx(1.0, rel=0.3)
    assert channels["R"]["dsnu_temporal_floor_dn"] > 0
    assert records[0].conditions["n_darks"] == 4


def _seed_bias_dn_only_then_ptc_and_linearity(st, isos=(100, 400, 1600)):
    """Bias records with read noise in DN only (bias was run before any PTC
    existed), then PTC gain + linearity full well on record."""
    gains = {100: 2.0, 400: 0.5, 1600: 0.125}
    for iso_value in isos:
        st.save(_iso_record("camera.bias", iso_value=iso_value, result={"read_noise_dn": {"G1": 3.0}}))
        st.save(_iso_record(
            "camera.ptc", iso_value=iso_value,
            result={"channels": {"G1": {"fit": {"gain_e_per_dn": gains[iso_value], "read_noise_e": None}}}},
        ))
        st.save(_iso_record("camera.linearity", iso_value=iso_value, result={"channels": {"G1": {"full_well_e": 20000.0}}}))


def test_camera_iso_converts_bias_read_noise_to_electrons_when_bias_ran_before_ptc(tmp_path, monkeypatch):
    """Run order used to matter: a bias run before its ISO's PTC only had DN,
    so `camera iso` never saw it (and the PTC intercept it fell back to is
    often unresolved). The gain is on record by the time `iso` runs, so it
    converts there instead of asking for a re-run."""
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    st = store.Store(tmp_path / "records")
    _seed_bias_dn_only_then_ptc_and_linearity(st)

    assert cli.main(["camera", "iso", "--device-id", "dev-1"]) == 0
    (rec,) = st.all(kind="camera.iso", device_id="dev-1")
    assert rec.result["read_noise_e_by_iso"] == {"100": pytest.approx(6.0), "400": pytest.approx(1.5), "1600": pytest.approx(0.375)}


def test_camera_iso_and_report_default_to_the_only_camera_with_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    st = store.Store(tmp_path / "records")
    _seed_bias_dn_only_then_ptc_and_linearity(st)

    assert cli.main(["camera", "iso"]) == 0
    assert "using device dev-1" in capsys.readouterr().out

    # The seeded PTC/bias stubs lack the fields the report plots; report from
    # a store holding only the real camera.iso record just produced.
    (iso_rec,) = st.all(kind="camera.iso", device_id="dev-1")
    report_dir = tmp_path / "report-records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(report_dir))
    store.Store(report_dir).save(iso_rec)
    assert cli.main(["camera", "report"]) == 0
    # Written next to the records (like lens/display reports), not dumped to the terminal.
    assert (report_dir / "dev-1" / "sensor-report.html").is_file()


def test_camera_iso_without_device_id_lists_the_cameras_when_there_are_several(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    st = store.Store(tmp_path / "records")
    st.save(_iso_record("camera.bias", iso_value=100, device_id="cam-a"))
    st.save(_iso_record("camera.bias", iso_value=100, device_id="cam-b"))

    assert cli.main(["camera", "iso"]) == 1
    out = capsys.readouterr().out
    assert "cam-a" in out and "cam-b" in out
    assert not list(st.all(kind="camera.iso"))


def test_camera_commands_print_the_device_id_they_recorded_under(synthetic_capture_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path / "records"))
    assert cli.main(["camera", "bias", "--from", str(synthetic_capture_dir)]) == 0
    out = capsys.readouterr().out
    (rec,) = store.Store(tmp_path / "records").all(kind="camera.bias")
    assert f"(device {rec.device['id']})" in out
    assert "no camera.ptc record" in out  # and says why read noise is DN-only

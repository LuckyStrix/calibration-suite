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


def test_camera_color_subcommand_is_wired_in(capsys):
    # camera/commands.py must call color_commands.register(camera_subparsers)
    # so the color area's own subcommand (built independently, in the same
    # wave) stays reachable through `calsuite camera color ...` -- this only
    # checks that the wiring exists and doesn't crash, not what the color
    # area's own subcommand does with its arguments.
    rc = cli.main(["camera", "color"])
    assert rc in (0, 1)
    assert "camera color" in capsys.readouterr().out

from pathlib import Path

import pytest

from calsuite import cli, store as storemod


def test_synthetic_measure_profile_validate_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    device_id = "synthetic-display"

    rc = cli.main(["display", "measure", "--backend", "synthetic", "--device-id", device_id, "--steps", "9"])
    assert rc == 0

    rc = cli.main(["display", "profile", "--device-id", device_id, "--out", str(tmp_path / "profile.icc")])
    assert rc == 0
    assert (tmp_path / "profile.icc").exists()

    rc = cli.main(["display", "validate", "--backend", "synthetic", "--device-id", device_id])
    assert rc == 0

    rc = cli.main(["display", "report", "--device-id", device_id, "--out", str(tmp_path / "report.html")])
    assert rc == 0
    html = (tmp_path / "report.html").read_text()
    assert "Tone response curves" in html
    assert "Validation" in html


def test_nominal_writes_record_from_edid_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    drm_dir = tmp_path / "drm"
    connector_dir = drm_dir / "card0-eDP-1"
    connector_dir.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures" / "edid_csot_mng007ja1.bin"
    (connector_dir / "edid").write_bytes(fixture.read_bytes())

    from calsuite.display import commands as display_commands

    orig_list_linux_edids = display_commands.devicesmod.list_linux_edids
    monkeypatch.setattr(display_commands.devicesmod, "list_linux_edids", lambda: orig_list_linux_edids(drm_dir))

    class Args:
        connector = None

    rc = display_commands._cmd_nominal(Args())
    assert rc == 0

    st = storemod.Store(tmp_path)
    records = list(st.all(kind="display.nominal"))
    assert len(records) == 1
    assert records[0].provenance == "nominal"
    assert records[0].result["chromaticity"]["r"] == pytest.approx([0.638, 0.334], abs=1e-3)

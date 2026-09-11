import pytest

from calsuite import cli


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "calsuite" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    rc = cli.main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_devices_runs(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    rc = cli.main(["devices"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Camera" in out
    assert "Displays" in out
    assert "Known records" in out


@pytest.mark.parametrize("area", ["camera", "lens", "display"])
def test_area_stubs_report_not_built(capsys, area):
    rc = cli.main([area])
    assert rc == 1
    assert "not built yet" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["export", "doctor", "demo"])
def test_top_level_stubs_report_not_built(capsys, cmd):
    rc = cli.main([cmd])
    assert rc == 1
    assert "not built yet" in capsys.readouterr().out

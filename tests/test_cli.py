import subprocess
import sys

import pytest

from calsuite import cli


def test_building_the_parser_does_not_import_heavy_libs():
    # Regression guard: `calsuite --help`/`calsuite doctor` used to take
    # ~2.4s and print pygame's "Hello from the pygame community" banner
    # plus colour-science's matplotlib warning, because cli.py (and each
    # area's commands.py) imported cv2/colour/pygame-dependent analysis
    # modules unconditionally just to build the argparse tree -- fixed by
    # moving those imports inside the specific `_cmd_*` functions that
    # actually need them. Run in a fresh subprocess, not in-process:
    # sys.modules is shared across this whole pytest session, so another
    # test file that already imported pygame/cv2/colour earlier would
    # contaminate an in-process check regardless of whether *this* fix
    # holds.
    code = (
        "import sys\n"
        "from calsuite import cli\n"
        "cli._build_parser()\n"
        "leaked = sorted(m for m in ('pygame', 'cv2', 'colour') if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"building the CLI parser imported: {leaked}\nstderr:\n{result.stderr}"


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


@pytest.mark.parametrize(
    "area,subcommands",
    [
        ("camera", ["bias", "ptc", "linearity", "darks", "shutter", "bulb", "iso", "report", "color"]),
        ("lens", ["distortion", "tca", "flats", "mtf", "psf", "export", "report"]),
        ("display", ["nominal", "measure", "profile", "install", "validate", "report"]),
    ],
)
def test_area_subcommands_exist_and_help_works(capsys, area, subcommands):
    # These used to be stubs ("not built yet"); each area now registers
    # real subcommands (Wave 2/3), so this asserts the actual surface
    # exists and every subcommand's --help exits cleanly, rather than the
    # old "prints not built yet" behaviour that no longer applies.
    for sub in subcommands:
        with pytest.raises(SystemExit) as exc:
            cli.main([area, sub, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert sub in out or "usage" in out.lower()


def test_area_with_no_subcommand_does_not_crash(capsys):
    for area in ("camera", "lens", "display"):
        rc = cli.main([area])
        assert rc == 1
        capsys.readouterr()  # each area prints its own "pass a subcommand" style message


@pytest.mark.parametrize(
    "cmd,subcommands",
    [
        ("export", ["lensfun", "dcp", "icc"]),
    ],
)
def test_export_subcommands_exist_and_help_works(capsys, cmd, subcommands):
    for sub in subcommands:
        with pytest.raises(SystemExit) as exc:
            cli.main([cmd, sub, "--help"])
        assert exc.value.code == 0
        capsys.readouterr()


def test_export_with_no_subcommand_prints_help(capsys):
    rc = cli.main(["export"])
    assert rc == 1
    assert "export" in capsys.readouterr().out.lower()


def test_doctor_runs(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    rc = cli.main(["doctor"])
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "external tools" in out.lower()


def test_demo_help_works(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["demo", "--help"])
    assert exc.value.code == 0
    capsys.readouterr()

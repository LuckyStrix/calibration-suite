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
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
    )
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


def test_demo_cli_exit_code_reflects_run_demo_warnings(monkeypatch, tmp_path, capsys):
    """`_cmd_demo` used to return 0 whenever *any* area produced a report,
    even if others raised -- `calsuite demo` could print "wrote ..." with
    several report lines and still exit 0 with failures buried in the
    warning list. The exit code must instead reflect whether `run_demo`
    reported any warnings at all (its own return value already lists every
    failure mode: a raised exception, an unexpected `_run_cli` exit code, a
    report that was never produced, an internal consistency check that
    didn't hold)."""
    from calsuite import demo as demomod

    index_path = tmp_path / "index.html"
    index_path.write_text("<html></html>", encoding="utf-8")

    def _fake_run_demo(out_dir, *, warnings):
        return {"index": index_path, "reports": {"Sensor": tmp_path / "sensor.html", "Lens": None}, "warnings": warnings}

    monkeypatch.setattr(demomod, "run_demo", lambda out_dir: _fake_run_demo(out_dir, warnings=[]))
    rc = cli.main(["demo", "--out", str(tmp_path)])
    assert rc == 0

    monkeypatch.setattr(
        demomod, "run_demo", lambda out_dir: _fake_run_demo(out_dir, warnings=["lens demo failed: RuntimeError('boom')"])
    )
    rc = cli.main(["demo", "--out", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    # the failure list is the last thing printed, not buried above a
    # misleadingly-successful-looking report listing.
    assert lines[-1].lstrip().startswith("WARNING:")


def test_demo_cli_exits_nonzero_when_an_area_is_actually_forced_to_fail(monkeypatch, tmp_path, capsys):
    """Same fix, exercised end to end through the real `run_demo` pipeline
    (not a faked result): force one real area (`_run_display`) to raise,
    and confirm the areas that don't depend on it still run (resilience is
    still correct -- see demo.py's own module docstring), but the CLI's
    exit code is now truthful about the failure."""
    from calsuite import demo as demomod

    def _boom(out_dir, warnings):
        raise RuntimeError("forced failure for this test")

    monkeypatch.setattr(demomod, "_run_display", _boom)

    rc = cli.main(["demo", "--out", str(tmp_path)])
    assert rc == 1

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert any("display demo failed" in line for line in lines)
    assert "Sensor:" in out  # the other areas still ran and still get reported
    assert lines[-1].lstrip().startswith("WARNING:")

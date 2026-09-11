import os
import sys

import pytest

from calsuite.display import install_linux, install_windows


def _make_script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def test_install_colormgr_full_chain(tmp_path, monkeypatch):
    _make_script(
        tmp_path,
        "colormgr",
        "#!/bin/bash\n"
        'case "$1" in\n'
        '  import-profile) echo "Object Path: /org/freedesktop/ColorManager/profiles/p1";;\n'
        '  get-devices-by-kind) echo "Object Path: /org/freedesktop/ColorManager/devices/d1";;\n'
        "  *) exit 0;;\n"
        "esac\n",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    report = install_linux.install_colormgr(tmp_path / "profile.icc")
    assert report.ok
    step_names = [s["step"] for s in report.steps]
    assert "colormgr import-profile" in step_names
    assert "colormgr device-add-profile" in step_names
    assert "colormgr device-make-profile-default" in step_names


def test_install_colormgr_missing_binary(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    report = install_linux.install_colormgr("x.icc")
    assert not report.ok
    assert "not found" in report.steps[0]["detail"]


def test_install_dispwin_success(tmp_path, monkeypatch):
    _make_script(tmp_path, "dispwin", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    report = install_linux.install_dispwin(tmp_path / "p.icc")
    assert report.ok


def test_write_autostart_entry_is_idempotent(tmp_path):
    autostart = tmp_path / "autostart"
    install_linux.write_autostart_entry(tmp_path / "a.icc", autostart_path=autostart)
    first = autostart.read_text()
    assert 'a.icc' in first
    assert first.count(install_linux.AUTOSTART_MARKER) == 1

    install_linux.write_autostart_entry(tmp_path / "b.icc", autostart_path=autostart)
    second = autostart.read_text()
    assert second.count(install_linux.AUTOSTART_MARKER) == 1
    assert "a.icc" not in second
    assert "b.icc" in second


def test_write_autostart_entry_preserves_other_lines(tmp_path):
    autostart = tmp_path / "autostart"
    autostart.parent.mkdir(parents=True, exist_ok=True)
    autostart.write_text("xset s off\n")
    install_linux.write_autostart_entry(tmp_path / "a.icc", autostart_path=autostart)
    text = autostart.read_text()
    assert "xset s off" in text
    assert "a.icc" in text


def test_install_full_linux_only_writes_autostart_when_asked(tmp_path, monkeypatch):
    _make_script(tmp_path, "dispwin", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    autostart = tmp_path / "autostart"
    install_linux.install(tmp_path / "p.icc", write_autostart=False, autostart_path=autostart)
    assert not autostart.exists()
    install_linux.install(tmp_path / "p.icc", write_autostart=True, autostart_path=autostart)
    assert autostart.exists()


def test_install_windows_guard_off_windows():
    if sys.platform == "win32":
        pytest.skip("this test only checks the non-Windows guard")
    report = install_windows.install("x.icc")
    assert not report.ok
    assert report.steps[0]["step"] == "platform"

import sys

import pytest

from calsuite.display import install_linux, install_windows

_FAKE_COLORMGR = """
import sys

sub = sys.argv[1] if len(sys.argv) > 1 else ""
if sub == "import-profile":
    print("Object Path: /org/freedesktop/ColorManager/profiles/p1")
elif sub == "get-devices-by-kind":
    print("Object Path: /org/freedesktop/ColorManager/devices/d1")
"""


def test_install_colormgr_full_chain(fake_bin, tmp_path):
    fake_bin("colormgr", _FAKE_COLORMGR)
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


def test_install_dispwin_success(fake_bin, tmp_path):
    fake_bin("dispwin", "")
    report = install_linux.install_dispwin(tmp_path / "p.icc")
    assert report.ok


def test_write_autostart_entry_is_idempotent(tmp_path):
    autostart = tmp_path / "autostart"
    install_linux.write_autostart_entry(tmp_path / "a.icc", autostart_path=autostart)
    first = autostart.read_text(encoding="utf-8")
    assert 'a.icc' in first
    assert first.count(install_linux.AUTOSTART_MARKER) == 1

    install_linux.write_autostart_entry(tmp_path / "b.icc", autostart_path=autostart)
    second = autostart.read_text(encoding="utf-8")
    assert second.count(install_linux.AUTOSTART_MARKER) == 1
    assert "a.icc" not in second
    assert "b.icc" in second


def test_write_autostart_entry_preserves_other_lines(tmp_path):
    autostart = tmp_path / "autostart"
    autostart.parent.mkdir(parents=True, exist_ok=True)
    autostart.write_text("xset s off\n", encoding="utf-8")
    install_linux.write_autostart_entry(tmp_path / "a.icc", autostart_path=autostart)
    text = autostart.read_text(encoding="utf-8")
    assert "xset s off" in text
    assert "a.icc" in text


def test_install_full_linux_only_writes_autostart_when_asked(fake_bin, tmp_path):
    fake_bin("dispwin", "")
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

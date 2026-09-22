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


def test_load_vcgt_cal_success(fake_bin, tmp_path):
    fake_bin("dispwin", "")
    report = install_linux.load_vcgt_cal(tmp_path / "vcgt.cal")
    assert report.ok


def test_install_full_linux_loads_vcgt_when_cal_path_given(fake_bin, tmp_path):
    fake_bin("dispwin", "")
    report = install_linux.install(tmp_path / "p.icc", cal_path=tmp_path / "p.cal")
    step_names = [s["step"] for s in report.steps]
    assert "dispwin <calfile>" in step_names

    report_none = install_linux.install(tmp_path / "p.icc", cal_path=None)
    assert "dispwin <calfile>" not in [s["step"] for s in report_none.steps]


def test_write_autostart_entry_with_cal_path_writes_both_lines_and_is_idempotent(tmp_path):
    autostart = tmp_path / "autostart"
    install_linux.write_autostart_entry(tmp_path / "a.icc", cal_path=tmp_path / "a.cal", autostart_path=autostart)
    first = autostart.read_text(encoding="utf-8")
    assert "a.icc" in first
    assert "a.cal" in first
    assert first.count(install_linux.AUTOSTART_MARKER) == 1
    assert first.count(install_linux.AUTOSTART_MARKER_END) == 1

    # Idempotent even when the new call has *fewer* lines than the old
    # block (no cal_path this time) -- the old 2-line block must be fully
    # removed, not just partially overwritten.
    install_linux.write_autostart_entry(tmp_path / "b.icc", cal_path=None, autostart_path=autostart)
    second = autostart.read_text(encoding="utf-8")
    assert "a.icc" not in second
    assert "a.cal" not in second
    assert "b.icc" in second
    assert second.count(install_linux.AUTOSTART_MARKER) == 1


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


def test_install_colormgr_refuses_to_guess_between_two_displays(fake_bin, tmp_path):
    """`find_display_device` returned whichever display colord enumerated
    first, and `install_colormgr` attached the profile to it -- so on a
    multi-monitor machine the profile measured for one panel was made the
    default for another, silently. The measured panel's identity is in the
    record; colord's enumeration order says nothing about it.
    """
    fake_bin(
        "colormgr",
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'import-profile':\n"
        "    print('Object Path: /org/freedesktop/ColorManager/profiles/p1')\n"
        "elif args[0] == 'get-devices-by-kind':\n"
        "    print('Object Path: /org/freedesktop/ColorManager/devices/d1')\n"
        "    print('Object Path: /org/freedesktop/ColorManager/devices/d2')\n",
    )
    icc = tmp_path / "p.icc"
    icc.write_bytes(b"not a real profile")

    report = install_linux.install_colormgr(icc)
    assert not report.ok
    step = [s for s in report.steps if s["step"] == "colormgr get-devices-by-kind display"][0]
    assert "2 display devices" in step["detail"]
    assert "--colord-device" in step["detail"]

    # Told which one, it proceeds.
    chosen = install_linux.install_colormgr(icc, device_path="/org/freedesktop/ColorManager/devices/d2")
    assert chosen.ok, chosen.steps

    # An unknown device path is an error, not a silent fallback to the first.
    wrong = install_linux.install_colormgr(icc, device_path="/org/freedesktop/ColorManager/devices/nope")
    assert not wrong.ok

import pytest

from calsuite.capture import gphoto2

# A fake gphoto2 on PATH, dispatching on which flag it was given -- lets
# capture/gphoto2.py's subprocess wrapper be tested with no camera attached
# (docs/implementation-plan.md: "testable with a fake gphoto2 script on PATH").
# Written in Python (via the shared `fake_bin` fixture, tests/conftest.py)
# rather than shell so the exact same fixture works on Windows CI too.
_FAKE_GPHOTO2 = """
import sys

argv = sys.argv[1:]
filename = ""
for arg in argv:
    if arg.startswith("--filename="):
        filename = arg[len("--filename="):]

if any(a == "--auto-detect" for a in argv):
    print("Model                          Port")
    print("----------------------------------------------------------")
    print("Canon EOS R100                 usb:001,004")
elif any(a == "--capture-image-and-download" for a in argv):
    print(f"Saving file as {filename}")
elif any(a.startswith("--set-config") for a in argv):
    print("ok")
else:
    print("unrecognized")
    sys.exit(1)
"""


@pytest.fixture
def fake_gphoto2_on_path(fake_bin):
    return fake_bin("gphoto2", _FAKE_GPHOTO2)


def test_is_camera_connected_true(fake_gphoto2_on_path):
    assert gphoto2.is_camera_connected() is True


def test_is_camera_connected_false_when_gphoto2_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert gphoto2.is_camera_connected() is False


def test_detect_returns_raw_text(fake_gphoto2_on_path):
    assert "Canon EOS R100" in gphoto2.detect()


def test_set_config_runs_without_error(fake_gphoto2_on_path):
    gphoto2.set_config(iso="800", aperture="4")  # must not raise


def test_capture_and_download_returns_path(fake_gphoto2_on_path, tmp_path):
    out_dir = tmp_path / "out"
    saved = gphoto2.capture_and_download(out_dir, filename="test.cr3")
    assert saved == out_dir / "test.cr3"

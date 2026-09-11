import os

import pytest

from calsuite.capture import gphoto2

# A fake gphoto2 on PATH, dispatching on which flag it was given -- lets
# capture/gphoto2.py's subprocess wrapper be tested with no camera attached
# (docs/implementation-plan.md: "testable with a fake gphoto2 script on PATH").
_FAKE_GPHOTO2 = """#!/bin/sh
FN=""
for arg in "$@"; do
  case "$arg" in
    --filename=*) FN="${arg#--filename=}" ;;
  esac
done
case "$*" in
  *--auto-detect*)
    printf '%s\\n' 'Model                          Port'
    printf '%s\\n' '----------------------------------------------------------'
    printf '%s\\n' 'Canon EOS R100                 usb:001,004'
    ;;
  *--capture-image-and-download*)
    echo "Saving file as $FN"
    ;;
  *--set-config*)
    echo ok
    ;;
  *)
    echo unrecognized
    exit 1
    ;;
esac
"""


@pytest.fixture
def fake_gphoto2_on_path(tmp_path, monkeypatch):
    script = tmp_path / "gphoto2"
    script.write_text(_FAKE_GPHOTO2)
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    return script


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

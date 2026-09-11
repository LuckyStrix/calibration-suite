import os
import sys

import pytest

from calsuite.display import osstate
from calsuite.fit import Refusal


def _make_script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def test_reset_gamma_table_missing_dispwin(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    done, method = osstate.reset_gamma_table()
    assert done is False
    assert "not found" in method


def test_reset_gamma_table_success(tmp_path, monkeypatch):
    _make_script(tmp_path, "dispwin", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    done, method = osstate.reset_gamma_table()
    assert done is True
    assert method == "dispwin -c"


def test_reset_gamma_table_failure(tmp_path, monkeypatch):
    _make_script(tmp_path, "dispwin", "#!/bin/sh\necho boom 1>&2\nexit 1\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    done, method = osstate.reset_gamma_table()
    assert done is False
    assert "failed" in method


def test_check_x11_icc_profile_unset(tmp_path, monkeypatch):
    _make_script(tmp_path, "xprop", "#!/bin/sh\necho '_ICC_PROFILE:  not found.'\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    atom, warning = osstate.check_x11_icc_profile()
    assert atom is None
    assert warning is None


def test_check_x11_icc_profile_set(tmp_path, monkeypatch):
    _make_script(tmp_path, "xprop", "#!/bin/sh\necho '_ICC_PROFILE(CARDINAL) = 26, 0, 0, 0'\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    atom, warning = osstate.check_x11_icc_profile()
    assert atom is not None
    assert "may be active" in warning


def test_check_x11_icc_profile_missing_xprop(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    atom, warning = osstate.check_x11_icc_profile()
    assert atom is None
    assert "not found" in warning


def test_check_hdr_windows_off_windows():
    if sys.platform == "win32":
        pytest.skip("this test only checks the non-Windows guard")
    hdr_on, method = osstate.check_hdr_windows()
    assert hdr_on is None
    assert method == "not running on Windows"


def test_refuse_if_hdr_on_true():
    state = osstate.OSState(
        platform="win32", gamma_reset=True, gamma_reset_method="x", icc_profile_atom=None,
        profile_loader_warning=None, hdr_on=True, hdr_method="test",
    )
    refusal = osstate.refuse_if_hdr_on(state)
    assert isinstance(refusal, Refusal)
    assert refusal.check == "hdr_on"


def test_refuse_if_hdr_on_unknown_without_confirm():
    state = osstate.OSState(
        platform="win32", gamma_reset=True, gamma_reset_method="x", icc_profile_atom=None,
        profile_loader_warning=None, hdr_on=None, hdr_method="test",
    )
    refusal = osstate.refuse_if_hdr_on(state)
    assert isinstance(refusal, Refusal)
    assert refusal.check == "hdr_state_unknown"


def test_refuse_if_hdr_on_false_passes():
    state = osstate.OSState(
        platform="linux", gamma_reset=True, gamma_reset_method="x", icc_profile_atom=None,
        profile_loader_warning=None, hdr_on=False, hdr_method="test",
    )
    assert osstate.refuse_if_hdr_on(state) is None


def test_gather_on_linux_sets_hdr_false(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("linux-specific behavior")
    _make_script(tmp_path, "dispwin", "#!/bin/sh\nexit 0\n")
    _make_script(tmp_path, "xprop", "#!/bin/sh\necho '_ICC_PROFILE:  not found.'\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    state = osstate.gather(osd={"brightness": "80%"})
    assert state.platform.startswith("linux")
    assert state.hdr_on is False
    assert state.gamma_reset is True
    assert state.osd == {"brightness": "80%"}
    assert osstate.refuse_if_hdr_on(state) is None

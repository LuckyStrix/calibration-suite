"""The Windows spotread path (``spotread_oneshot.py``): calibration in its own
window, then one ``spotread -e -N -O`` process per reading, against a fake
spotread that prints one result and exits -- what the real binary did on the
laptop with ``-N -O``. Runs on every OS: ``one_shot=True`` is forced, and the
"own console window" launcher is replaced, since a test cannot press keys
in a new window."""

from __future__ import annotations

import pytest

from calsuite import tools
from calsuite.display.backends import argyll as argyllmod
from calsuite.display.backends import spotread_oneshot as oneshotmod
from calsuite.display.backends.spotread_session import DeadReading, NeedsRecalibration, SessionAborted
from calsuite.display.window import WindowAborted

FAKE_SPOTREAD_ONESHOT = r'''
import os, sys, time
from pathlib import Path

if sys.argv[1:] == ["-?"]:
    print("usage: spotread [-options] [logfile]\n    1 = 'usb (X-Rite ColorMunki)'\n")
    sys.exit(1)
log = Path(os.environ["FAKE_LOG"])
with log.open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
n = len(log.read_text().splitlines())
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "hang":
    time.sleep(30)
if mode == "needs_cal":
    print("\nSpot read needs a calibration before continuing\n")
    sys.exit(1)
if mode == "zeros":
    print(" Result is XYZ: 0.000000 0.000000 0.000000, D50 Lab: 0 0 0")
    sys.exit(0)
print("\nPlace instrument on spot to be measured,")
print(f" Result is XYZ: {10 * n:.6f} {20 * n:.6f} {30 * n:.6f}, D50 Lab: 1.0 2.0 3.0")
'''


@pytest.fixture
def spotread(fake_bin, monkeypatch, tmp_path):
    log = tmp_path / "calls.txt"
    windows = []

    def install(**env):
        fake_bin("spotread", FAKE_SPOTREAD_ONESHOT)
        monkeypatch.setenv("FAKE_LOG", str(log))
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        monkeypatch.setattr(oneshotmod, "run_in_own_console", lambda argv: windows.append(argv[1:]) or 0)
        monkeypatch.setattr(argyllmod, "_default_one_shot", lambda: True)

    def calls():
        return log.read_text().splitlines() if log.exists() else []

    install.calls = calls
    install.windows = windows
    return install


def test_each_reading_is_its_own_dash_n_dash_o_process(spotread):
    spotread()
    with oneshotmod.OneShotSpotread(["-e"]) as s:
        assert s.measure() == pytest.approx((10, 20, 30))
        assert s.measure() == pytest.approx((20, 40, 60))
    assert spotread.calls() == ["-e -N -O", "-e -N -O"]


def test_session_calibrates_in_its_own_window_before_any_reading(spotread):
    from calsuite.display import patches as patchesmod

    spotread()
    said = []
    backend = argyllmod.ArgyllBackend()
    with backend.session(say=said.append, ask=lambda _p: "") as session:
        assert spotread.windows == [["-e"]]  # interactive spotread, no -N/-O
        assert spotread.calls() == []  # nothing read yet
        results = backend.measure(patchesmod.primaries_secondaries()[:3], session)
    assert [v for r in results for v in r.xyz] == pytest.approx([10, 20, 30, 20, 40, 60, 30, 60, 90])
    assert "separate spotread window" in "\n".join(said)
    assert "X-Rite ColorMunki" in backend.accuracy().basis


def test_display_type_reaches_both_the_calibration_window_and_the_readings(spotread):
    spotread()
    backend = argyllmod.ArgyllBackend(instrument_display_type="l")
    with backend.session(say=lambda _t: None, ask=lambda _p: "") as session:
        session.measure()
    assert spotread.windows == [["-y", "l", "-e"]]
    assert spotread.calls() == ["-y l -e -N -O"]


def test_quitting_after_calibration_aborts(spotread):
    spotread()
    with pytest.raises(SessionAborted):
        argyllmod.ArgyllBackend().session(say=lambda _t: None, ask=lambda _p: "q")


def test_power_cycle_calibration_uses_the_window_once_then_skips_it(spotread):
    spotread()
    asked = []
    backend = argyllmod.ArgyllBackend()
    backend.calibrate_with_power_cycle(lambda _t: None, lambda p: asked.append(p) or "")
    assert backend.skip_calibration is True
    with backend.session(say=lambda _t: None, ask=lambda _p: "") as session:
        session.measure()
    assert spotread.windows == [["-e"]]  # not a second time for the session
    assert len(asked) == 3  # power-cycle 1, "window showed a reading?", power-cycle 2


def test_a_reading_that_wants_calibration_raises_needs_recalibration(spotread):
    spotread(FAKE_MODE="needs_cal")
    with pytest.raises(NeedsRecalibration):
        oneshotmod.OneShotSpotread().measure()


def test_an_all_zero_reading_raises(spotread):
    spotread(FAKE_MODE="zeros")
    with pytest.raises(DeadReading):
        oneshotmod.OneShotSpotread().measure()


def test_a_hung_reading_times_out_instead_of_freezing(spotread):
    spotread(FAKE_MODE="hang")
    with pytest.raises(tools.ToolError, match="did not finish a reading"):
        oneshotmod.OneShotSpotread().measure(timeout=1.0)


def test_poll_can_abort_a_reading_in_progress(spotread):
    spotread(FAKE_MODE="hang")

    def poll():
        raise WindowAborted("esc")

    with pytest.raises(WindowAborted):
        oneshotmod.OneShotSpotread().measure(poll=poll)


def test_command_path_reads_every_patch_through_the_window(spotread, monkeypatch, tmp_path):
    import argparse

    from calsuite import store as storemod
    from calsuite.display import commands as cmds
    from calsuite.display import patches as patchesmod

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    spotread()
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt.strip()) or "")
    args = argparse.Namespace(backend="argyll", width=64, height=48, windowed=True)
    patches = patchesmod.primaries_secondaries()[:3]

    measurements, _accuracy = cmds._measure_via_backend(args, patches, store=storemod.Store(tmp_path))

    assert [v for m in measurements for v in m.xyz] == pytest.approx([10, 20, 30, 20, 40, 60, 30, 60, 90])
    assert len(prompts) == 2 and prompts[-1].endswith("start measuring:")

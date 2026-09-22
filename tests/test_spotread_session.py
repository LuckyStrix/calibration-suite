"""``SpotreadSession`` against a fake spotread that replays the transcript
captured from a real ColorMunki Photo (see ``spotread_session.py``'s
docstring): calibration prompt repeating until the dial is right, then an
idle "take a reading" prompt, ``Result is XYZ:`` per key, and quit on ``q``.

The fake is stateful and reads one key at a time from stdin, like the real
thing -- the old fake just printed a result and exited, which is exactly the
one-process-per-reading model that could not work on a real instrument."""

from __future__ import annotations

import pytest

from calsuite import tools
from calsuite.display.backends import argyll as argyllmod
from calsuite.display.backends.spotread_session import SessionAborted, SpotreadSession
from calsuite.display.window import WindowAborted

FAKE_SPOTREAD = r'''
import os, sys, time
try:
    import tty
    if sys.stdin.isatty():
        tty.setcbreak(0)   # real spotread reads keys unbuffered
except ImportError:
    pass

def say(s):
    sys.stdout.write(s); sys.stdout.flush()

def key():
    return os.read(0, 1)

MODE = os.environ.get("FAKE_MODE", "ok")
WRONG = int(os.environ.get("FAKE_DIAL_WRONG_TRIES", "1"))
READY = ("\nPlace instrument on spot to be measured,\nand hit [A-Z] to read white and setup FWA compensation\n"
         "Hit ESC or Q to exit, instrument switch or any other key to take a reading: ")

if sys.argv[1:] == ["-?"]:
    # The real usage text lists instruments found on USB first, then serial ports.
    say("usage: spotread [-options] [logfile]\n -c listno            Set instrument port from the following list (default 1)\n"
        "    1 = '/dev/bus/usb/003/008 (X-Rite ColorMunki)'\n    2 = '/dev/ttyS0'\n")
    sys.exit(1)
if os.environ.get("FAKE_BANNER") == "1":   # the real spotread only prints this with -v
    say("Connecting to the instrument ..\nInstrument Type:   ColorMunki\nInit instrument success !\n")
say("\nSpot read needs a calibration before continuing\n")
wrong = 0
while True:
    say("\nSet instrument sensor to calibration position,\n and then hit any key to continue,\n"
        " or hit Esc or Q to abort: ")
    k = key()
    if k in (b"q", b""):
        sys.exit(0)
    if wrong >= WRONG:
        break
    wrong += 1
say("\nCalibration complete\n")
say(READY)
n = 0
while True:
    k = key()
    if k in (b"q", b"\x1b", b""):
        sys.exit(0)
    n += 1
    if MODE == "hang":
        time.sleep(30)
    if MODE == "misread" and n == 2:
        say("\nSpot read failed due to misread\n"); say(READY); continue
    say(f"\n Result is XYZ: {10 * n:.6f} {20 * n:.6f} {30 * n:.6f}, D50 Lab: 1.0 2.0 3.0\n"); say(READY)
'''


@pytest.fixture
def spotread(fake_bin, monkeypatch):
    def install(**env):
        fake_bin("spotread", FAKE_SPOTREAD)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))

    return install


def test_calibrates_relaying_prompts_then_reads(spotread):
    spotread(FAKE_BANNER=1)  # dial wrong once, then right
    said, asked = [], []
    with SpotreadSession() as s:
        s.prepare(said.append, lambda p: asked.append(p) or "")
        assert s.instrument == "ColorMunki"
        assert len(asked) == 2  # the human was asked again after the wrong dial position
        assert all("calibration position" in t for t in said)
        assert s.measure() == pytest.approx((10, 20, 30))
        assert s.measure() == pytest.approx((20, 40, 60))


def test_gives_up_when_the_dial_never_reaches_position(spotread):
    spotread(FAKE_DIAL_WRONG_TRIES=99)
    with SpotreadSession() as s:
        with pytest.raises(tools.ToolError, match="dial"):
            s.prepare(lambda _t: None, lambda _p: "")


def test_human_can_quit_at_a_prompt(spotread):
    spotread()
    with SpotreadSession() as s:
        with pytest.raises(SessionAborted):
            s.prepare(lambda _t: None, lambda _p: "q")


def test_a_misread_raises_but_the_session_stays_usable(spotread):
    spotread(FAKE_DIAL_WRONG_TRIES=0, FAKE_MODE="misread")
    with SpotreadSession() as s:
        s.prepare(lambda _t: None, lambda _p: "")
        assert s.measure() == pytest.approx((10, 20, 30))
        with pytest.raises(tools.ToolError, match="no reading"):
            s.measure()
        assert s.measure() == pytest.approx((30, 60, 90))


def test_poll_can_abort_a_reading_in_progress(spotread):
    spotread(FAKE_DIAL_WRONG_TRIES=0, FAKE_MODE="hang")
    with SpotreadSession() as s:
        s.prepare(lambda _t: None, lambda _p: "")

        def poll():
            raise WindowAborted("esc")

        with pytest.raises(WindowAborted):
            s.measure(poll=poll)


def test_missing_spotread_raises(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    with pytest.raises(tools.ToolError):
        argyllmod.ArgyllBackend().session()


def test_backend_measures_patches_and_names_the_instrument(spotread):
    from calsuite.display import patches as patchesmod

    spotread(FAKE_DIAL_WRONG_TRIES=0)
    backend = argyllmod.ArgyllBackend()
    patches = patchesmod.primaries_secondaries()[:3]
    with backend.session(say=lambda _t: None, ask=lambda _p: "") as session:
        results = backend.measure(patches, session)
    assert [tuple(r.xyz) for r in results] == pytest.approx([(10, 20, 30), (20, 40, 60), (30, 60, 90)])
    # spotread prints no banner without -v: the name has to come from its port list
    assert "X-Rite ColorMunki" in backend.accuracy().basis


def test_command_path_calibrates_before_the_window_and_reads_every_patch(spotread, monkeypatch, tmp_path):
    """The real wiring in display/commands.py: session (prompts) -> Enter ->
    window -> one reading per patch, against a dummy SDL video driver."""
    import argparse

    from calsuite import store as storemod
    from calsuite.display import commands as cmds
    from calsuite.display import patches as patchesmod

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    spotread()
    events = []
    monkeypatch.setattr("builtins.input", lambda prompt="": events.append(prompt.strip()) or "")
    args = argparse.Namespace(backend="argyll", width=64, height=48, windowed=True)
    patches = patchesmod.primaries_secondaries()[:3]

    measurements, accuracy = cmds._measure_via_backend(args, patches, store=storemod.Store(tmp_path))

    assert [tuple(m.xyz) for m in measurements] == pytest.approx([(10, 20, 30), (20, 40, 60), (30, 60, 90)])
    assert len(events) == 3  # two dial prompts (one wrong try), then "place it on the screen"
    assert events[-1].endswith("start measuring:")
    assert "X-Rite ColorMunki" in accuracy.basis


def test_identify_instrument_reads_the_default_port_without_a_serial(spotread):
    from calsuite.display.backends.spotread_session import identify_instrument

    spotread()
    assert identify_instrument() == "X-Rite ColorMunki"


def test_identify_instrument_is_none_when_nothing_is_listed(fake_bin):
    from calsuite.display.backends.spotread_session import identify_instrument

    fake_bin("spotread", "print(\"    1 = '/dev/ttyS0'\")\n")
    assert identify_instrument() is None

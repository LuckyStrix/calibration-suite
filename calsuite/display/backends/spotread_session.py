"""One long-lived interactive ``spotread`` process, driven key by key.

Why a session and not one ``spotread -O`` per patch: spotread has to
*calibrate* before its first reading -- on a ColorMunki that means the human
turns the instrument's dial to the calibration position and confirms -- and
it re-does that for every fresh process. Run per patch with its output
captured (what this backend first did), the calibration prompt was invisible
behind a fullscreen black patch window, and spotread waited on a dial nobody
knew to turn until the timeout, once per patch.

The transcript this parses was captured from a real ColorMunki Photo
(``spotread -e``, ArgyllCMS 2.3.1), not written from the docs::

    Spot read needs a calibration before continuing

    Set instrument sensor to calibration position,
     and then hit any key to continue,
     or hit Esc or Q to abort:          <- repeats until the dial is right
    Calibration complete

    Place instrument on spot to be measured,
    and hit [A-Z] to read white and setup FWA compensation (keyed to letter)
    ...
    Hit ESC or Q to exit, instrument switch or any other key to take a reading:
     Result is XYZ: 6.322786 5.570381 2.072769, D50 Lab: 28.30 10.67 17.80
    Place instrument on spot to be measured, ...   <- ready for the next patch

Letters are *commands* at that prompt (``r`` sets a reference, ``k``
recalibrates, ``s`` saves a spectrum...), so the only key this module ever
sends is a space -- "any other key" -- and ``q`` to quit.

spotread reads its keys from stdin as a terminal would, so on POSIX it runs
under a pty (a plain pipe is not what the transcript above was captured
from). Windows has no ``pty``; there it falls back to plain pipes, which is
enough for a fake spotread in CI but is **unverified against a real
instrument** -- ArgyllCMS on Windows reads the console API directly.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time

from calsuite import tools
from calsuite.display import constants as dc

try:
    import pty
except ImportError:  # Windows
    pty = None

READY_PROMPT = "any other key to take a reading"
# The last line of the prompt spotread shows whenever it is idle and
# waiting to be told to measure ("Hit ESC or Q to exit, instrument switch or
# any other key to take a reading:").

ACTION_PROMPT_RE = re.compile(r"any key to continue,\s*or hit Esc or Q to abort:\s*$")
# A prompt that needs a *human* to do something physical first (the
# calibration one above; any other dial-position prompt has the same shape).

RESULT_RE = re.compile(
    r"Result is XYZ:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)"
)
INSTRUMENT_RE = re.compile(r"Instrument Type:\s*(.+)")
# Only printed by ``spotread -v``; a normal run's transcript starts at the
# calibration prompt, so this alone left records saying "instrument not
# identified". ``identify_instrument`` below is what actually works.

PORT_LIST_RE = re.compile(r"^\s*1 = '.*?\((.+?)\)'\s*$", re.MULTILINE)

MAX_ACTION_PROMPTS = 6
# A wrong dial position makes spotread re-ask forever. Each ask is a real
# human round trip, so give up after a handful and say why rather than loop.


def identify_instrument() -> str | None:
    """Name of the instrument spotread will use (its default: port 1), from
    the port list in ``spotread -?`` -- e.g. ``"X-Rite ColorMunki"``. No
    serial number is in that line, and none is recorded (public repo).
    ``None`` if it can't be read (no USB instrument, or an unexpected format)."""
    try:
        out = tools.run(["spotread", "-?"], check=False, timeout=15)
    except tools.ToolError:
        return None
    m = PORT_LIST_RE.search(out.stdout + "\n" + out.stderr)
    return m.group(1).strip() if m else None


class SessionAborted(RuntimeError):
    """The human chose to quit at a prompt."""


def _clean(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r", "").split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())


class SpotreadSession:
    def __init__(self, args=("-e",)):
        self.args = [str(a) for a in args]
        self.instrument: str | None = None
        self._proc: subprocess.Popen | None = None
        self._rfd: int | None = None
        self._buf = ""
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._eof = False

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return  # already running -- `with backend.session() as s` enters an already-prepared session
        exe = tools.which("spotread")
        if exe is None:
            raise tools.ToolError("'spotread' is not on PATH")
        argv = [exe, *self.args]
        if pty is not None:
            master, slave = pty.openpty()
            self._proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
            os.close(slave)
            self._rfd = self._wfd = master
        else:
            self._proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
            )
            self._rfd = self._proc.stdout.fileno()
            self._wfd = self._proc.stdin.fileno()
        threading.Thread(target=self._reader, daemon=True).start()

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is None:
            try:
                self._send(b"q")
                proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if pty is not None and self._rfd is not None:
            try:
                os.close(self._rfd)
            except OSError:
                pass

    def __enter__(self) -> SpotreadSession:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- plumbing ------------------------------------------------------

    def _reader(self) -> None:
        while True:
            try:
                data = os.read(self._rfd, 4096)
            except OSError:
                data = b""
            if not data:
                with self._lock:
                    self._eof = True
                self._changed.set()
                return
            with self._lock:
                self._buf += data.decode("utf-8", "replace")
                if self.instrument is None:
                    m = INSTRUMENT_RE.search(self._buf)
                    if m:
                        self.instrument = m.group(1).strip()
            self._changed.set()

    def _send(self, data: bytes) -> None:
        os.write(self._wfd, data)

    def _take(self) -> str:
        with self._lock:
            text, self._buf = self._buf, ""
        return text

    def _wait_for(self, pred, timeout: float, poll=None) -> str:
        """Block until ``pred(buffer)`` is true, then return the buffer
        (without clearing it). ``poll`` is called ~20x/s while waiting --
        the caller's chance to pump a window's events and raise to abort."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                buf, eof = self._buf, self._eof
            if pred(buf):
                return buf
            if eof:
                raise tools.ToolError(f"spotread exited unexpectedly. Its output:\n{_clean(buf)[-800:]}")
            if time.monotonic() > deadline:
                raise tools.ToolError(f"spotread did not respond within {timeout:.0f}s. Its output:\n{_clean(buf)[-800:]}")
            if poll is not None:
                poll()
            self._changed.wait(0.05)
            self._changed.clear()

    # -- the two phases ------------------------------------------------

    def prepare(self, say=None, ask=None) -> None:
        """Get spotread to its idle "take a reading" prompt, relaying every
        prompt that needs the human (dial position, mostly) through
        ``say``/``ask``. Run this **before** any patch window opens -- the
        person must be able to see the terminal."""
        say, ask = say or print, ask or input
        actions = 0
        while True:
            buf = self._wait_for(
                lambda b: READY_PROMPT in b or ACTION_PROMPT_RE.search(b) is not None, dc.SPOTREAD_STARTUP_TIMEOUT_S
            )
            text = self._take()
            if READY_PROMPT in buf and not ACTION_PROMPT_RE.search(buf):
                return
            actions += 1
            if actions > MAX_ACTION_PROMPTS:
                raise tools.ToolError(
                    f"spotread still refusing after {MAX_ACTION_PROMPTS} attempts -- the instrument's dial "
                    f"never reached the position it asked for. Last prompt:\n{_clean(text)[-400:]}"
                )
            say(_clean(text)[-400:])
            answer = ask("  >> Press Enter when done (or q + Enter to quit): ")
            if answer.strip().lower().startswith("q"):
                raise SessionAborted("quit at an instrument prompt")
            self._send(b" ")

    def measure(self, poll=None, timeout: float = dc.SPOTREAD_TIMEOUT_S):
        """Take one reading of whatever is under the instrument; returns
        ``(X, Y, Z)`` in cd/m^2. Waits for spotread's *next* idle prompt
        before returning, so the following call's key can't be swallowed
        mid-reading."""
        self._take()
        self._send(b" ")

        def settled(b: str) -> bool:
            m = RESULT_RE.search(b)
            if m:
                return READY_PROMPT in b[m.end():]
            # No result, but back at the idle prompt or asking for the dial
            # again: the reading failed or the instrument wants recalibrating.
            return READY_PROMPT in b or ACTION_PROMPT_RE.search(b) is not None

        buf = self._wait_for(settled, timeout, poll)
        m = RESULT_RE.search(buf)
        if not m:
            raise tools.ToolError(f"spotread gave no reading (misread, or it wants recalibrating). Its output:\n{_clean(buf)[-600:]}")
        return tuple(float(g) for g in m.groups())

"""spotread on Windows: calibrate in its own console window, then one
``spotread -N -O`` process per reading.

Why not the interactive session (``spotread_session.py``) on Windows:
ArgyllCMS reads keys through the console API, not stdin. Sharing calsuite's
console, spotread puts it in raw mode and takes the keys meant for
calsuite's own prompts (Enter never arrives, Ctrl+C does nothing). Run under a
ConPTY via ``pywinpty`` instead (tried on branch ``windows-spotread-conpty``),
spotread's output arrived, but on a real Windows laptop with pywinpty 3.1.2
the *parent* console still stopped delivering Enter and Ctrl+C to Python's
``input()`` -- pywinpty opens and reconfigures the real console while
setting the pseudoconsole up. Neither shares a console safely.

So nothing here shares calsuite's console:

- **Calibration** runs ``spotread -e`` in a *new, visible* console window
  (``CREATE_NEW_CONSOLE``). The human calibrates, takes a test reading and
  quits it there, typing into spotread directly; calsuite only waits for it
  to exit. ArgyllCMS saves the instrument's calibration on exit.
- **Each reading** is ``spotread -e -N -O``: ``-N`` trusts that saved
  calibration instead of asking for one, ``-O`` takes one reading and exits.
  It runs with its own hidden console (``CREATE_NO_WINDOW``) and stdin from
  ``NUL``, so it cannot touch calsuite's console even if it does prompt.
  Its stdout is fully buffered when piped, but ``-O`` exits after the
  reading, which flushes it -- the reason a pipe works here and hangs for the
  long-lived session.

Verified on the real laptop (ColorMunki, ArgyllCMS, PowerShell) by hand:
``spotread -e`` calibrates interactively, and ``spotread -e -N -O > out.txt``
exits on its own with a ``Result is XYZ:`` line in ``out.txt``. The cost is
instrument start-up per reading (a second or two each).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

from calsuite import console, tools
from calsuite.display import constants as dc
from calsuite.display.backends.spotread_session import (
    RESULT_RE,
    DeadReading,
    NeedsRecalibration,
    SessionAborted,
)

NEEDS_CALIBRATION_TEXT = "needs a calibration"


def _creationflags(name: str) -> int:
    return getattr(subprocess, name, 0) if sys.platform == "win32" else 0


def run_in_own_console(argv: list[str]) -> int:
    """Run ``argv`` in a new console window and wait for it to exit. On
    POSIX there is no separate console to open, so it runs in this terminal
    (only reachable from tests, which replace this function)."""
    return subprocess.run(argv, creationflags=_creationflags("CREATE_NEW_CONSOLE")).returncode


def calibrate_in_own_window(base_args: list[str], say=None, ask=None) -> None:
    """Have the human calibrate the instrument in a separate spotread window
    (see the module docstring), then confirm before going on."""
    say, ask = say or print, ask or console.prompt
    exe = tools.which("spotread")
    if exe is None:
        raise tools.ToolError("'spotread' is not on PATH")
    say(
        "\nCalibrating the instrument in a separate spotread window.\n"
        "  In THAT window: turn the dial to the calibration position and press a key when it asks.\n"
        "  Then turn the dial to the measure position, hold the instrument on a white part of the screen\n"
        "  and press a key once to check it gives a reading. Press q to close the window."
    )
    rc = run_in_own_console([exe, *base_args])
    say(f"spotread window closed (exit code {rc}).")
    answer = ask("  >> Dial on MEASURE and the window showed a reading? Press Enter (or q + Enter to quit): ")
    if answer.strip().lower().startswith("q"):
        raise SessionAborted("quit after calibration")


class OneShotSpotread:
    """Same surface the Argyll backend uses from ``SpotreadSession``
    (``start``/``prepare``/``measure``/``close``, context manager,
    ``instrument``), one ``spotread -N -O`` process per ``measure()``."""

    def __init__(self, base_args=("-e",)):
        self.args = [str(a) for a in base_args]
        if "-N" not in self.args:
            self.args.append("-N")
        if "-O" not in self.args:
            self.args.append("-O")
        self.instrument: str | None = None

    def start(self) -> None:
        if tools.which("spotread") is None:
            raise tools.ToolError("'spotread' is not on PATH")

    def prepare(self, say=None, ask=None) -> int:
        return 0  # calibration happened in its own window, before this

    def close(self) -> None:
        pass

    def __enter__(self) -> OneShotSpotread:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def measure(self, poll=None, timeout: float = dc.SPOTREAD_TIMEOUT_S):
        """One reading of whatever is under the instrument, ``(X, Y, Z)`` in
        cd/m^2. ``poll`` is called ~20x/s while spotread runs (ESC in the
        patch window raises from it); the process is killed if it raises."""
        exe = tools.which("spotread")
        if exe is None:
            raise tools.ToolError("'spotread' is not on PATH")
        proc = subprocess.Popen(
            [exe, *self.args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=_creationflags("CREATE_NO_WINDOW"),
        )
        out: list[bytes] = []
        reader = threading.Thread(target=lambda: out.append(proc.stdout.read()), daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        try:
            while reader.is_alive():
                if time.monotonic() > deadline:
                    raise tools.ToolError(
                        f"spotread did not finish a reading within {timeout:.0f}s -- it is probably waiting to be "
                        "recalibrated. Calibrate again and restart the run."
                    )
                if poll is not None:
                    poll()
                reader.join(0.05)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        text = (out[0] if out else b"").decode("utf-8", "replace").replace("\r", "")
        m = RESULT_RE.search(text)
        if not m:
            if NEEDS_CALIBRATION_TEXT in text:
                raise NeedsRecalibration(
                    f"spotread wants the instrument recalibrated before it will read. Its output:\n{text[-600:]}"
                )
            raise tools.ToolError(f"spotread gave no reading (exit code {proc.returncode}). Its output:\n{text[-600:]}")
        xyz = tuple(float(g) for g in m.groups())
        if not any(xyz):
            raise DeadReading(
                "the instrument returned an all-zero reading (X=Y=Z=0). Its dial is probably at the calibration "
                "position, or the sensor is blocked. Turn it to the measure position and start the run again."
            )
        return xyz

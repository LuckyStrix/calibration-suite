"""ArgyllCMS ``spotread`` backend (docs/design.md §5.1: "Colorimeter via
ArgyllCMS spotread ... the reference").

Flags verified against ArgyllCMS's own documentation, fetched 2026-09-11:
https://www.argyllcms.com/doc/spotread.html --

- ``-e``: "Use emissive measurement mode (absolute results)" -- a display
  is emissive, not reflective/transmissive, so this is required for every
  reading this backend takes.
- ``-O``: "Do one cal. or measure and exit" -- spotread's default is an
  interactive loop that takes a reading each time a key is hit; ``-O`` is
  what turns it into a single non-looping measurement, the shape scripted
  driving needs.
- ``-y <X>``: "Display type - instrument specific list to choose from" --
  optional, left unset unless the caller supplies one (the right value is
  instrument-specific and unknown without a real colorimeter attached to
  this build machine).
- Output: XYZ prints as an "XYZ: ..." line, "0..100 for reflective or
  transmissive readings, and absolute cd/m^2 for display, emissive and
  ambient readings" (per the fetched doc). The exact surrounding text
  format (spacing/punctuation) is Argyll's own well-known spotread
  transcript convention from real-world use, not itself shown verbatim in
  the fetched page, so ``_XYZ_RE`` below matches tolerantly (any
  separator) rather than a single exact string -- ArgyllCMS is not
  installed on this build machine (docs/design.md §0), so this parser is
  exercised only against a fake ``spotread`` script (``tests/
  test_display_backends_argyll.py``) that prints output in this shape, not
  against a real instrument.

Uses a raw ``subprocess.run`` rather than ``calsuite.tools.run``: spotread's
interactive prompt (in real-world use) commonly waits for a keypress/Enter
before it actually triggers the instrument, and ``tools.run`` doesn't feed
a process any stdin. Sending one newline satisfies that if it's needed and
is harmless if it isn't (with ``-O``, spotread takes one reading and exits
either way).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import numpy as np

from calsuite import tools
from calsuite.display import constants as dc
from calsuite.display import window as windowmod
from calsuite.display.backends.base import Accuracy, Measurement

_XYZ_RE = re.compile(
    r"XYZ:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)"
)


def parse_xyz(stdout: str) -> tuple:
    match = _XYZ_RE.search(stdout)
    if not match:
        raise ValueError(f"could not find an 'XYZ: ...' reading in spotread output:\n{stdout}")
    return tuple(float(g) for g in match.groups())


def _run_spotread(extra_args: list, timeout: float = dc.SPOTREAD_TIMEOUT_S) -> str:
    exe = tools.which("spotread")
    if exe is None:
        raise tools.ToolError("'spotread' is not on PATH")
    args = [exe, *extra_args]
    try:
        proc = subprocess.run(args, input="\n", capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise tools.ToolError(f"{' '.join(args)} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise tools.ToolError(f"{' '.join(args)} exited {proc.returncode}\nstderr:\n{proc.stderr}")
    return proc.stdout


@dataclass
class ArgyllBackend:
    name: str = "argyll-spotread"
    instrument_display_type: str | None = None  # spotread -y <X>; instrument-specific, unset by default
    cross_checked_against: str | None = None

    def accuracy(self) -> Accuracy:
        from calsuite.constants import ESTIMATED_ACCURACY  # foundation constant, not a display/-specific one

        return Accuracy(
            de00_estimate=ESTIMATED_ACCURACY["display_colorimeter_de00"],
            basis="ArgyllCMS spotread, i1Display-Pro-class colorimeter (design §5.1/§10 estimate)",
            cross_checked_against=self.cross_checked_against,
        )

    def measure_one(self) -> np.ndarray:
        """Trigger exactly one spotread reading of whatever patch is
        currently on screen. Returns raw XYZ (cd/m^2)."""
        args = ["-e", "-O"]
        if self.instrument_display_type:
            args = ["-y", self.instrument_display_type, *args]
        return np.array(parse_xyz(_run_spotread(args)))

    def measure(self, patches: list) -> list:
        """Reads one spotread measurement per patch with no window
        interaction of its own -- for direct backend testing (a fake
        `spotread` on PATH) and for callers that have already arranged for
        the right patch to be on screen. The real measurement command
        (``display/commands.py``) instead drives ``measure_via_window``,
        which interleaves showing each patch with triggering a reading.
        """
        out = []
        for patch in patches:
            xyz = self.measure_one()
            out.append(Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)))
        return out

    def measure_via_window(self, screen, patches: list, *, settle_s: float = dc.SETTLE_TIME_S, sleep=None) -> list:
        """Show each patch (``display.window.run_patch_sequence``), settle,
        then trigger one spotread reading -- the real driven-measurement
        path."""
        xyzs = windowmod.run_patch_sequence(
            screen, patches, lambda _patch: self.measure_one(), settle_s=settle_s, sleep=sleep
        )
        return [
            Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)) for patch, xyz in zip(patches, xyzs, strict=True)
        ]

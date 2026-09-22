"""ArgyllCMS ``spotread`` backend (docs/design.md §5.1: "Colorimeter via
ArgyllCMS spotread ... the reference").

Driven as one long-lived interactive process (``spotread_session.py`` has
the reasons and the real transcript), not one process per patch:

- ``-e``: "Use emissive measurement mode (absolute results)" -- a display
  is emissive, not reflective/transmissive.
- No ``-O``: that flag ("do one cal. or measure and exit") makes every
  reading a fresh process with a fresh calibration -- on a real ColorMunki
  Photo that meant a dial-position prompt per patch, invisible behind the
  fullscreen patch window.
- ``-y <X>``: "Display type - instrument specific list to choose from" --
  optional and left unset. The ColorMunki Photo this was first run against
  did not ask for one.
- Output: ``Result is XYZ: X Y Z, D50 Lab: ...`` -- absolute cd/m^2.

Verified against a real ColorMunki Photo (spectrophotometer, ArgyllCMS
2.3.1) for the prompt/reading protocol only. **Not** yet verified: that the
readings are *accurate* on this laptop panel (see ``accuracy()``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from calsuite.display import constants as dc
from calsuite.display import window as windowmod
from calsuite.display.backends.base import Accuracy, Measurement
from calsuite.display.backends.spotread_session import SpotreadSession, identify_instrument

_XYZ_RE = re.compile(
    r"XYZ:\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)[,\s]+([+-]?[\d.]+(?:[eE][+-]?\d+)?)"
)


def parse_xyz(stdout: str) -> tuple:
    match = _XYZ_RE.search(stdout)
    if not match:
        raise ValueError(f"could not find an 'XYZ: ...' reading in spotread output:\n{stdout}")
    return tuple(float(g) for g in match.groups())


@dataclass
class ArgyllBackend:
    name: str = "argyll-spotread"
    instrument_display_type: str | None = None  # spotread -y <X>; instrument-specific, unset by default
    cross_checked_against: str | None = None
    instrument: str | None = field(default=None, init=False)  # e.g. "ColorMunki", read from spotread's banner

    def accuracy(self) -> Accuracy:
        from calsuite.constants import ESTIMATED_ACCURACY  # foundation constant, not a display/-specific one

        return Accuracy(
            de00_estimate=ESTIMATED_ACCURACY["display_colorimeter_de00"],
            basis=(
                f"ArgyllCMS spotread, {self.instrument or 'instrument not identified'} "
                "(design §5.1/§10 colorimeter estimate -- not measured for this instrument on this panel)"
            ),
            cross_checked_against=self.cross_checked_against,
        )

    def session(self, say=None, ask=None) -> SpotreadSession:
        """Start spotread and bring it to "ready to measure", relaying its
        prompts (calibration, dial position) to the human through
        ``say``/``ask``. Call this **before** opening a patch window. Use
        as a context manager; the process is stopped on exit."""
        args = ["-e"]
        if self.instrument_display_type:
            args = ["-y", self.instrument_display_type, *args]
        session = SpotreadSession(args)
        session.start()
        try:
            session.prepare(say, ask)
        except BaseException:
            session.close()
            raise
        self.instrument = session.instrument or identify_instrument()
        return session

    def measure(self, patches: list, session: SpotreadSession | None = None) -> list:
        """One reading per patch with no window interaction of its own --
        for direct backend testing and for callers that have already
        arranged for the right patch to be on screen. Opens (and closes) its
        own session when none is passed."""
        if session is None:
            with self.session() as own:
                return self.measure(patches, own)
        return [
            Measurement(rgb=patch.rgb, xyz=np.array(session.measure()), uncertainty=np.zeros(3)) for patch in patches
        ]

    def measure_via_window(
        self, screen, patches: list, session: SpotreadSession, *, settle_s: float = dc.SETTLE_TIME_S, sleep=None
    ) -> list:
        """Show each patch (``display.window.run_patch_sequence``), settle,
        then take one reading from the already-prepared ``session``. Small
        squares (the uniformity grid) wait for the person to move the
        instrument there and press SPACE first. ESC stays live *during* a
        reading, not only between patches."""

        def poll() -> None:
            if windowmod.check_abort():
                raise windowmod.WindowAborted("aborted during a reading")

        xyzs = windowmod.run_patch_sequence(
            screen,
            patches,
            lambda _patch: np.array(session.measure(poll=poll)),
            settle_s=settle_s,
            sleep=sleep,
            confirm_placement=True,  # a handheld instrument has to be moved onto each uniformity square
        )
        return [
            Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)) for patch, xyz in zip(patches, xyzs, strict=True)
        ]

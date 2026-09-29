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
from calsuite.display.backends.spotread_session import SessionAborted, SpotreadSession, identify_instrument

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
    skip_calibration: bool = False  # spotread -N; for a unit calibrated separately (see session())
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

    def calibrate_with_power_cycle(self, say=None, ask=None) -> None:
        """For a unit whose dial-position report goes stale (it only updates
        at power-up): power-cycle at the calibration position, force a
        calibration, then power-cycle at the measure position. Afterwards
        ``session()`` runs with ``-N`` so spotread trusts the calibration
        it just saved (Argyll caches it on disk, per instrument) instead of
        asking about a dial position the unit can't report."""
        say, ask = say or print, ask or input

        def step(text: str) -> None:
            say(text)
            if ask("  >> Press Enter when done (or q + Enter to quit): ").strip().lower().startswith("q"):
                raise SessionAborted("quit during power-cycle calibration")

        step(
            "\nStep 1 of 2 -- calibrate.\n  Unplug the instrument, turn its dial fully to the CALIBRATION "
            "position (it clicks), then plug it back in."
        )
        with SpotreadSession(["-e"]) as cal:
            if cal.prepare(say, ask) == 0:  # already calibrated per its cache: calibrate anyway
                cal.recalibrate(say, ask)
        say("Calibration complete.")
        step(
            "\nStep 2 of 2 -- measure.\n  Unplug the instrument, turn its dial to the MEASURE position, "
            "then plug it back in. Do not turn the dial after this."
        )
        self.skip_calibration = True

    def session(self, say=None, ask=None) -> SpotreadSession:
        """Start spotread and bring it to "ready to measure", relaying its
        prompts (calibration, dial position) to the human through
        ``say``/``ask``. Call this **before** opening a patch window. Use
        as a context manager; the process is stopped on exit."""
        args = ["-e"]
        if self.instrument_display_type:
            args = ["-y", self.instrument_display_type, *args]
        if self.skip_calibration:
            # -N: "Disable auto calibration of instrument". For a unit whose dial-position
            # report goes stale (it only updates after a power cycle), calibrate once with
            # `spotread -c 1` with the dial at calibration, turn to measure, then run with this.
            args = ["-N", *args]
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

        def read_one(patch):
            # Only a placement patch (a small square) has free screen area
            # to draw "Measuring..." on without changing the exact color a
            # full-screen patch is being read as.
            if patch.placement is not None:
                windowmod.draw_measuring_indicator(screen, patch)
            xyz = np.array(session.measure(poll=poll))
            if patch.placement is not None:
                windowmod.show_patch(screen, patch.rgb, position=patch.position, size_frac=patch.size_frac)
            return xyz

        xyzs = windowmod.run_patch_sequence(
            screen,
            patches,
            read_one,
            settle_s=settle_s,
            sleep=sleep,
            confirm_placement=True,  # a handheld instrument has to be moved onto each uniformity square
        )
        return [
            Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)) for patch, xyz in zip(patches, xyzs, strict=True)
        ]

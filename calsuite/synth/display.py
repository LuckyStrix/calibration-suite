"""Synthetic display with known primaries, white, per-channel TRC, black
level, optional non-additivity, spatial non-uniformity, warm-up drift and
PWM dimming -- the foundation every ``display/`` analysis's round-trip test
is built on (docs/design.md house rule 4). Nothing here is measured; the
defaults are plausible values for an sRGB-ish LCD panel, not a claim about
any real display's actual (currently-unmeasured) parameters.

Owned by Wave 2C (display) alongside ``calsuite/display/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _xy_to_xyz_unit_y(x: float, y: float) -> np.ndarray:
    """Chromaticity (x, y) -> XYZ normalized to Y=1 -- the standard
    construction also used by formats/icc.py's own test helper
    (``_srgb_matrix_d50``) and by ``display/analysis.py``."""
    return np.array([x / y, 1.0, (1 - x - y) / y])


@dataclass
class DisplayModel:
    """A simulated display. XYZ throughout is **absolute**, cd/m^2 (Y in
    cd/m^2), matching what a real colorimeter/spectro reports in emissive
    mode (docs/design.md §5.1) -- backends and analysis code normalize to
    Y=1-for-white only at the points that actually need it (the .ti3 file,
    a Lab conversion), not universally.
    """

    # Panel primaries/white, defaulted to this laptop's EDID-nominal values
    # (docs/design.md §0) purely as a realistic starting point -- a
    # synthetic display's whole point is that these are *known exactly*,
    # unlike a real panel's.
    primaries_xy: dict = field(
        default_factory=lambda: {"r": (0.638, 0.334), "g": (0.300, 0.596), "b": (0.141, 0.058)}
    )
    white_xy: tuple = (0.312, 0.329)
    white_luminance_cdm2: float = 250.0
    black_luminance_cdm2: float = 0.25  # -> contrast ratio 1000:1, a plausible LCD figure

    # Per-channel effective gamma (a float, or a dict of one float per
    # channel for a display whose channels don't track each other exactly)
    # or a callable ``level -> normalized_output`` for a piecewise TRC.
    gamma: dict = field(default_factory=lambda: {"r": 2.2, "g": 2.2, "b": 2.2})

    # Non-additivity knob (docs/design.md §5.2: "measured W vs. R+G+B ...
    # if it fails, the display needs a LUT profile"): extra luminance that
    # appears only when R, G and B are *simultaneously* driven above zero
    # (a common real-world cause -- backlight/driver interaction at full
    # white), as a fraction of white_luminance_cdm2. Zero means perfectly
    # additive. This deliberately leaves single-channel primary
    # measurements (R alone, G alone, B alone) untouched, so it shows up
    # only in the additivity check, not in the primaries fit.
    white_boost_frac: float = 0.0

    # Spatial non-uniformity: a simple radial falloff from screen center,
    # `amplitude` in [0, 1] is the fractional luminance drop at the
    # corners (radius = 1) relative to center -- good enough to exercise
    # display/analysis.py's uniformity grid without a full measured map.
    nonuniformity_amplitude: float = 0.0

    # Warm-up: luminance rises from `warmup_initial_frac` of its final
    # value at t=0 towards 1.0 with time constant `warmup_tau_s`, an
    # exponential approach -- the simplest curve shape consistent with
    # "backlight output stabilizes after power-on" and good enough for
    # display/analysis.py's stable-time fit to recover a known tau.
    warmup_tau_s: float = 300.0
    warmup_initial_frac: float = 0.85

    # PWM backlight dimming: None disables it (a display with no PWM, or
    # too high a frequency/low enough duty variation to matter); when set,
    # synth.display.render_rolling_shutter_rows uses it to produce banding.
    pwm_hz: float | None = None
    pwm_duty: float = 0.5

    noise_std_frac: float = 0.0  # multiplicative Gaussian noise, fraction of XYZ magnitude
    seed: int = 0

    def primary_matrix(self) -> np.ndarray:
        """3x3, columns R/G/B, each column the primary's XYZ *at full
        drive* -- built by the same "solve for per-primary scale so the
        columns sum to the white point" construction ICC matrix/TRC
        profiles use (formats/icc.py's docstring; ICC.1:2010 Annex F),
        scaled here to `white_luminance_cdm2` rather than Y=1.
        """
        r = _xy_to_xyz_unit_y(*self.primaries_xy["r"])
        g = _xy_to_xyz_unit_y(*self.primaries_xy["g"])
        b = _xy_to_xyz_unit_y(*self.primaries_xy["b"])
        w = _xy_to_xyz_unit_y(*self.white_xy) * self.white_luminance_cdm2
        primaries = np.column_stack([r, g, b])
        scale = np.linalg.solve(primaries, w)
        return primaries * scale

    def black_xyz(self) -> np.ndarray:
        """Black point: same chromaticity as white (typical LCD backlight
        leakage through closed shutters/crossed polarizers), scaled to
        `black_luminance_cdm2`."""
        return _xy_to_xyz_unit_y(*self.white_xy) * self.black_luminance_cdm2

    def _trc(self, channel: str, level: float) -> float:
        level = max(0.0, min(1.0, level))
        g = self.gamma[channel] if isinstance(self.gamma, dict) else self.gamma
        if callable(g):
            return float(g(level))
        return level**g

    def warmup_factor(self, t_s: float | None) -> float:
        """Multiplier on total luminance at `t_s` seconds since power-on;
        1.0 (fully warmed) when `t_s` is None -- most measurements don't
        care about warm-up state, only display/patches.py's dedicated
        warm-up series does."""
        if t_s is None:
            return 1.0
        return self.warmup_initial_frac + (1.0 - self.warmup_initial_frac) * (
            1.0 - np.exp(-t_s / self.warmup_tau_s)
        )

    def uniformity_factor(self, position: tuple) -> float:
        if self.nonuniformity_amplitude <= 0:
            return 1.0
        x, y = position
        r = float(np.hypot(x - 0.5, y - 0.5) / np.hypot(0.5, 0.5))
        return 1.0 - self.nonuniformity_amplitude * r**2

    def measure(self, rgb: tuple, *, position: tuple = (0.5, 0.5), t_s: float | None = None, rng=None) -> np.ndarray:
        """XYZ (cd/m^2) the display would show for `rgb` in [0,1]^3, at
        `position` (fractional screen coords) and `t_s` seconds after
        power-on. `rng`, if given, adds this instance's `noise_std_frac`
        multiplicative Gaussian noise -- omit it (the default) for a
        noise-free "ground truth" reading, e.g. to compute the true
        primaries a fitted result should recover.
        """
        matrix = self.primary_matrix()
        lin = np.array([self._trc("r", rgb[0]), self._trc("g", rgb[1]), self._trc("b", rgb[2])])
        xyz = self.black_xyz() + matrix @ lin
        if self.white_boost_frac > 0:
            w_unit = _xy_to_xyz_unit_y(*self.white_xy)
            xyz = xyz + self.white_boost_frac * self.white_luminance_cdm2 * float(np.min(lin)) * w_unit
        xyz = xyz * self.uniformity_factor(position) * self.warmup_factor(t_s)
        if self.noise_std_frac > 0 and rng is not None:
            xyz = xyz * (1.0 + rng.normal(0.0, self.noise_std_frac, size=3))
        return xyz


def _pwm_on_time(t: float, period: float, on_time: float) -> float:
    """Total time a square wave (period `period`, high for `on_time` out of
    each period, starting high at t=0) has spent high over [0, t], t >= 0.
    A closed form (whole periods times `on_time`, plus a clamped remainder)
    rather than numerically integrating a sampled waveform -- exact, and
    fast to call once per row."""
    whole_periods, remainder = divmod(t, period)
    return whole_periods * on_time + min(remainder, on_time)


def render_rolling_shutter_rows(
    display: DisplayModel,
    rgb: tuple,
    *,
    n_rows: int,
    row_period_s: float,
    exposure_s: float,
    phase0_s: float = 0.0,
    read_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a fast-shutter rolling-readout capture of a flat `rgb`
    patch (docs/design.md §5.2: "rolling-shutter banding in a 1/8000 s
    frame of a white screen"): one intensity value per row.

    Row `r` starts exposing at ``phase0_s + r * row_period_s`` and
    integrates for `exposure_s`. If `display.pwm_hz` is set, each row's
    average brightness depends on where in the PWM cycle its exposure
    window landed -- the source of the banding ``display.analysis.
    pwm_banding`` is meant to detect. Row values are normalized so that an
    exposure spanning many full PWM periods (no banding, the ordinary
    case) reproduces the patch's true luminance, matching what a real
    long-exposure or PWM-free reading would show.

    Only a single value per row (not a full 2D frame) -- that row-mean is
    exactly what ``display.analysis.pwm_banding`` consumes, so there's
    nothing to gain from simulating column-wise detail no analysis step
    reads.
    """
    y_true = float(display.measure(rgb)[1])
    rows = np.empty(n_rows, dtype=np.float64)
    if display.pwm_hz:
        period = 1.0 / display.pwm_hz
        on_time = display.pwm_duty * period
        for r in range(n_rows):
            t0 = phase0_s + r * row_period_s
            on_in_window = _pwm_on_time(t0 + exposure_s, period, on_time) - _pwm_on_time(t0, period, on_time)
            frac_on = on_in_window / exposure_s
            rows[r] = y_true * frac_on / display.pwm_duty
    else:
        rows[:] = y_true
    if read_noise_std > 0 and rng is not None:
        rows = rows + rng.normal(0.0, read_noise_std, size=n_rows)
    return rows

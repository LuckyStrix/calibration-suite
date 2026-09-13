"""Synthetic sensor frames with known noise parameters -- the foundation
every camera/ analysis's round-trip test is built on (house rule 4: "true
gain 2.0 e-/DN fits to 1.997" is the same move as calibrate.py's synthetic
checks). Nothing here is measured; every default is a plausible order-of-
magnitude value for a modern APS-C CMOS sensor, chosen so a default
``SensorModel()`` produces sane frames for fast tests -- not a claim about
the R100's actual, currently-unmeasured parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calsuite.raw import FrameMeta, RawFrame

DARK_CURRENT_DOUBLING_C = 6.0
# Silicon dark current roughly doubles every ~6-7C near room temperature --
# a standard rule of thumb (docs/design.md §3.1's dark-current-vs-temperature
# row). Used here only to shape the synthetic dark current's temperature
# dependence for round-trip tests of camera/darks.py's temperature fit, not
# as a measured constant of any real sensor.

# The R100's real geometry (docs/design.md §0/§3.1): 6288x4056 full raw,
# 6000x4000 visible, 56 extra rows / 288 extra columns of optical black
# margin. SensorModel.r100_like() uses these; the plain default is much
# smaller so the rest of the test suite runs in well under a second.
R100_VISIBLE_SHAPE = (4000, 6000)
R100_TOP_MARGIN = 56
R100_LEFT_MARGIN = 288


@dataclass
class SensorModel:
    """Everything needed to generate a synthetic raw frame. Electrons are
    the unit of truth throughout; DN only appears at the final gain step,
    matching how a real sensor works (docs/design.md §3.1's Janesick-style
    photon transfer language)."""

    gain_e_per_dn: float = 2.0  # electrons per output DN
    read_noise_e: float = 3.0  # read noise, electrons RMS
    black_dn: float = 512.0  # bias level, DN -- used on all 4 CFA channels unless black_dn_by_channel overrides
    black_dn_by_channel: dict | tuple | None = None
    # Per-CFA-channel bias level, overriding `black_dn`: a dict keyed by
    # canonical channel name ("R"/"G1"/"G2"/"B") or a plain 4-tuple in that
    # (R, G1, G2, B) order. Real on some CMOS designs, where the two green
    # amplifier chains (G1/G2) read a few DN apart from each other and from
    # R/B (raw.black_level_by_channel's docstring) -- before this existed,
    # every synthetic frame had one scalar black_dn on all four positions,
    # so no test in the suite could exercise a sensor whose channels
    # genuinely disagree. Leave at the default `None` to keep every
    # existing flat-`black_dn` frame byte-for-byte unchanged.
    full_well_e: float = 40000.0  # electrons at saturation
    dark_current_e_per_s_at_20c: float = 0.05  # e-/s at 20C; see dark_current()
    pattern: str = "RGGB"
    shape: tuple = (256, 384)  # visible (rows, cols); small by default for fast tests -- see r100_like()
    top_margin: int = 8
    left_margin: int = 16
    prnu_std: float = 0.01  # fractional photoresponse nonuniformity (1% is a common rule-of-thumb figure)
    dsnu_std_e_per_s: float = 0.01  # per-pixel dark-current-RATE spread, e-/s (so its effect scales with exposure)
    hot_pixel_fraction: float = 0.0005
    hot_pixel_extra_e_per_s: float = 200.0
    row_banding_std_dn: float = 0.0  # disabled by default; a temporal (per-frame) readout artifact when > 0
    white_level_dn: float | None = None  # defaults to black_dn + full_well_e / gain_e_per_dn
    fixed_pattern_seed: int = 0  # seeds PRNU/DSNU/hot-pixel maps -- fixed per "sensor", unlike per-frame noise

    def dark_current(self, temp_c: float) -> float:
        """Dark current at `temp_c`, e-/s, using the ~6C-doubling rule."""
        return self.dark_current_e_per_s_at_20c * 2.0 ** ((temp_c - 20.0) / DARK_CURRENT_DOUBLING_C)

    def black_reference(self) -> float:
        """The single scalar black level used for the frame's shared
        clipping ceiling (`white_level`) -- the mean of the per-channel
        values when `black_dn_by_channel` is given, otherwise plain
        `black_dn`. A real sensor's analog-to-digital full scale is shared
        across channels; only the bias offset below it varies per channel."""
        if self.black_dn_by_channel is None:
            return self.black_dn
        return float(np.mean(list(_resolve_black_by_channel(self).values())))

    @property
    def white_level(self) -> float:
        if self.white_level_dn is not None:
            return self.white_level_dn
        return self.black_reference() + self.full_well_e / self.gain_e_per_dn

    @classmethod
    def r100_like(cls, **overrides) -> SensorModel:
        """A model with the R100's real pixel-array geometry, so lens/
        capture-pipeline tests can exercise realistic frame sizes. Every
        electronic parameter (gain, read noise, ...) still takes
        SensorModel's plausible-but-unmeasured defaults unless overridden
        -- geometry and electronics are independent knobs here."""
        defaults = dict(shape=R100_VISIBLE_SHAPE, top_margin=R100_TOP_MARGIN, left_margin=R100_LEFT_MARGIN)
        defaults.update(overrides)
        return cls(**defaults)


_CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _plane_positions(pattern: str) -> dict:
    """Which (row-phase, col-phase) in a 2x2 CFA tile is R/G1/G2/B -- a
    local copy of ``raw._plane_positions``'s convention (first 'G' in
    raster order is G1, second is G2), same as ``synth/color.py``'s
    ``plane_positions``: re-deriving 6 lines locally beats reaching into
    another module's underscore-prefixed private API for it."""
    if len(pattern) != 4:
        raise ValueError(f"expected a 2x2 (4-char) CFA pattern, got {pattern!r}")
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    names, g_count = {}, 0
    for pos, ch in zip(positions, pattern, strict=True):
        if ch == "G":
            g_count += 1
            names[pos] = f"G{g_count}"
        else:
            names[pos] = ch
    return names


def _resolve_black_by_channel(model: SensorModel) -> dict:
    """``{"R": ..., "G1": ..., "G2": ..., "B": ...}`` -- `model.black_dn`
    broadcast to all four channels when `model.black_dn_by_channel` is
    `None` (the always-supported default), otherwise `black_dn_by_channel`
    itself, accepting either a dict keyed by canonical channel name or a
    plain 4-tuple/list in (R, G1, G2, B) order."""
    spec = model.black_dn_by_channel
    if spec is None:
        return {name: model.black_dn for name in _CHANNEL_NAMES}
    if isinstance(spec, dict):
        if set(spec) != set(_CHANNEL_NAMES):
            raise ValueError(f"black_dn_by_channel dict must have exactly keys {set(_CHANNEL_NAMES)}, got {set(spec)!r}")
        return {name: float(value) for name, value in spec.items()}
    values = tuple(spec)
    if len(values) != 4:
        raise ValueError("black_dn_by_channel must be a dict keyed R/G1/G2/B, or a 4-tuple (R, G1, G2, B)")
    return dict(zip(_CHANNEL_NAMES, (float(v) for v in values), strict=True))


def _black_level_map(pattern: str, top: int, left: int, total_rows: int, total_cols: int, channel_black: dict) -> np.ndarray:
    """Per-pixel black level across the *entire* physical array, margins
    included -- the CFA color filter continues over the masked margin
    exactly like the visible area (the margin sees no light, not no
    filter), so a channel's black offset applies wherever that channel's
    amplifier reads out, not just inside `visible`. `pattern` is anchored
    at the visible origin (`top`, `left`), matching `RawFrame.pattern`'s
    own convention, so each phase's absolute-array parity is shifted by
    (-top, -left) before naming it -- the same `(row - top) % 2` rule
    ``raw._channel_at`` uses."""
    positions = _plane_positions(pattern)
    black_map = np.empty((total_rows, total_cols), dtype=np.float64)
    for dr in (0, 1):
        for dc in (0, 1):
            phase = ((dr - top) % 2, (dc - left) % 2)
            name = positions[phase]
            black_map[dr::2, dc::2] = channel_black[name]
    return black_map


def _black_level_tuple(pattern: str, top: int, left: int, channel_black: dict) -> tuple:
    """`RawFrame.black_level`'s 4 values, in the absolute-(0,0)-origin
    raster order real raw files use (``raw.black_level_by_channel``'s
    docstring) -- the mirror image of that function: given a value per
    canonical channel name, produce the positional tuple a real loader
    would have produced. The row/col-shift step is its own inverse (adding
    the same 0/1 shift twice mod 2 is the identity), so reusing it here
    exactly undoes what ``black_level_by_channel`` does to read it back."""
    positions = _plane_positions(pattern)
    row_shift, col_shift = top % 2, left % 2
    absolute_positions = ((0, 0), (0, 1), (1, 0), (1, 1))  # raster order -- black_level's own order
    values = []
    for r, c in absolute_positions:
        visible_pos = ((r + row_shift) % 2, (c + col_shift) % 2)
        values.append(channel_black[positions[visible_pos]])
    return tuple(values)


def _fixed_pattern_maps(model: SensorModel):
    """PRNU/DSNU/hot-pixel maps: a real sensor's fixed pattern does not
    change frame to frame, so these are derived from `model.fixed_pattern_seed`
    alone -- independent of the `rng` passed to `frame()`, which only
    governs the noise that *does* vary from exposure to exposure (shot
    noise, read noise, row banding)."""
    rows, cols = model.shape
    rs = np.random.RandomState(model.fixed_pattern_seed)
    prnu = rs.normal(0.0, model.prnu_std, size=(rows, cols))
    dsnu_e_per_s = rs.normal(0.0, model.dsnu_std_e_per_s, size=(rows, cols))
    hot_mask = (rs.random_sample((rows, cols)) < model.hot_pixel_fraction).astype(np.float64)
    return prnu, dsnu_e_per_s, hot_mask


def frame(
    model: SensorModel,
    exposure_s: float,
    flux_e_per_s,
    temp_c: float,
    rng: np.random.Generator,
) -> RawFrame:
    """Generate one synthetic ``RawFrame``.

    ``flux_e_per_s`` is the photon-generated signal rate, electrons per
    second per pixel -- a scalar for a uniform flat field, or an array
    matching ``model.shape`` for a non-uniform scene. ``rng`` is a
    ``numpy.random.Generator`` the caller seeds; nothing here seeds its own
    randomness, so two calls with independently-advanced draws from the
    same ``rng`` are correlated the way two real exposures in one session
    are (same fixed pattern, independent shot/read noise) -- exactly what
    the bias-pair and PTC-pair tests need.

    The masked (optical-black) margin gets **only** dark current, read
    noise and the bias level -- no flux, no PRNU, no DSNU, no hot pixels --
    matching ``raw.optical_black()``'s premise that those pixels see no
    light and represent a pure electronic reference (docs/design.md §3.1).
    """
    rows, cols = model.shape
    top, left = model.top_margin, model.left_margin
    total_rows, total_cols = rows + top, cols + left
    visible = (slice(top, top + rows), slice(left, left + cols))

    prnu, dsnu_e_per_s, hot_mask = _fixed_pattern_maps(model)
    flux = np.broadcast_to(np.asarray(flux_e_per_s, dtype=np.float64), (rows, cols))

    dark_rate_global = model.dark_current(temp_c)
    dark_rate_visible = np.clip(dark_rate_global + dsnu_e_per_s + hot_mask * model.hot_pixel_extra_e_per_s, 0.0, None)
    dark_e_visible = dark_rate_visible * exposure_s
    dark_e_masked = max(dark_rate_global, 0.0) * exposure_s

    photo_e_visible = np.clip(flux, 0.0, None) * exposure_s * (1.0 + prnu)
    mean_e_visible = np.clip(photo_e_visible + dark_e_visible, 0.0, None)

    electrons = np.empty((total_rows, total_cols), dtype=np.float64)
    electrons[visible] = rng.poisson(mean_e_visible)

    masked = np.ones((total_rows, total_cols), dtype=bool)
    masked[visible] = False
    electrons[masked] = rng.poisson(dark_e_masked, size=int(masked.sum()))

    electrons += rng.normal(0.0, model.read_noise_e, size=electrons.shape)

    channel_black = _resolve_black_by_channel(model)
    black_map = _black_level_map(model.pattern, top, left, total_rows, total_cols, channel_black)
    dn = black_map + electrons / model.gain_e_per_dn

    if model.row_banding_std_dn > 0:
        # A per-row additive offset that repeats across every column in a
        # row but is independent row-to-row and frame-to-frame -- models a
        # temporal readout artifact (amplifier/ADC drift), not a fixed
        # pattern, so it's drawn from `rng`, not the fixed-pattern seed.
        dn = dn + rng.normal(0.0, model.row_banding_std_dn, size=(total_rows, 1))

    white_level = min(model.white_level, 65535.0)
    cfa = np.clip(np.round(dn), 0, white_level).astype(np.uint16)

    meta = FrameMeta(
        model="calsuite-synthetic",
        exposure_s=exposure_s,
        sensor_temp_c=temp_c,
    )
    return RawFrame(
        cfa=cfa,
        pattern=model.pattern,
        visible=visible,
        black_level=_black_level_tuple(model.pattern, top, left, channel_black),
        white_level=float(white_level),
        meta=meta,
        path="<synthetic>",
        sha256="",
    )

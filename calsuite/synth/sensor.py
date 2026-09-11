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
    black_dn: float = 512.0  # bias level, DN
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

    @property
    def white_level(self) -> float:
        if self.white_level_dn is not None:
            return self.white_level_dn
        return self.black_dn + self.full_well_e / self.gain_e_per_dn

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

    dn = model.black_dn + electrons / model.gain_e_per_dn

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
        black_level=(float(model.black_dn),) * 4,
        white_level=float(white_level),
        meta=meta,
        path="<synthetic>",
        sha256="",
    )

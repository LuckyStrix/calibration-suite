"""Thresholds for the camera/sensor analysis modules (Wave 2A), each with the
*reason* for its value -- same convention as ``calsuite/constants.py`` and
``store.py``'s ``SHELF_LIFE_DAYS``. Kept in ``camera/`` rather than the
top-level ``constants.py`` because these are specific to how *this area's*
refusal checks work, not shared foundation constants.
"""

from __future__ import annotations

# -- bias / read noise (camera/bias.py) -------------------------------------

BIAS_MAX_EXPOSURE_S = 1.0 / 1000
# A frame with a longer exposure than this is a dark, not a bias -- dark
# current would leak into the read-noise estimate. Matches
# capture.manual.BIAS_MAX_EXPOSURE_S's own reasoning (bias frames are
# conventionally shot at the camera's shortest speed) but is kept as this
# area's own constant rather than importing across the capture/analysis
# boundary (house rule 5: pure analysis doesn't reach into capture/).

BIAS_MIN_FRAMES = 2
# std(A-B)/sqrt(2) needs at least one pair; fewer than 2 frames can't form one.

BLACK_LEVEL_METADATA_DISCREPANCY_DN = 5.0
# How far (in DN) the measured black level may drift from the metadata's
# reported black level before the report calls it out. Not a refusal --
# manufacturers' reported black levels are themselves round-number estimates
# -- just the bar for "worth a reader's attention" in the report.

# -- photon transfer curve (camera/ptc.py) -----------------------------------

PTC_CLIP_NEAR_WHITE_FRACTION = 0.999
# A pixel counts as "at white" once it reaches 99.9% of white_level, not
# exactly white_level -- real ADCs and rawpy's reported white level rarely
# line up to the last count, so a strict `== white_level` test would miss
# pixels that are visually and statistically already clipped.

PTC_CLIP_FRACTION_MAX = 0.001
# A flat-pair level is refused (dropped from the fit) once more than 0.1% of
# its pixels are clipped. A handful of hot/edge pixels near full well
# shouldn't kill an otherwise-good level; a few tenths of a percent is where
# clipping starts pulling the pair's mean and variance measurably off the
# true (unclipped) values Janesick's method assumes.

PTC_MIN_LEVELS = 6
# Below this many usable (unclipped) signal levels, a "line" is just
# connecting a handful of dots -- design doc §3.1 calls for "20-30 signal
# levels"; 6 is the floor below which the fit is refused outright, not the
# recommended count.

PTC_MIN_SHOT_NOISE_LEVELS = 4
# After the iterative residual trim (below) removes the high-signal levels
# where PRNU/nonlinearity start dominating over shot noise, at least this
# many points must remain for the slope (=1/gain) to be trustworthy. Fewer
# than 4 points gives a line fit with no real check on its own straightness.

PTC_RESIDUAL_FRACTION_MAX = 0.08
# A level is trimmed from the shot-noise region once its variance departs
# from the running linear fit by more than 8% (relative). PRNU contributes a
# variance term that grows with the *square* of signal, so it shows up first
# and most strongly at the high-signal end; 8% is loose enough not to trim
# points on ordinary shot-noise scatter (a handful of pixels' sampling noise)
# but tight enough to catch the curve's obvious upward bend before it biases
# the gain estimate.

# -- linearity / full well (camera/linearity.py) -----------------------------

LINEARITY_MIN_POINTS = 4
# Need at least this many exposure steps to say anything about a "linear
# range" at all -- 2 points are always exactly linear.

LINEARITY_LOW_SIGNAL_FRACTION = 0.3
# The baseline slope (flux rate) is fit only from points below 30% of white
# level -- comfortably below where a real sensor's nonlinearity or PRNU-driven
# excess variance would bias a slope fit, so the baseline represents the
# sensor's true low/mid-signal response.

LINEARITY_DEVIATION_PCT = 2.0
# A point counts as still within the "linear range" while it deviates less
# than 2% from the baseline line -- a commonly cited linearity spec for
# machine-vision sensors (ISO 12232-style full-well/linearity work), used
# here as the report's own bar rather than a vendor number.

LINEARITY_CLIP_FRACTION = 0.999
# Matches PTC_CLIP_NEAR_WHITE_FRACTION's reasoning: a point at/above this
# fraction of white level is excluded from both the baseline fit and the
# deviation calculation as clipped, not merely "nonlinear".

# -- dark current / hot pixels (camera/darks.py) -----------------------------

DARK_MIN_EXPOSURES_PER_BIN = 3
# A temperature bin needs at least 3 distinct exposure times to fit a
# mean-vs-exposure slope with any check on its own linearity (2 points
# always fit a line exactly).

DARK_MIN_TEMP_BINS = 2
# The doubling-temperature fit (log2(dark rate) vs temperature) needs at
# least two distinct temperatures to define a slope at all.

DARK_TEMP_BIN_WIDTH_C = 1.0
# Frames are grouped into the same temperature bin if their reported sensor
# temperature rounds to the same whole degree C -- finer than that chases
# sensor-temperature-sensor noise rather than a real ambient difference.

HOT_PIXEL_SIGMA_K = 6.0
# A pixel counts as "hot" once it exceeds median + 6 * (1.4826 * MAD) of its
# stacked-dark frame. For a Gaussian population, P(> 6 sigma) ~= 1e-9 --
# standard practice in hot-pixel/cosmic-ray rejection literature -- which
# keeps the false-positive count negligible even across a many-megapixel
# sensor. MAD-based (not plain std) so the hot pixels themselves, which are
# real outliers, don't inflate the very sigma used to detect them.

STAR_EATER_MIN_COUNT = 5
# Below this many hot pixels detected in the long-exposure dark, there isn't
# enough of a population to say anything statistically -- the check reports
# "inconclusive" rather than guessing from noise.

STAR_EATER_RATIO_THRESHOLD = 2.0
# A healthy (non-suppressing) camera's dark current excess grows with
# exposure time, so a long dark should show clearly more hot pixels than a
# short one at the same temperature -- "clearly more" is set at 2x here. A
# long/short ratio below this is the signature of in-camera single-pixel
# suppression ("star eater"): the camera is filtering out isolated bright
# pixels before writing the raw file, so the long exposure's hot-pixel count
# doesn't grow the way physics alone predicts.

# -- ISO invariance (camera/iso.py) ------------------------------------------

ISO_INVARIANCE_TOLERANCE_PCT = 5.0
# The recommended ISO is the lowest ISO whose input-referred read noise is
# within this tolerance of the global minimum across the sweep. Set equal to
# ESTIMATED_ACCURACY["read_noise_pct"] (calsuite/constants.py) on purpose:
# there's no point recommending a *higher* ISO to chase a read-noise
# difference finer than the suite's own read-noise measurement can resolve.

ISO_MIN_COUNT = 3
# Need at least 3 ISOs to distinguish "still falling" from "has flattened" --
# 2 points are always monotonic.

# -- shutter accuracy / bulb timing (camera/shutter.py, camera/bulb.py) ------

# Both pool all four CFA planes into one signal-vs-time relationship rather
# than fitting each channel separately: shutter timing is a property of the
# whole sensor's exposure window (the mechanical or electronic shutter),
# not the color filter over a given photosite, unlike bias/PTC/linearity/
# darks, whose relevant properties (bias level, gain, dark current) are
# per-amplifier-chain and genuinely differ by channel.

SHUTTER_MIN_SPEEDS = 4
# Need several speeds spread across the range to trust a cross-calibrated
# flux-rate slope and see whether % error trends with speed (typically worse
# at the fastest mechanical speeds).

BULB_MIN_POINTS = 3
# Fewer than 3 commanded bulb lengths can't separate a latency (intercept)
# term from a scale (slope) term in the fit.

# -- settings check (camera/settings.py) -------------------------------------

_ON_STRINGS = {"on", "enabled", "enable", "1", "yes", "true"}
_OFF_STRINGS = {"off", "disabled", "disable", "0", "no", "false"}
# Recognized spellings of a boolean camera setting across exiftool's Canon
# tag values (which vary: "On"/"Off", 1/0, True/False depending on the tag
# and exiftool version -- see raw.py's own disclaimer that these tag names
# are unverified against a real Canon file). Anything not in either set
# (including a genuinely missing/None value, e.g. from the dcraw metadata
# fallback, which reports no settings at all) is treated as *unknown*, not
# as "off" -- a refusal must never fire on the *absence* of information.

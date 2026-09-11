"""Display-specific thresholds, each with the reason for its value -- same
convention as ``calsuite/constants.py`` (foundation) and ``store.py``'s
``SHELF_LIFE_DAYS``. These live here rather than in the foundation module
because they're analysis-specific to ``display/`` (``constants.py``'s own
docstring: "Wave 2 areas ... add their own analysis-specific thresholds in
their own modules, next to the checks that use them").
"""

from __future__ import annotations

# -- patch sets (design §5.2) ------------------------------------------------

RAMP_STEPS_DEFAULT = 17
# Design's own range is 17-33 steps; 17 is the low end, kept as the default
# so a full measurement run (one physical patch shown + one instrument
# reading per step, times 3 channels plus gray) stays a reasonable session
# length. Callers that want denser TRC sampling pass a higher `steps`.

UNIFORMITY_GRID_N = 5
# Design §5.2 specifies a 5x5 grid explicitly.

UNIFORMITY_PATCH_SIZE_FRAC = 0.15
# Small enough that a uniformity grid position genuinely samples the panel
# area under it (not most of the screen), big enough that a colorimeter
# aperture or a camera ROI comfortably fits inside it without edge spill.

VALIDATION_EXTRA_NEUTRAL_L = (12.5, 37.5, 62.5, 87.5)
# ColorChecker24's own 6 neutral patches (white 9.5, neutral 8/6.5/5/3.5,
# black 2) sit at irregular L* steps roughly spanning the low-20s to
# mid-90s. These four extra values interleave denser gray-axis coverage
# between them (design §5.2's "extra neutrals") -- chosen as roughly the
# midpoints of that spacing, not values taken from the CC24 data itself.

# -- patch window (design §5.3) ----------------------------------------------

SETTLE_TIME_S = 0.3
# LCD pixel response plus an instrument's own settle time before a reading
# is trustworthy. ArgyllCMS's dispcal/dispread default patch delay is
# commonly around 0.1-0.2s for LCDs; 0.3s adds margin for slower panels
# (this laptop's included) without materially lengthening a session.

# -- additivity (design §5.2, §5.4) ------------------------------------------

ADDITIVITY_DE00_MAX = 2.0
# Below the commonly-cited "just perceptible under close comparison" range
# for ΔE2000 (~1-2). A display whose measured white sits within this of
# black-corrected R+G+B is additive enough for a matrix/TRC profile; above
# it, design §5.4 says the display needs a LUT profile instead, and
# display/profile.py refuses the built-in matrix/TRC fallback (which has
# no LUT capability at all -- formats/icc.py is matrix/TRC-only) when
# Argyll is also absent.

# -- warm-up (design §5.2) ---------------------------------------------------

WARMUP_STABLE_FRACTION = 0.02
# "Within 2% of final luminance" -- a commonly cited stabilization
# criterion for LCD warm-up drift (backlight output settling to within a
# couple of percent before critical color work).

# -- PWM banding (design §5.2) -----------------------------------------------

PWM_FFT_MIN_PROMINENCE = 3.0
# A candidate banding peak in the row-mean FFT must be at least this many
# times the median magnitude of the rest of the (non-DC) spectrum to be
# reported as a detected PWM frequency rather than sensor read noise --
# a simple prominence-over-floor test, not a formal peak-detection
# algorithm; chosen to comfortably reject a flat noise floor (whose peak
# bin, by chance, still runs a small multiple of the median) while catching
# a real square-wave-driven banding signal, which concentrates most of its
# energy in one bin.

# -- validation (design §5.6, §10) -------------------------------------------
# Pass thresholds are multiples of the *backend's own stated accuracy*
# (backend.accuracy()'s "de00_estimate", sourced from
# calsuite.constants.ESTIMATED_ACCURACY until a backend is itself
# measured) -- house rule 7, "accuracy is stated, not implied", so there is
# deliberately no fixed ΔE00 number here. The multipliers below have their
# own reason: a validation run stacks the profile-build step and a second,
# independent measurement pass on top of whatever uncertainty the backend
# already reports for a single reading, so some margin beyond the raw
# backend accuracy is expected even from a working profile. p95/max get
# progressively looser multipliers because a handful of outlier patches
# (e.g. near the gamut edge, where a matrix/TRC profile's error is
# structurally largest) are expected even from a good profile.
VALIDATION_MEAN_DE00_MULTIPLIER = 1.5
VALIDATION_P95_DE00_MULTIPLIER = 2.5
VALIDATION_MAX_DE00_MULTIPLIER = 4.0

# -- subprocess timeouts ------------------------------------------------------

SPOTREAD_TIMEOUT_S = 60.0
# One spotread invocation covers instrument positioning plus a real
# integration time, longer than tools.DEFAULT_TIMEOUT_S's generic 30s
# (chosen for quick version/identify-style calls).

COLPROF_TIMEOUT_S = 120.0
# colprof builds a profile from potentially 100+ patches; matrix/shaper
# profiles are fast but LUT profiles at higher -q settings are not.

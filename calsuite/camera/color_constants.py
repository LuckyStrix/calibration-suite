"""Every numeric threshold used by ``camera/{chart,color,ssf,dcp}.py``, with
the *reason* for its value -- same convention as ``calsuite/constants.py``
(house style, docs/design.md house rule 7: "accuracy is stated, not
implied"). Thresholds specific to color characterization live here, next to
the checks that use them, rather than in the foundation's ``constants.py``
(implementation-plan.md: "Your thresholds go in
``calsuite/camera/color_constants.py``, each with the reason for its
value.").
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# chart layout
# ---------------------------------------------------------------------------

DEFAULT_CHART_ROWS = 4
DEFAULT_CHART_COLS = 6
# The classic X-Rite/GretagMacbeth ColorChecker Classic (24-patch) is laid
# out 4 rows x 6 columns -- docs/design.md's "6x4 ColorChecker". Also the
# default when a custom chart's CSV doesn't say otherwise.

PATCH_CENTRAL_FRACTION = 0.5
# "sample the central 50% of each patch" (docs/design.md/instructions,
# verbatim) -- interpreted here as 50% of the local patch-to-patch pitch
# along each grid axis (so a half-width sampling box centered on each patch),
# not 50% of area and not the physical tile size net of its gutter (which
# this module has no way to know from 4 corner clicks alone). This is a
# deliberately conservative inset: it stays well clear of a patch's edge
# and the gap between patches even when the corners were clicked slightly
# off-center, at the cost of not using every pixel the true tile offers.

# ---------------------------------------------------------------------------
# refusals (docs/design.md sec 3.2)
# ---------------------------------------------------------------------------

REFERENCE_ILLUMINANT_XY_MAX_DELTA = 0.005
# How far the chart's own reference illuminant may sit from the illuminant
# the chart was photographed under, as a Euclidean distance in CIE 1931 xy,
# before the fit refuses. A reference chart's XYZ are the product of its
# patches' *reflectance* and its stated illuminant's spectrum; change the
# illuminant and those numbers change in a way no chromatic adaptation can
# undo (a CAT moves a white point, it does not re-integrate a reflectance).
# Measured on colour-science's own ColorChecker24 data: its D50 values
# versus the same reflectances under D65 differ by ΔE00 2.1 mean / 5.7 max
# (worst on the blues) even after each is placed in its own Lab space --
# half the suite's entire mean-ΔE00 acceptance budget, spent before the
# camera is involved. 0.005 in xy is roughly the disagreement between two
# published chromaticities of the *same* illuminant, i.e. small enough to
# tolerate bookkeeping differences and nothing else. D50 vs D65 is 0.044.

GLARE_MIN_SIGNAL_FRACTION = 0.25
# The glare check only looks at neutrals whose *black-subtracted* signal is
# at least this fraction of the brightest neutral's. Below that, photon
# shot noise alone is a several-percent CV (a patch at 5% of full scale on
# a gain-2 e-/DN sensor sits at a few hundred electrons, i.e. CV a few
# percent), so a single flat CV ceiling can't separate glare from noise
# down there -- and the dark neutrals are the ones where a real glare hot
# spot is proportionally largest, so the honest move is to say the check
# doesn't cover them rather than to loosen the ceiling for everyone.

GLARE_NEUTRAL_CV_MAX = 0.04
# Within a single neutral patch's sampled region, the coefficient of
# variation (std/mean) of a clean, evenly-lit, diffuse patch is set by
# photon shot noise plus PRNU: measured on this suite's own synthetic
# chart, a well-exposed neutral above GLARE_MIN_SIGNAL_FRACTION sits at
# 1.2-2.2% (the PRNU floor is ~1.1%, and it does not average away with
# exposure -- only the shot-noise part does). A glare hot spot (a specular
# reflection off the chart's coating, or a reflected light source) adds
# spatial structure on top of that: a +50% hot spot over a quarter of the
# patch measures 6-8%. 4% sits between the two.
#
# This is a CV of the *black-subtracted* signal. Computed on raw DN, as it
# was, the black pedestal (2048 of 16383 on the R100) dilutes it by
# signal/(signal + black), so the same 7.7% non-uniformity measured 1.3%
# on a dark neutral and passed a 2% ceiling, while on a bright one it
# measured 6.1% and refused -- an 8x swing in sensitivity across the
# neutral ramp, always understating, and blindest exactly where a hot spot
# is proportionally largest.

UNEVEN_LIGHTING_GRADIENT_MAX = 0.08
# "Uneven lighting" is checked by regressing each neutral patch's raw
# response, normalized by its own reference reflectance (so a perfect
# gray-ramp shot on perfectly even light would give a *constant* value
# across patches), against the patches' 2-D grid position. If the best-fit
# plane's peak-to-peak variation across the chart exceeds 8% of the mean
# normalized response, the light is judged uneven enough to bias the fit.
# 8% is chosen as roughly 2x the corner-shading a decent lighting setup with
# two diffused sources should achieve, per common photometric-lighting
# practice for product/chart photography -- not a measured figure, and
# reports show the achieved gradient next to this threshold so it can be
# tightened once real charts are shot.

CLIPPED_FRACTION_MAX = 0.001
# A patch/channel is refused as clipped if more than 0.1% of its sampled
# pixels sit at or above (white_level - 1 DN). A little slack below the
# literal white level absorbs read-noise-driven single-pixel excursions
# near a bright-but-not-clipped patch without masking real, widespread
# clipping (which affects a large fraction of the patch's pixels, not a
# handful at the tail of the noise distribution).

MODEL_TERMS = {"matrix": 3, "rp2": 6, "rp3": 13}
# Number of regression terms per output channel (X, Y or Z) for each model:
# "matrix" is a plain 3x3 (R, G, B); "rp2"/"rp3" are Finlayson root-
# polynomial expansions of degree 2 (R, G, B, sqrt(RG), sqrt(GB), sqrt(RB) =
# 6 terms) and degree 3 (+ 6 cube-root cross terms + sqrt(RGB) cube root = 13
# terms), matching ``colour.characterisation.polynomial_expansion_Finlayson2015``'s
# term layout for ``degree=2``/``degree=3`` with ``root_polynomial_expansion=True``.

MIN_PATCHES_PER_TERM = 4
# "n_patches >= 4 x terms per output channel" (docs/design.md sec 3.2 /
# instructions, verbatim rule): a 3x3 matrix (3 terms) needs 12 patches, rp2
# (6 terms) needs 24, rp3 (13 terms) needs 52. 4x is a standard rule-of-thumb
# safety margin for a least-squares fit (enough degrees of freedom left over
# for the residual to be a meaningful check of fit quality, not just enough
# equations to solve the system exactly).


def min_patches(model: str) -> int:
    if model not in MODEL_TERMS:
        raise ValueError(f"unknown model {model!r}, expected one of {sorted(MODEL_TERMS)}")
    return MIN_PATCHES_PER_TERM * MODEL_TERMS[model]


# ---------------------------------------------------------------------------
# Tier A fit
# ---------------------------------------------------------------------------

WHITE_PRESERVING_WEIGHT = 1000.0
# The "white-preserving" option is implemented as a heavily up-weighted
# duplicate of the neutral/reference patch's residual in both the linear
# least-squares initialization and the nonlinear DeltaE00 refinement (see
# camera/color.py), rather than an exact linear equality constraint -- a
# soft constraint that is exact to within floating-point precision at this
# weight (1000x every other patch's unit weight) while keeping the same
# unconstrained-least-squares code path for both cases. Solved this way
# because scipy.optimize.least_squares has no first-class linear equality
# constraint support outside of trust-constr, which is far slower for a
# 9-30 parameter problem run interactively.

# ---------------------------------------------------------------------------
# Tier C validation -- provenance "measured" only below these
# ---------------------------------------------------------------------------

VALIDATION_MEAN_DE00_MAX = 4.0
# A mean DeltaE2000 of 4 on held-out patches is the ISO 17321-1-adjacent
# "acceptable for casual colorimetry" ballpark quoted across camera-
# profiling literature (dcamprof, Argyll's ccxxmake docs) for a 3x3 matrix
# fit to a 24-patch chart under a single illuminant -- a much stricter bound
# would refuse essentially every real 3x3 fit, defeating the point of
# having a Tier A tier at all; a much looser one would let a genuinely bad
# fit through as "measured". Revisit once a real chart shot exists to
# compare against.
VALIDATION_P95_DE00_MAX = 8.0
VALIDATION_MAX_DE00_MAX = 12.0
# p95/max allow a few harder patches (saturated chromatic colors are the
# usual offenders for a 3x3 matrix, root-polynomial models do better) to run
# hotter than the mean without failing the whole profile, while still
# catching a fit that's fine on average but wildly wrong on part of the
# gamut.

# ---------------------------------------------------------------------------
# Tier B: Luther-Ives / SMI
# ---------------------------------------------------------------------------

LUTHER_IVES_EXACT_TOL = 1e-6
# Used only by tests: SSFs built as an *exact* linear combination of the CIE
# 1931 2-degree CMFs should reproduce them via ssf.luther_ives_deviation to
# within ordinary floating-point/least-squares solver precision -- 1e-6 is
# comfortably above numpy's float64 lstsq residual floor for a well-
# conditioned 3x3 system and comfortably below anything a real, non-Luther
# sensor would ever score.

SMI_SLOPE = 5.5
# ISO 17321-1's sensor metamerism index: SMI = 100 - 5.5 * mean(dE*ab) over
# the 18 chromatic (non-neutral) ColorChecker patches under D65. Verified
# 2026-09-11 via web search against a DPReview Forums summary of the ISO
# 17321-1 SMI experiment and DxOMark's "Color depth" glossary entry
# (https://www.dxomark.com/glossary/color-depth/,
# https://www.dpreview.com/forums/thread/3618930), both independently
# stating "SMI = 100 - 5.5 x mean dE" over the ColorChecker's chromatic
# patches; the standard itself (ISO 17321-1:2012) was not directly
# accessible (paywalled). Treat as verified-by-secondary-source, not
# verified against the primary standard text.

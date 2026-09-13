"""Every numeric threshold and lensfun-format constant used by ``calsuite/lens``
and ``targets/``, with the *reason* for its value -- same convention as
``calsuite/constants.py`` (docs/design.md house rule, hydrationTracker style).

The lensfun-format section (models, radius normalization, XML attribute
names, user-database paths) is not a "threshold" in the numeric sense, but it
carries the same obligation: every fact here was verified against a primary
source, and the source is cited in the comment next to it, per the Wave 2B
brief ("YOU MUST VERIFY lensfun's model definitions and coordinate
normalization... and cite the source").

Sources consulted (fetched to this build machine 2026-09-11):
  - lensfun manual, "Correcting lens distortions" page (lensfun.github.io/
    manual/latest/corrections.html): confirms distortion-model formulae map
    the *undistorted* coordinate to the *distorted* one.
  - lensfun source, ``libs/lensfun/modifier.cpp`` (the ``lfModifier``
    constructor's leading comment "About coordinate systems in Lensfun"):
    the authoritative statement of which radius is "1" for which model
    family. Quoted directly where used below.
  - lensfun source, ``libs/lensfun/mod-coord.cpp`` (poly3/ptlens formulae
    and direction), ``libs/lensfun/mod-subpix.cpp`` (TCA linear/poly3
    formulae), ``libs/lensfun/mod-color.cpp`` (vignetting "pa" formula).
  - lensfun source, ``libs/lensfun/database.cpp`` (XML attribute name ->
    internal term mapping, and the user database directory:
    ``g_build_filename(g_get_user_data_dir(), CONF_PACKAGE, NULL)`` with
    ``CONF_PACKAGE == "lensfun"``, loaded unconditionally at the end of
    ``lfDatabase::Load()``).
  - The locally-installed ``/usr/share/lensfun/version_1/mil-canon.xml``
    (lensfun 0.3.3 data, per docs/implementation-plan.md's "Facts checked"):
    used as a real, schema-valid example of ``<camera>``/``<lens>`` blocks,
    and as the vendor comparison target for "Canon RF 50mm F1.8 STM".
  - GLib docs (docs.gtk.org/glib/func.get_user_data_dir.html) for
    ``g_get_user_data_dir()``'s Windows behavior (``FOLDERID_LocalAppData``,
    i.e. ``%LOCALAPPDATA%``), corroborated by community reports
    (discuss.pixls.us) of darktable's Windows lensfun user directory at
    ``%LOCALAPPDATA%\\lensfun``, alongside darktable's own bundled/system
    copy under its install directory (``<install>\\share\\lensfun\\
    version_1\\``).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# R100 sensor geometry (docs/design.md §0 / the Wave 2B task prompt, which
# states these two facts directly: "22.3 mm / 6000 px" pixel pitch, and the
# R100 body is absent from mil-canon.xml so a crop factor must be derived).
# ---------------------------------------------------------------------------

R100_SENSOR_WIDTH_MM = 22.3
R100_SENSOR_HEIGHT_MM = 14.9
# APS-C-class dimensions consistent with the visible 6000x4000px raw
# (docs/design.md §0) at the pixel pitch below; this is the commonly cited
# Canon APS-C sensor active area, not a value read off an R100 datasheet
# (Canon does not publish one) -- flagged here rather than silently assumed.

R100_PIXEL_PITCH_MM = R100_SENSOR_WIDTH_MM / 6000.0
# Given directly in the Wave 2B task prompt ("R100: 22.3 mm / 6000 px --
# put in constants with source"). Used by mtf.py to convert cycles/px to
# lp/mm.

R100_CROP_FACTOR = (36.0**2 + 24.0**2) ** 0.5 / (R100_SENSOR_WIDTH_MM**2 + R100_SENSOR_HEIGHT_MM**2) ** 0.5
# hypot(36,24)/hypot(22.3,14.9) = diag(full-frame 35mm) / diag(R100 sensor)
# ~= 1.613. Not read off a Canon datasheet (none is published for the R100)
# -- derived from the sensor size above, the same way lensfun itself defines
# "cropfactor" (see mod-coord.cpp's hugin_scale_in_millimeters, which is
# built from hypot(36,24)/CropFactor). Cross-checked against
# /usr/share/lensfun/version_1/mil-canon.xml's own "Canon EOS M"-family
# <camera> blocks (Canon APS-C, EF-M mount), which record cropfactor
# "1.613" verbatim -- the same APS-C sensor class as the R100, so this
# agreement is a real corroboration, not a coincidence of rounding.

# ---------------------------------------------------------------------------
# ChArUco target board geometry -- shared by targets/generate_targets.py
# (which rasterizes it), lens/charuco.py (which detects it) and
# synth/lens.py (which renders a synthetic version of it). A single
# definition here is the reason all three agree on square count and aspect.
# ---------------------------------------------------------------------------

BOARD_DICT_NAME = "DICT_5X5_50"
# 5x5-bit markers, 50 of them -- more than the <=15 squares_x*squares_y/2
# markers this board ever needs (10x6 squares -> 30 markers max), with
# enough inter-marker Hamming distance at 5x5 to be robust to the JPEG-like
# artifacts a photographed (not screen-captured) target picks up, while
# staying small enough to print/display markers clearly at a few cm per
# square.

BOARD_SQUARES_X = 10
BOARD_SQUARES_Y = 6
# 10x6 squares (9x5 = 45 interior corners) is wide enough to give real
# coverage of the sensor's outer 20% (the region distortion.py's coverage
# refusal checks) at a practical few-tens-of-cm shooting distance, while
# 10 divides the display target's 1920px width evenly (192px/square) with
# no fractional-pixel square edges to alias against.

BOARD_MARKER_RATIO = 0.7
# ArUco marker side as a fraction of the enclosing chessboard square --
# OpenCV's own CharucoBoard examples and lensfun-adjacent calibration
# tutorials use 0.7 as the standard leaving a visible white square border
# (needed for the corner-interpolation step) without shrinking the marker
# so much it becomes hard to decode at a distance.

DISPLAY_PIXEL_PITCH_MM = 344.0 / 1920.0
# The laptop panel's physical pitch (docs/design.md §4.1: "344 mm / 1920 =
# 0.179 mm"), used to convert the ChArUco board's on-screen square size
# (pixels) to real-world mm for cv2.calibrateCamera's object points when the
# target is displayed rather than printed.

DISPLAY_SQUARE_PX = 192
# BOARD_SQUARES_X (10) * 192 = 1920 -- exactly the display target's width,
# so the printed/displayed board has no fractional-pixel square edges.
DISPLAY_SQUARE_MM = DISPLAY_SQUARE_PX * DISPLAY_PIXEL_PITCH_MM
# ~34.4mm/square on the laptop panel.

PRINT_SQUARE_MM = 20.0
# A convenient, ruler-measurable square size that still fits a 10x6 board
# (200x120mm) on an A4 sheet (210x297mm) with margin to spare.
PRINT_DPI = 300
# Standard "photo quality" laser/inkjet print resolution -- fine enough that
# the printer, not the raster, is the sharpness bottleneck for a target
# whose corners only need to be a few px wide.

# ---------------------------------------------------------------------------
# lensfun coordinate conventions (verified against source -- see module
# docstring for exactly which files/pages).
# ---------------------------------------------------------------------------

# Quoting libs/lensfun/modifier.cpp's coordinate-system comment verbatim:
#
#   "(2) The Hugin-based distortion and TCA calibration models "ptlens",
#   "poly3", "poly5", and "linear" use the Hugin coordinate system. r = 1 is
#   the middle of the long edge, in other words, the half height of the
#   image (in landscape mode)."
#
#   "(3) For the vignetting model "pa", r = 1 is the corner of the image."
#
# i.e. distortion/TCA radii are normalized by half of the *shorter* image
# dimension in pixels; vignetting ("pa") radii are normalized by half the
# image *diagonal*. Both are computed from the image being fit, in
# lens/distortion.py / lens/tca.py / lens/flats.py, as
# ``min(width, height) / 2`` and ``hypot(width, height) / 2`` respectively.

# Direction: libs/lensfun/mod-coord.cpp's own comment and code confirm the
# formulae map *undistorted* radius Ru to *distorted* radius Rd (matching
# the manual's corrections.html page: "the formulae for distortion models...
# map the undistorted coordinate to the distorted coordinate"):
#   ptlens:  Rd = Ru * (a*Ru^3 + b*Ru^2 + c*Ru + d), d = 1-a-b-c  [XML: a,b,c]
#   poly3:   Rd = Ru * ((1-k1) + k1*Ru^2)                         [XML: k1]
#            (poly3 is exactly ptlens with a=0, c=0, b=k1 -- confirmed by
#            comparing ``ModifyCoord_Dist_Poly3``'s ``one_minus_k1 + k1*ru2``
#            with ``ModifyCoord_Dist_PTLens``'s ``a*ru2*r+b*ru2+c*r+d``, both
#            in libs/lensfun/mod-coord.cpp.)
#
# CORRECTION (2026-09-12, closing the Wave 2B "lensfun export is never
# verified to actually correct an image" gap): the constant term above is
# **d = 1-a-b-c, not 1** -- i.e. lensfun's ptlens/poly3 models are
# structurally pinned so that Rd == Ru exactly at the Hugin r=1 edge, for
# *any* (a,b,c). An earlier version of this comment said "+ 1", sourced from
# a copy of mod-coord.cpp fetched to /tmp in a previous session that turned
# out to be from a much newer/unreleased lensfun source tree (it also had a
# completely different, RealFocal/rescale_polynomial_coefficients-based
# NormScale scheme with no equivalent in any released version). Verified
# instead directly against both real, running copies used by this suite:
# `/usr/share/lensfun/version_1` uses system liblensfun 0.3.3
# (`dpkg -s liblensfun1`), and lensfunpy 1.18.0 bundles its own liblensfun
# 0.3.4 (`lensfunpy.lensfun_version() == (0,3,4,0)`, in
# `lensfunpy.libs/liblensfun-*.so.0.3.4`, NOT the system copy) -- fetched
# `libs/lensfun/{modifier,mod-coord}.cpp` from github.com/lensfun/lensfun at
# tags `v0.3.3` and `v0.3.4` directly: byte-identical between the two tags
# for every function this suite depends on, so what lensfunpy actually runs
# (0.3.4) matches what darktable/system tooling runs here (0.3.3).
#
# Consequence for `distortion.refit_ptlens_poly3`: it correctly targets the
# "+1" convention (matching OpenCV/Brown-Conrady's own r=1-coefficient-of-1
# convention -- a real lens's true curve has no reason to satisfy Rd(1)=1),
# which is confirmed (see that function's docstring and
# tests/test_lens_export_lensfun.py's per-pixel round trip) to be a *better*
# approximation of a real distortion curve than forcing a+b+c=0 would be.
# The two conventions coincide exactly only when a+b+c==0; lensfun's actual,
# exported-XML-driven correction therefore carries a bounded, *explained*
# systematic error of magnitude ~|a+b+c| * (Hugin-normalized radius) beyond
# our own fit's residual -- not a bug in the fit, a real and now-quantified
# limitation of the ptlens/poly3 export formats themselves. This is also
# why the direction/model-family choice above ("+1", not "+d") is left
# unchanged: switching the fit to target "+d" instead does not remove this
# error, it relocates it (verified numerically) into a *worse* overall
# approximation of the true curve, because it is a fundamentally more
# constrained 3-parameter family, not a reparameterization of the same one.
#
# This is also OpenCV's own Brown-Conrady direction: distorted = undistorted
# scaled by (1 + k1 r^2 + k2 r^4 + k3 r^6) + tangential terms, in the
# camera's normalized (undistorted) coordinate system -- the same
# undistorted-to-distorted sense, so no direction flip is needed when
# refitting an OpenCV ``calibrateCamera`` result to ptlens/poly3.

# TCA (libs/lensfun/mod-subpix.cpp), same r=1 convention as distortion:
#   linear:  Rd_r = Ru * kr,  Rd_b = Ru * kb                [XML: kr, kb]
#   poly3:   Rd_r = Ru * (br*Ru^2 + cr*Ru + vr)              [XML: vr,cr,br]
#            Rd_b = Ru * (bb*Ru^2 + cb*Ru + vb)              [XML: vb,cb,bb]
# (mil-canon.xml's own RF 50mm entry uses poly3 with only vr/vb set --
# cr=cb=br=bb default to 0, which makes poly3 degenerate to the same pure
# radial scale as "linear". lens/tca.py follows that convention: it always
# fits the linear scale kr/kb and writes it out under the poly3 element as
# vr=kr, vb=kb, matching the existing vendor entry's shape.)

# Vignetting "pa" (libs/lensfun/mod-color.cpp's ModifyColor_Vignetting_PA):
#   V(r) = 1 + k1*r^2 + k2*r^4 + k3*r^6                     [XML: k1,k2,k3]
# applied multiplicatively -- i.e. a flat-field map normalized to 1.0 at the
# image center is fit *directly* against this polynomial in r (no direction
# flip needed), matching docs/design.md §4.3's own statement of the "pa"
# formula.

LENSFUN_DISTORTION_PTLENS_TERMS = ("a", "b", "c")
LENSFUN_DISTORTION_POLY3_TERM = "k1"
LENSFUN_TCA_LINEAR_TERMS = ("kr", "kb")
LENSFUN_TCA_POLY3_TERMS = ("vr", "vb", "cr", "cb", "br", "bb")
LENSFUN_VIGNETTING_PA_TERMS = ("k1", "k2", "k3")

# User database locations (libs/lensfun/database.cpp: ``UserLocation =
# g_build_filename(g_get_user_data_dir(), "lensfun", NULL)``, loaded
# unconditionally by ``lfDatabase::Load()`` in addition to the system
# database -- no "version_1" subdirectory, unlike the system location).
LENSFUN_USER_DIR_LINUX = Path.home() / ".local" / "share" / "lensfun"
# g_get_user_data_dir() on Linux is $XDG_DATA_HOME, defaulting to
# ~/.local/share, when XDG_DATA_HOME is unset (freedesktop.org base-dir
# spec, which is what GLib implements).
LENSFUN_USER_DIR_WINDOWS_RELATIVE = Path("lensfun")
# Combine with %LOCALAPPDATA% at call time (os.environ, not resolved here
# so this module stays importable/testable on Linux). GLib's
# g_get_user_data_dir() on Windows resolves to the FOLDERID_LocalAppData
# shell folder (docs.gtk.org/glib/func.get_user_data_dir.html: "the folder
# to use for local (as opposed to roaming) application data"), i.e.
# %LOCALAPPDATA%\lensfun -- corroborated by community reports (darktable
# users on discuss.pixls.us) of darktable actually reading/writing
# C:\Users\<user>\AppData\Local\lensfun. darktable's *bundled* system copy
# lives separately, under its own install directory
# (<install>\share\lensfun\version_1\), and is not something this suite
# writes to -- ``--out`` is the safe way to hand a file to a Windows
# darktable install whose layout isn't verified on this machine.

SYSTEM_LENSFUN_DB_LINUX = Path("/usr/share/lensfun/version_1")
# Confirmed present on this build machine (docs/implementation-plan.md
# "Facts checked"): SystemLocation = SYSTEM_DB_PATH + DATABASE_SUBDIR in
# database.cpp, and DATABASE_SUBDIR is literally "version_1" -- this path
# is where mil-canon.xml (used by export_lensfun.compare_with_vendor) is
# read from by default.

VENDOR_COMPARISON_LENS_MODEL = "Canon RF 50mm F1.8 STM"
VENDOR_COMPARISON_MOUNT = "Canon RF"

# ---------------------------------------------------------------------------
# distortion.py
# ---------------------------------------------------------------------------

DISTORTION_MIN_VIEWS = 4
# cv2.calibrateCamera's own docs recommend >= 10 views for a well-conditioned
# fit; 4 is the bare floor below which the intrinsic+distortion system is
# usually under-determined for a single-camera, single-focal-length fit
# (each view contributes 2 pose DOF constraints against 5 distortion + 4
# intrinsic unknowns shared across all views) -- below it we refuse outright
# rather than return a number nobody should trust.

DISTORTION_MIN_CORNERS_PER_VIEW = 8
# A charuco view with fewer than ~8 of the (up to 45) interior corners
# detected is usually a glancing or edge-of-frame shot that contributes
# almost no coverage -- cheap to drop before it can skew the coverage grid.

COVERAGE_OUTER_FRACTION = 0.2
# "the outer 20% of the field" -- taken directly from docs/design.md §4.1's
# own refusal rule, not chosen independently here.

COVERAGE_RADIAL_BINS = 5
COVERAGE_ANGULAR_BINS = 12
# A 5 (radius) x 12 (angle, i.e. 30 degree wedges) polar grid: fine enough
# that a hole confined to one side of the frame (e.g. only the left edge
# was ever covered) is caught, coarse enough that a handful of well-spread
# corners per outer wedge is a realistic thing to ask a real capture session
# to achieve -- an angular bin every few degrees would refuse every
# realistic dataset.

REPROJECTION_OUTLIER_FACTOR = 3.0
# A view whose own reprojection RMS exceeds 3x the median of all views' RMS
# is treated as an outlier (motion blur, a missed corner, board flex) and
# dropped before the final refit -- 3x is the classic "obviously worse than
# the pack" multiple used for this kind of robust one-pass trim (not a
# formal MAD estimator, which would be overkill for what's usually <30
# views).

DISTORTION_FIT_MAX_RU = 1.3
# Hugin-normalized radius (1.0 = half the shorter image dimension) out to
# which the ptlens/poly3 refit samples the OpenCV radial curve -- corners
# near the image diagonal corner reach roughly hypot(w,h)/min(w,h)/... ~=
# 1.2-1.3x for a typical 3:2 sensor, so this comfortably covers every corner
# actually seen without extrapolating the polynomial far past the data.

# ---------------------------------------------------------------------------
# flats.py -- self-calibrating flat field
# ---------------------------------------------------------------------------

FLAT_V_RADIAL_ORDER = 2
# V(r) modeled as exp(v2*r^2 + v4*r^4) in Hugin-normalized sensor radius --
# a 4th-order even polynomial (2 free coefficients) is the same order as
# lensfun's own "pa" vignetting model (k1,k2,k3 -> here v2,v4 plus the pa
# refit adds k3/r^6 back in export_lensfun's refit step), enough to capture
# cos^4-law-like falloff without overfitting the handful of poses a
# self-calibrating flat session realistically has.

FLAT_S_ORDER = 2
# S(u,v) modeled as a full 2nd-order 2D polynomial (constant + linear +
# quadratic, 6 terms) in source/world coordinates -- enough to represent a
# tilted or off-center source (the realistic failure mode of "a cheap A4 LED
# tracing pad", docs/design.md §3.1) without enough free parameters to
# start absorbing genuine lens vignetting.

FLAT_GRID_SAMPLES = 24
# Each pose image is subsampled on a FLAT_GRID_SAMPLES x FLAT_GRID_SAMPLES
# grid before being added to the joint least-squares system -- 576 points
# per pose is far more than the <=8 unknowns need, while keeping the joint
# design matrix small enough to build and condition-check in well under a
# second even for many poses.

FLAT_COND_THRESHOLD = 1.0e6
# Condition number above which the joint (V, S) design matrix is treated as
# degenerate ("degeneracy refusal", docs/design.md §4.3: rotation-only poses
# make a radially-symmetric source term indistinguishable from vignetting).
# 1e6 is a standard rule-of-thumb boundary for "numerically rank-deficient
# in float64 least squares" (double precision carries ~15-16 significant
# digits; a condition number near 1e6 already leaves only single-digit
# significant figures of headroom before roundoff dominates the solve).

# ---------------------------------------------------------------------------
# mtf.py -- slanted-edge e-SFR (ISO 12233)
# ---------------------------------------------------------------------------

MTF_OVERSAMPLE = 4
# "4x oversampled ESF" -- specified directly in the Wave 2B task prompt,
# and the traditional ISO 12233 e-SFR oversample factor (each of the ~10-20
# degrees of edge tilt per row contributes a slightly different sub-pixel
# phase; 4x bins those phases finely enough to reconstruct the edge profile
# without needing an impractically long edge).

MTF_EDGE_MIN_ANGLE_DEG = 1.5
MTF_EDGE_MAX_ANGLE_DEG = 10.0
# The target is built at ~5 degrees (docs/design.md §4.4). Below ~1.5
# degrees, consecutive rows land in nearly the same oversample phase bin
# (not enough sub-pixel diversity to beat single-pixel sampling); above ~10
# degrees, a fixed-height ROI captures many fewer *independent* rows worth
# of edge before the edge exits the ROI horizontally, and hugin/ISO-12233
# style tools converge on the same few-degree window for this reason.

MTF_MIN_CONTRAST = 0.15
# Minimum (bright - dark) plateau contrast, as a fraction of the local
# dynamic range, below which the edge is refused as "not really a step" --
# guards against an ROI that missed the edge entirely (e.g. landed in a
# uniform region) producing a meaningless flat MTF=1 curve.

MTF_MIN_ROWS = 20
# An ROI shorter than this many rows along the edge doesn't give the 4x
# oversampling enough independent phase samples to average down row noise
# meaningfully -- below it, the fit is more sensitive to a single row's
# corner-of-target imperfection than to the lens.

MTF_FIELD_GRID = (3, 5)
# (rows, cols) -- "5x3 field grid" per docs/design.md §4.4.

# ---------------------------------------------------------------------------
# psf.py
# ---------------------------------------------------------------------------

PSF_MIN_SNR = 5.0
# A candidate star/pinhole blob whose peak is below 5 sigma over the local
# background is refused as noise, not a real point source -- 5 sigma is the
# conventional astronomical detection threshold (false-positive rate over a
# full frame's worth of pixels is still small at 5 sigma, unlike the more
# common 3 sigma single-pixel threshold which would false-trigger constantly
# over a multi-megapixel frame).

PSF_WINDOW_RADIUS_PX = 15
# Half-width of the square window used to compute a blob's second/third
# moments around its centroid -- wide enough to include a lens's coma wing
# a few px out from the core, narrow enough that two stars closer than
# ~30px don't contaminate each other's moment sums in a typical field test.

# ---------------------------------------------------------------------------
# export_lensfun.py
# ---------------------------------------------------------------------------

DEFAULT_LENS_MOUNT = "Canon RF"
DEFAULT_LENS_MAKER = "Canon"
DEFAULT_CAMERA_MAKER = "Canon"
DEFAULT_CAMERA_MODEL = "Canon EOS R100"

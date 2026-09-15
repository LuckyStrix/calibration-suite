# calsuite — working notes

Camera sensor, lens and display calibration, with provenance on every
number. Full plan: `docs/design.md`. How it was built, in waves:
`docs/implementation-plan.md`.

## The rule

> **A record's `provenance` names the *method*; its `status` names whether
> you can trust it. `store.require_exportable` demands both.**

`provenance` is one of `measured` / `derived` / `vendor` / `nominal`
(`provenance.py`), and it is set once, by *how* a record was produced, never
adjusted after the fact because a fit happened to come out badly:

- `measured` — a fresh capture/measurement attempt: bias frames, a PTC pair
  series, a photographed chart, a display measurement session, a lens
  capture session. A command that runs one of these **always** stamps
  `measured`, whether or not the analysis's own checks passed.
- `derived` — computed from *other records*, not a fresh capture: an ISO
  invariance record (from `camera.ptc` + `camera.linearity`), a display
  profile (from a `display.measurement`), Tier B colour (from SSFs + a
  reflectance set, no chart shot).
- `vendor` — a third-party database entry for this exact device (lensfun's
  `mil-canon.xml`).
- `nominal` — read off a label with no measurement behind it (EDID).

`status` (`"ok"` | `"refused"`) is the orthogonal axis: it is `"refused"`
whenever the analysis's own refusal checks fail, **or** — where the kind has
one — its validation step fails. `store.require_exportable(record)` raises
unless `status == "ok"` **and** `provenance` is `measured`/`derived`
(`provenance.EXPORTABLE`). A record can be honestly `measured` (a chart
*was* photographed) and still `status="refused"` (the fit shouldn't be
trusted) — that's not a contradiction, it's the whole point of splitting the
two axes. See `store.py`'s module/`Record`/`require_exportable` docstrings
for the same statement in one place, and `camera/color.py::fit()` for where
this actually bit: a chart that failed Tier-A/C validation used to come back
`status="ok"` with no refusal at all, because validation-failure was only a
quiet `result["validation_passed"] = False` — that's now a real
`analysis.refuse("validation_failed", ...)`.

Every command that saves a record (`camera/commands.py::_save`,
`lens/commands.py`, `camera/color_commands.py`, `display/commands.py`)
follows this. `camera/commands.py::_save` takes `provenance="measured"` as
its default and the one caller with a different story (`_cmd_iso`, which
computes from existing `camera.ptc`/`camera.linearity` records rather than a
fresh capture) passes `provenance="derived"` explicitly.

Refused records **are saved**, not discarded (house rule 3, `docs/design.md`
§2) — omission would hide exactly the finding a refusal exists to report.

## Layering: pure analysis, separate I/O

`camera/`, `lens/` and `display/` split cleanly into two kinds of module:

- **Pure analysis** (`camera/bias.py`, `lens/distortion.py`,
  `display/analysis.py`, ...): arrays and metadata in, `fit.Analysis` out.
  No file I/O, no subprocess, no `Store`. Every one of these has a synthetic
  round-trip test that builds its input from `synth/*`, not from a file.
- **`commands.py`** (one per area, plus `camera/color_commands.py` for the
  color sub-area): the *only* place that loads a raw file, runs a capture
  backend, or touches `store.Store`. It turns `--from DIR` into loaded
  frames, calls the pure function, and saves the `Analysis` as a `Record`.

If you're adding a check or a fit, it goes in the pure module. If you're
adding a flag or a file format, it goes in `commands.py`. A test that needs
to monkeypatch `raw.load` to fake a frame is a sign the *command* is doing
too much, not that the pure function is hard to test.

## Where the synthetic round trips live

Every analysis is checked against `synth/{sensor,lens,color,display}.py`,
which build a `RawFrame`/backend output from a *known* ground truth (a
gain, a distortion, a matrix, a set of primaries) with no file on disk —
`synth.sensor.frame()` is the root every other generator calls through.

That used to be where the round trip *stopped*: a synthetic frame had no
file format, so nothing that needed a real `--from DIR` (most CLI commands,
`capture.manual.scan_folder`, `demo.py`) could be exercised without either
monkeypatching `raw.load` or writing a fragile fake raw file. `raw.save_npz`
/ `raw.load_npz` close that gap — every `RawFrame` field, `FrameMeta`
(`settings` included), round-trips through a `.npz`, and `raw.load()`
dispatches to `load_npz` for a `.npz` path. `capture.manual.scan_folder` and
every area's `--from DIR` (`_RAW_EXTENSIONS` in `camera/commands.py`,
`lens/commands.py`) accept `.npz` right alongside `.cr3`/`.dng`/`.nef`, so a
synthetic session behaves exactly like a folder of real captures. `demo.py`
is built entirely on this: it writes synthetic frames to a temp dir and
drives the *real* CLI commands against them, except `lens distortion`/`lens
tca`, which detect a ChArUco board in a *rendered* image — rendering and
detecting a synthetic board is `lens/charuco.py`'s own test's job, so `demo.py`
calls `distortion.fit_distortion`/`tca.fit_tca` directly against
`synth.lens.synthetic_views`' point correspondences instead of re-deriving
image rendering a second time.

## Things that are load-bearing

- **A tight fit on a degenerate dataset is the real failure mode, not a
  loose one.** Every refusal exists because the alternative — a confident
  number computed from data that couldn't have supported it — is worse than
  refusing outright (house rule 3). The self-calibrating flat's degeneracy
  check is the sharpest example: a set of poses that are pure rotation (no
  shift) makes a radially-symmetric source non-uniformity term
  *mathematically indistinguishable* from lens vignetting — the joint
  least-squares system still solves, and solves confidently, for the wrong
  split between "the lens" and "the light source". `lens/flats.py` checks
  the joint design matrix's condition number (`FLAT_COND_THRESHOLD`) for
  exactly this, because a low residual on that fit means nothing about which
  half of V×S it actually recovered.
- **The distortion coverage check normalizes radius *per angle*, not by
  the frame's half-diagonal.** `lens/distortion.py::coverage_grid` used to
  divide every corner's distance from center by the fixed half-diagonal
  (only reached exactly at the image's 4 corners) before binning it into
  the outer-ring/angular-sector grid the coverage refusal reads. That
  makes every cardinal-direction edge *midpoint* — top, bottom, left,
  right — structurally unreachable at any realistic outer-ring threshold,
  for *any* rectangular sensor: a corner detected exactly at the
  top-center of the frame, as close to that edge as a point can ever be,
  still measured short of the "outer field" cutoff, so the fit's own
  sanity test (`test_coverage_refusal_does_not_fire_for_a_deliberately_full_grid`)
  could only pass with hand-placed points *outside* the image bounds —
  real (in-frame) corner data could never satisfy it, on a synthetic
  dataset or a real capture. `coverage_grid` now normalizes each corner's
  radius by the frame boundary's own distance from center *in that
  corner's direction* (`min(half_width/|cos θ|, half_height/|sin θ|)`),
  which asks the question the refusal is actually for: how close is this
  corner to the nearest edge of the frame, not to the far corner overall.
  `demo.py`'s lens step is what surfaced this — its first, honest attempt
  at "cover the field" with only random full-frame poses reliably left
  6-9 of 12 sectors empty regardless of view count, which is what led to
  digging into the normalization rather than just adding more random
  views.
- **The PTC shot-noise region is a *moving target*, not a fixed range.**
  `camera/ptc.py` doesn't take "the top N% of levels are shot-noise-limited"
  as given — PRNU's variance contribution grows with the *square* of signal,
  so it dominates first at the high end, and where that crossover happens
  depends on the sensor. The iterative residual trim
  (`PTC_RESIDUAL_FRACTION_MAX`) walks in from the high-signal end dropping
  points that depart from the running linear fit, rather than trusting a
  pre-picked cutoff — and then refuses outright if too few shot-noise-region
  points survive (`PTC_MIN_SHOT_NOISE_LEVELS`).
- **Black is subtracted before a gamma (or additivity) fit, never after.**
  `display/analysis.py::trc_fit` takes `black_y` and subtracts it from every
  ramp step first, because a real display's raw luminance is
  `black_y + contribution * level**gamma`, not a pure power law — fitting
  the raw curve fits gamma *and* the black level's leverage on the curve's
  shape at once, and gets neither right. `additivity()` does the same thing
  for the opposite reason: a raw R/G/B/W measurement each already carries
  one copy of the black offset, so naively summing R+G+B triples it against
  a W that only has one; each channel is black-corrected before summing,
  and the black term is added back exactly once.
- **A reference chart's XYZ belong to *its* illuminant.** A chart's values
  are its patches' reflectances integrated against the illuminant it states
  (D50, for colour-science's ColorChecker24), and no chromatic adaptation
  re-integrates a reflectance: a CAT moves a white point. `color.fit`
  therefore refuses when `--illuminant` disagrees materially with
  `ReferenceChart.illuminant_xy` (`REFERENCE_ILLUMINANT_XY_MAX_DELTA`), and
  the record carries both. Fitting the D50 chart under D65 — which every
  test and `demo.py` used to do — costs ΔE00 2.1 mean / 5.7 max in the fit
  targets before the camera is involved. The escape hatch for shooting under
  something else is a reference measured under it
  (`chart.reference_from_csv(..., illuminant_xy=...)`).
- **Display validation compares in the profile's own space.**
  `display/profile.py` Bradford-adapts to D50 (what an ICC matrix/TRC
  profile's tags mean, and what lcms reproduces), so `validate()` must too —
  `analysis.xyz_to_lab_pcs`, not `xyz_to_lab`'s normalize-by-the-display's-
  own-white. The two disagree by ~1.2 ΔE00 mean / 4.5 p95 on a *perfect*
  profile of a noise-free synthetic display, concentrated in the blues,
  which is enough to refuse a correct profile against a real colorimeter's
  thresholds. `xyz_to_lab` stays right for same-white comparisons
  (`uniformity`, where every cell is measured against the center cell).
- **The leave-one-out-fold caveat is not a footnote.** `camera/color.py`'s
  Tier-A/C validation, when no explicit held-out set is given, refits the
  matrix once per left-out patch — but each fold uses the cheap *linear*
  (raw-XYZ-error) fit, not the full DeltaE00-refined nonlinear fit the
  *reported* matrix gets, because refitting 24+ patches with scipy's
  DeltaE00 refinement on every fold is too slow to run in a test suite.
  `validation_method` is named `"leave_one_out_linear_folds"`, not
  `"leave_one_out"`, specifically so a reader of a record (or the HTML
  report, which spells this out again next to the chart) doesn't assume the
  per-patch numbers there are an exact replay of the fit above them — they're
  a faithful but slightly more conservative generalization estimate.
- **Never demosaic. Ever.** `raw.RawFrame` holds the untouched Bayer mosaic;
  `raw.planes()`/`raw.optical_black()` are the *only* sanctioned ways
  analysis code reads pixel values out of it, and both return the four CFA
  planes separately (`R`/`G1`/`G2`/`B`), never an interpolated image. This
  is why lens MTF/TCA can show green and red/blue disagreeing honestly (and
  why that disagreement is itself useful — it's longitudinal CA), and why a
  color matrix fit never has a demosaic algorithm's own color-fringing
  baked invisibly into its residuals.
- **`RawFrame.black_level`'s 4 values are indexed by *color*, not by
  position.** They're LibRaw's `cblack[0..3]`: one value per color-filter
  index of `color_desc`, i.e. **R, first-G, B, second-G** — verified against
  this project's own reference capture, where rawpy reports
  `color_desc=b"RGBG"` and `raw_pattern=[[0,1],[3,2]]`, so B is index 2 at
  raster position (1,1) while the second green is index 3 at (1,0). Zipping
  the four values onto raster positions — which `black_level_by_channel`
  did, against its own docstring — therefore swaps **B and G2 on every
  standard Bayer sensor**, silently wrong in exactly the case the function
  exists for. Only the two greens are affected by the visible-origin phase
  shift (R is R and B is B wherever they sit), and that shift is undone
  before they're named. `synth.sensor._black_level_tuple` re-derives the
  order rather than mirroring the same table, because written as a literal
  mirror the round-trip test could only prove the mapping was a bijection —
  which is how the swap survived. `raw.black_level_by_channel(frame)` is the
  one place this is done; use it, never a `mean(black_level)` scalar.
- **A slanted-edge/ChArUco target can land in the exact same mean-DN band as
  a flat.** `capture.manual.classify()` doesn't stop at "is the mean signal
  10-90% of the DN range" — it also checks spatial uniformity
  (`FLAT_MAX_CV`, a coefficient-of-variation cap): a real flat's spread is
  shot noise plus a few percent PRNU, while a two-tone target at a similar
  *mean* is bimodal and has a much higher CV. A caller that already knows
  every frame in a folder is one role (`lens/commands.py`'s own commands, for
  exactly this reason) can skip the heuristic entirely via
  `expected_role=`/`classify(frame, expected_role=...)`. A near-black frame
  with **no exposure time** (what `raw._read_metadata` gives you when neither
  exiftool nor dcraw is installed) is classified `near_black`, not `dark`:
  bias and dark are the same picture without an exposure time to tell them
  apart, and calling it either one hid a whole folder of bias frames from
  `camera bias`.

## Conventions

- **Python 3.11+**, `numpy`/`scipy`/`opencv-python-headless`/`colour-science`
  for analysis, `pygame` for the patch window, `rawpy` for real raw decode.
  `ruff` rules are narrow (`E4,E7,E9,F,B`, line length 110) — same as
  hydrationTracker's.
- **Every threshold has a reason, next to the constant** (`constants.py` at
  each level: foundation, `camera/constants.py`, `camera/color_constants.py`,
  `lens/constants.py`, `display/constants.py`), same convention as
  `store.py`'s `SHELF_LIFE_DAYS` table. A constant with no comment is a bug
  waiting to be filed.
- **Privacy (public repo).** A device id is `slug(model)-sha256(serial)[:8]`
  (`devices.device_id`) — never a raw serial. `devices.edid_hash` hashes the
  *whole* EDID block (serial bytes included) for change detection only;
  it's one-way and never paired with a human-readable serial in a record.
  Captures (`*.cr3`/`*.CR3`/`*.dng`/`captures/`) and `.venv/` are
  git-ignored; the default captures dir (`~/calsuite-captures`) is outside
  this repo on purpose.
- **`records/` is committed.** It's the suite's actual output, not a cache —
  small JSON plus `.npz` sidecars. `records/csot-t3-unknown/display.nominal-*.json`
  is the first real one (this laptop's panel, written by `calsuite devices`,
  provenance `nominal`).
- **Records dir / captures dir are both overridable** via `CALSUITE_RECORDS`
  / `CALSUITE_CAPTURES` (`config.py`) — every test that touches the store
  sets `CALSUITE_RECORDS` to a `tmp_path`, and `demo.py` does the same
  (saving/restoring whatever was there before) so a `calsuite demo` run
  never touches the real store.
- **`doctor.py`'s six checks are each an independent pure function of a
  `Store`** (`check_tools`, `check_stale_records`, `check_firmware_changes`,
  `check_edid_changes`, `check_profiles_without_recent_validation`,
  `check_unsuperseded_refusals`) — `run()` just calls all six and pools the
  `Finding`s. Add a seventh the same way, not by growing `run()`'s own logic.

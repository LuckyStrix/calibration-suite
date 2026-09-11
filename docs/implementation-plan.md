# Implementation plan — calibration suite (`calsuite`)

## Context

`project_planning/calibration_suite.md` is a design-only plan for one tool that
measures everything between a photon and a number: the Canon R100 sensor, each
lens, and each display, on Linux and Windows. It turns those measurements into
files that real software reads (lensfun XML, DCP/ICC camera profiles, installed
display ICC profiles), and every number carries its provenance, the way
filamentdb's values do. starTracker, 3mf_scripts, EFIP and the DIY
spectrophotometer are all waiting on its numbers.

This plan turns that design into something buildable: module contracts, how the
work is split, what gets built versus deferred, and how we verify it with no
hardware attached. It answers the design's open question §11.5: the project is
its own **public GitHub repo** (`LuckyStrix/calibration-suite`, name confirmed
free) living at `shared/calibration_suite/`, and it's kept out of Syncthing the
same way hydrationTracker is.

**Model split, as requested:** Opus 5 (this session) plans, orchestrates,
reviews, commits and pushes. **All code is written by Sonnet 5 subagents**
(`Agent(model: "sonnet", subagent_type: "general-purpose")`).

### Facts checked on this machine (2026-09-11)
- `gh` is logged in as **LuckyStrix** with the `repo` + `workflow` scopes. Git identity: Carter Laubach <cjlaubach@gmail.com>.
- Python 3.11.2, `uv` available; numpy 1.26, scipy 1.17, OpenCV 4.11 (with `cv2.aruco`), Pillow, pytest 7.2.
- **Missing:** rawpy, colour-science, pygame (all pip-installable into a venv); exiftool and ArgyllCMS (system packages that need sudo, so the user installs them; tests use fakes).
- `dcraw -i -v` works, so it can serve as the metadata fallback when exiftool is absent. `~/capt0000.cr3` exists (R100 + RF 50 mm) for a local-only smoke test.
- lensfun 0.3.3 data is installed. `mil-canon.xml` **already has "Canon RF 50mm F1.8 STM"** (ptlens a=0.002 b=−0.009 c=0.014, poly3 TCA, **no vignetting**, cropfactor 1.0). The R100 body isn't in it. The export therefore adds an R100 `<camera>` block and compares our fit against the existing entry (provenance `vendor`).
- There's an EDID at `/sys/class/drm/card0-eDP-1/edid` (CSOT MNG007JA1-2).
- Sibling conventions to copy: hydrationTracker's `pyproject.toml` (setuptools, dynamic deps from requirements.txt), narrow ruff rules (`E4,E7,E9,F,B`, line length 110), GitHub Actions CI (py3.11/3.12 matrix), MIT license "Carter Laubach", and a `CLAUDE.md` of working notes. filamentdb's `PROVENANCE` tuple pattern (`3mf_scripts/filamentdb.py:51`). The stockroom's server-side SVG chart style (`CIS_Stockroom_Inventory_System/src/stockroom/reports.py:312` `bar_chart`).

---

## Scope

**Built:** all *software* for design phases 0–5, the pure math of phase 6 (Tier B
spectral color, spectro display backend reading spectra files), `doctor`, and a
`demo` command. Every analysis is verified by a synthetic round trip, every
refusal by a test that triggers it, and every export format by reading it back
with an independent parser (Pillow/littleCMS for ICC, lensfunpy for lensfun XML
where it's installable, our own TIFF reader for DCP, a CGATS parser for `.ti3`).

**Not built (listed in the README as open work):**
- Real measurements. They need the user and the hardware, and no record gets `measured` without them.
- The starTracker README sentence (it's written once a real PTC exists).
- Canon EDSDK (the SDK needs registration; manual import covers Windows).
- A dispcal-style VCGT calibration loop.
- Photodiode/ESP32 flicker and latency.
- Adobe LCP.
- Wayland.

---

## Repo layout

```
calibration_suite/                       (git repo → github.com/LuckyStrix/calibration-suite, public)
  pyproject.toml  requirements.txt  requirements-dev.txt  LICENSE (MIT)  README.md  CLAUDE.md  .gitignore
  .github/workflows/ci.yml                ruff + pytest on ubuntu py3.11/3.12 AND windows-latest py3.11
  docs/design.md                          copy of project_planning/calibration_suite.md
  docs/implementation-plan.md             this plan
  calsuite/
    __init__.py (version)  config.py (records dir, captures dir; env CALSUITE_RECORDS / CALSUITE_CAPTURES)
    provenance.py   PROVENANCE = ("measured","derived","vendor","nominal"); ordering; rules
    fit.py          Refusal, Analysis(result, residuals, uncertainty, refusals, ok)
    store.py        Record schema v1, JSON store, sidecar artifacts, SHELF_LIFE table (with reasons), require_exportable()
    devices.py      camera/lens/display identity; EDID parser (pure) + Linux sysfs + Windows registry readers
    raw.py          RawFrame, FrameMeta, load() via rawpy, planes(), optical_black(); metadata: exiftool -j → dcraw -i -v fallback
    tools.py        find external binaries (exiftool, dcraw, gphoto2, dispwin, spotread, colprof, profcheck, colormgr), run with timeout
    constants.py    every threshold, with the *reason* for its value (hydration style); §10 accuracy estimates tagged "estimate"
    synth/          sensor.py (foundation) · lens.py · color.py · display.py (wave 2 owners)
    capture/        manual.py (folder import, EXIF role sorting: bias/dark/flat/target) · gphoto2.py (subprocess wrapper)
    camera/         bias.py ptc.py linearity.py darks.py fixed_pattern.py iso.py shutter.py report.py commands.py
                    color.py chart.py ssf.py dcp.py (color owner) — see waves
    lens/           charuco.py distortion.py tca.py flats.py mtf.py psf.py export_lensfun.py report.py commands.py
    display/        patches.py window.py osstate.py backends/{base,argyll,camera,spectro,synthetic}.py
                    measure.py analysis.py profile.py install_linux.py install_windows.py validate.py report.py commands.py
    formats/        icc.py (matrix/TRC ICC writer+reader) · cgats.py (.ti3 read/write) — foundation
    report/         svg.py (line, log-log, scatter, heatmap, bar; no JS) · html.py (single-file page, provenance badge, refusal box, inputs+SHA table, error budget)
    doctor.py  demo.py  cli.py (argparse; `calsuite` entry point; each area's commands.register(sub))
  targets/          generated ChArUco / slanted edge / gray ramp (PNG at 1920×1200 + PDF), plus the script that makes them
  records/          .gitkeep (real records land here and are committed)
  tests/            one file per module; fixtures/ (EDID with serial bytes zeroed)
```

A flat `calsuite/` package with no `src/` keeps the paths the design doc
uses. pytest gets `pythonpath = ["."]`.

**Privacy (public repo):** a device id is `slug(model)-sha256(serial)[:8]`. Raw
serial numbers are never written to records, and EDID test fixtures have their
serial bytes zeroed. Captures (`*.cr3`, `*.CR3`, `*.dng`, `captures/`) and
`.venv/` are git-ignored, and the default captures dir is `~/calsuite-captures`,
outside the synced folder.

## Core contracts (Wave 1 defines these; everyone else builds on them)

- **`RawFrame`** (frozen dataclass): `cfa` (full raw including masked margins), `pattern` (e.g. `"RGGB"` at the visible origin), `visible` (row/col slices), `black_level` (4 values from metadata), `white_level`, `meta: FrameMeta` (model, serial, firmware, lens, focal, aperture, exposure_s, iso, timestamp, sensor_temp_c, settings{LENR, high-ISO NR, HTP, shutter mode}), `path`, `sha256`. `planes(frame)` returns `{"R","G1","G2","B"}` float arrays. **Nothing ever demosaics.**
- **`Record`** JSON: `schema, id, kind, device, devices[], created, provenance, status ("ok"|"refused"), refusals[{check,message,value,threshold}], conditions{…}, method{name, calsuite_version, params}, inputs[{name, sha256}], derived_from[ids], result{}, residuals{}, uncertainty{}, artifacts[{name, sha256}]`. Stored at `records/<device-id>/<kind>-<UTC stamp>.json`, with sidecars (`.npz` maps) next to it. Refused records **are saved**, because a refusal is a finding. `require_exportable(record)` raises unless `status=="ok"` and the provenance is measured/derived. A record gets `measured` only when its analysis passes its refusal checks **and** its validation step, where the kind has one.
- **Analysis layering (house rule 5):** functions in `camera/`, `lens/` and `display/` take arrays + metadata and return `Analysis`. Only `commands.py` touches files, capture and the store.
- **Synthetic generators:** `synth.sensor.SensorModel(gain_e_per_dn, read_noise_e, black_dn, full_well_e, dark_current(temp), prnu, dsnu, hot_pixels, row_banding, pattern, shape)` and `synth.sensor.frame(model, exposure_s, flux, temp_c, rng) -> RawFrame`. Every generator is seeded.

## Work waves (Sonnet 5 builders)

**Wave 1: foundation** (1 agent, runs to completion first). Scaffolding, all the
core contracts above, `formats/icc.py` and `formats/cgats.py`, `report/`,
`capture/`, `devices.py` (EDID tested against a zeroed copy of this laptop's
EDID, which must decode to the chromaticities in the design §0 table),
`synth/sensor.py`, stub `commands.register` per area, CI, `.gitignore`, the
venv (`uv venv && uv pip install -r requirements-dev.txt`, which brings in
rawpy, colour-science, opencv-python-headless, pygame, lensfunpy when a wheel
exists, and ruff). **Opus reviews, runs the tests and commits.**

**Wave 2: four agents in parallel, one per area, in the same tree.** Each owns
only its own directories (plus its `synth/` module and tests), adds no
dependencies without asking, and doesn't commit.

| agent | owns | builds (design §) | key round trips / refusals |
|---|---|---|---|
| **2A sensor** | `camera/{bias,ptc,linearity,darks,fixed_pattern,iso,shutter,report}` + sensor commands | §3.1 all rows; the settings check (LENR on ⇒ dark series refused); gphoto2-guided capture + `--from DIR` | gain/read noise/full well/dark current recovered within §10 budgets; PTC refuses clipped pairs; ISO-invariance recommendation on a synthetic ISO sweep; hot-pixel map exact on injected pixels; the sensor HTML report emits the starTracker sentence |
| **2B lens** | `lens/*`, `synth/lens.py`, `targets/` generator | §4.1–4.7: ChArUco (green-plane detection, coordinates mapped back to sensor px), Brown–Conrady → lensfun ptlens/poly3 refit, TCA per CFA plane, self-calibrating flat (poses × rotation/shift, log-domain joint LSQ), slanted-edge e-SFR (ISO 12233) on a 5×3 grid, PSF moments/ellipticity map, relative T-stop, lensfun XML export incl. R100 `<camera>` block | synthetic distortion recovered < 0.05 px; **coverage refusal** when the outer 20% has holes; **degeneracy refusal** for rotation-only poses with a radial source term; Gaussian-blurred edge MTF50 within 5% of the analytic value; exported XML loads in lensfunpy and reproduces the distortion map (skip if lensfunpy is unavailable). The agent must check lensfun's radius-normalization and user-DB-path conventions against lensfun docs and cite them in comments |
| **2C display** | `display/*`, `synth/display.py` | §5: patch sets, pygame fullscreen patch window, OS state (VCGT reset via `dispwin -c`, X11 `_ICC_PROFILE` check, Windows HDR check via ctypes, falling back to a required `--confirm-hdr-off` and recording which method was used), backends (argyll `spotread`, camera via a `camera.color` record, spectro from spectra CSV + CIE CMFs, synthetic), TRC/additivity/primaries-vs-EDID/black/contrast/uniformity/warm-up/rolling-shutter banding, `.ti3` → `colprof` (matrix/shaper, or LUT when additivity fails) with a **built-in matrix/TRC ICC fallback** when Argyll is absent, install (colormgr + dispwin on Linux, with the Openbox autostart only behind `--write-autostart`; dispwin on Windows), validation through the profile (Pillow ImageCms Lab→RGB) → ΔE00 mean/p95/max | synthetic display's primaries/TRC recovered; a non-additive synthetic display ⇒ LUT recommended; built-in profile validates against the synthetic display under the backend's threshold; fake `spotread`/`colprof`/`dispwin` scripts on PATH exercise the subprocess paths; patch window smoke test with `SDL_VIDEODRIVER=dummy` |
| **2D color** | `camera/{chart,color,ssf,dcp}`, `synth/color.py` | §3.2: chart patch sampling from 4 user-given corners (6×4 grid, central 50%, per CFA plane), reference data from colour-science (ColorChecker 24, post-2014) or a custom CSV, a 3×3 fit minimizing ΔE2000 (white-preserving) plus root-polynomial options, Tier C leave-one-out validation, DCP writer (dual illuminant StdA+D65, ColorMatrix/ForwardMatrix per DNG spec normalization) + reader, ICC input profile via `formats/icc.py`, Tier B: SSF → matrix for any illuminant, Luther–Ives deviation, SMI (ISO 17321-1) | a known matrix is recovered; glare/gradient/clipping/too-few-patches refusals fire (rule: n_patches ≥ 4 × terms per channel); DCP round trip + DNG normalization properties (ForwardMatrix·1 = D50); ICC opened by ImageCms maps camera RGB → Lab matching our matrix; SSFs that are an exact linear combination of the CMFs ⇒ Luther–Ives deviation 0, SMI 100 |

Wave 2 ends with Opus running the whole suite and ruff and reviewing each
area's refusal logic. Findings go back to the owning agent via SendMessage.
Opus then commits.

**Wave 3: integration** (1 agent):
- `doctor.py` checks: tool availability; records past `SHELF_LIFE`; the camera firmware has changed since the last record; the current EDID hash differs from the display's records; an installed profile has no passing validation newer than it.
- `demo.py`: `calsuite demo --out DIR` runs every analysis on synthetic data into a temporary store and writes all the reports.
- The full CLI from design §7.
- The first real record: `calsuite devices` writes a `display.nominal` record (EDID primaries, provenance `nominal`) for the laptop panel.
- README (install incl. `sudo apt install libimage-exiftool-perl argyll`, workflow per phase, what's not built), CLAUDE.md (house rules + load-bearing details), and an e2e test of `demo`.

## Changes outside the new folder
- `shared/.stignore`: add `/calibration_suite` with a comment like the hydrationTracker one. It lives on GitHub, and syncing it too would copy `.venv` and captures between machines.
- `project_planning/calibration_suite.md` status line and `project_planning/README.md` row: point both at the repo.

## Git / GitHub
`git init -b main` in `shared/calibration_suite/`. There's one commit per wave,
each ending with the session's Co-Authored-By/Claude-Session trailer. After
local verification passes:
`gh repo create LuckyStrix/calibration-suite --public --source . --remote origin --push --description "..."`.
Then `gh run watch` the CI. If CI is red (py3.12 or Windows), a Sonnet agent
fixes it and the fix is pushed until CI is green.

## Verification
1. `.venv/bin/python -m pytest -q` passes. Every analysis module has a synthetic round-trip test and a test for each of its refusals.
2. `.venv/bin/ruff check calsuite tests` is clean.
3. `calsuite demo --out <scratchpad>` produces every HTML report. Opus opens a sample and checks the charts, provenance badges, refusal boxes and error budgets.
4. Local-only smoke tests (skipped in CI):
   - `calsuite devices` reads the real eDP EDID, and its primaries match the design §0 table (R 0.638,0.334 …).
   - `raw.load(~/capt0000.cr3)` returns 6288×4056 CFA data, an RGGB-type pattern, the RF 50 mm lens and ISO/exposure metadata via the dcraw fallback.
   - `calsuite lens` export writes lensfun XML that parses, and the report compares it with the RF 50 entry already in `mil-canon.xml`.
5. GitHub CI is green on ubuntu py3.11/3.12 and windows py3.11. The repo is public and visible at github.com/LuckyStrix/calibration-suite.

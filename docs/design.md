# Plan: calibration suite — camera, lenses, displays

Status: **design only.** Nothing is built. Proposed home:
`shared/calibration_suite/` (name open — see §11).

One tool that measures the things between a photon and a number: the R100's
sensor, each lens, and each display, on both Linux and Windows. It produces
files that real software reads (lensfun, DCP/ICC, installed display profiles)
and **records where every number came from**, the way filamentdb does.

---

## 0. What is actually here

Checked on this machine, 2026-09-11:

| thing | what we know | source |
|---|---|---|
| Camera | Canon EOS R100, raw 6288 × 4056 (6000 × 4000 visible), CR3 | `dcraw -i -v ~/capt0000.cr3` |
| Lens | RF 50 mm f/1.8 STM (50 mm, f/1.8 in that file) | same file |
| Other lenses | **unknown** | — |
| Laptop panel | CSOT MNG007JA1-2 (EDID "CSOT T3"), 1920 × 1200, 344 × 215 mm, 2021 | EDID from `/sys/class/drm/card0-eDP-1/edid` |
| …its claimed primaries | R (0.638, 0.334) G (0.300, 0.596) B (0.141, 0.058) W (0.312, 0.329), γ 2.2 | EDID — the maker's *nominal*, not a measurement |
| Windows PC monitors | **unknown** | — |
| Linux session | X11, BunsenLabs (Openbox, no colord-aware settings daemon) | `$XDG_SESSION_TYPE` |
| Tools present | gphoto2 2.5.28, darktable (uses lensfun), colormgr, dcraw, OpenCV 4.11, numpy, scipy, Pillow | `command -v` / imports |
| Tools missing | exiftool, ArgyllCMS, rawpy, colour-science | — |
| GPU | Radeon 680M iGPU, 30 GB RAM | `lspci` |

`capt0000.cr3` is gphoto2's default capture name, so tethered capture on
Linux already works.

---

## 1. Goals, and what it feeds

The suite is worth building because other projects are waiting on its numbers:

| consumer | needs | from |
|---|---|---|
| starTracker | recommended ISO, darks, flats, hot-pixel map, bulb timing | §3.1 |
| 3mf_scripts | the camera as a colorimeter, so a photographed wedge can be `measured` rather than `matched` | §3.2 |
| Digital-Twins (EFIP) | linear radiometry + flat field for photometric stereo; distortion for photogrammetry | §3.1, §4 |
| DIY spectrophotometer | linearity, shot-noise model, lens distortion for wavelength mapping | §3.1, §4 |
| darktable / RawTherapee | lensfun entries, DCP/ICC camera profiles | §4.7, §3.2 |
| every photo you edit | a profiled display on both OSes | §5 |

Non-goals: instrument drivers (ArgyllCMS already does colorimeters well on
both OSes), print profiling, video.

---

## 2. House rules

These come from how your other projects already work.

1. **Raw only, linear only.** No analysis ever reads a JPEG or a demosaiced
   image. Lens and sensor measurements work on the four CFA planes
   separately, so demosaicing never biases a result.
2. **Every result is a record with provenance.** A record names the device,
   the date, the firmware, the conditions (ISO, aperture, focus distance,
   sensor temperature, display brightness setting), the method, the SHA-256 of
   every input file, the fit residuals, and an uncertainty. Provenance runs
   `measured` → `derived` (computed from other measurements) → `vendor` →
   `nominal` (EDID, spec sheet). A record gets `measured` only by being
   measured.
3. **Refuse bad fits, and say why.** A distortion fit whose corners were never
   covered, a color matrix from a chart with glare on it, a PTC with clipped
   frames: each gets a red report and no export. As with `calibrate.py`, a
   tight residual on a degenerate dataset is the failure mode to guard
   against.
4. **Every analysis has a synthetic round trip.** Generate frames from a known
   gain, read noise, distortion or MTF; recover it; assert it comes back.
   That's the same move as "true td 0.30 fits to 0.2997", and the tests are
   where the risk lives.
5. **Pure analysis, separate I/O.** `camera/`, `lens/` and `display/` analysis
   functions take arrays and metadata and return results; capture and storage
   live elsewhere. Same layering as hydration's `model/`.
6. **Staleness is a finding.** `calsuite doctor` reports calibrations older
   than their shelf life, a firmware change since the last one, an EDID that
   no longer matches, or a display profile that is installed but no longer
   validates.
7. **Accuracy is stated, not implied.** Every report carries its own error
   budget, and every estimate in this plan gets replaced by a measured value
   once it exists.

---

## 3. Camera

### 3.1 Sensor characterization (no special equipment)

Everything here needs a lens cap, a uniform light, and patience.

| measurement | method | output |
|---|---|---|
| Black level / bias | optical-black margin (the 288 extra columns and 56 extra rows outside the visible 6000 × 4000) and cap-on bias frames, per CFA channel, per ISO; compare with the black level in metadata | offset table |
| Read noise | pairs of shortest-exposure bias frames; σ = std(A − B)/√2 cancels fixed pattern | e⁻ and DN, per ISO |
| Gain + photon transfer | pairs of flats at 20–30 signal levels; variance of difference/2 vs. mean; slope in the shot-noise region = 1/gain (Janesick) | e⁻/DN per ISO |
| Full well + linearity | mean vs. exposure time up to clipping; residual from a line, in % | saturation DN, linear range, max deviation |
| Dark current + hot pixels | dark series 30–300 s at several ambient temperatures; sensor temperature from Canon's maker notes (`CameraTemperature` via exiftool) | e⁻/s vs. °C, hot-pixel map per exposure length |
| Fixed pattern | DSNU from stacked darks, PRNU from stacked flats; row/column banding from the FFT of bias frames | maps + summary numbers |
| ISO invariance | input-referred read noise vs. ISO; the ISO above which it stops falling | **recommended astro ISO** |
| Dynamic range | engineering DR = log₂(full well / read noise), per ISO | stops |
| Shutter accuracy | measured signal vs. nominal time across the range | % error per speed |
| Bulb timing (ESP32) | a flat through the star tracker's optocoupler path at commanded bulb lengths | latency + scale error of the tracker's intervalometer |
| "Star eater" check | statistics of isolated hot pixels in long darks: does the camera filter single bright pixels in raw? | yes / no, and at what exposure |

**Uniform light source.** PTC needs *stable*, not uniform, light: differencing
a pair of frames cancels spatial structure. A cheap A4 LED tracing pad behind
two diffusers is enough, or the laptop display at full white. Flats (§4.3) do
need uniformity; that's handled there.

**Settings that silently change raw data**, recorded in every record and
checked against the metadata: long-exposure noise reduction (in-camera dark
subtraction), high-ISO NR, highlight tone priority, and the electronic
first-curtain shutter setting.

Deliverable: an R100 sensor report (single self-contained HTML, server-side
SVG charts, like the stockroom's), plus one sentence for the star tracker
README: *"ISO N is the invariance point; above it you're only losing
headroom."*

### 3.2 Color characterization

Three tiers, each built on the one before:

**Tier A — chart matrix.** Photograph a chart (ColorChecker Classic, or a DIY
chart measured by the spectrophotometer) under a known illuminant. Fit raw →
XYZ as a 3 × 3 matrix, with a root-polynomial option (Finlayson). Minimize
ΔE2000, not RGB error. Shoot under two illuminants (daylight and tungsten) for
a dual-illuminant DCP. Refusals:

- glare: non-uniformity inside the neutral patches above a threshold;
- uneven lighting: gray-patch luminance gradient across the chart;
- clipping in any channel of any patch;
- too few patches for the model order (the root-polynomial fit needs far more
  than 24).

**Tier B — spectral.** With the camera's spectral sensitivities (SSFs, from
the spectrophotometer plan §7), a matrix can be computed for *any* illuminant
from spectral data, with no chart shot per light. It also gives the sensor's
deviation from the Luther–Ives condition and a sensor metamerism index
(ISO 17321-1): a number for *how well this camera can ever do* colorimetry.

**Tier C — validation.** Always against patches the fit didn't see: hold-out
patches, a second target, or printed filament swatches measured by the
spectrophotometer. The profile is only labeled `measured` if validation
passes.

Outputs: DCP (darktable, RawTherapee, Lightroom) and an ICC input profile.
Either write DCP directly (the format is documented) or wrap `dcamprof`;
decide in Phase 5.

### 3.3 Capture backends

| backend | Linux | Windows |
|---|---|---|
| gphoto2 | installed; already works with the R100 | painful to build (see "Building gphoto2 for Windows") |
| Canon EDSDK | — | official, free, needs a Canon developer registration |
| Manual import | yes | yes |

**Manual import is first-class.** Shoot to the card, then point the suite at a
folder; it reads the EXIF to sort frames into roles (bias, dark, flat,
target). Tethering is a convenience, never a requirement, so every measurement
works on both OSes from day one.

---

## 4. Lenses

Per lens, per focal length (zooms), per aperture, and per focus distance where
it matters.

### 4.1 Geometric distortion

- **Target:** a ChArUco board. Either printed and glued to something flat, or
  **displayed on a monitor.** A display is perfectly flat at this scale and
  its pixel pitch is known (344 mm / 1920 = 0.179 mm on the laptop), but its
  cover glass adds a small refraction offset at steep angles. Use both once and
  compare.
- **Model:** Brown–Conrady (k₁, k₂, k₃, p₁, p₂) via OpenCV for internal use.
  Refit to lensfun's `ptlens` (a, b, c) or `poly3` for export.
- **Refusals:** coverage. Map where corners were detected; refuse the fit if
  the outer 20% of the field has holes, because distortion is measured where
  it's largest. Report reprojection RMS per image and drop outliers visibly.
- **Caveat worth writing down:** distortion changes with focus distance, and
  lensfun records per focal length only. Calibrate at the distance you use
  (infinity for astro, near for the EFIP rigs) and store the distance in the
  record.

### 4.2 Lateral chromatic aberration

Fit R and B scale against G from the same ChArUco corners, detected per CFA
plane. Export as lensfun `tca` (vr, vb).

### 4.3 Vignetting and flat fields

- Flats at each aperture (f/1.8, 2.8, 4, 5.6, 8), fitted to lensfun's `pa`
  model (1 + k₁r² + k₂r⁴ + k₃r⁶) for export. Full flat-field maps are kept for
  astro (they also capture dust).
- **The source is never perfectly uniform, so don't assume it is.** Shoot the
  same source with the camera rotated through 0/90/180/270° and shifted
  slightly. Lens vignetting is fixed to the sensor and the source's
  non-uniformity moves with each pose, so solve for both jointly. This is a
  self-calibrating flat field, and it produces a display-uniformity map (§5)
  as a by-product.
- Twilight sky flats as an independent check.

### 4.4 Sharpness (MTF)

- **Method:** slanted-edge e-SFR (ISO 12233). Edge at ~5°, supersampled ESF,
  derivative → LSF, FFT → MTF, per CFA plane (so green and red/blue MTF differ
  honestly, which also shows longitudinal CA).
- **Target:** a razor blade backlit by the display is straighter and sharper
  than anything printed. A laser print on matte paper is the fallback. Check
  that the target is sharper than the lens can resolve at the chosen
  magnification.
- **Grid:** a 5 × 3 field grid × apertures, giving an MTF50 map in cycles/px
  and lp/mm, with a "sweet spot" chart per lens.

### 4.5 PSF / coma / astigmatism (astro-relevant)

Artificial star: a pinhole in foil, LED behind it, 10+ m away, or real stars
on a tracked frame. Corner PSF shape vs. aperture answers "what do I stop down
to for astro at 50 mm?" with pictures.

### 4.6 Optional

Relative T-stop between lenses (same source, same exposure, signal ratio),
through-focus field curvature, focus breathing.

### 4.7 Export

- **lensfun XML** into the user data dir
  (`~/.local/share/lensfun/` on Linux; darktable's lensfun data dir on
  Windows). Verified by loading a test raw in darktable and checking that
  corrections apply. lensfun's `mil-canon.xml` already covers some RF gear;
  check coverage first and contribute upstream what's missing.
- **Adobe LCP:** stretch goal; the format is XMP-based but the details are
  unverified.

---

## 5. Displays

### 5.1 Measurement backends

Pluggable, and each carries its own accuracy statement:

| backend | how | accuracy (estimate until measured) | cost |
|---|---|---|---|
| Colorimeter via ArgyllCMS `spotread` | i1Display Pro class | ΔE00 ≈ 1; the reference | buy, or **borrow from the CIS stockroom** |
| DIY spectrophotometer, emission mode | spectra of the primaries → XYZ via the CIE CMFs | depends on wavelength accuracy and radiometric calibration (spectro plan §6) | built there |
| The camera as a colorimeter | characterized R100 (Tier B best) photographs patches | ΔE00 2–5 on sRGB-ish panels, worse on narrow-band wide-gamut | free |

The camera backend is honest only if it's cross-checked once against one of
the other two; the report says which backends it was checked against.

### 5.2 What gets measured

- **Tone response per channel:** 17–33 step ramps per channel plus gray.
- **Additivity:** measured W vs. R + G + B. If it fails, the display needs a
  LUT profile rather than matrix/TRC, and the report says so.
- **Primaries and white point vs. EDID claims:** "EDID says R (0.638, 0.334);
  measured (…, …)".
- **Black level and contrast ratio:** camera exposure bracketing is fine for
  relative luminance.
- **Uniformity:** luminance and chromaticity on a 5 × 5 grid, using the camera
  with the lens flat field from §4.3 applied. That's a dependency: lens before
  display.
- **Viewing-angle shift:** the camera at angles; laptop panels are notorious.
- **Warm-up drift:** measure for 30 min after power-on; tells you how long to
  wait before profiling.
- **PWM flicker:** rolling-shutter banding in a 1/8000 s frame of a white
  screen, or a photodiode on an ESP32 (idea 43 hardware).
- **Latency / response time:** photodiode + ESP32, optional.

### 5.3 Showing exact pixel values

Measurement is only as good as the certainty that the panel received the RGB
we asked for.

- Patch window: SDL2 (pygame) fullscreen on both OSes; no browser, because
  browsers color-manage.
- Before measuring, reset the video-card gamma table to linear
  (`dispwin -c`). Record the Windows HDR / Auto Color Management state and
  refuse to measure with HDR on. On Linux / X11, check nothing else is loading
  a VCGT.
- Record the OSD brightness / mode settings the user typed in; they're part
  of the display's identity for this profile.

### 5.4 Building the profile

All backends write measurements as **Argyll `.ti3` (CGATS text)**, and
ArgyllCMS's `colprof` builds the ICC. The pipeline is identical whatever
measured the patches, and Argyll's `profcheck` is available for self-checks.
An optional calibration loop (target white point and gamma → VCGT curves)
follows `dispcal`'s approach.

### 5.5 Installing it

| | Linux (this laptop, X11) | Windows |
|---|---|---|
| install | `colormgr import-profile` + `device-add-profile` + `device-make-profile-default` | `dispwin -I profile.icc`, or the Color Management control panel |
| load VCGT | `dispwin -I` (sets the `_ICC_PROFILE` X atom and loads the curves) at session start. No colord-aware daemon on Openbox, so an autostart entry is needed | Color Management → Advanced → "Use Windows display calibration", or Argyll's loader |
| gotchas | Wayland sessions are compositor-specific; noted, not supported at first | HDR / ACM on changes what apps see; profile SDR with HDR off |

### 5.6 Validation

After install, display a validation patch set **through the profile, in a
color-managed app**, measure it, and report ΔE2000 (mean, p95, max). The
profile is `measured` only if validation passes; otherwise it's installed
with a warning, or not at all.

---

## 6. Cross-platform summary

| capability | Linux | Windows |
|---|---|---|
| raw decode | rawpy (LibRaw ≥ 0.20 decodes CR3; dcraw here reads the header but don't trust it for CR3 pixels) | same |
| metadata | exiftool (to install) | exiftool.exe |
| capture | gphoto2 or manual import | EDSDK or manual import |
| patch window | pygame / SDL2 | same, HDR off |
| instruments | ArgyllCMS | ArgyllCMS (some instruments need its driver) |
| profile install | colord + dispwin | dispwin / Color Management |
| display identity | EDID from `/sys/class/drm/*/edid` | EDID from the registry (`HKLM\SYSTEM\CurrentControlSet\Enum\DISPLAY\…\Device Parameters\EDID`) |

**Sync:** `shared/` is a Syncthing folder. Raw captures are big: a PTC session
is 100+ frames at ~25–30 MB each. Keep captures outside the synced folder, or
in `.stignore`, and keep `.venv` out of sync for the same reason
hydrationTracker lives on GitHub instead. Records (small JSON) and code sync
fine.

---

## 7. Layout

```
calibration_suite/
  calsuite/
    devices.py      identity: body (model, serial, firmware), lens, display (EDID, both OSes)
    store.py        records, provenance, schema version, staleness rules
    raw.py          rawpy, CFA planes, optical-black margin, metadata via exiftool
    synth.py        synthetic frames with known parameters, for every round-trip test
    capture/        gphoto2.py  edsdk.py  manual.py
    camera/         bias.py  ptc.py  darks.py  linearity.py  ptc_report.py  color.py  ssf.py
    lens/           distortion.py  tca.py  flats.py  mtf.py  psf.py  export_lensfun.py
    display/        patches.py  backends/{argyll,camera,spectro}.py  measure.py
                    profile.py  install_linux.py  install_windows.py  validate.py
    spectro/        (see diy_spectrophotometer.md)
    report/         single-file HTML reports, server-side SVG charts
    doctor.py
    cli.py          `calsuite ...`
  targets/          ChArUco, slanted edge, gray ramp (PDF + PNG at exact display resolution)
  records/          <device-id>/<kind>-<date>.json  (in git)
  tests/
```

CLI sketch:

```sh
calsuite devices                       # what's connected, what's known, what's stale
calsuite camera ptc --iso 100..6400    # guided capture (or --from DIR)
calsuite camera darks --minutes 5
calsuite lens distortion --lens rf50 --from ~/captures/charuco/
calsuite lens flats --apertures 1.8,2.8,4,5.6,8
calsuite lens mtf --grid 5x3
calsuite display measure --backend argyll|camera|spectro
calsuite display profile && calsuite display install && calsuite display validate
calsuite export lensfun
calsuite doctor
```

---

## 8. Equipment

| have | cheap to get | borrow |
|---|---|---|
| R100, RF 50 mm, laptop panel, ESP32s, star tracker electronics, 4-head printer | LED tracing pad + diffusers; razor blades; ChArUco on foam board; PTFE sheet (white reference); exiftool, ArgyllCMS (free) | colorimeter and/or ColorChecker from the **CIS stockroom** — your own system probably knows whether they have one |

---

## 9. Phases

Each phase ends with a report and its synthetic round-trip tests passing.

| phase | contents | needs |
|---|---|---|
| 0 | skeleton, device identity (EDID both OSes, EXIF), record store, raw loader, `synth.py`, CI | nothing |
| 1 | bias, read noise, PTC, linearity, darks, hot pixels, ISO invariance → **R100 sensor report**, star tracker README updated | lens cap, uniform light |
| 2 | distortion, TCA, flats (self-calibrating) → **lensfun export verified in darktable** | ChArUco |
| 3 | slanted-edge MTF, astro PSF | razor blade |
| 4 | display: patch window, argyll + camera backends, TRC / primaries / uniformity, ICC via colprof, install + validate on **both** OSes | colorimeter ideally |
| 5 | camera color Tier A (chart → DCP/ICC) | chart or DIY chart |
| 6 | spectral: SSF-based Tier B, spectro display backend | spectrophotometer plan |

---

## 10. Expected accuracy (estimates — to be replaced)

| quantity | expect | limited by |
|---|---|---|
| gain | ±2–3% with 20+ pairs | source stability, pair count |
| read noise | ±5% | bias pair count |
| distortion | < 0.3 px RMS reprojection | coverage, target flatness |
| vignetting | ±1% | source non-uniformity (hence §4.3) |
| MTF50 | ±5% | edge quality, focus repeatability |
| display via colorimeter | ΔE00 ≈ 1 | instrument |
| display via camera | ΔE00 2–5 | camera metamerism vs. narrow primaries |

---

## 11. Open questions

1. **Which other lenses?** (the R100 kit is usually RF-S 18-45 or 18-150)
2. **What are the Windows PC's monitors and GPU**, and does it run HDR?
3. **Can you borrow a colorimeter and/or a ColorChecker** from the CIS stockroom?
4. **Windows tethering:** is a Canon developer registration for EDSDK fine, or
   is manual import enough there?
5. **Name and home:** `shared/calibration_suite/` synced by Syncthing with
   captures ignored, or its own GitHub repo like hydration-tracker?

# calsuite

Camera sensor, lens and display calibration — with provenance on every
number.

`calsuite` measures the things between a photon and a number: a camera
body's sensor (gain, read noise, dark current, linearity, ISO invariance), a
lens (distortion, chromatic aberration, vignetting, sharpness, PSF), and a
display (tone response, primaries, uniformity, an installable ICC profile).
It writes files real software reads — lensfun XML, DCP/ICC camera profiles,
installed display ICC profiles — and it records *where every number came
from*, the same way `filamentdb` does for filament colors.

Full design: [`docs/design.md`](docs/design.md). How it was built, wave by
wave: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Provenance and refusal, in one paragraph

Every record has a `provenance` (`measured` / `derived` / `vendor` /
`nominal` — how the number was produced) and a `status` (`ok` / `refused` —
whether it can be trusted). The two are independent: a chart genuinely was
photographed (`measured`), but if its fit fails its own quality checks or
its own hold-out validation, `status` is `refused` and
`store.require_exportable()` will not let it near a lensfun/DCP/ICC export.
Refused records are **saved, not discarded** — a refusal is a finding (a
distortion fit whose corners never covered the frame's edge, a color matrix
from a glare-lit chart, a PTC with clipped frames), and the HTML report
shows it in a red box with the reason, not silence. `CLAUDE.md` has the
full policy and the load-bearing details behind several of the refusals.

## Install

```sh
uv venv
uv pip install -e .
uv pip install -r requirements-dev.txt   # pytest, ruff, pygame, lensfunpy
```

(or `python -m venv .venv && .venv/bin/pip install -e . -r requirements-dev.txt`
if you don't have `uv`.)

That gets you every *software* path: all analysis, all synthetic round
trips, `calsuite demo`, the built-in ICC writer. Two more things unlock real
measurement:

```sh
sudo apt install libimage-exiftool-perl argyll
```

- **`exiftool`** — full raw metadata (serial, firmware, lens, sensor
  temperature). Without it, `raw.py` falls back to `dcraw -i -v`, which
  gives camera model/ISO/shutter/aperture/focal length only — good enough
  to run, not to distinguish two firmware versions of the same body.
- **ArgyllCMS** (`spotread`, `colprof`, `profcheck`, `dispwin`) —
  `spotread` is the colorimeter display-measurement backend; `colprof`
  builds an ICC profile from a `.ti3` (calsuite has a **built-in
  matrix/TRC fallback ICC writer** when Argyll is absent — a LUT profile
  for a non-additive display still needs Argyll); `dispwin` resets the
  video card's gamma table before measuring and installs a profile;
  `profcheck` self-checks a built profile.

`calsuite doctor` reports exactly which of these are present on your
machine and what each one unlocks — run it any time.

## Walkthrough: capture → analyse → report → export

Every command below is real and runs against a folder of already-captured
raw files (`--from DIR`) — tethered capture via gphoto2 (`--capture`, Linux
only so far) is a convenience layered on top, never a requirement (design
§3.3: "manual import is first-class"). A folder can hold real
`.cr3`/`.dng`/`.nef` files or `.npz` files written by `raw.save_npz` — the
two are interchangeable everywhere.

**Sensor** (needs a lens cap and a stable light source):

```sh
calsuite camera bias --from ~/calsuite-captures/bias/          # black level, read noise
calsuite camera ptc --from ~/calsuite-captures/ptc/            # gain, read noise (photon transfer)
calsuite camera linearity --from ~/calsuite-captures/flats/    # full well, linear range
calsuite camera darks --from ~/calsuite-captures/darks/        # dark current, hot pixels, star-eater check
calsuite camera fixed-pattern --from ~/calsuite-captures/fpn/  # DSNU, PRNU, banding (darks + flats + biases in one folder)
calsuite camera iso --device-id <id>                           # ISO invariance, from the bias/ptc/linearity records above
calsuite camera report --device-id <id> --out sensor.html
```

**Lens** (needs a ChArUco target — printed, or displayed full-screen):

```sh
calsuite lens distortion --from ~/calsuite-captures/charuco/ --square-mm 34.4
calsuite lens tca --from ~/calsuite-captures/charuco/
calsuite lens flats --from ~/calsuite-captures/flats-poses/ --aperture 1.8 \
    --poses "0:0:0,90:0.08:0,180:0:0.08,270:0.08:0.08"
calsuite lens mtf --from ~/calsuite-captures/slanted-edge/ --grid 3x5
calsuite lens psf --from ~/calsuite-captures/stars/
calsuite lens export --device-id <id> --lens-model "Canon RF 50mm F1.8 STM" --out rf50.xml
calsuite lens report --device-id <id> --lens-model "Canon RF 50mm F1.8 STM" --out lens.html
```

**Camera colour** (needs a ColorChecker or a DIY chart, one photo):

```sh
calsuite camera color fit --from ~/calsuite-captures/chart/ \
    --corners 60,60,660,60,660,420,60,420 --illuminant D65
calsuite camera color report --record records/<id>/camera.color-*.json --out color.html
calsuite export dcp --record records/<id>/camera.color-*.json --out camera.dcp
calsuite export icc --record records/<id>/camera.color-*.json --out camera-input.icc
```

**Display**:

```sh
calsuite display nominal                                        # EDID -> display.nominal (provenance "nominal")
calsuite display measure --backend argyll --device-id <id>       # or --backend camera / spectro / synthetic
                                                                 # argyll: calibrate the instrument at the terminal first, then the screen
                                                                 # goes fullscreen. The last 25 patches are small white squares (uniformity):
                                                                 # each one waits for you to put the instrument on it and press SPACE --
                                                                 # the instructions are drawn on the screen, since the terminal is hidden.
calsuite display profile --device-id <id> --out profile.icc
calsuite display install --device-id <id>
calsuite display validate --backend argyll --device-id <id>
calsuite display report --device-id <id> --out display.html
```

**Anywhere:**

```sh
calsuite devices    # what's connected, what's known, what's stale -- and writes
                     # a display.nominal record for every detected display
calsuite doctor     # tool availability, stale records, firmware/EDID drift, unvalidated profiles
```

## `calsuite demo`

```sh
calsuite demo --out /tmp/calsuite-demo
```

Runs every analysis — sensor PTC/bias/linearity/darks/ISO, lens
distortion/TCA/flats/MTF/PSF, camera colour Tier A (chart matrix) and Tier B
(spectral), display measure/profile/validate — on **synthetic** data, with
no camera, no instrument, and no ArgyllCMS attached, through the real CLI
commands against synthetic frames written to disk (`raw.save_npz`). It
writes one HTML report per area plus an index page linking them, in well
under 10 seconds. This is the fastest way to see what a report looks like,
and it's what CI (and every reviewer without a camera on hand) actually
runs.

## Accuracy (estimates — not yet measured)

From `docs/design.md` §10, carried verbatim into `constants.ESTIMATED_ACCURACY`
and shown on every relevant HTML report's error-budget table. These are
*targets a well-executed capture session should hit*, not measurements —
house rule 7: "accuracy is stated, not implied," and every one of these gets
replaced by a real measured value once one exists.

| quantity | expected | limited by |
|---|---|---|
| gain | ±2–3% (20+ PTC pairs) | source stability, pair count |
| read noise | ±5% | bias pair count |
| distortion | < 0.3 px RMS reprojection | target coverage, flatness |
| vignetting | ±1% | source non-uniformity (why the flat field is self-calibrating) |
| MTF50 | ±5% | edge quality, focus repeatability |
| display via colorimeter | ΔE00 ≈ 1 | instrument |
| display via camera | ΔE00 2–5 | camera metamerism vs. narrow display primaries |

## Cross-platform (design §6)

| capability | Linux | Windows |
|---|---|---|
| raw decode | rawpy (LibRaw) | same |
| metadata | exiftool, dcraw fallback | exiftool.exe |
| capture | gphoto2, or manual import | EDSDK (unbuilt), or manual import |
| patch window | pygame / SDL2 | same, HDR off |
| instruments | ArgyllCMS | ArgyllCMS |
| profile install | colord + dispwin | dispwin / Color Management |
| display identity | EDID via `/sys/class/drm/*/edid` | EDID via the registry |

Manual import (a folder of raw files) works identically on both OSes for
every measurement — tethering is the only OS-specific convenience.

## Not built / unverified

Honestly, in one place:

- **The display side has real measurements now; the camera and lens sides
  don't.** `records/csot-t3-unknown/` has a real measure → profile → report
  pass against this laptop's panel (`measured`/`derived`, `status="ok"`)
  plus a VCGT-bearing ICC profile. The one `display.validation` record on
  file predates that profile, though — it validated the profile build
  before the VCGT correction fix, not the current one — so `calsuite
  doctor` correctly flags the current profile as unvalidated; rerun
  `calsuite display validate` against it before trusting it end to end.
  The camera side has one real capture attempt —
  `records/canon-eos-r100-cf6f80d8/camera.darks-*.json` — but it's
  `status="refused"` (long-exposure NR was on for the series), so no
  camera number has cleared its own checks yet. No lens has been shot at
  all. `docs/implementation-plan.md`'s "local-only smoke tests" section
  names what's still outstanding: a lensfun export compared against the
  installed `mil-canon.xml`, and (now that dark frames exist) a PTC/
  linearity/ISO-invariance pass with LENR off.
- **Canon EDSDK** (Windows tethered capture) — needs a Canon developer
  registration; manual import covers Windows fully in the meantime.
- **A `dispcal`-style VCGT calibration loop** (target white point + gamma →
  VCGT curves) is not implemented; `calsuite display install` loads a
  profile's existing curves, it doesn't fit new ones.
- **Photodiode/ESP32 flicker and response-time measurement** — design §5.2
  lists these as optional hardware; not built.
- **Adobe LCP export** — a stretch goal in the design doc; only lensfun XML
  and DCP/ICC are implemented.
- **Wayland** — not supported; the patch window and OS-state checks assume
  X11 on Linux.
- **The Windows code paths** (`devices.read_edid_windows`,
  `display/install_windows.py`) are written from documented registry
  layouts and API behaviour, not run on a real Windows machine as part of
  this build — treat a mismatch as a detail to fix, not an architecture
  problem.
- **The `argyll` backend's prompt protocol is verified only on Linux, with
  a ColorMunki Photo.** `spotread` runs as one long-lived session under a
  pty (`display/backends/spotread_session.py` has the real transcript);
  Windows falls back to plain pipes, which works against the test fake but
  is unverified against real ArgyllCMS (it reads the console API directly).
  The *readings'* accuracy on this laptop panel is likewise unmeasured —
  the ΔE00 figure on the record is the design's colorimeter estimate, and a
  spectrophotometer without a display-specific correction can be off on a
  white-LED backlight.
- **The `camera` display-measurement backend** is only trustworthy once
  cross-checked against `argyll` or `spectro` on a real panel (design §5.1)
  — that cross-check hasn't happened yet.

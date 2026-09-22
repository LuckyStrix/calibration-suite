"""Display patch sets (docs/design.md §5.2/§5.3): pure data -- a ``Patch``
just names the RGB to show, where to show it, and (for the validation set)
a target Lab it should reproduce through a profile. Nothing here touches a
window, a backend, or a file; ``display/window.py`` draws a ``Patch``,
``display/backends/*`` measure one, ``display/validate.py`` fills in
``rgb`` for the validation set once an ICC profile exists to map its
``lab_target`` through.
"""

from __future__ import annotations

from dataclasses import dataclass

from calsuite.display import constants as dc

# The chart dataset colour-science ships, matching docs/design.md §3.2's
# "ColorChecker 24 ... post-2014" (the post-Nov-2014 reprint has slightly
# different pigments/values than the older chart colour-science also
# carries under a different key).
_CC24_DATASET = "ColorChecker24 - After November 2014"


@dataclass(frozen=True)
class Patch:
    """One patch to display and measure.

    ``rgb``: the exact (r, g, b) drive level in [0, 1]^3 to show, or
    ``None`` for a validation patch whose RGB isn't known yet (it's
    computed by ``display/validate.py`` from ``lab_target`` through the
    device's own ICC profile -- the whole point of that patch set).
    ``position``: fractional screen coordinates ((0.5, 0.5) = center) the
    patch should be centered at; only ``uniformity_grid`` varies this.
    ``size_frac``: fraction of the shorter screen dimension the patch
    square covers; 1.0 (full screen) for everything except uniformity,
    which needs a genuinely local sample.
    ``placement``: see the field's own comment; set only by
    ``uniformity_grid``.
    ``lab_target``: target CIE Lab, referenced to the same white used to
    build the display's ICC profile (D50, matching the ICC PCS -- see
    ``formats/icc.py``'s ``ICC_PCS_ILLUMINANT_D50``), for the validation set.
    """

    rgb: tuple | None
    position: tuple = (0.5, 0.5)
    size_frac: float = 1.0
    label: str = ""
    lab_target: tuple | None = None
    placement: str | None = None
    # ``None`` for a patch that fills the screen -- the instrument can sit
    # anywhere on it. Otherwise a human-readable "where" (``"row 2 of 5,
    # column 4 of 5"``): this patch is a small square the instrument has to
    # be *moved onto*, so a run driving a real instrument must stop and wait
    # for a person before measuring it.


def channel_ramp(channel: str, steps: int = dc.RAMP_STEPS_DEFAULT) -> list:
    """A `steps`-step ramp (design §5.2: "17-33 step ramps per channel")
    driving only `channel` ("r"/"g"/"b"), the other two held at zero."""
    if channel not in ("r", "g", "b"):
        raise ValueError(f"channel must be 'r', 'g' or 'b', got {channel!r}")
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    patches = []
    for i in range(steps):
        level = i / (steps - 1) if steps > 1 else 1.0
        rgb = [0.0, 0.0, 0.0]
        rgb[idx] = level
        patches.append(Patch(rgb=tuple(rgb), label=f"{channel}-ramp-{level:.4f}"))
    return patches


def gray_ramp(steps: int = dc.RAMP_STEPS_DEFAULT) -> list:
    """A `steps`-step neutral (r=g=b) ramp, plus gray -- design §5.2."""
    patches = []
    for i in range(steps):
        level = i / (steps - 1) if steps > 1 else 1.0
        patches.append(Patch(rgb=(level, level, level), label=f"gray-ramp-{level:.4f}"))
    return patches


def primaries_secondaries() -> list:
    """Full-drive R, G, B, C, M, Y, W, K (design §5.2: "primaries and white
    point vs. EDID claims" needs the primaries; secondaries and K come
    along for free and are useful gamut/black sanity checks)."""
    named = {
        "r": (1.0, 0.0, 0.0),
        "g": (0.0, 1.0, 0.0),
        "b": (0.0, 0.0, 1.0),
        "c": (0.0, 1.0, 1.0),
        "m": (1.0, 0.0, 1.0),
        "y": (1.0, 1.0, 0.0),
        "w": (1.0, 1.0, 1.0),
        "k": (0.0, 0.0, 0.0),
    }
    return [Patch(rgb=rgb, label=name) for name, rgb in named.items()]


def additivity_set() -> list:
    """R, G, B, W, K at full drive -- exactly the five measurements
    ``display.analysis.additivity`` needs (black-corrected R+G+B vs.
    measured W, design §5.2)."""
    named = {"r": (1.0, 0.0, 0.0), "g": (0.0, 1.0, 0.0), "b": (0.0, 0.0, 1.0), "w": (1.0, 1.0, 1.0), "k": (0.0, 0.0, 0.0)}
    return [Patch(rgb=rgb, label=f"additivity-{name}") for name, rgb in named.items()]


def uniformity_grid(n: int = dc.UNIFORMITY_GRID_N, rgb: tuple = (1.0, 1.0, 1.0)) -> list:
    """An n x n grid of `rgb` patches (design §5.2's 5x5 default) covering
    the panel from ``UNIFORMITY_GRID_INSET`` in from one edge to the same
    distance in from the other, so the outermost squares (the corners, where
    non-uniformity is usually worst) are still fully on screen and an
    instrument can be put on them. Returned row-major, shape (n, n),
    matching what ``display.analysis.uniformity`` expects; index
    [n // 2][n // 2] is the center cell it compares every other cell
    against (an odd n keeps it at exactly (0.5, 0.5)).
    """
    lo, hi = dc.UNIFORMITY_GRID_INSET, 1.0 - dc.UNIFORMITY_GRID_INSET
    positions = [lo + (hi - lo) * i / (n - 1) if n > 1 else 0.5 for i in range(n)]
    grid = []
    for yi, y in enumerate(positions):
        row = []
        for xi, x in enumerate(positions):
            row.append(
                Patch(
                    rgb=rgb,
                    position=(x, y),
                    size_frac=dc.UNIFORMITY_PATCH_SIZE_FRAC,
                    label=f"uniformity-{yi}-{xi}",
                    placement=f"row {yi + 1} of {n}, column {xi + 1} of {n}",
                )
            )
        grid.append(row)
    return grid


def warmup_patch(rgb: tuple = (1.0, 1.0, 1.0)) -> Patch:
    """The single patch (full white by default) re-measured repeatedly for
    the warm-up series."""
    return Patch(rgb=rgb, label="warmup")


def warmup_schedule(duration_s: float = 1800.0, interval_s: float = 60.0) -> list:
    """Time offsets (seconds since power-on) at which to re-measure
    ``warmup_patch()`` -- design §5.2's "measure for 30 min after
    power-on". Pure data: `display/commands.py` (or a test) is responsible
    for actually waiting between measurements; this just says when."""
    times = []
    t = 0.0
    while t <= duration_s:
        times.append(t)
        t += interval_s
    return times


def validation_set() -> list:
    """ColorChecker 24 (colour-science) plus extra neutral steps (design
    §5.2/§3.2), as Lab targets with no RGB assigned yet -- ``display/
    validate.py`` fills `rgb` in by mapping each `lab_target` through the
    device's ICC profile. Two of CC24's own 24 patches ("dark skin",
    "light skin") already cover the "skin tones" design asks for
    alongside its six built-in neutrals; `VALIDATION_EXTRA_NEUTRAL_L`
    adds denser gray-axis sampling between them.
    """
    import colour

    cc = colour.CCS_COLOURCHECKERS[_CC24_DATASET]
    patches = []
    for name, xyy in cc.data.items():
        xyz = colour.xyY_to_XYZ(xyy)
        lab = colour.XYZ_to_Lab(xyz, illuminant=cc.illuminant)
        patches.append(Patch(rgb=None, label=f"cc24-{name}", lab_target=tuple(float(v) for v in lab)))
    for l_star in dc.VALIDATION_EXTRA_NEUTRAL_L:
        patches.append(Patch(rgb=None, label=f"neutral-L{l_star:.1f}", lab_target=(float(l_star), 0.0, 0.0)))
    return patches

"""Build an ICC display profile from a measurement set (docs/design.md
§5.4): write ``.ti3``, then either run ArgyllCMS's ``colprof`` (+
``profcheck`` as a self-check) when it's installed, or fall back to a
built-in matrix/TRC profile (``formats/icc.py``) when it isn't -- but only
if the display measured additive; a non-additive display with no Argyll
gets an explicit refusal, since the built-in writer has no LUT capability.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from calsuite import tools
from calsuite.display import constants as dc
from calsuite.formats import cgats, icc


class ProfileRefused(RuntimeError):
    """No profile can legitimately be built: Argyll is absent and the
    display's additivity check failed, so the built-in matrix/TRC
    fallback (no LUT support) cannot represent it honestly."""


# ---------------------------------------------------------------------------
# .ti3
# ---------------------------------------------------------------------------


def samples_from_measurements(rgb_list: list, xyz_list: list) -> list:
    """Build ``formats.cgats.write_ti3``-ready samples: XYZ normalized so
    the full-white (1,1,1) patch lands at Y=1 -- the .ti3/DISPLAY
    convention (``formats/cgats.py``'s docstring: "XYZ with Y=1 for a
    perfect white"). Every backend in this wave reports XYZ in its own
    absolute-ish units (cd/m^2 for argyll/synthetic, relative-exposure
    units for camera, integrated-spectrum units for spectro); this
    normalization is what makes them all comparable at the .ti3 boundary.
    """
    white_y = None
    for rgb, xyz in zip(rgb_list, xyz_list, strict=True):
        if tuple(rgb) == (1.0, 1.0, 1.0):
            white_y = float(np.asarray(xyz)[1])
            break
    if white_y is None or white_y <= 0:
        raise ValueError("no full-white (1, 1, 1) sample found (or its Y <= 0) to normalize the .ti3 against")
    return [
        {"rgb": tuple(float(v) for v in rgb), "xyz": tuple(float(v) / white_y for v in xyz)}
        for rgb, xyz in zip(rgb_list, xyz_list, strict=True)
    ]


def write_ti3(path: Path, rgb_list: list, xyz_list: list, *, descriptor: str = "calsuite display measurement") -> None:
    samples = samples_from_measurements(rgb_list, xyz_list)
    cgats.write_ti3(path, samples, device_class="DISPLAY", descriptor=descriptor)


# ---------------------------------------------------------------------------
# colprof / profcheck
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileResult:
    path: Path
    method: str  # "colprof -as (matrix/shaper)" | "colprof (LUT)" | "built-in matrix/TRC"
    profcheck_ok: bool | None  # None if profcheck wasn't run (not installed)
    profcheck_output: str
    cal_path: Path | None = None  # VCGT correction curve (dispwin <calfile>), None if not built
    vcgt_note: str | None = None  # why cal_path is None, when it is


def build_with_colprof(
    ti3_base: Path,
    *,
    recommend_lut: bool,
    description: str,
    out_path: Path | None = None,
    timeout: float = dc.COLPROF_TIMEOUT_S,
) -> ProfileResult:
    """Run ``colprof`` against ``<ti3_base>.ti3`` (already written by
    ``write_ti3``). Flags verified against ArgyllCMS's own documentation,
    fetched 2026-09-11: https://www.argyllcms.com/doc/colprof.html --

    - ``-as``: "shaper curve + matrix profile ... three independent
      curves" -- used when the display measured additive (design §5.4:
      "matrix/shaper"), the simpler, more robust model when it applies.
    - no ``-a`` flag: colprof's own documented default (``-al``, a
      cLUT-based L*a*b* PCS profile) -- used when additivity failed, since
      only a LUT can represent behavior a matrix/TRC model assumes away.
    - ``-D "<description>"``: profile description tag.
    - table resolution is left at colprof's own default ("Medium", ``-q
      m``) -- nothing about a display's own achievable accuracy calls for
      "Ultra", and it would only slow the build down.

    Then runs ``profcheck`` (design §5.4: "Argyll's profcheck is available
    for self-checks"), if present, against the same ``.ti3``/profile pair.
    """
    ti3_base = Path(ti3_base)
    ti3_path = ti3_base.with_suffix(".ti3")
    icc_path = Path(out_path) if out_path is not None else ti3_base.with_suffix(".icc")

    args = ["colprof", "-D", description]
    if not recommend_lut:
        args.append("-as")
    args.append(str(ti3_base))
    tools.run(args, timeout=timeout)

    if not icc_path.exists():
        # colprof writes <basename>.icc (or .icm on some builds) next to
        # the .ti3 by default; fall back to whichever extension actually
        # landed rather than assuming.
        alt = ti3_base.with_suffix(".icm")
        if alt.exists():
            icc_path = alt

    profcheck_ok, profcheck_output = None, ""
    if tools.which("profcheck"):
        result = tools.run(["profcheck", str(ti3_path), str(icc_path)], check=False)
        profcheck_output = result.stdout + result.stderr
        profcheck_ok = result.returncode == 0

    method = "colprof (LUT)" if recommend_lut else "colprof -as (matrix/shaper)"
    return ProfileResult(path=icc_path, method=method, profcheck_ok=profcheck_ok, profcheck_output=profcheck_output)


# ---------------------------------------------------------------------------
# built-in matrix/TRC fallback
# ---------------------------------------------------------------------------


def build_fallback_matrix_trc(
    out_path: Path,
    *,
    primaries_measured: dict,
    trc_analysis,
    description: str = "calsuite built-in display profile",
) -> ProfileResult:
    """Built-in matrix/TRC ICC (``formats/icc.py``), used only when
    Argyll's ``colprof`` isn't available *and* the display's additivity
    check passed. ``primaries_measured``: ``{"r": xyz, "g": xyz, "b": xyz,
    "w": xyz, "k": xyz}`` full-drive measurements (the same shape
    ``display.analysis.primaries_vs_edid`` takes). ``trc_analysis``: the
    ``fit.Analysis`` from ``display.analysis.trc_fit`` -- its
    ``effective_gamma`` per channel becomes the profile's TRC (falling
    back to 2.2, a conventional display gamma, for any channel whose fit
    was refused).

    The matrix is built from black-corrected R/G/B, normalized to the
    measured white's own Y=1, then Bradford chromatic-adapted from that
    measured white chromaticity to the ICC PCS illuminant D50 (formats/
    icc.py's ``ICC_PCS_ILLUMINANT_D50`` / ICC.1:2010 §7.2.16) -- exactly
    the adaptation an ICC matrix/TRC profile's rXYZ/gXYZ/bXYZ tags require,
    since those are always expressed in the PCS's own D50 space regardless
    of the device's native white point.
    """
    import colour

    black = np.asarray(primaries_measured["k"], dtype=np.float64)
    r = np.asarray(primaries_measured["r"], dtype=np.float64) - black
    g = np.asarray(primaries_measured["g"], dtype=np.float64) - black
    b = np.asarray(primaries_measured["b"], dtype=np.float64) - black
    w = np.asarray(primaries_measured["w"], dtype=np.float64)
    y_white = float(w[1])
    if y_white <= 0:
        raise ProfileRefused("measured white luminance is zero or negative; cannot build a profile from it")

    matrix_native = np.column_stack([r, g, b]) / y_white
    white_xy = colour.XYZ_to_xy(w / y_white)
    d50_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    adapt = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        colour.xy_to_XYZ(white_xy), colour.xy_to_XYZ(d50_xy), transform="Bradford"
    )
    matrix_d50 = adapt @ matrix_native

    gammas = trc_analysis.result.get("effective_gamma", {})
    trc = {ch: float(gammas[ch]) if ch in gammas else 2.2 for ch in ("r", "g", "b")}

    icc.write_profile(out_path, device_class="mntr", description=description, matrix=matrix_d50, trc=trc)
    return ProfileResult(path=Path(out_path), method="built-in matrix/TRC", profcheck_ok=None, profcheck_output="")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def build_profile(
    out_path: Path,
    *,
    rgb_list: list,
    xyz_list: list,
    additivity_analysis,
    trc_analysis,
    primaries_measured: dict,
    description: str = "calsuite display profile",
) -> ProfileResult:
    """The single entry point ``display/commands.py`` calls: writes the
    ``.ti3``, then picks a build path per design §5.4's decision tree --
    colprof (matrix/shaper or LUT, by additivity) when installed;
    otherwise the built-in matrix/TRC writer if additivity passed, or an
    explicit refusal if it didn't.
    """
    out_path = Path(out_path)
    ti3_base = out_path.with_suffix("")
    write_ti3(ti3_base.with_suffix(".ti3"), rgb_list, xyz_list, descriptor=description)
    recommend_lut = bool(additivity_analysis.result.get("recommend_lut"))

    if tools.which("colprof"):
        result = build_with_colprof(ti3_base, recommend_lut=recommend_lut, description=description, out_path=out_path)
    elif recommend_lut:
        de00 = additivity_analysis.result.get("additivity_de00")
        raise ProfileRefused(
            "ArgyllCMS's colprof is not installed and this display failed its additivity check "
            f"(ΔE00={de00:.2f} > {dc.ADDITIVITY_DE00_MAX}), so it needs a LUT profile -- the "
            "built-in ICC writer (formats/icc.py) is matrix/TRC only. Install ArgyllCMS "
            "(`colprof`) to build a LUT profile for this display."
        )
    else:
        result = build_fallback_matrix_trc(out_path, primaries_measured=primaries_measured, trc_analysis=trc_analysis, description=description)

    return _attach_vcgt(result, ti3_base.with_suffix(".cal"), trc_analysis)


def _attach_vcgt(result: ProfileResult, cal_path: Path, trc_analysis) -> ProfileResult:
    """Build+write the VCGT correction curve (docs/design.md gap: `colprof`
    here never embeds a `vcgt` tag -- see CLAUDE.md) and attach it to
    `result`, or attach why it couldn't be built. Never raises: a missing
    VCGT curve doesn't invalidate the ICC profile itself, so this can't turn
    an otherwise-good profile build into a refusal (`_cmd_profile` keeps the
    two independent for exactly that reason)."""
    from calsuite.display import analysis as displayanalysis

    vcgt_analysis = displayanalysis.vcgt_correction(trc_analysis)
    if not vcgt_analysis.ok:
        note = "; ".join(r.message for r in vcgt_analysis.refusals)
        return replace(result, cal_path=None, vcgt_note=note)

    cgats.write_cal(cal_path, vcgt_analysis.result["curves"])
    return replace(result, cal_path=cal_path, vcgt_note=None)

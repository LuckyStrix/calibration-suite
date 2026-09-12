"""Tier B: spectral camera-color characterization from measured spectral
sensitivities (SSFs), docs/design.md sec 3.2 / the DIY spectrophotometer
plan's sec 7 ("Measuring the camera's own spectral sensitivities"). Given an
SSF CSV (``nm, r, g, b``, from that separate spectrophotometer pipeline) and
any illuminant, this computes a raw-to-XYZ matrix *without a chart shot*,
plus two "how good can this sensor ever be at colorimetry" numbers: the
Luther-Ives deviation and the ISO 17321-1 sensor metamerism index.

Pure analysis (house rule 5): arrays in (an ``SSF``, spectral reflectance
data from colour-science), ``fit.Analysis``/plain floats out. No file I/O
except ``load_ssf_csv``, which is the one sanctioned place an SSF CSV is
read from disk -- ``color_commands.py`` calls it, nothing downstream does.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calsuite.camera import color as colormod
from calsuite.camera import color_constants as cc
from calsuite.fit import Analysis

# The 18 "chromatic" (non-neutral) patches of the 24-patch ColorChecker --
# the ISO 17321-1 SMI experiment's own patch set (the 6 neutral/gray-ramp
# patches are excluded because SMI specifically measures how a sensor's
# spectral response confuses *different hues*, which a gray patch can't
# probe). Named to match colour-science's "ColorChecker24 - After November
# 2014" / "ISO 17321-1" dataset keys.
SMI_CHROMATIC_PATCH_NAMES = (
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
)


@dataclass(frozen=True)
class SSF:
    wavelengths: np.ndarray  # nm, strictly increasing
    r: np.ndarray
    g: np.ndarray
    b: np.ndarray


def load_ssf_csv(path: Path | str) -> SSF:
    """Read ``nm,r,g,b`` (header required, any column order/case) -- the
    format the DIY spectrophotometer plan's sec 7 pipeline produces."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldmap = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        required = {"nm", "r", "g", "b"}
        missing = required - set(fieldmap)
        if missing:
            raise ValueError(f"{path}: SSF CSV missing column(s) {sorted(missing)}")
        wl, r, g, b = [], [], [], []
        for row in reader:
            wl.append(float(row[fieldmap["nm"]]))
            r.append(float(row[fieldmap["r"]]))
            g.append(float(row[fieldmap["g"]]))
            b.append(float(row[fieldmap["b"]]))
    order = np.argsort(wl)
    return SSF(
        wavelengths=np.asarray(wl)[order],
        r=np.asarray(r)[order],
        g=np.asarray(g)[order],
        b=np.asarray(b)[order],
    )


def _interp_sd(ssf_wavelengths: np.ndarray, sd) -> np.ndarray:
    """A ``colour.SpectralDistribution``/``MultiSpectralDistributions``,
    resampled onto ``ssf_wavelengths`` via linear interpolation with
    zero-fill outside its native range -- the SSF's own measured grid is
    almost always narrower than a CIE illuminant/CMFS table, so this never
    extrapolates a physically meaningless value past the SSF's own support."""
    return np.interp(ssf_wavelengths, sd.wavelengths, sd.values, left=0.0, right=0.0)


def camera_response(ssf: SSF, illuminant_sd, reflectance_sd) -> tuple:
    """Raw camera (r, g, b) for one reflectance spectrum under one
    illuminant: sum over the SSF's own wavelength grid of
    ``SSF(lambda) * illuminant(lambda) * reflectance(lambda)``, trapezoidal
    integration. The overall scale is arbitrary (real sensor gain isn't
    modeled) -- ``color.fit_matrix_terms``'s linear system absorbs any
    constant scale factor, same as it would absorb a chart's unknown
    exposure level in Tier A."""
    illum = _interp_sd(ssf.wavelengths, illuminant_sd)
    refl = _interp_sd(ssf.wavelengths, reflectance_sd)
    weight = illum * refl
    # np.trapz was removed in numpy 2.0 in favor of np.trapezoid.
    r = np.trapezoid(ssf.r * weight, ssf.wavelengths)
    g = np.trapezoid(ssf.g * weight, ssf.wavelengths)
    b = np.trapezoid(ssf.b * weight, ssf.wavelengths)
    return float(r), float(g), float(b)


def _reflectance_set(name: str) -> dict:
    import colour

    return colour.SDS_COLOURCHECKERS[name]


def _xyz_for_reflectance(reflectance_sd, illuminant_sd, cmfs) -> np.ndarray:
    import colour

    xyz100 = colour.sd_to_XYZ(reflectance_sd, cmfs, illuminant_sd, method="Integration")
    return np.asarray(xyz100, dtype=np.float64) / 100.0  # -> Y=1 for a perfect diffuser (color.py's convention)


def fit_from_ssf(
    ssf: SSF,
    *,
    illuminant_name: str,
    illuminant_xy: tuple,
    model: str = "matrix",
    reflectance_set_name: str = "ISO 17321-1",
    white_preserving: bool = False,
) -> Analysis:
    """Tier B: fit raw->XYZ for ``illuminant_name`` from SSFs + a spectral
    reflectance set, with no chart photograph at all. Provenance is
    "derived" (see color_commands.py), not "measured" -- there is nothing
    captured here to refuse on the way a chart shot can be refused for
    glare/clipping/lighting, so this has no chart-quality refusal checks;
    it can still refuse for having too few reflectance samples for the
    requested model order (the same rule Tier A uses).
    """
    import colour

    illuminant_sd = colour.SDS_ILLUMINANTS[illuminant_name]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    reflectances = _reflectance_set(reflectance_set_name)

    names = list(reflectances.keys())
    analysis = Analysis()
    needed = cc.min_patches(model)
    if len(names) < needed:
        analysis.refuse(
            "too_few_reflectance_samples",
            f"{len(names)} reflectance samples in {reflectance_set_name!r} is below the minimum for "
            f"model {model!r} ({cc.MIN_PATCHES_PER_TERM}x{cc.MODEL_TERMS[model]} terms)",
            value=len(names),
            threshold=needed,
        )

    rgb = np.array([camera_response(ssf, illuminant_sd, reflectances[n]) for n in names])
    xyz = np.array([_xyz_for_reflectance(reflectances[n], illuminant_sd, cmfs) for n in names])

    # The brightest sample (highest reference Y) stands in for "the chart's
    # neutral" that Tier A uses -- Tier B has no chart, so this is the
    # closest analog: the reflectance sample closest to a perfect white
    # diffuser. Always located (not just when white_preserving) so its raw
    # RGB can be carried on the record for camera/dcp.py's export, the same
    # way color.fit does.
    white_idx = int(np.argmax(xyz[:, 1]))
    white_local = white_idx if white_preserving else None

    M = colormod.fit_matrix_terms(rgb, xyz, model=model, illuminant_xy=illuminant_xy, white_local_idx=white_local)
    matrix_3x3 = (
        M if model == "matrix" else colormod.fit_matrix_terms(rgb, xyz, model="matrix", illuminant_xy=illuminant_xy, white_local_idx=white_local)
    )

    pred = colormod.predict_xyz(rgb, model, M)
    de = colormod.delta_e00(pred, xyz, illuminant_xy)

    analysis.result = {
        "model": model,
        "illuminant": illuminant_name,
        "illuminant_xy": [float(illuminant_xy[0]), float(illuminant_xy[1])],
        "white_preserving": bool(white_preserving),
        "reflectance_set": reflectance_set_name,
        "matrix_raw_to_xyz": matrix_3x3.tolist(),
        "white_patch_raw_rgb": rgb[white_idx].tolist(),
        "n_samples": len(names),
    }
    if model != "matrix":
        analysis.result["root_polynomial"] = {"degree": {"rp2": 2, "rp3": 3}[model], "matrix": M.tolist()}
    analysis.residuals = {
        "delta_e00_mean": float(np.mean(de)),
        "delta_e00_max": float(np.max(de)),
        "delta_e00_per_sample": dict(zip(names, (float(d) for d in de), strict=True)),
    }
    return analysis


# ---------------------------------------------------------------------------
# Luther-Ives deviation
# ---------------------------------------------------------------------------


def luther_ives_deviation(ssf: SSF, cmfs=None) -> float:
    """How well *some* linear combination of the SSFs reproduces the CIE
    1931 2-degree CMFs (the Luther-Ives condition for colorimetric
    accuracy): solve the best-fit 3x3 ``A`` with ``SSF @ A ~= CMFS`` over
    wavelength (unweighted least squares across the SSF's own grid), then
    report the RMS residual normalized by the CMFs' own RMS magnitude --
    0 means the SSFs are an exact linear recombination of the CMFs (perfect
    colorimetric camera, e.g. by construction in a synthetic test);
    realistic camera SSFs (narrower, overlapping color-filter-array
    transmission curves, not the CMFs' negative lobes) always score above
    0.
    """
    import colour

    cmfs = cmfs or colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    xbar = np.interp(ssf.wavelengths, cmfs.wavelengths, cmfs.values[:, 0])
    ybar = np.interp(ssf.wavelengths, cmfs.wavelengths, cmfs.values[:, 1])
    zbar = np.interp(ssf.wavelengths, cmfs.wavelengths, cmfs.values[:, 2])

    S = np.column_stack([ssf.r, ssf.g, ssf.b])
    CMF = np.column_stack([xbar, ybar, zbar])
    A, *_ = np.linalg.lstsq(S, CMF, rcond=None)
    residual = S @ A - CMF
    denom = float(np.sqrt(np.mean(CMF**2)))
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.mean(residual**2)) / denom)


# ---------------------------------------------------------------------------
# ISO 17321-1 sensor metamerism index
# ---------------------------------------------------------------------------


def sensor_metamerism_index(ssf: SSF, *, reflectance_set_name: str = "ISO 17321-1") -> dict:
    """ISO 17321-1 sensor metamerism index: SMI = 100 - 5.5 * mean(dE*ab)
    over the 18 chromatic ColorChecker patches under D65, where dE*ab is
    the plain CIE76 Lab distance between (a) the camera's own best-fit
    (linear, unweighted least squares -- no DeltaE2000 refinement; ISO
    17321-1's definition is of the *linear* colorimetric error) XYZ
    prediction and (b) the patches' true D65 XYZ. Verified 2026-09-11 by
    web search against DxOMark's "Color depth" glossary and a DPReview
    Forums summary of the ISO 17321-1 SMI experiment (both independently
    give "SMI = 100 - 5.5 x mean dE" over the ColorChecker's 18 chromatic
    patches); ISO 17321-1:2012 itself was not directly accessible (it's a
    paywalled standard) -- see color_constants.SMI_SLOPE.
    """
    import colour

    illuminant_sd = colour.SDS_ILLUMINANTS["D65"]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    d65_xy = tuple(float(v) for v in colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    reflectances = _reflectance_set(reflectance_set_name)

    missing = [n for n in SMI_CHROMATIC_PATCH_NAMES if n not in reflectances]
    if missing:
        raise ValueError(f"{reflectance_set_name!r} is missing chromatic patch(es) {missing}")

    rgb = np.array([camera_response(ssf, illuminant_sd, reflectances[n]) for n in SMI_CHROMATIC_PATCH_NAMES])
    xyz = np.array(
        [_xyz_for_reflectance(reflectances[n], illuminant_sd, cmfs) for n in SMI_CHROMATIC_PATCH_NAMES]
    )

    # Plain (unweighted) linear least squares -- SMI measures the sensor's
    # intrinsic colorimetric error, not how well a DeltaE2000-optimized fit
    # can paper over it.
    M, *_ = np.linalg.lstsq(rgb, xyz, rcond=None)
    pred_xyz = rgb @ M

    pred_lab = colour.XYZ_to_Lab(pred_xyz, illuminant=d65_xy)
    true_lab = colour.XYZ_to_Lab(xyz, illuminant=d65_xy)
    de_ab = colour.delta_E(pred_lab, true_lab, method="CIE 1976")

    mean_de = float(np.mean(de_ab))
    smi = 100.0 - cc.SMI_SLOPE * mean_de
    return {
        "smi": smi,
        "mean_delta_e_ab": mean_de,
        "per_patch_delta_e_ab": dict(zip(SMI_CHROMATIC_PATCH_NAMES, (float(d) for d in de_ab), strict=True)),
    }

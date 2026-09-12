"""Tier A (chart matrix) and Tier C (validation) camera color
characterization (docs/design.md sec 3.2). Pure analysis: ``chart.ChartSample``
+ ``chart.ReferenceChart`` + metadata in, ``fit.Analysis`` out (house rule
5) -- ``color_commands.py`` is the only place that loads a frame, runs
``chart.sample_chart`` and saves a ``Record``.

**XYZ convention used throughout this module**: Y = 1.0 for a perfect
reflecting diffuser under the shooting illuminant (the same convention
``colour.xyY_to_XYZ``/``colour.CCS_COLOURCHECKERS`` already use -- a
ColorChecker's white patch has reference Y around 0.88, not 1.0, because
real chart material isn't a perfect reflector). The fitted
``matrix_raw_to_xyz`` is therefore *not* renormalized so that camera white
balance maps to Y=1 -- it reproduces the chart's own measured/published
colorimetry as directly as a linear model can. ``camera/dcp.py`` does its
own, separate renormalization into the DNG spec's white-balanced
ForwardMatrix convention when exporting.
"""

from __future__ import annotations

import numpy as np

from calsuite.camera import chart
from calsuite.camera import color_constants as cc
from calsuite.fit import Analysis

MODELS = ("matrix", "rp2", "rp3")


def _design_matrix(rgb: np.ndarray, model: str) -> np.ndarray:
    """``rgb``: (n, 3) linear camera RGB -> (n, terms) expansion. "matrix" is
    the identity expansion (3 terms = R, G, B); "rp2"/"rp3" are Finlayson
    root-polynomial expansions, degree 2 (6 terms) / degree 3 (13 terms),
    reusing colour-science's own implementation
    (``colour.characterisation.polynomial_expansion_Finlayson2015``) rather
    than re-deriving the term list, since its term count and ordering are
    exactly what ``color_constants.MODEL_TERMS`` documents and is verified
    against by this module's own tests."""
    if model == "matrix":
        return rgb
    import colour

    degree = {"rp2": 2, "rp3": 3}[model]
    return colour.characterisation.polynomial_expansion_Finlayson2015(rgb, degree=degree, root_polynomial_expansion=True)


def black_subtracted_rgb(chart_sample: chart.ChartSample, patch_indices=None) -> np.ndarray:
    """Per-patch camera-native RGB, black level subtracted per channel, G =
    mean(G1, G2) (the display-agent contract's stated convention). Each of
    R/G1/G2/B is subtracted with its *own* black level
    (``chart_sample.black_level_by_channel``, from
    ``raw.black_level_by_channel`` -- see that function's docstring for why
    a single mean(black_level) scalar is wrong on a sensor with a
    genuinely different black level per channel), not a scalar mean across
    all 4. ``patch_indices`` restricts to a subset (used for held-out/
    leave-one-out splits); default is every patch in ``chart_sample``.
    """
    idx = range(len(chart_sample.patches)) if patch_indices is None else patch_indices
    black = chart_sample.black_level_by_channel or {ch: float(np.mean(chart_sample.black_level)) for ch in chart.CHANNELS}
    rows = []
    for i in idx:
        p = chart_sample.patches[i]
        r = p.mean["R"] - black["R"]
        g = 0.5 * ((p.mean["G1"] - black["G1"]) + (p.mean["G2"] - black["G2"]))
        b = p.mean["B"] - black["B"]
        rows.append((r, g, b))
    return np.array(rows, dtype=np.float64)


def _valid_indices(chart_sample: chart.ChartSample) -> list:
    """Patches where every channel actually landed at least one pixel
    (``sample_chart`` returns 0 px for a box that fell off the frame edge,
    e.g. a corner typo) -- excluded from the fit rather than propagating a
    literal 0.0 mean as if it were a real (black) measurement."""
    return [
        i
        for i, p in enumerate(chart_sample.patches)
        if all(p.n_px[ch] > 0 for ch in chart.CHANNELS)
    ]


def _target_xyz(reference: chart.ReferenceChart, indices) -> np.ndarray:
    return np.array([reference.patches[i].XYZ for i in indices], dtype=np.float64)


def _white_index_within(indices: list, white_idx_global: int) -> int | None:
    return indices.index(white_idx_global) if white_idx_global in indices else None


def _weights(n: int, white_local_idx: int | None) -> np.ndarray:
    w = np.ones(n)
    if white_local_idx is not None:
        w[white_local_idx] = cc.WHITE_PRESERVING_WEIGHT
    return w


def _linear_init(terms: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(weights)[:, None]
    solution, *_ = np.linalg.lstsq(terms * sw, target * sw, rcond=None)
    return solution.T  # (3, n_terms): M @ terms_row(column) = xyz_row(column)


def _refine_delta_e00(
    terms: np.ndarray, target_xyz: np.ndarray, weights: np.ndarray, illuminant_xy: tuple, M0: np.ndarray
) -> np.ndarray:
    import colour
    from scipy.optimize import least_squares

    target_lab = colour.XYZ_to_Lab(target_xyz, illuminant=illuminant_xy)
    sw = np.sqrt(weights)

    def residuals(x):
        M = x.reshape(M0.shape)
        pred_xyz = terms @ M.T
        pred_lab = colour.XYZ_to_Lab(pred_xyz, illuminant=illuminant_xy)
        de = colour.delta_E(pred_lab, target_lab, method="CIE 2000")
        return de * sw

    result = least_squares(residuals, M0.ravel(), method="trf")
    return result.x.reshape(M0.shape)


def fit_matrix_terms(
    rgb: np.ndarray,
    target_xyz: np.ndarray,
    *,
    model: str,
    illuminant_xy: tuple,
    white_local_idx: int | None = None,
) -> np.ndarray:
    """Fit an (3, terms) matrix minimizing DeltaE2000 between ``rgb``
    (already black-subtracted, G-averaged) run through the model's term
    expansion and ``target_xyz``. Initializes with linear least squares (raw
    XYZ error, cheap and always well-posed), then refines against DeltaE2000
    with ``scipy.optimize.least_squares`` -- the two-stage approach the
    instructions call for ("initialize by linear least squares, refine with
    scipy least_squares")."""
    terms = _design_matrix(rgb, model)
    weights = _weights(len(rgb), white_local_idx)
    M0 = _linear_init(terms, target_xyz, weights)
    return _refine_delta_e00(terms, target_xyz, weights, illuminant_xy, M0)


def predict_xyz(rgb: np.ndarray, model: str, M: np.ndarray) -> np.ndarray:
    """Apply a fitted (3, terms) matrix to (black-subtracted) camera RGB,
    expanding through the model's term set first. Public because
    ``camera/ssf.py`` (Tier B) reuses it -- fit/predict are the same
    operation whether the target XYZ came from a chart shot or a spectral
    reflectance set."""
    terms = _design_matrix(rgb, model)
    return terms @ M.T


def delta_e00(pred_xyz: np.ndarray, target_xyz: np.ndarray, illuminant_xy: tuple) -> np.ndarray:
    """CIE DeltaE2000 between two (n, 3) XYZ arrays, converted to Lab
    against the same ``illuminant_xy`` white. Public for the same reason as
    ``predict_xyz``."""
    import colour

    pred_lab = colour.XYZ_to_Lab(pred_xyz, illuminant=illuminant_xy)
    target_lab = colour.XYZ_to_Lab(target_xyz, illuminant=illuminant_xy)
    return colour.delta_E(pred_lab, target_lab, method="CIE 2000")


def _validate_leave_one_out(rgb, target_xyz, model, illuminant_xy, white_pos) -> np.ndarray:
    """Refit ``len(rgb)`` times, each time holding out one patch, and
    return that patch's DeltaE2000 -- "leave-one-out ... on patches the fit
    didn't see" when no separate held-out set was given. ``white_pos`` is
    the *local* position of the white-preserving reference patch within
    ``rgb``/``target_xyz`` (or -1 to disable the constraint), not a global
    reference-chart index -- every fold's training array is a different
    subset with its own local positions, recomputed per fold below.

    Each fold uses the cheap **linear** (raw-XYZ-error) fit only, not the
    full DeltaE2000 nonlinear refinement ``fit_matrix_terms`` otherwise
    does -- with a chart-sized patch count this means dozens of folds, and
    scipy's iterative refinement (each iteration re-running colour-science's
    Lab/DeltaE2000 conversions) made a full N-fold refit-per-patch far too
    slow for a test suite (multi-minute for one 24-patch chart). The linear
    fit is still a faithful "how well would an unseen patch be predicted"
    estimate -- it just uses the same initialization step the final,
    reported matrix's own nonlinear refinement starts from -- so leave-one-
    out validation stays a meaningful generalization check, just against a
    slightly more conservative (linear-error-optimal rather than DeltaE00-
    optimal) per-fold model.
    """
    n = len(rgb)
    de = np.empty(n)
    all_idx = np.arange(n)
    for i in range(n):
        train = np.delete(all_idx, i)
        white_local = _white_index_within(list(train), white_pos)
        terms = _design_matrix(rgb[train], model)
        weights = _weights(len(train), white_local)
        M = _linear_init(terms, target_xyz[train], weights)
        pred = predict_xyz(rgb[i : i + 1], model, M)
        de[i] = delta_e00(pred, target_xyz[i : i + 1], illuminant_xy)[0]
    return de


def fit(
    chart_sample: chart.ChartSample,
    reference: chart.ReferenceChart,
    *,
    illuminant_xy: tuple,
    illuminant_name: str = "",
    model: str = "matrix",
    white_preserving: bool = False,
    held_out_names: list | None = None,
) -> Analysis:
    """Tier A fit + Tier C validation in one pass.

    ``chart_sample`` must already be patch-named (``chart.label_patches``)
    so held-out selection by name and neutral/white lookup work.
    ``held_out_names``, when given, names patches reserved for validation
    only (never used to fit); otherwise validation is leave-one-out over
    every valid patch.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    analysis = Analysis()

    quality_refusals = chart.refusals_for_fit(chart_sample, reference, model)
    for r in quality_refusals:
        analysis.refusals.append(r)

    valid = _valid_indices(chart_sample)
    if len(valid) < 4:  # not even enough to attempt a 3x3 -- nothing more to compute
        analysis.refuse(
            "insufficient_valid_patches",
            f"only {len(valid)} patches sampled any pixels at all",
            value=len(valid),
            threshold=4,
        )
        return analysis

    white_global = reference.brightest_neutral_index()
    held_out_global = set()
    if held_out_names:
        name_to_idx = {reference.patches[i].name: i for i in valid}
        held_out_global = {name_to_idx[n] for n in held_out_names if n in name_to_idx}
    fit_idx = [i for i in valid if i not in held_out_global]

    rgb_all = black_subtracted_rgb(chart_sample, valid)
    target_all = _target_xyz(reference, valid)
    pos_in_valid = {g: k for k, g in enumerate(valid)}

    fit_pos = [pos_in_valid[i] for i in fit_idx]
    rgb_fit, target_fit = rgb_all[fit_pos], target_all[fit_pos]
    white_local = _white_index_within(fit_idx, white_global) if white_preserving else None

    M = fit_matrix_terms(rgb_fit, target_fit, model=model, illuminant_xy=illuminant_xy, white_local_idx=white_local)

    # Contract with the display agent (implementation-plan.md): a
    # camera.color record MUST carry a plain 3x3 raw->XYZ matrix at
    # result["matrix_raw_to_xyz"], regardless of which model was actually
    # requested -- for "matrix" that IS the fitted model; for the root-
    # polynomial models, a companion plain-matrix fit is run on the same
    # patches purely to populate this field, since a 3xN root-polynomial
    # matrix isn't a drop-in "3x3 raw->XYZ" the display agent could apply.
    if model == "matrix":
        matrix_3x3 = M
    else:
        matrix_3x3 = fit_matrix_terms(
            rgb_fit, target_fit, model="matrix", illuminant_xy=illuminant_xy, white_local_idx=white_local
        )

    if held_out_global:
        held_idx = [i for i in valid if i in held_out_global]
        held_pos = [pos_in_valid[i] for i in held_idx]
        pred = predict_xyz(rgb_all[held_pos], model, M)
        de = delta_e00(pred, target_all[held_pos], illuminant_xy)
        validation_method = "held_out"
    else:
        white_pos_in_fit = fit_idx.index(white_global) if (white_preserving and white_global in fit_idx) else -1
        de = _validate_leave_one_out(rgb_fit, target_fit, model, illuminant_xy, white_pos_in_fit)
        # Named for what it actually validates, not just "leave one out":
        # each fold's own matrix is the cheap linear (raw-XYZ-error) fit,
        # not the full DeltaE00 nonlinear refinement the *reported* matrix
        # gets (see _validate_leave_one_out's docstring for why) -- a
        # slightly more conservative generalization estimate than a true
        # "refit exactly like the real thing N times" would give, and the
        # record should say so rather than imply the two used the same fit.
        validation_method = "leave_one_out_linear_folds"

    mean_de, p95_de, max_de = float(np.mean(de)), float(np.percentile(de, 95)), float(np.max(de))
    validation_passed = (
        mean_de <= cc.VALIDATION_MEAN_DE00_MAX
        and p95_de <= cc.VALIDATION_P95_DE00_MAX
        and max_de <= cc.VALIDATION_MAX_DE00_MAX
    )
    if not validation_passed:
        # House rule 2 / store.require_exportable: "measured" (the method
        # -- a chart WAS photographed) is a different question from
        # "trustworthy" (status). A fit whose own validation fails is not
        # trustworthy regardless of which tier's checks it happened to
        # clear, so it must be refused (status="refused"), not merely
        # carry a quiet validation_passed=False buried in result -- the
        # earlier version of this function left it exactly that quiet,
        # which meant a chart that failed validation but triggered no
        # *other* refusal still had status="ok" and would pass
        # store.require_exportable().
        analysis.refuse(
            "validation_failed",
            f"{validation_method} validation exceeds its DeltaE00 threshold: "
            f"mean={mean_de:.3f} (max {cc.VALIDATION_MEAN_DE00_MAX}), "
            f"p95={p95_de:.3f} (max {cc.VALIDATION_P95_DE00_MAX}), "
            f"max={max_de:.3f} (max {cc.VALIDATION_MAX_DE00_MAX})",
            value=round(mean_de, 4),
            threshold=cc.VALIDATION_MEAN_DE00_MAX,
        )

    fit_pred = predict_xyz(rgb_fit, model, M)
    fit_de = delta_e00(fit_pred, target_fit, illuminant_xy)

    # Carried on the record so camera/dcp.py's export can build a DNG
    # ForwardMatrix/ColorMatrix later without needing the original
    # ChartSample again -- the white/reference patch's own black-subtracted
    # raw RGB is exactly what ``dcp.build_forward_and_color_matrices`` needs
    # as its ``raw_white`` argument.
    white_patch_raw_rgb = rgb_all[pos_in_valid[white_global]].tolist() if white_global in valid else None

    analysis.result = {
        "model": model,
        "illuminant": illuminant_name or f"xy={illuminant_xy[0]:.4f},{illuminant_xy[1]:.4f}",
        "illuminant_xy": [float(illuminant_xy[0]), float(illuminant_xy[1])],
        "white_preserving": bool(white_preserving),
        "reference_chart": reference.name,
        "reference_patch_names": [reference.patches[i].name for i in valid],
        "matrix_raw_to_xyz": matrix_3x3.tolist(),
        "white_patch_raw_rgb": white_patch_raw_rgb,
        "n_patches_fit": len(fit_idx),
        "n_patches_held_out": len(held_out_global),
        "validation_method": validation_method,
        "validation_passed": validation_passed,
    }
    if model != "matrix":
        analysis.result["root_polynomial"] = {"degree": {"rp2": 2, "rp3": 3}[model], "matrix": M.tolist()}

    analysis.residuals = {
        "delta_e00_fit_mean": float(np.mean(fit_de)),
        "delta_e00_fit_max": float(np.max(fit_de)),
        "delta_e00_per_patch": {reference.patches[i].name: float(d) for i, d in zip(fit_idx, fit_de, strict=True)},
    }
    analysis.uncertainty = {
        "delta_e00_validation_mean": mean_de,
        "delta_e00_validation_p95": p95_de,
        "delta_e00_validation_max": max_de,
        "delta_e00_validation_per_patch": {
            reference.patches[i].name: float(d)
            for i, d in zip(held_idx if held_out_global else fit_idx, de, strict=True)
        },
    }
    return analysis

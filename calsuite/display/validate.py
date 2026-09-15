"""Validate an installed profile against the real display (docs/design.md
§5.6): map target Lab values through the profile to device RGB (Pillow
``ImageCms``, perceptual/relative colorimetric), display and measure each
one, and report ΔE00 mean/p95/max. A profile record is `measured` only if
this passes (design house rule 2 / ``store.require_exportable``'s
provenance rule) -- ``display/commands.py`` is what actually sets that
provenance bit; this module just computes the numbers and refusals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from calsuite.display import analysis as analysismod
from calsuite.display import constants as dc
from calsuite.fit import Analysis


def lab_to_rgb_via_profile(profile_path: Path, lab_targets: list, *, intent=None) -> list:
    """Map each (L*, a*, b*) in `lab_targets` to device RGB in [0, 1]^3
    through the ICC profile at `profile_path`, via Pillow/littleCMS
    (design §5.6). Pillow has no single-call Lab->RGB convenience, so this
    builds LittleCMS's own Lab profile (D50-referenced, matching the ICC
    PCS -- ``formats/icc.py``'s ``ICC_PCS_ILLUMINANT_D50``) and an
    explicit transform, the same approach ``formats/icc.py``'s own test
    suite uses to open a profile with Pillow as an independent check.

    Pillow's "LAB" image mode packs L* as a byte 0-255 (not 0-100) and
    a*/b* as bytes offset by +128 (documented Pillow/LittleCMS
    convention) -- confirmed by round-tripping a known (L*, a*, b*)
    through ``PIL.Image.new("LAB", ...)`` + ``ImageCms.applyTransform``
    against Pillow's own built-in sRGB profile while building this module.
    """
    intent = intent if intent is not None else ImageCms.Intent.RELATIVE_COLORIMETRIC
    lab_profile = ImageCms.createProfile("LAB", colorTemp=5000)
    dst_profile = ImageCms.ImageCmsProfile(str(profile_path))
    transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, dst_profile, "LAB", "RGB", renderingIntent=intent)

    out_rgb = []
    for l_star, a_star, b_star in lab_targets:
        px = (
            max(0, min(255, round(l_star / 100.0 * 255))),
            max(0, min(255, round(a_star + 128))),
            max(0, min(255, round(b_star + 128))),
        )
        swatch = Image.new("LAB", (1, 1), px)
        transformed = ImageCms.applyTransform(swatch, transform)
        r, g, b = transformed.getpixel((0, 0))
        out_rgb.append((r / 255.0, g / 255.0, b / 255.0))
    return out_rgb


def validate(measured_xyz: list, lab_targets: list, white_xyz, accuracy_de00_estimate: float) -> Analysis:
    """`measured_xyz`: XYZ measurements of the validation patches as
    actually displayed through the profile, in the same order as
    `lab_targets` (the target each patch was supposed to reproduce).
    `white_xyz`: the display's own measured white -- the media-relative
    reference each measurement is normalized to before being Bradford-
    adapted to the ICC PCS's D50 Lab (``display.analysis.xyz_to_lab_pcs``),
    which is the space `lab_targets` (D50-referenced CC24 Lab) live in and
    the one the profile itself was built in. `accuracy_de00_estimate`:
    the measuring backend's own ``accuracy().de00_estimate`` -- pass
    thresholds scale from this (house rule 7; the multipliers themselves,
    with their reasons, are ``constants.VALIDATION_*_DE00_MULTIPLIER``).
    """
    analysis = Analysis()
    if len(measured_xyz) != len(lab_targets):
        analysis.refuse(
            "validation_length_mismatch",
            "measured_xyz and lab_targets must be the same length",
            len(measured_xyz),
            len(lab_targets),
        )
        return analysis

    de00s = [
        analysismod.delta_e00(lab_target, analysismod.xyz_to_lab_pcs(xyz, white_xyz))
        for xyz, lab_target in zip(measured_xyz, lab_targets, strict=True)
    ]
    de00s = np.asarray(de00s, dtype=np.float64)
    mean_de00, p95_de00, max_de00 = float(de00s.mean()), float(np.percentile(de00s, 95)), float(de00s.max())
    mean_thr = accuracy_de00_estimate * dc.VALIDATION_MEAN_DE00_MULTIPLIER
    p95_thr = accuracy_de00_estimate * dc.VALIDATION_P95_DE00_MULTIPLIER
    max_thr = accuracy_de00_estimate * dc.VALIDATION_MAX_DE00_MULTIPLIER

    analysis.result.update(
        {
            "de00_mean": mean_de00,
            "de00_p95": p95_de00,
            "de00_max": max_de00,
            "de00_per_patch": [float(v) for v in de00s],
            "backend_de00_estimate": accuracy_de00_estimate,
            "thresholds": {"mean": mean_thr, "p95": p95_thr, "max": max_thr},
        }
    )
    if mean_de00 > mean_thr:
        analysis.refuse("validation_mean_de00", "mean ΔE00 exceeds the backend-accuracy-scaled threshold", mean_de00, mean_thr)
    if p95_de00 > p95_thr:
        analysis.refuse("validation_p95_de00", "p95 ΔE00 exceeds the backend-accuracy-scaled threshold", p95_de00, p95_thr)
    if max_de00 > max_thr:
        analysis.refuse("validation_max_de00", "max ΔE00 exceeds the backend-accuracy-scaled threshold", max_de00, max_thr)
    return analysis

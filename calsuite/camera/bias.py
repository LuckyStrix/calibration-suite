"""Black level and read noise, per CFA channel, from bias frames (docs/
design.md §3.1's first two rows). Pure analysis (house rule 5): everything
here takes already-loaded ``raw.RawFrame`` objects and returns a
``fit.Analysis``; ``commands.py`` is the only thing that reads files.

Black level is estimated two ways from the same bias frames -- the masked
optical-black margin (no light, no lens, via ``raw.optical_black``) and the
visible area itself (also dark, because a bias frame is shot with the lens
capped) -- and both are compared against the black level the camera itself
reported in metadata (``RawFrame.black_level``).

Read noise is the classic sigma = std(A-B)/sqrt(2) pair difference, which
cancels each channel's fixed pattern (identical in both frames of a pair)
and leaves only the frame-to-frame (temporal) read noise.
"""

from __future__ import annotations

import numpy as np

from calsuite import raw as rawmod
from calsuite.camera.constants import (
    BIAS_MAX_EXPOSURE_S,
    BIAS_MIN_FRAMES,
    BLACK_LEVEL_METADATA_DISCREPANCY_DN,
)
from calsuite.fit import Analysis

CHANNELS = ("R", "G1", "G2", "B")


def _channel_means(frames, area: str) -> dict:
    """``{channel: [per-frame mean, ...]}`` over ``area`` ("visible" or, via
    ``raw.optical_black``, the masked margin)."""
    out = {ch: [] for ch in CHANNELS}
    for f in frames:
        planes = rawmod.optical_black(f) if area == "optical_black" else rawmod.planes(f, area="visible")
        for ch, arr in planes.items():
            out[ch].append(float(arr.mean()))
    return out


def analyze_bias(frames: list) -> Analysis:
    """One ISO's worth of bias frames in, an ``Analysis`` out.

    ``result`` carries, per channel: the optical-black-margin estimate, the
    visible-area estimate, their combined mean (``black_level_dn``), the
    metadata's reported value, and the discrepancy between them; plus
    read noise in DN (and in electrons, if the caller already knows this
    ISO's gain -- ``camera.ptc`` measures that, so it isn't available yet
    the first time a bias series is analyzed, hence the optional arg).
    """
    a = Analysis()
    if len(frames) < BIAS_MIN_FRAMES:
        a.refuse(
            "insufficient_frames",
            f"need at least {BIAS_MIN_FRAMES} bias frames to form a difference pair, got {len(frames)}",
            value=len(frames),
            threshold=BIAS_MIN_FRAMES,
        )
        return a

    long_frames = [f for f in frames if (f.meta.exposure_s or 0.0) > BIAS_MAX_EXPOSURE_S]
    if long_frames:
        a.refuse(
            "not_bias_exposure",
            f"{len(long_frames)} frame(s) have exposure_s above the bias threshold "
            f"({BIAS_MAX_EXPOSURE_S}s) -- dark current would leak into the read-noise estimate",
            value=max(f.meta.exposure_s for f in long_frames),
            threshold=BIAS_MAX_EXPOSURE_S,
        )
        return a

    ob_means = _channel_means(frames, "optical_black")
    vis_means = _channel_means(frames, "visible")

    metadata_black = float(np.mean(frames[0].black_level)) if frames[0].black_level else None

    black_level_dn = {}
    black_level_optical_black_dn = {}
    black_level_visible_dn = {}
    black_level_discrepancy_dn = {}
    for ch in CHANNELS:
        ob = float(np.mean(ob_means[ch]))
        vis = float(np.mean(vis_means[ch]))
        black_level_optical_black_dn[ch] = ob
        black_level_visible_dn[ch] = vis
        black_level_dn[ch] = 0.5 * (ob + vis)
        if metadata_black is not None:
            black_level_discrepancy_dn[ch] = black_level_dn[ch] - metadata_black

    # Read noise: pair up frames (0,1), (2,3), ... -- non-overlapping pairs
    # so no frame contributes to more than one difference, keeping the
    # pairs statistically independent of each other. An odd frame out is
    # dropped (documented, not silently mis-paired).
    n_pairs = len(frames) // 2
    planes_list = [rawmod.planes(f, area="visible") for f in frames[: 2 * n_pairs]]
    read_noise_dn = {}
    for ch in CHANNELS:
        sigmas = []
        for i in range(n_pairs):
            diff = planes_list[2 * i][ch] - planes_list[2 * i + 1][ch]
            sigmas.append(float(np.std(diff)) / np.sqrt(2.0))
        read_noise_dn[ch] = float(np.mean(sigmas))

    a.result = {
        "n_frames": len(frames),
        "n_pairs": n_pairs,
        "black_level_dn": black_level_dn,
        "black_level_optical_black_dn": black_level_optical_black_dn,
        "black_level_visible_dn": black_level_visible_dn,
        "black_level_metadata_dn": metadata_black,
        "black_level_discrepancy_dn": black_level_discrepancy_dn,
        "black_level_metadata_discrepancy_threshold_dn": BLACK_LEVEL_METADATA_DISCREPANCY_DN,
        "read_noise_dn": read_noise_dn,
    }
    a.residuals = {
        "optical_black_vs_visible_dn": {
            ch: black_level_optical_black_dn[ch] - black_level_visible_dn[ch] for ch in CHANNELS
        }
    }
    return a


def add_read_noise_electrons(analysis: Analysis, gain_e_per_dn) -> None:
    """Extend an already-run bias ``Analysis`` in place with read noise in
    electrons, once a gain (from ``camera.ptc``) is known. ``gain_e_per_dn``
    is either one float (applied to every channel) or a ``{channel: gain}``
    dict, matching how ``camera.ptc`` reports gain per channel."""
    read_noise_dn = analysis.result.get("read_noise_dn", {})
    gains = gain_e_per_dn if isinstance(gain_e_per_dn, dict) else {ch: gain_e_per_dn for ch in read_noise_dn}
    analysis.result["read_noise_e"] = {ch: read_noise_dn[ch] * gains[ch] for ch in read_noise_dn if ch in gains}

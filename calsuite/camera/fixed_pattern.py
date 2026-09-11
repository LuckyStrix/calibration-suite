"""Fixed pattern noise, per CFA channel (docs/design.md §3.1): DSNU from
stacked darks, PRNU from stacked flats, row/column banding from the FFT of
stacked bias frames. Pure analysis (house rule 5) -- each ``*_map`` function
returns plain ``numpy`` arrays (for a caller to route to a ``.npz`` sidecar,
per ``fit.py``'s convention), while ``analyze_fixed_pattern`` reports only
JSON-safe summary numbers.
"""

from __future__ import annotations

import numpy as np

from calsuite import raw as rawmod
from calsuite.fit import Analysis

CHANNELS = ("R", "G1", "G2", "B")

FIXED_PATTERN_MIN_FRAMES = 3
# A single stacked frame can't distinguish "this pixel's fixed pattern" from
# "this pixel's noise that frame" -- a handful of frames averaged brings the
# per-pixel noise down enough that the map that's left over is dominated by
# the pattern, not by residual shot/read noise. Matches the order of
# magnitude used elsewhere in this area (BIAS_MIN_FRAMES, DARK_MIN_EXPOSURES_PER_BIN).


def _black_for(black_dn, channel: str) -> float:
    return black_dn[channel] if isinstance(black_dn, dict) else black_dn


def _stack(frames: list) -> dict:
    """Per-channel mean across ``frames``, each already read via
    ``raw.planes(area="visible")``."""
    out = {ch: None for ch in CHANNELS}
    for ch in CHANNELS:
        stacked = np.mean([rawmod.planes(f, area="visible")[ch] for f in frames], axis=0)
        out[ch] = stacked
    return out


def dsnu_map(darks: list, black_dn) -> dict:
    """``{channel: 2D array}`` -- stacked dark signal (DN, black-subtracted)
    per pixel. Its per-channel std is the DSNU figure."""
    stacked = _stack(darks)
    return {ch: stacked[ch] - _black_for(black_dn, ch) for ch in CHANNELS}


def prnu_map(flats: list, black_dn) -> dict:
    """``{channel: 2D array}`` -- fractional deviation from each channel's
    own mean signal, i.e. photoresponse non-uniformity with the mean flux
    level normalized out so different flats/exposures are comparable."""
    stacked = _stack(flats)
    out = {}
    for ch in CHANNELS:
        signal = stacked[ch] - _black_for(black_dn, ch)
        mean_signal = float(signal.mean())
        out[ch] = (signal / mean_signal - 1.0) if mean_signal != 0 else np.zeros_like(signal)
    return out


def banding_spectrum(biases: list) -> dict:
    """``{channel: {"row": 1D magnitude array, "col": 1D magnitude array}}``
    -- FFT magnitude of the stacked bias frame's row-means and column-means.
    A real per-row/per-column readout artifact shows up as a peak (above the
    noise floor) somewhere in these spectra; a clean sensor's spectra are
    flat/noisy with no dominant peak.
    """
    stacked = _stack(biases)
    out = {}
    for ch in CHANNELS:
        arr = stacked[ch]
        row_means = arr.mean(axis=1)
        col_means = arr.mean(axis=0)
        out[ch] = {
            "row": np.abs(np.fft.rfft(row_means - row_means.mean())),
            "col": np.abs(np.fft.rfft(col_means - col_means.mean())),
        }
    return out


def _peak_excluding_dc(spectrum: np.ndarray) -> tuple:
    """(peak magnitude, its index, median magnitude) over bins 1.. (bin 0 is
    the DC/mean term, already removed before the FFT, but rfft's bin 0 can
    still carry residual rounding; excluded here for clarity too)."""
    if len(spectrum) <= 1:
        return 0.0, 0, 0.0
    ac = spectrum[1:]
    idx = int(np.argmax(ac)) + 1
    return float(spectrum[idx]), idx, float(np.median(ac))


def analyze_fixed_pattern(darks: list, flats: list, biases: list, black_dn) -> Analysis:
    a = Analysis()
    missing = [
        name
        for name, frames in (("darks", darks), ("flats", flats), ("biases", biases))
        if len(frames) < FIXED_PATTERN_MIN_FRAMES
    ]
    if missing:
        a.refuse(
            "insufficient_frames",
            f"need at least {FIXED_PATTERN_MIN_FRAMES} frames each of darks/flats/biases; "
            f"short on: {', '.join(missing)}",
            value={name: len(frames) for name, frames in (("darks", darks), ("flats", flats), ("biases", biases))},
            threshold=FIXED_PATTERN_MIN_FRAMES,
        )
        return a

    dsnu = dsnu_map(darks, black_dn)
    prnu = prnu_map(flats, black_dn)
    banding = banding_spectrum(biases)

    channels_result = {}
    for ch in CHANNELS:
        row_peak, row_idx, row_median = _peak_excluding_dc(banding[ch]["row"])
        col_peak, col_idx, col_median = _peak_excluding_dc(banding[ch]["col"])
        channels_result[ch] = {
            "dsnu_std_dn": float(dsnu[ch].std()),
            "prnu_std_pct": float(prnu[ch].std()) * 100.0,
            "row_banding_peak_ratio": row_peak / row_median if row_median else float("inf"),
            "row_banding_peak_cycles_per_frame": row_idx,
            "col_banding_peak_ratio": col_peak / col_median if col_median else float("inf"),
            "col_banding_peak_cycles_per_frame": col_idx,
        }

    a.result = {"channels": channels_result}
    return a

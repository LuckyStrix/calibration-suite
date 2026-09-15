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
from calsuite.camera.constants import FIXED_PATTERN_DSNU_MIN_EXCESS, FIXED_PATTERN_MIN_FRAMES
from calsuite.fit import Analysis

CHANNELS = ("R", "G1", "G2", "B")



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
    per pixel. Note that its raw per-channel std is *not* the DSNU figure:
    it still contains the temporal-noise floor stacking leaves behind. Use
    ``dsnu_stats`` for the number."""
    stacked = _stack(darks)
    return {ch: stacked[ch] - _black_for(black_dn, ch) for ch in CHANNELS}


def dsnu_stats(darks: list, black_dn) -> dict:
    """``{channel: {"dsnu_std_dn", "temporal_floor_dn", "observed_std_dn",
    "resolved"}}`` -- DSNU with the residual temporal noise removed.

    Averaging N darks does not remove per-pixel temporal noise, it divides
    its variance by N; the std of the stacked image is therefore
    ``sqrt(dsnu^2 + var_temporal / N)``, not the DSNU. Reporting that std
    directly (as this module did) reports the floor: at N = 3 with a
    realistic 3 e- read noise and gain 2, "dsnu_std_dn" came back as
    0.89 DN whether the injected DSNU was 0.125 DN or exactly zero -- it
    was measuring 1.5/sqrt(3) = 0.87 DN of read noise both times.

    The floor is estimated from the frames themselves (the mean over pixels
    of each pixel's variance across the stack, divided by N) and subtracted
    in quadrature. When what's left isn't at least
    ``FIXED_PATTERN_DSNU_MIN_EXCESS`` of the floor, `dsnu_std_dn` is None
    and `resolved` is False: this many frames cannot see a pattern that
    small.
    """
    n = len(darks)
    out = {}
    for ch in CHANNELS:
        stack = np.stack([rawmod.planes(f, area="visible")[ch] for f in darks])
        mean_image = stack.mean(axis=0) - _black_for(black_dn, ch)
        observed_var = float(mean_image.var())
        temporal_var = float(stack.var(axis=0, ddof=1).mean()) if n > 1 else 0.0
        floor_var = temporal_var / n if n else 0.0
        pattern_var = observed_var - floor_var
        resolved = pattern_var > FIXED_PATTERN_DSNU_MIN_EXCESS * floor_var and pattern_var > 0
        out[ch] = {
            "dsnu_std_dn": float(np.sqrt(pattern_var)) if resolved else None,
            "temporal_floor_dn": float(np.sqrt(floor_var)),
            "observed_std_dn": float(np.sqrt(observed_var)),
            "resolved": bool(resolved),
        }
    return out


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

    dsnu = dsnu_stats(darks, black_dn)
    prnu = prnu_map(flats, black_dn)
    banding = banding_spectrum(biases)

    channels_result = {}
    for ch in CHANNELS:
        row_peak, row_idx, row_median = _peak_excluding_dc(banding[ch]["row"])
        col_peak, col_idx, col_median = _peak_excluding_dc(banding[ch]["col"])

        channels_result[ch] = {
            "dsnu_std_dn": dsnu[ch]["dsnu_std_dn"],
            "dsnu_resolved": dsnu[ch]["resolved"],
            "dsnu_temporal_floor_dn": dsnu[ch]["temporal_floor_dn"],
            "dsnu_observed_std_dn": dsnu[ch]["observed_std_dn"],
            "prnu_std_pct": float(prnu[ch].std()) * 100.0,
            # None, not inf: a zero median means the ratio is unbounded, and
            # `Store.save` (rightly) refuses to write a non-finite number
            # into a record -- `Infinity` is not valid JSON. The peak
            # magnitude itself is carried below either way.
            "row_banding_peak_ratio": (row_peak / row_median) if row_median else None,
            "row_banding_peak_cycles_per_frame": row_idx,
            "col_banding_peak_ratio": (col_peak / col_median) if col_median else None,
            "col_banding_peak_cycles_per_frame": col_idx,
        }

    a.result = {"channels": channels_result}
    return a

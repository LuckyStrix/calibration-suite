"""Synthetic backend: wraps ``calsuite.synth.display.DisplayModel`` behind
the ``base.Backend`` protocol, for every round-trip test in this wave
(docs/design.md house rule 4) and for ``calsuite demo``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calsuite.display.backends.base import Accuracy, Measurement
from calsuite.synth.display import DisplayModel


@dataclass
class SyntheticBackend:
    name: str = "synthetic"
    model: DisplayModel = None  # type: ignore[assignment]
    t_s: float | None = None  # time since power-on, for warm-up series measurements
    seed: int = 0

    def __post_init__(self):
        if self.model is None:
            self.model = DisplayModel()
        self._rng = np.random.default_rng(self.seed)

    def accuracy(self) -> Accuracy:
        # A synthetic display has no instrument to be wrong about -- its
        # own measurement noise (`noise_std_frac`, often 0) would put this
        # near-zero. But `display.validate`'s pass thresholds scale off
        # this number to cover *every* error source between "true color"
        # and "measured color", and for this backend that's dominated not
        # by measurement noise but by the matrix/TRC ICC profile's own
        # structural fidelity limit (e.g. it has no black-offset term at
        # all -- see profile.build_fallback_matrix_trc's docstring): a
        # perfectly-measured, noiseless round trip through a matrix/TRC
        # profile still shows a few ΔE00 of residual error on a
        # non-trivial validation set. 2.0 is an empirical figure (with
        # margin) from that round trip (this wave's own
        # `test_display_profile.py`), not a measurement-noise estimate.
        return Accuracy(
            de00_estimate=2.0,
            basis="synthetic model + matrix/TRC profile fidelity limit (no measurement noise of its own)",
            cross_checked_against=None,
        )

    def measure(self, patches: list) -> list:
        out = []
        for patch in patches:
            xyz = self.model.measure(patch.rgb, position=patch.position, t_s=self.t_s, rng=self._rng)
            uncertainty = np.abs(xyz) * self.model.noise_std_frac
            out.append(Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=uncertainty))
        return out

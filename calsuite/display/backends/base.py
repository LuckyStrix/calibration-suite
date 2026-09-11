"""The backend protocol every ``display/backends/*`` module implements
(docs/design.md §5.1: "Pluggable, and each carries its own accuracy
statement").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Accuracy:
    """A backend's own stated accuracy (design §5.1's table). `de00_estimate`
    is the expected ΔE00 under typical conditions -- an *estimate* until a
    backend is itself measured against a stronger reference (house rule 7).
    `cross_checked_against` names another backend's name this one has been
    validated against at least once (design §5.1: "the camera backend is
    honest only if it's cross-checked once against one of the other two");
    ``None`` means it hasn't been, which callers should surface, not hide.
    """

    de00_estimate: float
    basis: str
    cross_checked_against: str | None = None

    def to_dict(self) -> dict:
        return {
            "de00_estimate": self.de00_estimate,
            "basis": self.basis,
            "cross_checked_against": self.cross_checked_against,
        }


@dataclass(frozen=True)
class Measurement:
    """One measured patch."""

    rgb: tuple
    xyz: np.ndarray  # absolute, cd/m^2 (Y in cd/m^2) -- see synth.display.DisplayModel
    uncertainty: np.ndarray  # same shape as xyz (per-component 1-sigma), or a 0-d array for a scalar estimate


@runtime_checkable
class Backend(Protocol):
    name: str

    def accuracy(self) -> Accuracy: ...

    def measure(self, patches: list) -> list[Measurement]: ...

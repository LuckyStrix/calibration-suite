"""The return type every pure analysis function in ``camera/``, ``lens/`` and
``display/`` produces (docs/design.md house rule 5: "pure analysis, separate
I/O"). Nothing here touches a file, a subprocess, or the store -- that split
is the entire reason this module exists separately from ``store.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Refusal:
    """One reason an analysis declined to produce a trustworthy result.

    ``value`` and ``threshold`` are kept as plain JSON-safe types (floats,
    strings, small lists) rather than numpy scalars -- ``store.py`` writes
    ``Analysis.refusals`` straight into a record with no custom JSON
    encoder, and a stray ``numpy.float64`` would break that silently on
    some numpy versions and loudly on others.
    """

    check: str
    message: str
    value: Any = None
    threshold: Any = None

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class Analysis:
    """Uniform output of a camera/lens/display analysis function.

    ``result``, ``residuals`` and ``uncertainty`` are plain dicts of
    JSON-safe values -- an analysis function that wants to hand back a large
    array (a PRNU map, a distortion field) puts it under a key and the
    *caller* decides whether that becomes a ``store.Store.save(artifacts=…)``
    sidecar; ``fit.py`` itself has no opinion about storage.

    ``refusals`` is empty for a clean fit. A non-empty list means the
    eventual record is saved with ``status="refused"`` -- house rule 3: a
    refusal is still a finding, and the record is kept, not discarded.
    """

    result: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    refusals: list[Refusal] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.refusals) == 0

    def refuse(self, check: str, message: str, value: Any = None, threshold: Any = None) -> None:
        """Convenience for analysis functions: append a Refusal in place
        rather than constructing the list by hand at every call site."""
        self.refusals.append(Refusal(check, message, value, threshold))

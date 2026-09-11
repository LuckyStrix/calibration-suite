"""Provenance: where a record's numbers came from, and what that permits.

Ordering matters, from strongest evidence to weakest. Same idea as
filamentdb's ``PROVENANCE`` tuple (``3mf_scripts/filamentdb.py``), applied to
camera/lens/display records instead of filament colors:

    measured -> derived -> vendor -> nominal

- ``measured``   -- produced by an analysis run on captured data that passed
  its own refusal checks *and* its validation step, where the kind has one
  (docs/design.md house rule 2; see ``store.require_exportable``).
- ``derived``    -- computed from other records (e.g. a DCP's ForwardMatrix
  derived from a measured ColorMatrix), not from a fresh capture.
- ``vendor``     -- taken from a manufacturer datasheet or an existing
  third-party database entry (e.g. lensfun's ``mil-canon.xml`` distortion
  params for a lens we haven't measured ourselves).
- ``nominal``    -- read off a label with no measurement behind it at all.
  EDID primaries are the running example (docs/design.md §0): they are what
  the panel *claims*, not what it does.

A record's provenance is never upgraded by fiat -- only by rerunning the
analysis that would legitimately produce a stronger one.
"""

from __future__ import annotations

PROVENANCE = ("measured", "derived", "vendor", "nominal")

# Only these two are trustworthy enough to hand to something else (lensfun,
# a system-installed ICC profile, a DCP). "vendor" and "nominal" numbers are
# for comparison and staleness checks, not export -- exporting a nominal
# EDID primary as though it were a calibration is exactly the kind of
# quietly-wrong number this project exists to avoid.
EXPORTABLE = ("measured", "derived")


def rank(provenance: str) -> int:
    """Lower is stronger evidence. Raises ValueError on an unknown string,
    same as the list it wraps."""
    return PROVENANCE.index(provenance)


def is_exportable(provenance: str) -> bool:
    return provenance in EXPORTABLE


def stronger_or_equal(a: str, b: str) -> bool:
    """True if provenance ``a`` is at least as strong as ``b``."""
    return rank(a) <= rank(b)

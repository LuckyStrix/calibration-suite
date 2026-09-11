"""Where things live on disk, overridable by environment variable.

Two directories with very different sync policies:

- ``records/`` is small JSON (plus ``.npz`` sidecars), lives inside this
  repo, and is meant to be committed -- it *is* the suite's output
  (implementation-plan.md, "Core contracts").
- ``captures/`` holds raw camera files (CR3/DNG, ~25-30 MB each, 100+ frames
  in a single PTC session). It defaults to a directory OUTSIDE this repo so
  an accidental broad ``git add`` can't pull tens of gigabytes of raw frames
  along for the ride -- the same reason hydrationTracker keeps its database
  out of its own tree. ``.gitignore`` is a second line of defense, not the
  first.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def records_dir() -> Path:
    """The record store root. ``CALSUITE_RECORDS`` overrides; default is
    ``<repo>/records``."""
    override = os.environ.get("CALSUITE_RECORDS")
    return Path(override).expanduser() if override else REPO_ROOT / "records"


def captures_dir() -> Path:
    """Where manual-import capture folders default to. ``CALSUITE_CAPTURES``
    overrides; default is ``~/calsuite-captures`` -- deliberately outside
    both this repo and any synced folder (docs/design.md §6)."""
    override = os.environ.get("CALSUITE_CAPTURES")
    return Path(override).expanduser() if override else Path.home() / "calsuite-captures"

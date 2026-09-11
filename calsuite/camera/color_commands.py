"""`calsuite camera color ...` subcommands (camera color characterization,
docs/design.md section 3.2). Wired in from ``camera/commands.py`` so the
sensor and color halves of ``camera/`` can be built independently."""

from __future__ import annotations


def register(camera_subparsers) -> None:
    """Add the `color` subcommand to the `camera` command's subparsers."""
    p = camera_subparsers.add_parser("color", help="camera color characterization (not built yet)")
    p.set_defaults(func=lambda args: print("calsuite camera color: not built yet") or 1)

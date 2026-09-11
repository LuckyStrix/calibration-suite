"""``calsuite lens`` subcommands.

Stub: Wave 2B (lens -- docs/implementation-plan.md's work-waves table)
replaces this with real ``distortion``/``tca``/``flats``/``mtf``/``psf``/
``export``/``report`` subcommands. See camera/commands.py's docstring for
the wiring pattern every area follows.
"""

from __future__ import annotations


def register(subparsers) -> None:
    parser = subparsers.add_parser("lens", help="lens calibration (not built yet)")
    parser.set_defaults(func=_not_built)


def _not_built(args) -> int:
    print("calsuite lens: not built yet -- see docs/implementation-plan.md, Wave 2B.")
    return 1

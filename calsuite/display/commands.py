"""``calsuite display`` subcommands.

Stub: Wave 2C (display -- docs/implementation-plan.md's work-waves table)
replaces this with real ``measure``/``profile``/``install``/``validate``/
``report`` subcommands. See camera/commands.py's docstring for the wiring
pattern every area follows.
"""

from __future__ import annotations


def register(subparsers) -> None:
    parser = subparsers.add_parser("display", help="display calibration (not built yet)")
    parser.set_defaults(func=_not_built)


def _not_built(args) -> int:
    print("calsuite display: not built yet -- see docs/implementation-plan.md, Wave 2C.")
    return 1

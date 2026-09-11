"""``calsuite camera`` subcommands.

Stub: Wave 2A (camera/sensor -- docs/implementation-plan.md's work-waves
table) replaces this with real ``bias``/``ptc``/``linearity``/``darks``/
``fixed-pattern``/``iso``/``shutter``/``report`` subcommands, each backed
by a pure analysis function in this package and wired up the same way
``cli.py`` wires areas together: ``register(subparsers)`` adds a
subparser, the subparser's ``func`` does the file/capture/store I/O and
calls into the pure analysis.
"""

from __future__ import annotations


def register(subparsers) -> None:
    parser = subparsers.add_parser("camera", help="camera sensor calibration (not built yet)")
    parser.set_defaults(func=_not_built)


def _not_built(args) -> int:
    print("calsuite camera: not built yet -- see docs/implementation-plan.md, Wave 2A.")
    return 1

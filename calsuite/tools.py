"""Finding and running external binaries the suite shells out to.

Every optional external tool is named once, here, so ``doctor.py`` can
report on availability from one list instead of several ad hoc checks
scattered through ``raw.py``, ``capture/`` and ``display/``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# Every external binary the suite ever shells out to. Not all are needed for
# every command -- raw.py falls back from exiftool to dcraw, display/
# backends fall back to the built-in ICC writer without ArgyllCMS, etc.
# Centralizing the names means doctor.py's availability check can't drift
# out of sync with what the code actually calls.
KNOWN_TOOLS = (
    "exiftool",
    "dcraw",
    "gphoto2",
    "dispwin",
    "spotread",
    "colprof",
    "profcheck",
    "colormgr",
    "xprop",
)

DEFAULT_TIMEOUT_S = 30
# Long enough for any of the above to answer a version/identify/short-capture
# query; short enough that a hung subprocess (gphoto2 waiting on a camera
# that was unplugged mid-command, a colorimeter that lost its USB grip)
# doesn't stall a test run or a CLI invocation indefinitely.


class ToolError(RuntimeError):
    """Missing binary, timeout, or non-zero exit -- raised with the full
    command and stderr attached, so the failure is diagnosable from the
    exception message alone. A bare ``CalledProcessError`` repr doesn't
    show stderr, which is exactly the part you need mid-capture-session."""


@dataclass(frozen=True)
class ToolResult:
    args: tuple
    returncode: int
    stdout: str
    stderr: str


def which(name: str) -> str | None:
    """Path to ``name`` on PATH, or ``None``. Never raises -- callers that
    have a fallback (raw.py: exiftool -> dcraw) call this first and branch;
    callers with no fallback call ``run()`` directly and let it raise."""
    return shutil.which(name)


def run(args, *, timeout: float = DEFAULT_TIMEOUT_S, check: bool = True) -> ToolResult:
    """Run ``args`` (a list; ``args[0]`` the binary name or path) and
    capture its output.

    Raises ``ToolError`` if the binary isn't on PATH, if it times out, or
    (when ``check=True``, the default) if it exits non-zero. Tests exercise
    this against small fake scripts placed on ``PATH`` (see
    ``tests/test_tools.py`` and ``tests/test_capture_gphoto2.py``) rather
    than the real binaries, which aren't guaranteed to be installed.
    """
    args = [str(a) for a in args]
    exe = which(args[0])
    if exe is None:
        raise ToolError(f"{args[0]!r} is not on PATH (needed for: {' '.join(args)})")
    # Run the *resolved* path, not the bare name -- on Windows, a bare
    # extensionless name (e.g. "gphoto2") only launches via PATHEXT
    # resolution through a shell; `which()` (shutil.which) already did that
    # resolution once (e.g. to "gphoto2.cmd"), so handing that resolved
    # path straight to subprocess lets it launch directly on every OS
    # without needing `shell=True`.
    resolved_args = [exe, *args[1:]]
    try:
        proc = subprocess.run(
            resolved_args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"{' '.join(args)} timed out after {timeout}s") from exc
    if check and proc.returncode != 0:
        raise ToolError(f"{' '.join(args)} exited {proc.returncode}\nstderr:\n{proc.stderr}")
    return ToolResult(tuple(args), proc.returncode, proc.stdout, proc.stderr)

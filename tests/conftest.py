"""Shared test fixtures.

``fake_bin``: puts a small fake external-tool executable on PATH, for
tests that exercise ``calsuite.tools.run()``/``which()`` (and the couple of
callers that shell out directly, e.g. ``display/backends/argyll.py``)
against gphoto2, exiftool, dcraw, spotread, colprof, profcheck, dispwin,
colormgr or xprop -- none of which are installed on a CI runner.

Cross-platform, in one place, used by every test that needs a fake tool
(instead of each test file rolling its own ``_make_script`` -- the old
per-file version wrote a ``#!/bin/sh`` script, which Windows cannot
execute at all: no shebang support, and PATH lookup only ever finds a
PATHEXT extension -- ``.exe``/``.cmd``/``.bat``/... -- never a bare
extensionless file). The fake tool's own logic is always plain **Python**,
never shell, specifically so one fixture works unmodified on every OS
pytest itself runs on: read ``sys.argv``/``sys.stdin``, write to
``sys.stdout``/``sys.stderr``, call ``sys.exit(code)`` if needed.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest


def _install_fake_bin(tmp_path: Path, name: str, python_source: str) -> Path:
    """Write ``name`` under ``tmp_path`` such that ``shutil.which(name)``
    (``calsuite.tools.which``) and a subprocess launch of the resolved
    path both find and run ``python_source``.

    POSIX: a ``#!<this interpreter>``-shebang script named exactly
    ``name``, executable bit set. Windows: a ``<name>.py`` file holding
    the real logic, plus a ``<name>.cmd`` shim next to it
    (``@"<python>" "<name>.py" %*``) -- ``.cmd`` is always in the default
    ``PATHEXT``, so ``shutil.which("name")`` finds ``name.cmd``
    automatically, and CMD/``CreateProcess`` can launch a ``.cmd`` file
    directly with no shell wrapping needed from the caller.
    """
    py_path = tmp_path / f"{name}.py"
    py_path.write_text(python_source, encoding="utf-8")

    if sys.platform == "win32":
        shim = tmp_path / f"{name}.cmd"
        shim.write_text(f'@"{sys.executable}" "{py_path}" %*\r\n', encoding="utf-8")
        return shim

    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n{python_source}", encoding="utf-8")
    mode = script.stat().st_mode
    script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """``fake_bin("name", python_source) -> Path``. Prepends ``tmp_path``
    to ``PATH`` the first time it's called -- safe to call more than once
    in the same test (e.g. to install both ``colprof`` and ``profcheck``),
    every fake tool then living in the one directory already on PATH.
    """
    state = {"on_path": False}

    def make(name: str, python_source: str) -> Path:
        path = _install_fake_bin(tmp_path, name, python_source)
        if not state["on_path"]:
            monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
            state["on_path"] = True
        return path

    return make

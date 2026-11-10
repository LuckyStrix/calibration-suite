"""Keep calsuite's own console usable for its prompts on Windows.

ArgyllCMS tools read keys through the Windows console API and switch the
console they run in to raw input (no line editing, no echo, Ctrl+C delivered
as a key) -- and do not always switch it back. When one shared calsuite's
console, the next ``input()`` never saw a whole line: Enter only moved the
cursor to the start of the line and Ctrl+C did nothing (seen on a real
laptop, 2026-11). ``tools.run`` now gives every tool its own hidden console,
and ``prompt()`` puts this console back into normal line mode before asking
anything, so neither depends on the other being enough.

No-ops off Windows, and on a stdin that is not a console (a pipe, a test).
"""

from __future__ import annotations

import sys

_LINE_MODE_BITS = 0x0001 | 0x0002 | 0x0004
# ENABLE_PROCESSED_INPUT (Ctrl+C is a signal) | ENABLE_LINE_INPUT (input()
# returns on Enter) | ENABLE_ECHO_INPUT (typed keys are shown): what an
# ordinary console prompt needs, and what raw mode turns off.

_saved_mode: int | None = None


def _stdin_console():  # pragma: no cover -- Windows only
    import ctypes

    kernel32 = ctypes.windll.kernel32
    return kernel32, kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE


def save_mode() -> None:
    """Remember the console's input mode as calsuite found it (``cli.main``)."""
    global _saved_mode
    if sys.platform != "win32":
        return
    try:  # pragma: no cover -- Windows only
        import ctypes
        from ctypes import wintypes

        kernel32, handle = _stdin_console()
        mode = wintypes.DWORD()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            _saved_mode = mode.value
    except Exception:  # pragma: no cover
        pass


def restore_mode() -> None:
    """Put the console back into line mode (the saved mode, with the three
    line-mode bits forced on) and drop keys queued while a tool owned it."""
    if sys.platform != "win32":
        return
    try:  # pragma: no cover -- Windows only
        kernel32, handle = _stdin_console()
        base = _saved_mode if _saved_mode is not None else 0
        if kernel32.SetConsoleMode(handle, base | _LINE_MODE_BITS):
            kernel32.FlushConsoleInputBuffer(handle)
    except Exception:  # pragma: no cover
        pass


def prompt(text: str = "") -> str:
    """``input(text)``, after making sure Enter and Ctrl+C work."""
    restore_mode()
    return input(text)

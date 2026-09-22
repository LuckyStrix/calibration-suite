"""Fullscreen patch window (docs/design.md §5.3: "SDL2 (pygame) fullscreen
on both OSes; no browser, because browsers color-manage"). Draws exact RGB
with no scaling and no color management -- a pygame software surface just
gets integer 0-255 values written straight into a pixel and blitted; there
is no ICC/VCGT step in this module for pygame to apply even if it wanted
to.

Testable headless via ``SDL_VIDEODRIVER=dummy``, set on the environment
*before* calling ``open_window`` (pygame reads it at
``pygame.display.init()`` time, which happens inside ``pygame.init()``).
"""

from __future__ import annotations

import os
import time as _time

from calsuite.display import constants as dc

# ``pygame`` itself is imported lazily, inside ``open_window`` below (the
# one function every caller in this module calls before any other,
# per its own pygame.init() sequencing requirement) -- importing it is
# slow and, without this, prints pygame's own "Hello from the pygame
# community" banner on every `import calsuite.display.window`, which
# includes commands that never touch a real window at all (`--help`,
# `doctor`, any non-display command). ``global pygame`` in ``open_window``
# binds the module into this file's namespace exactly once, after which
# every other function below (``close_window``, ``show_patch``, ...) can
# keep referring to the bare name ``pygame`` as if it had been imported at
# the top, because by the time any of them runs, ``open_window`` already
# has. Type annotations referencing ``pygame.Surface`` are fine unimported
# -- ``from __future__ import annotations`` (above) makes every annotation
# in this file a lazily-evaluated string, never touched at import time.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Must be set before pygame is ever imported (it reads this once, on
# import) -- the banner it silences is unconditional otherwise, harmless
# but noisy on every `calsuite display ...`/`calsuite demo` run.


class WindowAborted(RuntimeError):
    """Raised when the user hits ESC (or closes the window) mid-sequence
    (design §5.3: "ESC to abort")."""


def open_window(width: int | None = None, height: int | None = None, *, fullscreen: bool = True) -> pygame.Surface:
    """Open the patch window. With explicit `width`/`height` (the usual
    case -- callers pass the display's own resolution, e.g. from EDID or
    ``pygame.display.Info()``) it opens at exactly that size; with either
    omitted, it asks SDL for the current desktop size to fill in the gap.
    Tests under ``SDL_VIDEODRIVER=dummy`` should always pass both
    explicitly -- the dummy driver's desktop-size query is not meaningful.
    """
    global pygame
    import pygame

    pygame.init()
    pygame.mouse.set_visible(False)
    flags = pygame.FULLSCREEN if fullscreen else 0
    if width is None or height is None:
        info = pygame.display.Info()
        width = width or info.current_w
        height = height or info.current_h
    return pygame.display.set_mode((width, height), flags)


def close_window() -> None:
    pygame.quit()


def _rgb255(rgb: tuple) -> tuple:
    return tuple(max(0, min(255, round(c * 255))) for c in rgb)


def show_patch(
    screen: pygame.Surface,
    rgb: tuple,
    *,
    position: tuple = (0.5, 0.5),
    size_frac: float = 1.0,
    background: tuple = (0.0, 0.0, 0.0),
) -> None:
    """Fill the whole screen with `background` (also exact, no color
    management), then draw `rgb` as a square covering `size_frac` of the
    shorter screen dimension, centered at fractional `position` ((0.5,
    0.5) = screen center). `size_frac=1.0` (the default) fills the entire
    screen -- what every patch set except ``patches.uniformity_grid``
    wants; uniformity measurements pass a smaller `size_frac` so each grid
    position samples a genuinely local part of the panel.
    """
    w, h = screen.get_size()
    screen.fill(_rgb255(background))
    side = max(1, round(size_frac * min(w, h)))
    rect = pygame.Rect(0, 0, side, side)
    rect.center = (round(position[0] * w), round(position[1] * h))
    pygame.draw.rect(screen, _rgb255(rgb), rect)
    pygame.display.flip()


def check_abort() -> bool:
    """Pump the SDL event queue; True if ESC was pressed or the window was
    asked to close since the last call. Must be polled regularly during a
    measurement sequence -- SDL only surfaces events when its queue is
    pumped -- which ``run_patch_sequence`` does once per patch."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


PLACEMENT_TEXT_GRAY = (0.65, 0.65, 0.65)
PLACEMENT_KEY_GRAY = (0.95, 0.95, 0.95)
# Prompt text is drawn dim-ish and only on the half of the screen *away
# from* the square being measured (see ``draw_placement_prompt``); it is
# erased before the reading, so the instrument never sees it.

PLACEMENT_TEXT_WIDTH_FRAC = 0.38
# Widest a prompt line may be, as a fraction of screen width (also capped by
# the free space beside the square, below).

PLACEMENT_TEXT_MARGIN_FRAC = 0.06
# Keep the text at least this far (fraction of screen width; ~20 mm on a
# 344 mm panel) from the square's edge, so a puck sitting on the square --
# whose housing is wider than its aperture -- has the text well outside its
# footprint. Matters for the center-column squares, where the free space on
# either side is only ~45% of the width; the first version (fixed text
# width) left the text 10 mm from those squares, and *overlapping* the
# square on a portrait screen.


def _wrap(font, text: str, max_px: int) -> list:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and font.size(trial)[0] > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def draw_placement_prompt(screen: pygame.Surface, patch, index: int, total: int) -> None:
    """Show ``patch`` with instructions for the human next to it, on the half
    of the screen it is *not* in. The terminal is hidden behind this window
    during a run, so this is the only place the person can be told which
    square to move to and what key to press."""
    w, h = screen.get_size()
    show_patch(screen, patch.rgb, position=patch.position, size_frac=patch.size_frac)
    short = min(w, h)  # font sizes follow the short side, so a rotated (portrait) screen doesn't get giant text
    big = pygame.font.Font(None, max(20, round(short * 0.085)))
    small = pygame.font.Font(None, max(16, round(short * 0.055)))
    blocks = [
        (big, f"SQUARE {index} of {total}", PLACEMENT_TEXT_GRAY),
        (small, patch.placement or "", PLACEMENT_TEXT_GRAY),
        (small, "Put the instrument flat on the WHITE SQUARE.", PLACEMENT_TEXT_GRAY),
        (big, "Then press SPACE", PLACEMENT_KEY_GRAY),
        (small, "Hold still until the next square lights up.   ESC stops the run.", PLACEMENT_TEXT_GRAY),
    ]
    # The text lives in the free strip beside the square, on whichever side
    # has more room, never closer than PLACEMENT_TEXT_MARGIN_FRAC.
    side = round(patch.size_frac * min(w, h))
    sq_cx, margin = patch.position[0] * w, w * PLACEMENT_TEXT_MARGIN_FRAC
    if patch.position[0] < 0.5:
        lo, hi = sq_cx + side / 2 + margin, w - margin
    else:
        lo, hi = margin, sq_cx - side / 2 - margin
    max_px = max(1, min(round(w * PLACEMENT_TEXT_WIDTH_FRAC), round(hi - lo)))
    cx = round((lo + hi) / 2)
    rendered, gap = [], round(short * 0.03)
    for font, text, gray in blocks:
        for line in _wrap(font, text, max_px):
            rendered.append(font.render(line, True, _rgb255(gray)))
        rendered.append(None)  # gap between blocks
    total_h = sum((surf.get_height() if surf else gap) for surf in rendered)
    y = max(0, (h - total_h) // 2)
    for surf in rendered:
        if surf is None:
            y += gap
            continue
        screen.blit(surf, surf.get_rect(midtop=(cx, y)))
        y += surf.get_height()
    pygame.display.flip()


def wait_for_placement(screen: pygame.Surface, patch, index: int, total: int, *, sleep=None) -> None:
    """Show the placement prompt and block until the person presses SPACE or
    ENTER, then redraw ``patch`` *alone* (no text) so the reading is of a
    clean square. ESC or closing the window raises ``WindowAborted``. Keys
    pressed before the prompt existed are discarded, so a stray keypress
    can't skip a square."""
    sleep = sleep or _time.sleep
    confirm = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER)
    pygame.event.clear()
    draw_placement_prompt(screen, patch, index, total)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise WindowAborted("aborted at a placement prompt")
            if event.type == pygame.KEYDOWN and event.key in confirm:
                show_patch(screen, patch.rgb, position=patch.position, size_frac=patch.size_frac)
                return
        sleep(0.02)


def run_patch_sequence(
    screen: pygame.Surface,
    patches: list,
    on_patch,
    *,
    settle_s: float = dc.SETTLE_TIME_S,
    sleep=None,
    confirm_placement: bool = False,
) -> list:
    """Show each of `patches` in turn, wait `settle_s` for the panel (and
    an instrument) to settle, then call ``on_patch(patch)`` -- typically a
    backend's per-patch measurement -- and collect its return value.

    With ``confirm_placement`` (a handheld instrument), a patch whose
    ``placement`` is set (a small square the instrument has to be moved
    onto) first waits for the person to put it there and press a key, with
    the instructions drawn on the window itself. A camera pointed at the
    whole screen leaves this off.

    Raises ``WindowAborted`` (leaving the window open; the caller decides
    whether/when to close it) the moment ESC or a window-close is seen,
    either before showing a patch or after showing it but before
    measuring. `sleep` defaults to ``time.sleep``; tests pass a fast
    stand-in (or 0) to run a whole sequence without actually waiting.
    """
    sleep = sleep or _time.sleep
    results = []
    n_placements = sum(1 for p in patches if p.placement is not None) if confirm_placement else 0
    placed = 0
    for patch in patches:
        if check_abort():
            raise WindowAborted("aborted before showing all patches")
        show_patch(screen, patch.rgb, position=patch.position, size_frac=patch.size_frac)
        if confirm_placement and patch.placement is not None:
            placed += 1
            wait_for_placement(screen, patch, placed, n_placements, sleep=sleep)
        sleep(settle_s)
        if check_abort():
            raise WindowAborted("aborted after showing a patch, before measuring it")
        results.append(on_patch(patch))
    return results

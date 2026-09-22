import pytest


@pytest.fixture(autouse=True)
def _dummy_video_driver(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    yield


def test_open_and_close_window():
    from calsuite.display import window

    screen = window.open_window(64, 48, fullscreen=False)
    assert screen.get_size() == (64, 48)
    window.close_window()


def test_show_patch_draws_exact_rgb():
    from calsuite.display import window

    screen = window.open_window(20, 20, fullscreen=False)
    try:
        window.show_patch(screen, (0.2, 0.6, 1.0), size_frac=1.0)
        expected = tuple(round(c * 255) for c in (0.2, 0.6, 1.0))
        got = screen.get_at((10, 10))[:3]
        assert tuple(got) == expected
    finally:
        window.close_window()


def test_show_patch_background_outside_patch():
    from calsuite.display import window

    screen = window.open_window(40, 40, fullscreen=False)
    try:
        window.show_patch(screen, (1.0, 1.0, 1.0), position=(0.5, 0.5), size_frac=0.1, background=(0.0, 0.0, 0.0))
        corner = screen.get_at((0, 0))[:3]
        assert tuple(corner) == (0, 0, 0)
        center = screen.get_at((20, 20))[:3]
        assert tuple(center) == (255, 255, 255)
    finally:
        window.close_window()


def test_check_abort_false_with_no_events():
    from calsuite.display import window

    window.open_window(20, 20, fullscreen=False)
    try:
        assert window.check_abort() is False
    finally:
        window.close_window()


def test_run_patch_sequence_collects_results_and_settles(monkeypatch):
    from calsuite.display import patches, window

    screen = window.open_window(20, 20, fullscreen=False)
    slept = []
    try:
        seq = patches.gray_ramp(steps=3)
        results = window.run_patch_sequence(screen, seq, lambda p: p.rgb, settle_s=0.01, sleep=slept.append)
        assert results == [p.rgb for p in seq]
        assert slept == [0.01, 0.01, 0.01]
    finally:
        window.close_window()


def test_run_patch_sequence_aborts_on_escape(monkeypatch):
    from calsuite.display import patches, window

    screen = window.open_window(20, 20, fullscreen=False)
    try:
        calls = {"n": 0}

        def fake_check_abort():
            calls["n"] += 1
            return calls["n"] > 1  # abort after the first patch is shown

        monkeypatch.setattr(window, "check_abort", fake_check_abort)
        seq = patches.gray_ramp(steps=5)
        with pytest.raises(window.WindowAborted):
            window.run_patch_sequence(screen, seq, lambda p: p.rgb, settle_s=0.0, sleep=lambda _s: None)
    finally:
        window.close_window()


# -- placement prompt (handheld instrument, hidden terminal) --------------------


def _space():
    import pygame

    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)


def _rgb(screen):
    import numpy as np
    import pygame

    w, h = screen.get_size()
    return np.frombuffer(pygame.image.tostring(screen, "RGB"), dtype=np.uint8).reshape(h, w, 3)


def _grid_patch(row, col):
    from calsuite.display import patches

    return patches.uniformity_grid()[row][col]


@pytest.mark.parametrize("screen_size", [(640, 400), (1920, 1200), (1080, 1920)])
def test_placement_prompt_keeps_its_text_clear_of_the_square(screen_size):
    """The instrument sits on the square; the prompt text must not glow next
    to it -- at least PLACEMENT_TEXT_MARGIN_FRAC of the width from its edge,
    on every screen shape (including a rotated one, and the center column,
    where the free space beside the square is smallest)."""
    import numpy as np

    from calsuite.display import window

    w, h = screen_size
    screen = window.open_window(w, h, fullscreen=False)
    try:
        for row, col in [(0, 0), (2, 2), (4, 4), (0, 2)]:
            patch = _grid_patch(row, col)
            window.draw_placement_prompt(screen, patch, 1, 25)
            lit = _rgb(screen).sum(axis=2) > 0
            side = round(patch.size_frac * min(w, h))
            cx, cy = round(patch.position[0] * w), round(patch.position[1] * h)
            square = np.zeros_like(lit)
            square[max(0, cy - side // 2) : cy + side // 2 + 1, max(0, cx - side // 2) : cx + side // 2 + 1] = True
            ys, xs = np.where(lit & ~square)
            assert len(xs), "no prompt text was drawn"
            gap = min(abs(xs - cx + side // 2).min(), abs(xs - cx - side // 2).min())
            assert gap >= window.PLACEMENT_TEXT_MARGIN_FRAC * w * 0.9, (row, col, gap)
            assert xs.min() > 0 and xs.max() < w - 1 and ys.min() >= 0 and ys.max() < h - 1  # nothing clipped
    finally:
        window.close_window()


def test_placement_prompt_flips_sides_for_a_square_on_the_right():
    from calsuite.display import window

    screen = window.open_window(640, 400, fullscreen=False)
    try:
        window.draw_placement_prompt(screen, _grid_patch(4, 4), 25, 25)
        lit = _rgb(screen).sum(axis=2) > 0
        assert lit[:, : 640 // 2].any()
    finally:
        window.close_window()


def test_wait_for_placement_blocks_until_space_then_shows_the_square_alone(monkeypatch):
    import pygame

    from calsuite.display import window

    screen = window.open_window(640, 400, fullscreen=False)
    try:
        replies = iter([[], [], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)], [_space()]])
        calls = {"n": 0}

        def fake_get():
            calls["n"] += 1
            return next(replies)

        monkeypatch.setattr(pygame.event, "get", fake_get)
        patch = _grid_patch(0, 0)
        window.wait_for_placement(screen, patch, 1, 25, sleep=lambda _s: None)

        assert calls["n"] == 4  # an unrelated key did not release the wait
        img = _rgb(screen)
        lit = img.sum(axis=2) > 0
        assert not lit[:, 640 // 2 :].any(), "prompt text was still on screen for the reading"
        assert tuple(img[round(patch.position[1] * 400), round(patch.position[0] * 640)]) == (255, 255, 255)
    finally:
        window.close_window()


@pytest.mark.parametrize("which", ["escape", "quit"])
def test_wait_for_placement_aborts_on_escape_or_close(monkeypatch, which):
    import pygame

    from calsuite.display import window

    screen = window.open_window(320, 200, fullscreen=False)
    try:
        ev = (
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
            if which == "escape"
            else pygame.event.Event(pygame.QUIT)
        )
        monkeypatch.setattr(pygame.event, "get", lambda: [ev])
        with pytest.raises(window.WindowAborted):
            window.wait_for_placement(screen, _grid_patch(0, 0), 1, 25, sleep=lambda _s: None)
    finally:
        window.close_window()


def test_run_patch_sequence_pauses_only_on_placement_patches_and_numbers_them(monkeypatch):
    import pygame

    from calsuite.display import patches, window

    screen = window.open_window(320, 200, fullscreen=False)
    try:
        monkeypatch.setattr(pygame.event, "get", lambda: [_space()])
        seen = []
        real = window.wait_for_placement
        monkeypatch.setattr(
            window, "wait_for_placement", lambda scr, patch, i, n, **kw: (seen.append((patch.label, i, n)), real(scr, patch, i, n, **kw))
        )
        seq = patches.gray_ramp(2) + [_grid_patch(0, 0), _grid_patch(0, 1)]
        window.run_patch_sequence(screen, seq, lambda p: p.label, settle_s=0.0, sleep=lambda _s: None, confirm_placement=True)
        assert seen == [("uniformity-0-0", 1, 2), ("uniformity-0-1", 2, 2)]

        seen.clear()
        window.run_patch_sequence(screen, seq, lambda p: p.label, settle_s=0.0, sleep=lambda _s: None)
        assert seen == []  # a camera watching the whole screen never needs the pause
    finally:
        window.close_window()

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

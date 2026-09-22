import pytest

from calsuite.display import patches


def test_channel_ramp_isolates_channel():
    ramp = patches.channel_ramp("g", steps=5)
    assert len(ramp) == 5
    assert ramp[0].rgb == (0.0, 0.0, 0.0)
    assert ramp[-1].rgb == (0.0, 1.0, 0.0)
    assert all(p.rgb[0] == 0.0 and p.rgb[2] == 0.0 for p in ramp)


def test_gray_ramp_is_neutral():
    ramp = patches.gray_ramp(steps=9)
    assert len(ramp) == 9
    assert all(p.rgb[0] == p.rgb[1] == p.rgb[2] for p in ramp)


def test_channel_ramp_rejects_bad_channel():

    with pytest.raises(ValueError):
        patches.channel_ramp("x")


def test_primaries_secondaries_has_eight():
    ps = patches.primaries_secondaries()
    assert {p.label for p in ps} == {"r", "g", "b", "c", "m", "y", "w", "k"}


def test_additivity_set_has_five():
    aset = patches.additivity_set()
    assert {p.label for p in aset} == {"additivity-r", "additivity-g", "additivity-b", "additivity-w", "additivity-k"}


def test_uniformity_grid_shape_and_positions():
    from calsuite.display import constants as dc

    lo, hi = dc.UNIFORMITY_GRID_INSET, 1.0 - dc.UNIFORMITY_GRID_INSET
    grid = patches.uniformity_grid(n=5)
    assert len(grid) == 5
    assert all(len(row) == 5 for row in grid)
    assert grid[0][0].position == pytest.approx((lo, lo))
    assert grid[4][4].position == pytest.approx((hi, hi))
    assert grid[2][2].position == pytest.approx((0.5, 0.5))  # the center cell analysis.uniformity compares against
    assert all(p.rgb == (1.0, 1.0, 1.0) for row in grid for p in row)


@pytest.mark.parametrize("screen", [(1920, 1200), (1920, 1080), (1366, 768), (1080, 1920)])
def test_every_uniformity_square_fits_fully_on_screen(screen):
    """The grid used to run 0.0..1.0, centering the corner squares *on* the
    screen corners -- three quarters of each off-screen, unreachable by any
    instrument."""
    w, h = screen
    for row in patches.uniformity_grid():
        for p in row:
            side = round(p.size_frac * min(w, h))
            cx, cy = p.position[0] * w, p.position[1] * h
            assert cx - side / 2 >= 0 and cx + side / 2 <= w, p.label
            assert cy - side / 2 >= 0 and cy + side / 2 <= h, p.label


def test_only_the_uniformity_squares_ask_for_placement():
    grid = [p for row in patches.uniformity_grid() for p in row]
    assert all(p.placement for p in grid)
    assert grid[0].placement == "row 1 of 5, column 1 of 5"
    assert grid[-1].placement == "row 5 of 5, column 5 of 5"
    others = patches.gray_ramp(3) + patches.primaries_secondaries() + patches.additivity_set() + patches.validation_set()
    assert all(p.placement is None for p in others)


def test_warmup_schedule_covers_duration():
    times = patches.warmup_schedule(duration_s=300.0, interval_s=100.0)
    assert times == [0.0, 100.0, 200.0, 300.0]


def test_validation_set_has_cc24_plus_extra_neutrals():
    vset = patches.validation_set()
    labels = [p.label for p in vset]
    assert any("dark skin" in label for label in labels)
    assert any("light skin" in label for label in labels)
    assert sum(1 for label in labels if label.startswith("neutral-L")) == 4
    assert all(p.rgb is None for p in vset)
    assert all(p.lab_target is not None and len(p.lab_target) == 3 for p in vset)
    # L* should be a sane percept range for every patch, including the CC24 ones.
    assert all(0.0 <= p.lab_target[0] <= 100.0 for p in vset)

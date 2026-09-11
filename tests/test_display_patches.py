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
    import pytest

    with pytest.raises(ValueError):
        patches.channel_ramp("x")


def test_primaries_secondaries_has_eight():
    ps = patches.primaries_secondaries()
    assert {p.label for p in ps} == {"r", "g", "b", "c", "m", "y", "w", "k"}


def test_additivity_set_has_five():
    aset = patches.additivity_set()
    assert {p.label for p in aset} == {"additivity-r", "additivity-g", "additivity-b", "additivity-w", "additivity-k"}


def test_uniformity_grid_shape_and_positions():
    grid = patches.uniformity_grid(n=5)
    assert len(grid) == 5
    assert all(len(row) == 5 for row in grid)
    assert grid[0][0].position == (0.0, 0.0)
    assert grid[4][4].position == (1.0, 1.0)
    assert grid[2][2].position == (0.5, 0.5)
    assert all(p.rgb == (1.0, 1.0, 1.0) for row in grid for p in row)


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

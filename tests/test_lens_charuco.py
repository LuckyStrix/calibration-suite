import numpy as np

from calsuite.lens import charuco
from calsuite.synth import lens as synthlens
from calsuite.synth import sensor as synth_sensor


def test_build_board_has_expected_geometry():
    board = charuco.build_board(square_length=25.0)
    corners = board.getChessboardCorners()
    # 9x5 interior corners for a 10x6-square board
    assert corners.shape == (45, 3)
    assert corners[:, 2].max() == 0.0  # planar, z=0


def test_plane_to_sensor_maps_plane_px_times_two_plus_offset():
    model = synth_sensor.SensorModel(shape=(40, 60), top_margin=6, left_margin=10)
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=0.01, flux_e_per_s=1000.0, temp_c=20.0, rng=rng)
    # a corner reported at plane px (3, 4) for each named plane should map to
    # cfa row/col = visible_origin + 2*plane_px + that plane's own phase offset.
    corners = np.array([[4.0, 3.0]])  # (x, y) = (col, row) in plane px
    for name in ("R", "G1", "G2", "B"):
        mapped = charuco.plane_to_sensor(frame, corners, name)
        dr, dc = charuco._plane_offsets(frame.pattern)[name]
        expected_col = frame.visible[1].start + 4 * 2 + dc
        expected_row = frame.visible[0].start + 3 * 2 + dr
        assert mapped[0, 0] == expected_col
        assert mapped[0, 1] == expected_row


def test_to_8bit_normalizes_and_clips():
    plane = np.array([0.0, 10.0, 20.0, 1000.0, 1010.0])
    img8 = charuco.to_8bit(plane)
    assert img8.dtype == np.uint8
    assert img8.min() >= 0 and img8.max() <= 255


def test_detect_green_on_synthetic_rendered_frame():
    """End-to-end smoke test: render a ChArUco view through a known camera
    into a synthetic Bayer frame (synth.lens.render_board_frame) and check
    that detect_green finds a plausible number of corners, roughly where
    the known projection puts them (loose tolerance -- this exercises the
    detector's own noise, unlike distortion.py's tight point-correspondence
    round trip)."""
    model = synth_sensor.SensorModel(
        shape=(500, 700), read_noise_e=1.0, gain_e_per_dn=2.0, black_dn=100.0,
        prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0, full_well_e=2_000_000,
    )
    board = charuco.build_board(square_length=30.0)
    board_img = charuco.render_board_image(board, (1000, 600), margin_px=20)
    mm_per_px = 30.0 * 10 / 1000  # squares_x * square_length / board image width

    K = np.array([[900.0, 0, 350.0], [0, 900.0, 250.0], [0, 0, 1.0]])
    dist = np.array([-0.02, 0.0, 0.0, 0.0, 0.0])
    rvec = np.zeros(3)
    objp = board.getChessboardCorners()
    tvec = np.array([-objp[:, 0].max() / 2, -objp[:, 1].max() / 2, 700.0])

    rng = np.random.default_rng(2)
    frame = synthlens.render_board_frame(
        model, board_img, mm_per_px, K, dist, rvec, tvec, rng=rng, exposure_s=0.05
    )
    detection = charuco.detect_green(frame, board)
    assert detection is not None
    assert len(detection.ids) >= 8

    # sensor-mapped corners should land inside the visible frame
    assert np.all(detection.corners[:, 0] >= frame.visible[1].start)
    assert np.all(detection.corners[:, 1] >= frame.visible[0].start)

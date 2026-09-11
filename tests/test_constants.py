from calsuite import constants


def test_estimated_accuracy_has_expected_keys():
    for key in (
        "gain_pct",
        "read_noise_pct",
        "distortion_rms_px",
        "vignetting_pct",
        "mtf50_pct",
        "display_colorimeter_de00",
        "display_camera_de00",
    ):
        assert key in constants.ESTIMATED_ACCURACY


def test_edid_constants_are_internally_consistent():
    assert constants.EDID_BASE_BLOCK_LENGTH == 128
    assert len(constants.EDID_HEADER_MAGIC) == 8
    assert constants.EDID_CHROMATICITY_STEPS == 1024

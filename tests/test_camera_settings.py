from dataclasses import replace

import numpy as np

from calsuite.camera import settings
from calsuite.synth import sensor as synth_sensor


def _frame_with_settings(**kwargs):
    model = synth_sensor.SensorModel(shape=(16, 16))
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    return replace(frame, meta=replace(frame.meta, settings=kwargs))


def test_is_on_recognizes_common_spellings():
    assert settings.is_on("On") is True
    assert settings.is_on("Off") is False
    assert settings.is_on(1) is True
    assert settings.is_on(0) is False
    assert settings.is_on(True) is True
    assert settings.is_on(None) is None
    assert settings.is_on("banana") is None


def test_check_measurement_settings_refuses_only_when_flagged_and_on():
    frames = [_frame_with_settings(long_exposure_nr="On")]
    a = settings.check_measurement_settings(frames, refuse_if_lenr=True)
    assert not a.ok
    assert a.refusals[0].check == "long_exposure_nr_on"


def test_check_measurement_settings_does_not_refuse_when_flag_not_set():
    frames = [_frame_with_settings(long_exposure_nr="On")]
    a = settings.check_measurement_settings(frames, refuse_if_lenr=False)
    assert a.ok


def test_check_measurement_settings_does_not_refuse_on_unknown_metadata():
    # No settings at all (e.g. the dcraw metadata fallback) must never be
    # treated as "off is confirmed" or, worse, cause a false refusal.
    frames = [_frame_with_settings()]
    a = settings.check_measurement_settings(frames, refuse_if_lenr=True, refuse_if_high_iso_nr=True, refuse_if_htp=True)
    assert a.ok
    assert a.result["settings"][settings.LENR_KEY] is None


def test_setting_on_any_true_if_any_frame_has_it_on():
    frames = [_frame_with_settings(high_iso_nr="Off"), _frame_with_settings(high_iso_nr="On")]
    assert settings.setting_on_any(frames, settings.HIGH_ISO_NR_KEY) is True

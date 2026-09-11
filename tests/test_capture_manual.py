from dataclasses import replace
from pathlib import Path

import numpy as np

from calsuite.capture import manual
from calsuite.raw import FrameMeta
from calsuite.synth import sensor as synth_sensor


def _frame_with_signal(mean_dn_over_black, exposure_s, black_dn=512.0, gain=2.0):
    """A tiny synthetic frame whose visible mean sits close to
    `mean_dn_over_black` DN above black -- built via synth.sensor so
    classify() can be tested against known signal levels with no real raw
    file. `frame.meta.exposure_s` is overwritten afterward because
    synth.sensor's FrameMeta doesn't otherwise carry it in a form classify()
    reads consistently with the flux/exposure used to generate the frame.
    """
    flux = mean_dn_over_black * gain / max(exposure_s, 1e-9)
    model = synth_sensor.SensorModel(
        shape=(64, 64),
        black_dn=black_dn,
        gain_e_per_dn=gain,
        read_noise_e=2.0,
        prnu_std=0,
        dsnu_std_e_per_s=0,
        hot_pixel_fraction=0,
    )
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=exposure_s, flux_e_per_s=flux, temp_c=20.0, rng=rng)
    return replace(frame, meta=FrameMeta(exposure_s=exposure_s))


def test_classify_bias():
    assert manual.classify(_frame_with_signal(5, 0.0005)) == "bias"


def test_classify_dark():
    assert manual.classify(_frame_with_signal(5, 30.0)) == "dark"


def test_classify_flat():
    # Default model: black_dn=512, gain=2, full_well_e=40000 -> white_level
    # 20512, span (white - black) = 20000 DN, flat band [2000, 18000]. 5000
    # DN over black sits comfortably inside it.
    assert manual.classify(_frame_with_signal(5000, 0.01)) == "flat"


def test_classify_target():
    # Near the top of the range -- outside the flat band, and not near
    # black either. A coarse heuristic by design (docs/design.md §3.3).
    assert manual.classify(_frame_with_signal(19500, 0.01)) == "target"


def test_scan_folder_builds_manifest(tmp_path, monkeypatch):
    for name in ("bias0000.cr3", "flat0000.cr3"):
        (tmp_path / name).write_bytes(b"not a real raw file")

    frames = {
        "bias0000.cr3": _frame_with_signal(5, 0.0005),
        "flat0000.cr3": _frame_with_signal(5000, 0.01),
    }

    def fake_load(path):
        return frames[Path(path).name]

    monkeypatch.setattr(manual.rawmod, "load", fake_load)
    manifest = manual.scan_folder(tmp_path)

    roles = {Path(e.path).name: e.role for e in manifest.entries}
    assert roles == {"bias0000.cr3": "bias", "flat0000.cr3": "flat"}
    assert manifest.by_role("bias")[0].path.endswith("bias0000.cr3")
    assert manifest.by_role("target") == []


def test_scan_folder_ignores_non_raw_files(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "bias0000.cr3").write_bytes(b"x")
    monkeypatch.setattr(manual.rawmod, "load", lambda path: _frame_with_signal(5, 0.0005))
    manifest = manual.scan_folder(tmp_path)
    assert len(manifest.entries) == 1

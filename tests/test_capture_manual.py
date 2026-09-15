from dataclasses import replace
from pathlib import Path

import numpy as np

from calsuite import raw as rawmod
from calsuite.capture import manual
from calsuite.raw import FrameMeta, RawFrame
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
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "bias0000.cr3").write_bytes(b"x")
    monkeypatch.setattr(manual.rawmod, "load", lambda path: _frame_with_signal(5, 0.0005))
    manifest = manual.scan_folder(tmp_path)
    assert len(manifest.entries) == 1


def test_scan_folder_caches_the_decoded_frame_no_double_load(tmp_path, monkeypatch):
    (tmp_path / "bias0000.cr3").write_bytes(b"not a real raw file")
    frame = _frame_with_signal(5, 0.0005)
    calls = []

    def fake_load(path):
        calls.append(path)
        return frame

    monkeypatch.setattr(manual.rawmod, "load", fake_load)
    manifest = manual.scan_folder(tmp_path)
    assert len(calls) == 1  # scan_folder decoded it exactly once
    assert manifest.entries[0].frame is frame  # ...and handed that same object back on the entry


def _slanted_edge_frame(black_dn=512.0, exposure_s=0.01) -> RawFrame:
    """A crude but genuinely bimodal target: half the visible area near
    black, half near white, mean landing near 50% of the DN range -- lands
    in classify()'s "flat" DN band by mean alone, the exact case fix list
    item 6 calls out (a slanted-edge target at ~50% signal misclassified
    as "flat"). No PRNU/DSNU (irrelevant to this check; the bimodal split
    itself is what should trip the CV heuristic).
    """
    rows, cols = 64, 96
    top, left = 8, 16
    white = black_dn + 20000.0
    cfa = np.full((rows + top, cols + left), black_dn, dtype=np.float64)
    visible = cfa[top:, left:]
    visible[:, : cols // 2] = white  # left half bright, right half at black
    return RawFrame(
        cfa=cfa.astype(np.uint16),
        pattern="RGGB",
        visible=(slice(top, top + rows), slice(left, left + cols)),
        black_level=(black_dn,) * 4,
        white_level=white,
        meta=FrameMeta(exposure_s=exposure_s),
        path="<synthetic-edge>",
        sha256="",
    )


def test_classify_slanted_edge_target_not_misclassified_as_flat():
    frame = _slanted_edge_frame()
    assert manual.classify(frame) == "target"


def test_classify_expected_role_overrides_heuristic():
    frame = _slanted_edge_frame()
    assert manual.classify(frame, expected_role="flat") == "flat"


def test_scan_folder_expected_role_applies_to_every_frame(tmp_path, monkeypatch):
    for name in ("edge0000.cr3", "edge0001.cr3"):
        (tmp_path / name).write_bytes(b"not a real raw file")
    monkeypatch.setattr(manual.rawmod, "load", lambda path: _slanted_edge_frame())
    manifest = manual.scan_folder(tmp_path, expected_role="target")
    assert {e.role for e in manifest.entries} == {"target"}


def test_scan_folder_accepts_npz_frames(tmp_path):
    model = synth_sensor.SensorModel(shape=(48, 64), black_dn=512.0, prnu_std=0.0, dsnu_std_e_per_s=0.0, hot_pixel_fraction=0.0)
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=1e-4, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    rawmod.save_npz(frame, tmp_path / "bias0000.npz")

    manifest = manual.scan_folder(tmp_path)
    assert len(manifest.entries) == 1
    assert manifest.entries[0].role == "bias"
    assert manifest.entries[0].frame is not None


def test_near_black_frame_without_an_exposure_time_is_undecided_not_a_dark():
    """`raw._read_metadata` returns an empty FrameMeta when neither exiftool
    nor dcraw is installed -- a supported configuration ("pixels still
    loaded, metadata just empty"). A near-black frame then has no exposure
    time, and calling it "dark" made `camera bias --from DIR` report "no
    bias frames found" on a folder of perfectly good bias frames, while
    `camera darks` counted those same frames as darks. Bias and dark are
    the same picture without an exposure time to tell them apart.
    """
    model = synth_sensor.SensorModel(shape=(64, 64), black_dn=512.0, read_noise_e=3.0)
    rng = np.random.default_rng(0)
    frame = synth_sensor.frame(model, exposure_s=0.001, flux_e_per_s=0.0, temp_c=20.0, rng=rng)

    assert manual.classify(frame) == "bias"  # with its exposure time, it's a bias

    stripped = replace(frame, meta=rawmod.FrameMeta(model=frame.meta.model))
    assert stripped.meta.exposure_s is None
    assert manual.classify(stripped) == "near_black"

    # A long exposure with metadata is still a dark.
    long_dark = synth_sensor.frame(model, exposure_s=30.0, flux_e_per_s=0.0, temp_c=20.0, rng=rng)
    assert manual.classify(long_dark) == "dark"

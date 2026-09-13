from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from calsuite.fit import Analysis
from calsuite.store import (
    ArtifactTamperedError,
    ExportRefused,
    Record,
    RecordCorruptError,
    SCHEMA_VERSION,
    SHELF_LIFE_DAYS,
    Store,
    UnsupportedSchemaError,
    require_exportable,
    utcnow_stamp,
)


def _device():
    return {"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234", "firmware": ""}


def test_record_defaults_and_created_autofill():
    r = Record(schema=1, id="x", kind="camera.bias", device=_device())
    assert r.status == "ok"
    assert r.provenance == "nominal"
    assert r.created  # auto-filled by __post_init__


def test_record_rejects_bad_provenance():
    with pytest.raises(ValueError):
        Record(schema=1, id="x", kind="k", device=_device(), provenance="guessed")


def test_record_rejects_bad_status():
    with pytest.raises(ValueError):
        Record(schema=1, id="x", kind="k", device=_device(), status="maybe")


def test_record_to_dict_from_dict_round_trip():
    r = Record(schema=1, id="x", kind="camera.bias", device=_device(), result={"gain": 2.0})
    r2 = Record.from_dict(r.to_dict())
    assert r2 == r


def test_record_from_analysis_ok():
    a = Analysis(result={"gain_e_per_dn": 2.0})
    r = Record.from_analysis(
        kind="camera.ptc", device=_device(), analysis=a, provenance="measured", method={"name": "ptc"}
    )
    assert r.status == "ok"
    assert r.result["gain_e_per_dn"] == 2.0


def test_record_from_analysis_refused():
    a = Analysis()
    a.refuse("clipping", "channel saturated")
    r = Record.from_analysis(
        kind="camera.ptc", device=_device(), analysis=a, provenance="measured", method={"name": "ptc"}
    )
    assert r.status == "refused"
    assert r.refusals[0]["check"] == "clipping"


def test_store_save_and_load_round_trip(tmp_path):
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.bias",
        device=_device(),
        analysis=Analysis(result={"black_dn": 512.0}),
        provenance="measured",
        method={"name": "bias"},
    )
    path = store.save(r)
    assert path.exists()
    loaded = store.load(path)
    assert loaded.id == r.id
    assert loaded.result == {"black_dn": 512.0}


def test_store_save_with_artifacts(tmp_path):
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.fixed_pattern", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "fp"}
    )
    prnu_map = np.ones((4, 4))
    store.save(r, artifacts={"prnu": {"map": prnu_map}})
    assert len(r.artifacts) == 1
    loaded_arrays = store.load_artifact(r, "prnu")
    assert np.array_equal(loaded_arrays["map"], prnu_map)


def test_store_all_and_latest(tmp_path):
    store = Store(tmp_path)
    device = _device()
    r1 = Record.from_analysis(
        kind="camera.bias", device=device, analysis=Analysis(), provenance="measured", method={"name": "bias"}
    )
    store.save(r1)
    r2 = Record.from_analysis(
        kind="camera.bias", device=device, analysis=Analysis(), provenance="measured", method={"name": "bias"}
    )
    store.save(r2)

    all_records = list(store.all(kind="camera.bias", device_id=device["id"]))
    assert len(all_records) == 2

    latest = store.latest("camera.bias", device["id"])
    assert latest is not None
    assert latest.id in (r1.id, r2.id)


def test_store_iterate_is_all():
    assert Store.iterate is Store.all


def test_store_latest_returns_none_when_empty(tmp_path):
    store = Store(tmp_path)
    assert store.latest("camera.bias", "nope") is None


def test_store_all_on_missing_root_yields_nothing(tmp_path):
    store = Store(tmp_path / "does-not-exist")
    assert list(store.all()) == []


def test_require_exportable_ok():
    a = Analysis(result={"x": 1})
    r = Record.from_analysis(kind="camera.ptc", device=_device(), analysis=a, provenance="measured", method={"name": "ptc"})
    require_exportable(r)  # must not raise


def test_require_exportable_refuses_bad_status():
    a = Analysis()
    a.refuse("clipping", "saturated")
    r = Record.from_analysis(kind="camera.ptc", device=_device(), analysis=a, provenance="measured", method={"name": "ptc"})
    with pytest.raises(ExportRefused):
        require_exportable(r)


def test_require_exportable_refuses_weak_provenance():
    a = Analysis(result={"x": 1})
    r = Record.from_analysis(kind="camera.ptc", device=_device(), analysis=a, provenance="nominal", method={"name": "ptc"})
    with pytest.raises(ExportRefused):
        require_exportable(r)


def test_shelf_life_values_are_positive():
    for kind, days in SHELF_LIFE_DAYS.items():
        assert days > 0, kind


def test_display_nominal_shelf_life_is_effectively_permanent():
    # An EDID reading doesn't go stale with time alone -- only a panel swap
    # (doctor.py's EDID-hash check) invalidates it.
    assert SHELF_LIFE_DAYS["display.nominal"] > 365 * 5


def test_is_stale(tmp_path):
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.darks", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "darks"}
    )
    old_created = (
        datetime.now(timezone.utc) - timedelta(days=SHELF_LIFE_DAYS["camera.darks"] + 10)
    ).strftime("%Y%m%dT%H%M%SZ")
    r.created = old_created
    assert store.is_stale(r)

    r.created = utcnow_stamp()
    assert not store.is_stale(r)


# ---------------------------------------------------------------------------
# robustness: corrupt/truncated records, tampered artifacts, schema
# mismatches, duplicate timestamps, a read-only records dir
# ---------------------------------------------------------------------------


def test_store_load_truncated_json_raises_recordcorrupterror_with_path(tmp_path):
    device_dir = tmp_path / "canon-eos-r100-abcd1234"
    device_dir.mkdir()
    bad = device_dir / "camera.bias-20260101T000000Z-aaaaaa.json"
    bad.write_text('{"schema": 1, "id": "x", "kind": "cam')  # truncated mid-write
    store = Store(tmp_path)
    with pytest.raises(RecordCorruptError) as excinfo:
        store.load(bad)
    assert str(bad) in str(excinfo.value)


def test_store_all_raises_recordcorrupterror_on_one_bad_file_not_a_bare_jsondecodeerror(tmp_path):
    # A single corrupt record file anywhere under records/ must not surface
    # as an unlabelled json.JSONDecodeError -- every downstream consumer of
    # Store.all() (doctor.py's checks, every report) needs to know *which*
    # file is unreadable, not just that parsing failed somewhere.
    store = Store(tmp_path)
    ok = Record.from_analysis(
        kind="camera.bias", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "bias"}
    )
    store.save(ok)
    bad_path = store._device_dir(_device()["id"]) / "camera.bias-99999999T000000Z-bad000.json"
    bad_path.write_text("{not valid json")
    with pytest.raises(RecordCorruptError) as excinfo:
        list(store.all())
    assert "bad000" in str(excinfo.value) or str(bad_path) in str(excinfo.value)


def test_record_from_dict_rejects_schema_newer_than_supported():
    d = {
        "schema": SCHEMA_VERSION + 1,
        "id": "x",
        "kind": "camera.bias",
        "device": _device(),
    }
    with pytest.raises(UnsupportedSchemaError):
        Record.from_dict(d)


def test_record_from_dict_rejects_unknown_fields_as_corrupt_not_a_bare_typeerror():
    d = {
        "schema": SCHEMA_VERSION,
        "id": "x",
        "kind": "camera.bias",
        "device": _device(),
        "this_field_does_not_exist": True,
    }
    with pytest.raises(RecordCorruptError):
        Record.from_dict(d)


def test_store_load_artifact_detects_tampered_sidecar(tmp_path):
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.fixed_pattern", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "fp"}
    )
    store.save(r, artifacts={"prnu": {"map": np.ones((4, 4))}})

    # Tamper with the sidecar after save() recorded its hash -- e.g. disk
    # corruption, or the file being replaced out from under the record.
    artifact_path = store._artifact_path(r, "prnu")
    np.savez_compressed(artifact_path, map=np.zeros((4, 4)))

    with pytest.raises(ArtifactTamperedError):
        store.load_artifact(r, "prnu")


def test_store_load_artifact_untampered_sidecar_still_loads(tmp_path):
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.fixed_pattern", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "fp"}
    )
    prnu_map = np.ones((4, 4)) * 3.0
    store.save(r, artifacts={"prnu": {"map": prnu_map}})
    loaded = store.load_artifact(r, "prnu")
    assert np.array_equal(loaded["map"], prnu_map)


def test_store_latest_with_identical_timestamps_picks_one_deterministically(tmp_path):
    # Two records of the same kind/device created in the same second
    # (`created` is second-resolution -- utcnow_stamp) have no ordering
    # information beyond that; `latest()` must not crash and must return
    # one of them consistently (Python's max() keeps the first-seen
    # maximum), not silently prefer whichever happens to have a larger
    # random id suffix.
    store = Store(tmp_path)
    device = _device()
    stamp = "20260101T000000Z"
    r1 = Record(schema=1, id=f"camera.bias-{stamp}-aaaaaa", kind="camera.bias", device=device, created=stamp)
    r2 = Record(schema=1, id=f"camera.bias-{stamp}-bbbbbb", kind="camera.bias", device=device, created=stamp)
    store.save(r1)
    store.save(r2)

    latest = store.latest("camera.bias", device["id"])
    assert latest is not None
    assert latest.created == stamp
    assert latest.id in (r1.id, r2.id)
    # Deterministic: sorted(glob) always yields aaaaaa before bbbbbb, and
    # max() over a tie keeps the first-seen element.
    assert latest.id == r1.id


def test_store_save_to_read_only_records_dir_raises_clearly(tmp_path):
    device_dir = tmp_path / _device()["id"]
    device_dir.mkdir(parents=True)
    device_dir.chmod(0o500)  # read + execute, no write
    store = Store(tmp_path)
    r = Record.from_analysis(
        kind="camera.bias", device=_device(), analysis=Analysis(), provenance="measured", method={"name": "bias"}
    )
    try:
        with pytest.raises(OSError):
            store.save(r)
    finally:
        device_dir.chmod(0o700)  # restore so tmp_path cleanup can remove it

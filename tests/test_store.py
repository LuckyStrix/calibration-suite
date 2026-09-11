from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from calsuite.fit import Analysis
from calsuite.store import ExportRefused, Record, SHELF_LIFE_DAYS, Store, require_exportable, utcnow_stamp


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

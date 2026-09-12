"""The suite's first real record: a ``display.nominal`` for this machine's
laptop panel, written by ``calsuite devices`` and committed under
``records/`` (docs/implementation-plan.md Wave 3 Build item). This just
checks it parses through the same ``Store``/``Record`` path every other
record does, and that -- being a public-repo, EDID-derived record --
nothing that looks like a raw per-unit serial number made it into the
file. It does not (and should not) assert particular chromaticity values:
those describe *this* laptop's panel, not the suite.
"""

from __future__ import annotations

import json

from calsuite import config, store as storemod

RECORDS_DIR = config.REPO_ROOT / "records"


def _committed_record_paths() -> list:
    if not RECORDS_DIR.is_dir():
        return []
    return sorted(p for p in RECORDS_DIR.glob("*/*.json"))


def test_at_least_one_real_record_is_committed():
    assert _committed_record_paths(), f"expected at least one committed record under {RECORDS_DIR}"


def test_committed_records_parse_via_the_store():
    st = storemod.Store(RECORDS_DIR)
    for path in _committed_record_paths():
        record = st.load(path)
        assert record.schema == 1
        assert record.kind
        assert record.device.get("id")


def test_committed_display_nominal_record_has_nominal_provenance_and_no_serial():
    found_display_nominal = False
    for path in _committed_record_paths():
        data = json.loads(path.read_text())
        if data["kind"] != "display.nominal":
            continue
        found_display_nominal = True
        assert data["provenance"] == "nominal"
        # No raw per-unit serial anywhere in the file -- devices.py never
        # writes EDIDInfo.serial_number/serial_string into a record; this
        # is the check that promise actually holds for the committed file,
        # not just for the code path that produced it.
        text = json.dumps(data).lower()
        assert "serial" not in text
    assert found_display_nominal, "expected a committed display.nominal record"

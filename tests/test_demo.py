"""End-to-end test of ``calsuite demo``: every phase runs on synthetic
data, with no camera/instrument/ArgyllCMS, and every report gets written.
Uses the same small synthetic sizes demo.py's own generators use, so this
stays fast -- ``run_demo`` itself is run once (module-scoped fixture) and
every test below just asserts against that one result, rather than paying
for a fresh multi-phase run per assertion.
"""

from __future__ import annotations

import json
import os

import pytest

from calsuite import demo, store as storemod


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("calsuite-demo")
    return demo.run_demo(out_dir), out_dir


def test_index_written_and_links_every_report(demo_result):
    result, _out_dir = demo_result
    assert result["index"].exists()
    index_html = result["index"].read_text()
    assert "calsuite demo" in index_html


@pytest.mark.parametrize(
    "name",
    [
        "Sensor",
        "Lens",
        "Lens (refusal example: narrow pose coverage)",
        "Colour (Tier A -- chart matrix)",
        "Colour (Tier B -- spectral)",
        "Display",
    ],
)
def test_every_area_report_was_produced(demo_result, name):
    result, _out_dir = demo_result
    path = result["reports"].get(name)
    assert path is not None, f"{name} report was not produced; warnings={result['warnings']}"
    assert path.exists()
    assert path.stat().st_size > 500


def test_store_has_every_expected_record_kind(demo_result):
    _result, out_dir = demo_result
    st = storemod.Store(out_dir / "records")
    kinds = {r.kind for r in st.all()}
    expected = {
        "camera.bias", "camera.ptc", "camera.linearity", "camera.darks", "camera.iso",
        "lens.distortion", "lens.tca", "lens.flats", "lens.mtf", "lens.psf",
        "camera.color",
        "display.measurement", "display.profile", "display.validation",
    }
    missing = expected - kinds
    assert not missing, f"missing record kinds: {missing}"


def test_records_are_valid_json_with_a_known_provenance(demo_result):
    _result, out_dir = demo_result
    paths = list((out_dir / "records").glob("*/*.json"))
    assert paths
    for p in paths:
        data = json.loads(p.read_text())
        assert data["schema"] == 1
        assert data["provenance"] in ("measured", "derived", "vendor", "nominal")


def test_run_produces_no_warnings(demo_result):
    # The primary lens.distortion pose set is deliberately built to pass
    # its own coverage check (broad random views plus deliberate edge/
    # corner poses); the *second*, deliberately narrow pose set is
    # expected to refuse -- and does, on its own device id, with its own
    # report -- so nothing here should surface as a warning.
    result, _out_dir = demo_result
    assert result["warnings"] == []


def test_primary_distortion_passes_and_refusal_example_is_refused(demo_result):
    _result, out_dir = demo_result
    st = storemod.Store(out_dir / "records")
    from calsuite import devices as devicesmod
    from calsuite.demo import LENS_MODEL, LENS_MODEL_REFUSAL_EXAMPLE

    primary = st.latest("lens.distortion", devicesmod.device_id(LENS_MODEL, None))
    refused = st.latest("lens.distortion", devicesmod.device_id(LENS_MODEL_REFUSAL_EXAMPLE, None))
    assert primary is not None and primary.status == "ok"
    assert refused is not None and refused.status == "refused"
    assert any(r["check"] == "coverage" for r in refused.refusals)


def test_run_demo_restores_the_records_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", "/should/not/be/touched")
    demo.run_demo(tmp_path)
    assert os.environ["CALSUITE_RECORDS"] == "/should/not/be/touched"

"""Command-level wiring checks for ``lens/commands.py``.

These sit above the pure-analysis unit tests (``test_lens_flats.py`` etc.):
they exercise ``calsuite lens export``/``calsuite lens report`` through the
real ``cli.main`` + ``Store``, the same way ``test_camera_commands.py`` and
``test_display_commands.py`` do for their areas, to catch a command that
forgets to load or forward a record kind it has -- a bug the pure-function
tests (which only ever call ``export_lensfun.export_records``/
``report.render_lens_report`` directly with hand-built records) can't see.
"""

from __future__ import annotations

from calsuite import cli, store
from calsuite.fit import Analysis


def _device() -> dict:
    return {"kind": "lens", "model": "Test Lens 50mm", "id": "test-lens-50mm-unknown", "firmware": ""}


def _save_ok_flats_record(records_dir, *, aperture: float = 1.8) -> None:
    st = store.Store(records_dir)
    analysis = Analysis(
        result={
            "v_coeffs": [-0.35, 0.05],
            "s_coeffs": [0.0, 0.15, -0.10, -0.08, 0.02, -0.05],
            "image_shape": [180, 240],
            "n_poses": 4,
            "condition_number": 12.0,
            "poses": [],
        },
        residuals={"log_domain_rms": 0.01},
    )
    record = store.Record.from_analysis(
        kind="lens.flats",
        device=_device(),
        analysis=analysis,
        provenance="measured",
        method={"name": "flats.self_calibrate_flat"},
        conditions={"focal_mm": 50.0, "aperture": aperture},
    )
    # Mirrors what `_cmd_flats` adds after a successful fit (lens/commands.py):
    # the refit-to-lensfun's-"pa"-model dict, computed from v_coeffs.
    record.result["pa"] = {"k1": -0.34, "k2": 0.06, "k3": -0.01, "residual_rms": 0.001}
    st.save(record)


def test_lens_export_includes_vignetting_from_a_saved_flats_record(tmp_path, monkeypatch):
    """`calsuite lens export` must pick up an existing `ok` `lens.flats`
    record and write it as a `<vignetting>` element -- `export_lensfun.
    export_records` already accepts `flats_records` and turns each into one,
    but `_cmd_export` never fetched or passed them, so real vignetting data
    silently never reached the exported XML even when a flats session had
    already produced it."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    _save_ok_flats_record(records_dir, aperture=1.8)

    out_path = tmp_path / "out.xml"
    rc = cli.main(
        [
            "lens",
            "export",
            "--device-id",
            "test-lens-50mm-unknown",
            "--lens-model",
            "Test Lens 50mm",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    xml_text = out_path.read_text(encoding="utf-8")
    assert "<vignetting" in xml_text
    assert 'model="pa"' in xml_text
    assert 'aperture="1.8"' in xml_text
    assert 'k1="-0.340000"' in xml_text


def test_lens_report_includes_vignetting_section_for_a_saved_flats_record(tmp_path, monkeypatch):
    """Same wiring gap as the export command: `render_lens_report` already
    renders a "Vignetting" section whenever it's given
    `flats_records_by_aperture`, but `_cmd_report` never built that dict, so
    a lens report never showed vignetting even with an `ok` `lens.flats`
    record on file."""
    records_dir = tmp_path / "records"
    monkeypatch.setenv("CALSUITE_RECORDS", str(records_dir))
    _save_ok_flats_record(records_dir, aperture=2.8)

    out_path = tmp_path / "report.html"
    rc = cli.main(
        [
            "lens",
            "report",
            "--device-id",
            "test-lens-50mm-unknown",
            "--lens-model",
            "Test Lens 50mm",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    html = out_path.read_text(encoding="utf-8")
    assert "Vignetting" in html

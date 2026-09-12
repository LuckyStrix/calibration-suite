from calsuite.fit import Analysis
from calsuite.lens import report as R
from calsuite.store import Record


def _device():
    return {"kind": "lens", "model": "Test Lens 50mm", "id": "test-lens-50mm-unknown", "firmware": ""}


def test_render_lens_report_with_no_records_still_renders():
    html = R.render_lens_report(device=_device())
    assert "<html" in html
    assert "Test Lens 50mm" in html


def test_render_lens_report_includes_available_sections():
    distortion_analysis = Analysis(
        result={"ptlens": {"a": 0.001, "b": -0.004, "c": 0.002}},
        residuals={
            "overall_rms_px": 0.12,
            "coverage_grid": [[1, 2], [3, 4]],
            "per_view_names": ["a.cr3", "b.cr3"],
            "per_view_rms_px": [0.1, 0.14],
        },
    )
    distortion_record = Record.from_analysis(
        kind="lens.distortion", device=_device(), analysis=distortion_analysis, provenance="measured",
        method={"name": "distortion.fit_distortion"},
    )
    tca_analysis = Analysis(result={"model": "poly3", "vr": 1.0002, "vb": 0.9998, "kr": 1.0002, "kb": 0.9998})
    tca_record = Record.from_analysis(
        kind="lens.tca", device=_device(), analysis=tca_analysis, provenance="measured", method={"name": "tca.fit_tca"}
    )
    mtf_analysis = Analysis(result={"mtf50_lp_per_mm_map": [[30.0, 28.0], [25.0, 22.0]]})
    mtf_record = Record.from_analysis(
        kind="lens.mtf", device=_device(), analysis=mtf_analysis, provenance="measured", method={"name": "mtf"}
    )
    psf_analysis = Analysis(
        result={
            "stars": [
                {"x": 10.0, "y": 10.0, "sigma_major": 2.0, "sigma_minor": 1.0, "orientation_deg": 10.0, "orientation": "sagittal"},
                {"x": 90.0, "y": 40.0, "sigma_major": 3.0, "sigma_minor": 2.5, "orientation_deg": -20.0, "orientation": "meridional"},
            ]
        }
    )
    psf_record = Record.from_analysis(
        kind="lens.psf", device=_device(), analysis=psf_analysis, provenance="measured", method={"name": "psf"}
    )

    html = R.render_lens_report(
        device=_device(),
        distortion_record=distortion_record,
        tca_record=tca_record,
        mtf_record=mtf_record,
        psf_record=psf_record,
    )
    assert "Coverage" in html
    assert "Chromatic aberration" in html
    assert "MTF50" in html
    assert "PSF field map" in html
    assert "<ellipse" in html


def test_render_lens_report_shows_refusal_box_for_refused_distortion():
    a = Analysis()
    a.refuse("coverage", "outer field not covered")
    record = Record.from_analysis(
        kind="lens.distortion", device=_device(), analysis=a, provenance="measured", method={"name": "x"}
    )
    html = R.render_lens_report(device=_device(), distortion_record=record)
    assert "Refused" in html


def test_render_lens_report_with_empty_result_dicts_does_not_raise():
    # Regression guard: a record can be *present* (distortion/tca/mtf/psf
    # all supplied) with an *empty* `.result` -- a refusal that returned
    # before computing anything, or an older/partial record. Every section
    # here must degrade to "nothing to show" rather than raising on a
    # missing key.
    def _empty(kind, method_name):
        return Record.from_analysis(
            kind=kind, device=_device(), analysis=Analysis(result={}), provenance="measured",
            method={"name": method_name},
        )

    html = R.render_lens_report(
        device=_device(),
        distortion_record=_empty("lens.distortion", "distortion.fit_distortion"),
        tca_record=_empty("lens.tca", "tca.fit_tca"),
        mtf_record=_empty("lens.mtf", "mtf"),
        psf_record=_empty("lens.psf", "psf"),
    )
    assert html.startswith("<!doctype html>")


def test_tca_chart_plots_out_to_the_sensor_corner_not_a_fixed_short_range():
    """tca.fit_tca fits kr/kb against raw sensor-pixel radii (a dimensionless
    ratio -- see tca.py's module docstring), which run out to a couple
    thousand px at a real sensor's corner. The chart used to always plot
    over a fixed 0-1.2 range (a leftover Hugin-normalized-looking span this
    module never actually uses), so a real few-parts-in-a-thousand kr/kb
    produced an "offset" of a few thousandths of a pixel over that tiny
    span -- indistinguishable from zero, understating real chromatic
    aberration that's several px wide at the sensor's actual corner. With
    ``image_size`` on the record (as ``tca.fit_tca`` now always includes),
    the chart's x-axis must reach out near the image's own half-diagonal,
    not stop at 1.2."""
    from calsuite.lens import tca as T

    image_size = (6000, 4000)
    tca_analysis = Analysis(
        result={
            "model": "poly3", "vr": 1.0025, "vb": 0.9975, "kr": 1.0025, "kb": 0.9975,
            "image_size": list(image_size),
        }
    )
    tca_record = Record.from_analysis(
        kind="lens.tca", device=_device(), analysis=tca_analysis, provenance="measured", method={"name": "tca.fit_tca"}
    )
    html = R.render_lens_report(device=_device(), tca_record=tca_record)
    half_diag = T.half_diagonal(image_size)
    assert half_diag > 100  # sanity: this is a real few-thousand-px sensor
    # The chart's x-domain must reach out near the actual corner radius --
    # not stay pinned to the old fixed [0, 1.2] span regardless of sensor
    # size. The x-axis's rightmost tick label is exactly the series' own max
    # x value (report/svg.py's `_nice_ticks` includes the range's upper
    # bound), formatted the same way (`"{:.3g}"`) as here.
    assert f"{half_diag:.3g}" in html


def test_render_lens_report_error_budget_shows_not_measured_when_rms_absent():
    # `distortion_record.residuals.get("overall_rms_px")` can be `None`
    # (residuals is an empty dict for an early refusal); the error-budget
    # table used to splice that straight in as the literal text "None".
    a = Analysis(result={"ptlens": {"a": 0.0, "b": 0.0, "c": 0.0}})  # residuals stay empty
    record = Record.from_analysis(
        kind="lens.distortion", device=_device(), analysis=a, provenance="measured", method={"name": "x"}
    )
    html = R.render_lens_report(device=_device(), distortion_record=record)
    assert "not measured" in html
    assert ">None<" not in html

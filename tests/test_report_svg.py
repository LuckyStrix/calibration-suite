import xml.etree.ElementTree as ET

from calsuite.report import svg


def _parse(svg_string):
    return ET.fromstring(svg_string)


def test_line_chart_is_valid_xml():
    s = svg.line_chart(
        [{"name": "gain", "x": [1, 2, 3, 4], "y": [10, 20, 15, 30]}],
        title="Test",
        x_label="x",
        y_label="y",
    )
    root = _parse(s)
    assert root.tag.endswith("svg")


def test_line_chart_multi_series_has_legend_and_both_names():
    s = svg.line_chart(
        [
            {"name": "A", "x": [1, 2, 3], "y": [1, 2, 3]},
            {"name": "B", "x": [1, 2, 3], "y": [3, 2, 1]},
        ]
    )
    _parse(s)
    assert "A" in s and "B" in s


def test_line_chart_log_axes_are_valid_xml():
    s = svg.line_chart(
        [{"name": "s", "x": [1, 10, 100, 1000], "y": [0.1, 1, 10, 100]}],
        log_x=True,
        log_y=True,
    )
    _parse(s)


def test_line_chart_empty_series_still_valid_xml():
    _parse(svg.line_chart([]))


def test_line_chart_drops_none_points_instead_of_raising():
    # min()/max() over a list containing None raises TypeError the moment
    # Python tries to compare it against a real number -- a None point (an
    # optional per-sample value a partial/refused record doesn't have for
    # every sample) must be dropped, not passed through to axis-bounds math.
    s = svg.line_chart([{"name": "s", "x": [1, 2, 3, 4], "y": [1.0, None, 3.0, None]}])
    _parse(s)
    assert "<path" in s or "<circle" in s


def test_line_chart_all_none_points_is_the_empty_chart_not_a_crash():
    _parse(svg.line_chart([{"name": "s", "x": [1, 2], "y": [None, None]}]))


def test_scatter_chart_is_valid_xml_and_uses_circles():
    s = svg.scatter_chart([{"name": "pts", "x": [1, 2, 3], "y": [4, 5, 6]}])
    root = _parse(s)
    assert root.tag.endswith("svg")
    assert "<circle" in s


def test_heatmap_is_valid_xml_with_colorbar():
    grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    s = svg.heatmap(grid, title="Heat", colorbar=True)
    _parse(s)
    assert "Heat" in s


def test_heatmap_without_colorbar_is_valid_xml():
    _parse(svg.heatmap([[0, 1], [1, 0]], colorbar=False))


def test_heatmap_respects_explicit_value_range():
    s1 = svg.heatmap([[0, 10]], value_range=(0, 10))
    s2 = svg.heatmap([[0, 10]], value_range=(-10, 20))
    _parse(s1)
    _parse(s2)
    assert s1 != s2  # different ranges must produce different fills


def test_heatmap_tolerates_none_cells():
    # lens.mtf_field_grid's map carries None for a field-grid cell whose
    # own edge check failed -- heatmap() must render those as a flat "no
    # data" cell, not crash comparing None against a number.
    s = svg.heatmap([[0.5, None, 1.5], [None, None, None]])
    _parse(s)
    assert "no data" in s


def test_bar_chart_is_valid_xml():
    s = svg.bar_chart([("gain", 2.5), ("read noise", 3.0, "3.0 e-")])
    _parse(s)
    assert "gain" in s


def test_bar_chart_empty_returns_empty_string():
    assert svg.bar_chart([]) == ""


def test_bar_chart_none_value_renders_not_measured_instead_of_raising():
    # float(None) raises TypeError -- a None value (mean/p95/max ΔE00 that
    # a validation refused before computing) must render as a zero-length
    # "not measured" bar, not crash the whole report.
    s = svg.bar_chart([("mean ΔE00", 1.0), ("p95 ΔE00", None)])
    _parse(s)
    assert "not measured" in s


def test_escaping_of_untrusted_text():
    s = svg.line_chart(
        [{"name": "<script>alert(1)</script>", "x": [1, 2], "y": [1, 2]}],
        title="a & b \"quoted\"",
    )
    root = _parse(s)  # would raise ParseError if unescaped
    assert root.tag.endswith("svg")
    assert "<script>" not in s

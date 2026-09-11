import os

import pytest

from calsuite.display import analysis as analysismod
from calsuite.display import patches as patchesmod
from calsuite.display import profile as profilemod
from calsuite.display import validate as validatemod
from calsuite.formats import cgats
from calsuite.synth.display import DisplayModel


def _make_script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def _ramp_data(model, channel, steps=17):
    idx = {"r": 0, "g": 1, "b": 2}[channel]
    levels = [i / (steps - 1) for i in range(steps)]
    xyz = []
    for level in levels:
        rgb = [0.0, 0.0, 0.0]
        rgb[idx] = level
        xyz.append(list(model.measure(tuple(rgb))))
    return {"levels": levels, "xyz": xyz}


def test_write_ti3_normalizes_to_white_y1(tmp_path):
    rgb_list = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
    xyz_list = [(0.1, 0.1, 0.1), (95.0, 100.0, 108.0), (40.0, 21.0, 2.0)]
    path = tmp_path / "test.ti3"
    profilemod.write_ti3(path, rgb_list, xyz_list)
    result = cgats.read_ti3(path)
    assert len(result.samples) == 3
    white_sample = next(s for s in result.samples if s["rgb"] == (1.0, 1.0, 1.0))
    assert white_sample["xyz"][1] == pytest.approx(1.0)


def test_write_ti3_raises_without_white_patch(tmp_path):
    with pytest.raises(ValueError):
        profilemod.write_ti3(tmp_path / "x.ti3", [(0.5, 0.5, 0.5)], [(1.0, 1.0, 1.0)])


def test_build_with_colprof_matrix_shaper(tmp_path, monkeypatch):
    _make_script(
        tmp_path,
        "colprof",
        "#!/bin/bash\n"
        'base="${!#}"\n'
        'touch "$base.icc"\n',
    )
    _make_script(tmp_path, "profcheck", "#!/bin/sh\necho ok\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    ti3_base = tmp_path / "display"
    profilemod.write_ti3(ti3_base.with_suffix(".ti3"), [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)], [(95, 100, 108), (0.1, 0.1, 0.1)])
    result = profilemod.build_with_colprof(ti3_base, recommend_lut=False, description="test")
    assert result.method.startswith("colprof -as")
    assert result.path.exists()
    assert result.profcheck_ok is True


def test_build_with_colprof_lut_when_recommended(tmp_path, monkeypatch):
    _make_script(tmp_path, "colprof", '#!/bin/bash\nbase="${!#}"\ntouch "$base.icc"\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    ti3_base = tmp_path / "display"
    profilemod.write_ti3(ti3_base.with_suffix(".ti3"), [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)], [(95, 100, 108), (0.1, 0.1, 0.1)])
    result = profilemod.build_with_colprof(ti3_base, recommend_lut=True, description="test")
    assert result.method == "colprof (LUT)"


def test_build_profile_refuses_when_nonadditive_and_no_colprof(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")
    model = DisplayModel(white_boost_frac=0.3)
    black, r, g, b, w = (model.measure(c) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    additivity_analysis = analysismod.additivity(black, r, g, b, w)
    trc_analysis = analysismod.trc_fit({ch: _ramp_data(model, ch) for ch in ("r", "g", "b")}, black_y=float(black[1]))

    with pytest.raises(profilemod.ProfileRefused):
        profilemod.build_profile(
            tmp_path / "out.icc",
            rgb_list=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)],
            xyz_list=[list(black), list(r), list(g), list(b), list(w)],
            additivity_analysis=additivity_analysis,
            trc_analysis=trc_analysis,
            primaries_measured={"r": r, "g": g, "b": b, "w": w, "k": black},
        )


def test_builtin_fallback_profile_validates_against_synthetic_display(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-bin-dir")  # force the built-in fallback: no colprof
    model = DisplayModel(white_boost_frac=0.0)  # additive, so the fallback is legal

    black, r, g, b, w = (model.measure(c) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)))
    additivity_analysis = analysismod.additivity(black, r, g, b, w)
    assert additivity_analysis.result["recommend_lut"] is False
    trc_analysis = analysismod.trc_fit({ch: _ramp_data(model, ch) for ch in ("r", "g", "b")}, black_y=float(black[1]))
    primaries_measured = {"r": r, "g": g, "b": b, "w": w, "k": black}

    grid = patchesmod.gray_ramp(9) + patchesmod.channel_ramp("r", 9) + patchesmod.channel_ramp("g", 9) + patchesmod.channel_ramp("b", 9)
    grid += [patchesmod.Patch(rgb=(0, 0, 0)), patchesmod.Patch(rgb=(1, 1, 1))]
    rgb_list = [p.rgb for p in grid]
    xyz_list = [list(model.measure(p.rgb)) for p in grid]

    out_path = tmp_path / "builtin.icc"
    result = profilemod.build_profile(
        out_path,
        rgb_list=rgb_list,
        xyz_list=xyz_list,
        additivity_analysis=additivity_analysis,
        trc_analysis=trc_analysis,
        primaries_measured=primaries_measured,
    )
    assert result.method == "built-in matrix/TRC"
    assert result.path.exists()

    val_patches = patchesmod.validation_set()
    lab_targets = [p.lab_target for p in val_patches]
    rgb_values = validatemod.lab_to_rgb_via_profile(result.path, lab_targets)
    measured_xyz = [model.measure(rgb) for rgb in rgb_values]

    analysis = validatemod.validate(measured_xyz, lab_targets, w, accuracy_de00_estimate=2.0)
    assert analysis.result["de00_mean"] < 3.0
    assert analysis.ok, [r for r in analysis.refusals]

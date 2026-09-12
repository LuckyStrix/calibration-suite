"""DIY spectrophotometer backend (docs/design.md §5.1/§6): read a spectrum
file (wavelength_nm, value) per patch, integrate against the CIE 1931 2°
standard observer to get XYZ. `value` is whatever relative or radiometric
unit the spectrophotometer's own calibration produces (design's spectro
plan, §6); this module applies no further radiometric scale of its own --
it only does the CMF integration, which is backend-agnostic.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calsuite.display.backends.base import Accuracy, Measurement

_CMFS_NAME = "CIE 1931 2 Degree Standard Observer"


def read_spectrum_csv(path: Path | str) -> tuple:
    """Read a two-column (wavelength_nm, value) CSV. Any row that doesn't
    parse as two floats (e.g. a header) is skipped rather than raising --
    spectrophotometer export tools vary in whether they include one."""
    wavelengths, values = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                wavelengths.append(float(row[0]))
                values.append(float(row[1]))
            except ValueError:
                continue
    if not wavelengths:
        raise ValueError(f"{path}: no numeric (wavelength, value) rows found")
    return np.asarray(wavelengths, dtype=np.float64), np.asarray(values, dtype=np.float64)


def spectrum_to_xyz(wavelengths, values) -> np.ndarray:
    """Integrate an emissive spectral power distribution against the CIE
    1931 2° CMFs (colour-science's `MSDS_CMFS["CIE 1931 2 Degree Standard
    Observer"]`, 360-830nm at 1nm steps) via the trapezoidal rule. The
    input spectrum is linearly interpolated onto the CMF wavelength grid
    (zero outside the measured range) before integrating -- a DIY
    spectrophotometer's native sampling rarely matches the CMF grid
    exactly.
    """
    import colour

    cmfs = colour.MSDS_CMFS[_CMFS_NAME]
    grid = cmfs.wavelengths
    spd = np.interp(grid, wavelengths, values, left=0.0, right=0.0)
    xbar, ybar, zbar = cmfs.values[:, 0], cmfs.values[:, 1], cmfs.values[:, 2]
    x = np.trapezoid(spd * xbar, grid)
    y = np.trapezoid(spd * ybar, grid)
    z = np.trapezoid(spd * zbar, grid)
    return np.array([x, y, z])


@dataclass
class SpectroBackend:
    """`spectra_for_patch`: a callable ``patch -> Path`` locating the CSV
    file for a given patch (design leaves file layout to the caller; a
    typical `display/commands.py` wiring names files by patch label).
    `luminance_scale`: a single multiplicative factor converting the
    spectrometer's raw integrated Y into cd/m^2 -- the spectrophotometer's
    own radiometric calibration (design §6), supplied by the caller since
    this backend has no way to derive it itself.
    """

    spectra_for_patch: object  # Callable[[Patch], Path | str]
    luminance_scale: float = 1.0
    name: str = "spectro"
    cross_checked_against: str | None = None

    def accuracy(self) -> Accuracy:
        return Accuracy(
            de00_estimate=2.0,
            basis="DIY spectrophotometer, emission mode (design §5.1: depends on wavelength accuracy and "
            "radiometric calibration; 2.0 is a placeholder pending that calibration -- see "
            "calsuite.constants.ESTIMATED_ACCURACY for the suite-wide estimate table this should graduate into)",
            cross_checked_against=self.cross_checked_against,
        )

    def measure(self, patches: list) -> list:
        out = []
        for patch in patches:
            wavelengths, values = read_spectrum_csv(self.spectra_for_patch(patch))
            xyz = spectrum_to_xyz(wavelengths, values) * self.luminance_scale
            out.append(Measurement(rgb=patch.rgb, xyz=xyz, uncertainty=np.zeros(3)))
        return out

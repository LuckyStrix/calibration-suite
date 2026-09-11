"""Local-only smoke test: exercises raw.load() against a real Canon R100
CR3 file this build machine happens to have at ~/capt0000.cr3 (docs/design.md
§0). Skipped everywhere else, including CI -- there is no such file there.
"""

import os
from pathlib import Path

import pytest

CAPTURE_PATH = Path(os.path.expanduser("~/capt0000.cr3"))

pytestmark = pytest.mark.skipif(not CAPTURE_PATH.exists(), reason="local-only: ~/capt0000.cr3 not present")


def test_real_r100_raw_loads_with_expected_geometry():
    from calsuite import raw

    frame = raw.load(CAPTURE_PATH)

    total_rows, total_cols = frame.cfa.shape
    assert (total_rows, total_cols) == (4056, 6288)

    # RGGB-family pattern: some rotation/permutation of R, G, G, B.
    assert sorted(frame.pattern) == sorted("RGGB")

    assert frame.meta.iso == 800
    assert frame.meta.exposure_s == pytest.approx(1 / 83.0, rel=0.05)
    # "Lens metadata via the fallback": dcraw's `-i -v` (used here because
    # exiftool isn't installed on this build machine) gives focal length
    # and aperture but never a lens name string -- see raw.py's
    # _metadata_from_dcraw docstring. These two fields are what's actually
    # available about the lens through that fallback.
    assert frame.meta.focal == pytest.approx(50.0)
    assert frame.meta.aperture == pytest.approx(1.8)

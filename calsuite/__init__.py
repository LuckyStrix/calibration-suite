"""calsuite -- camera sensor, lens and display calibration with provenance.

Everything downstream of a photon and a number: the R100's sensor, each
lens, and each display, on Linux and Windows. See docs/design.md for the
full plan and docs/implementation-plan.md for how it's built.
"""

import warnings as _warnings

__version__ = "0.1.0"

# colour-science warns, at import time, that matplotlib isn't installed
# (this project never plots with colour-science -- every chart is
# report/svg.py's own hand-built inline SVG) -- silenced globally, here,
# once, rather than at every one of the many lazy `import colour` call
# sites across camera/lens/display that would otherwise each need to
# repeat the same `warnings.catch_warnings()` dance. This module runs
# before any of them, since every submodule import goes through this
# package's `__init__.py` first.
_warnings.filterwarnings("ignore", message=r".*Matplotlib.* related API features are not available.*")

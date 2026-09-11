"""Display measurement backends (docs/design.md §5.1): ``argyll`` (an
ArgyllCMS-driven colorimeter/spectro via ``spotread``), ``camera`` (a
characterized camera used as a colorimeter), ``spectro`` (the DIY
spectrophotometer's spectra files), and ``synthetic`` (wraps
``calsuite.synth.display`` for tests). Every backend implements the
``base.Backend`` protocol: ``measure(patches) -> (xyz, uncertainty)`` plus
an ``accuracy()`` statement.
"""

from __future__ import annotations

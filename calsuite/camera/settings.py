"""Settings that silently change raw data (docs/design.md §3.1): long-exposure
noise reduction, high-ISO NR, highlight tone priority, and shutter mode, read
from ``FrameMeta.settings``. Every ``analyze_*`` function in this area that
can be corrupted by one of these calls into here rather than re-implementing
its own truthy-parsing of whatever exiftool happened to report.
"""

from __future__ import annotations

from calsuite.camera.constants import _OFF_STRINGS, _ON_STRINGS
from calsuite.fit import Analysis

# FrameMeta.settings keys (raw.py's _metadata_from_exiftool), documented once
# here rather than as string literals scattered through this area.
LENR_KEY = "long_exposure_nr"
HIGH_ISO_NR_KEY = "high_iso_nr"
HTP_KEY = "highlight_tone_priority"
SHUTTER_MODE_KEY = "shutter_mode"


def is_on(value) -> bool | None:
    """Normalize a raw settings value to True/False/None (unknown). Handles
    the plain bool/int a fake test fixture gives and the string spellings
    exiftool's real Canon tags are documented to use (raw.py's own
    disclaimer: those tag names are unverified against a real camera).
    ``None`` covers both "tag absent" (dcraw fallback: no settings at all)
    and any spelling this function doesn't recognize -- both mean "can't
    tell", never "off".
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _ON_STRINGS:
        return True
    if text in _OFF_STRINGS:
        return False
    return None


def setting_on_any(frames, key: str) -> bool:
    """True if any frame in ``frames`` reports ``key`` as on. Used for a
    refusal check: one contaminated frame in a series is enough to corrupt
    it, so "any", not "all" or "majority"."""
    return any(is_on(f.meta.settings.get(key)) is True for f in frames)


def summarize(frames) -> dict:
    """One value per setting for a Record's ``conditions`` -- the *first*
    frame's value, normalized, with ``None`` standing for "unknown/absent"
    (not merged/voted across frames; ``check_measurement_settings`` below is
    what actually notices disagreement between frames)."""
    if not frames:
        return {}
    first = frames[0].meta.settings
    return {
        LENR_KEY: is_on(first.get(LENR_KEY)),
        HIGH_ISO_NR_KEY: is_on(first.get(HIGH_ISO_NR_KEY)),
        HTP_KEY: is_on(first.get(HTP_KEY)),
        SHUTTER_MODE_KEY: first.get(SHUTTER_MODE_KEY),
    }


def check_measurement_settings(
    frames,
    *,
    refuse_if_lenr: bool = False,
    refuse_if_high_iso_nr: bool = False,
    refuse_if_htp: bool = False,
) -> Analysis:
    """A reusable settings check other ``analyze_*`` functions fold into
    their own ``Analysis`` (by extending ``result``/``refusals``) or that
    ``commands.py`` runs standalone. Each ``refuse_if_*`` flag is opt-in per
    caller, because a setting that corrupts one measurement (LENR subtracts
    an in-camera dark, corrupting camera.darks) is irrelevant to another
    (LENR has no bearing on camera.ptc).
    """
    a = Analysis(result={"settings": summarize(frames)})
    if refuse_if_lenr and setting_on_any(frames, LENR_KEY):
        a.refuse(
            "long_exposure_nr_on",
            "long-exposure noise reduction is enabled on one or more frames; it subtracts an "
            "in-camera dark from the raw data, which corrupts this measurement",
            value=True,
            threshold=False,
        )
    if refuse_if_high_iso_nr and setting_on_any(frames, HIGH_ISO_NR_KEY):
        a.refuse(
            "high_iso_nr_on",
            "high-ISO noise reduction is enabled on one or more frames; it is a non-linear, "
            "signal-dependent filter applied before the raw data is written",
            value=True,
            threshold=False,
        )
    if refuse_if_htp and setting_on_any(frames, HTP_KEY):
        a.refuse(
            "highlight_tone_priority_on",
            "highlight tone priority is enabled on one or more frames; it shifts the effective "
            "ISO/gain and remaps the highlight response",
            value=True,
            threshold=False,
        )
    return a

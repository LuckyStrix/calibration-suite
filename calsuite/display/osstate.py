"""OS/display state checks that must happen before a measurement is
trustworthy (docs/design.md §5.3):

- reset the video-card gamma table to linear (``dispwin -c``) and record
  whether that actually happened;
- on Linux/X11, check nothing else (a colord-aware daemon, a manual VCGT
  loader) is already loading a profile, via the ``_ICC_PROFILE`` root
  window property;
- on Windows, best-effort detect HDR / Advanced Color via
  ``DisplayConfigGetDeviceInfo`` -- **unverified on Windows** (see
  ``check_hdr_windows``'s docstring), so any failure falls back to
  requiring an explicit ``--confirm-hdr-off`` from the user;
- always record *which method* determined the HDR state, so a report never
  implies more certainty than it has;
- refuse to measure with HDR on (``refuse_if_hdr_on``);
- carry the OSD brightness/mode settings the user typed in, as part of the
  display's identity for this profile (design: "part of the display's
  identity for this profile").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from calsuite import tools
from calsuite.fit import Refusal


@dataclass(frozen=True)
class OSState:
    platform: str
    gamma_reset: bool
    gamma_reset_method: str
    icc_profile_atom: str | None  # raw xprop output, when a profile/VCGT loader appears active
    profile_loader_warning: str | None
    hdr_on: bool | None  # None = unknown/unverified
    hdr_method: str
    osd: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "gamma_reset": self.gamma_reset,
            "gamma_reset_method": self.gamma_reset_method,
            "icc_profile_atom": self.icc_profile_atom,
            "profile_loader_warning": self.profile_loader_warning,
            "hdr_on": self.hdr_on,
            "hdr_method": self.hdr_method,
            "osd": self.osd,
        }


def reset_gamma_table() -> tuple:
    """``dispwin -c`` resets the video card's gamma LUT to linear (design
    §5.3). Returns ``(done, method)`` -- `done` is False (never raises) if
    `dispwin` isn't on PATH or the call fails, since a missing calibration
    loader shouldn't itself crash a measurement run; the caller decides
    whether an unreset gamma table is acceptable."""
    if tools.which("dispwin") is None:
        return False, "dispwin not found on PATH -- gamma table not reset"
    try:
        tools.run(["dispwin", "-c"])
    except tools.ToolError as exc:
        return False, f"dispwin -c failed: {exc}"
    return True, "dispwin -c"


def check_x11_icc_profile() -> tuple:
    """Query the root window's ``_ICC_PROFILE`` atom via `xprop` (design
    §5.3: "On Linux/X11, check nothing else is loading a VCGT"). Returns
    ``(raw_atom_text_or_None, warning_or_None)``. This laptop's session has
    no colord-aware settings daemon (docs/design.md §0), so on a clean
    session the atom is normally unset -- if it *is* set, something (a
    colord/colormgr daemon, a manual `xcalib`/`dispwin -I` load from a
    previous session, ...) is already applying a profile/VCGT, which could
    silently bias a "raw panel" measurement.
    """
    if tools.which("xprop") is None:
        return None, "xprop not found on PATH -- cannot check for an active ICC/VCGT loader"
    try:
        result = tools.run(["xprop", "-root", "_ICC_PROFILE"], check=False)
    except tools.ToolError as exc:
        return None, f"xprop failed: {exc}"
    text = result.stdout.strip()
    if not text or "not found" in text.lower():
        return None, None  # atom unset -- nothing is loading a profile
    return text, f"_ICC_PROFILE root window atom is set ({text}) -- a profile/VCGT loader may be active"


# ---------------------------------------------------------------------------
# Windows HDR / Advanced Color -- UNVERIFIED
# ---------------------------------------------------------------------------
#
# Written from the documented Win32 API shape (DISPLAYCONFIG_DEVICE_INFO_
# GET_ADVANCED_COLOR_INFO via DisplayConfigGetDeviceInfo, after enumerating
# paths with GetDisplayConfigBufferSizes/QueryDisplayConfig), but this build
# machine is Linux and there is no Windows box available to run it against
# (same caveat as devices.read_edid_windows). Every failure mode --
# "not Windows", "ctypes call failed", "struct layout wrong", "API not
# present on this Windows version" -- is caught and turned into
# ``(None, reason)`` rather than raised, and the caller is required to fall
# back to ``--confirm-hdr-off`` whenever the method returns None. Treat any
# True/False this actually returns on a real Windows machine with
# suspicion until confirmed there.

DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO = 9
QDC_ONLY_ACTIVE_PATHS = 0x00000002


def _win_query_advanced_color() -> bool:  # pragma: no cover -- only reachable on win32
    """Best-effort: True if any active display path reports HDR/advanced
    color enabled. Raises on any failure; callers must catch broadly."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DISPLAYCONFIG_PATH_TARGET_INFO(ctypes.Structure):
        _fields_ = [
            ("adapterId", LUID),
            ("id", wintypes.UINT),
            ("modeInfoIdx", wintypes.UINT),
            ("outputTechnology", wintypes.UINT),
            ("rotation", wintypes.UINT),
            ("scaling", wintypes.UINT),
            ("refreshRate", wintypes.UINT * 2),
            ("scanLineOrdering", wintypes.UINT),
            ("targetAvailable", wintypes.BOOL),
            ("statusFlags", wintypes.UINT),
        ]

    class DISPLAYCONFIG_PATH_SOURCE_INFO(ctypes.Structure):
        _fields_ = [
            ("adapterId", LUID),
            ("id", wintypes.UINT),
            ("modeInfoIdx", wintypes.UINT),
            ("statusFlags", wintypes.UINT),
        ]

    class DISPLAYCONFIG_PATH_INFO(ctypes.Structure):
        _fields_ = [
            ("sourceInfo", DISPLAYCONFIG_PATH_SOURCE_INFO),
            ("targetInfo", DISPLAYCONFIG_PATH_TARGET_INFO),
            ("flags", wintypes.UINT),
        ]

    class DISPLAYCONFIG_DEVICE_INFO_HEADER(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.UINT),
            ("size", wintypes.UINT),
            ("adapterId", LUID),
            ("id", wintypes.UINT),
        ]

    class DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO(ctypes.Structure):
        _fields_ = [
            ("header", DISPLAYCONFIG_DEVICE_INFO_HEADER),
            ("value", wintypes.UINT),  # bit 0 advancedColorSupported, bit 1 advancedColorEnabled, bit 2 wideColorEnforced
        ]

    path_count, mode_count = wintypes.UINT(0), wintypes.UINT(0)
    if user32.GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_count), ctypes.byref(mode_count)):
        raise OSError("GetDisplayConfigBufferSizes failed")
    paths = (DISPLAYCONFIG_PATH_INFO * path_count.value)()
    modes_buf = ctypes.create_string_buffer(mode_count.value * 64)  # DISPLAYCONFIG_MODE_INFO is opaque to us here
    if user32.QueryDisplayConfig(
        QDC_ONLY_ACTIVE_PATHS,
        ctypes.byref(path_count),
        paths,
        ctypes.byref(mode_count),
        modes_buf,
        None,
    ):
        raise OSError("QueryDisplayConfig failed")

    any_enabled = False
    for path in paths:
        info = DISPLAYCONFIG_GET_ADVANCED_COLOR_INFO()
        info.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO
        info.header.size = ctypes.sizeof(info)
        info.header.adapterId = path.targetInfo.adapterId
        info.header.id = path.targetInfo.id
        if user32.DisplayConfigGetDeviceInfo(ctypes.byref(info)):
            continue  # this path didn't answer; keep checking the others
        advanced_color_enabled = bool(info.value & 0x2)
        any_enabled = any_enabled or advanced_color_enabled
    return any_enabled


def check_hdr_windows(confirm_hdr_off: bool = False) -> tuple:
    """Returns ``(hdr_on_or_None, method_description)``. See the module
    docstring's UNVERIFIED note. Off Windows, always returns
    ``(None, "not running on Windows")``."""
    if sys.platform != "win32":
        return None, "not running on Windows"
    try:
        hdr_on = _win_query_advanced_color()
        return hdr_on, "DisplayConfigGetDeviceInfo (DISPLAYCONFIG_DEVICE_INFO_GET_ADVANCED_COLOR_INFO) -- unverified on this build machine, no Windows box available"
    except Exception as exc:  # noqa: BLE001 -- any failure here must degrade to "unknown", not crash a measurement run
        if confirm_hdr_off:
            return False, f"HDR query failed ({exc}); proceeding on --confirm-hdr-off"
        return None, f"HDR query failed ({exc}); pass --confirm-hdr-off to proceed"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def gather(*, confirm_hdr_off: bool = False, osd: dict | None = None) -> OSState:
    """Collect everything a measurement run needs to record about OS
    display state before taking a single reading."""
    gamma_reset, gamma_method = reset_gamma_table()
    icc_atom, profile_warning = (None, None)
    hdr_on, hdr_method = None, "not applicable on this platform"

    if sys.platform.startswith("linux"):
        icc_atom, profile_warning = check_x11_icc_profile()
        hdr_on, hdr_method = False, "not applicable on X11 (no system HDR/advanced-color path here)"
    elif sys.platform == "win32":
        hdr_on, hdr_method = check_hdr_windows(confirm_hdr_off=confirm_hdr_off)

    return OSState(
        platform=sys.platform,
        gamma_reset=gamma_reset,
        gamma_reset_method=gamma_method,
        icc_profile_atom=icc_atom,
        profile_loader_warning=profile_warning,
        hdr_on=hdr_on,
        hdr_method=hdr_method,
        osd=osd or {},
    )


def refuse_if_hdr_on(state: OSState) -> Refusal | None:
    """Design §5.3: "refuse to measure with HDR on." `hdr_on is None`
    (unknown/unverified, with no ``--confirm-hdr-off``) also refuses --
    proceeding on an unknown HDR state is exactly the silent-wrongness
    this project exists to avoid."""
    if state.hdr_on is True:
        return Refusal("hdr_on", "HDR/advanced color is enabled; measurements would not reflect SDR output", True, False)
    if state.hdr_on is None:
        return Refusal(
            "hdr_state_unknown",
            "HDR state could not be determined ("
            + state.hdr_method
            + "); pass --confirm-hdr-off after manually confirming HDR is off",
            None,
            False,
        )
    return None

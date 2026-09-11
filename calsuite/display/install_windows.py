"""Install a display profile on Windows (docs/design.md §5.5): `dispwin -I`
if ArgyllCMS is present, else explain the Color Management control panel
steps. UNVERIFIED: this build machine is Linux, so nothing in this module
has run against a real Windows install -- ``install()`` refuses to do
anything at all off `win32`, and the manual-steps text is written from
Windows' own documented Color Management UI, not confirmed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from calsuite import tools
from calsuite.display.install_linux import InstallReport

CONTROL_PANEL_STEPS = (
    "Control Panel > Color Management > Devices tab > select this display > Add... > "
    "browse to the .icc file > OK > check 'Use my settings for this device' > set it as "
    "the default profile. Advanced tab > 'Use Windows display calibration' loads the VCGT "
    "curves at login (design §5.5's Windows column). UNVERIFIED here (this build machine is "
    "Linux) -- exact wording/paths may differ across Windows versions; taken from Windows' "
    "documented Color Management UI, never run against a real install."
)


def install(icc_path: Path, *, confirm: bool = False) -> InstallReport:
    report = InstallReport()
    if sys.platform != "win32":
        report.add("platform", False, "install_windows.install() only runs on win32; nothing was done")
        return report
    if tools.which("dispwin") is not None:
        try:
            tools.run(["dispwin", "-I", str(icc_path)])
            report.add("dispwin -I", True)
            return report
        except tools.ToolError as exc:
            report.add("dispwin -I", False, str(exc))
    report.add("manual install required", False, CONTROL_PANEL_STEPS)
    return report

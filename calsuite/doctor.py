"""``calsuite doctor``: environment and record-store health checks (house
rule 6, docs/design.md: "staleness is a finding"). Every check below is a
pure function of a ``store.Store`` (plus, for the tool/EDID checks, the
live environment) -- ``run()`` just calls each in turn and pools the
results, so each is independently unit-testable against a constructed
store with no CLI involved.

Findings are grouped by category and printed readable; ``doctor`` exits
non-zero whenever there is at least one finding, because every finding
here names something a user should look at before trusting a record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from calsuite import config, devices as devicesmod, store as storemod, tools

# What each external tool unlocks, for the "present/absent" report --
# tools.KNOWN_TOOLS is the single source of truth for *which* tools exist;
# this just explains *why* a user would install the missing one.
TOOL_PURPOSE = {
    "exiftool": "raw metadata (serial/firmware/lens/sensor temperature) -- preferred over the dcraw fallback",
    "dcraw": "raw metadata fallback when exiftool is absent (CameraModel/ISO/shutter/aperture/focal only)",
    "gphoto2": "tethered capture on Linux (--capture); manual import always works without it",
    "dispwin": "loading/resetting the video-card gamma table (-c) before measuring, and installing a profile",
    "spotread": "the ArgyllCMS colorimeter measurement backend (display measure --backend argyll)",
    "colprof": "building an ICC profile from a .ti3 via ArgyllCMS (falls back to calsuite's built-in matrix/TRC writer when absent)",
    "profcheck": "self-checking a colprof-built ICC profile against its own .ti3",
    "colormgr": "installing/activating a display profile on Linux (colord)",
    "xprop": "checking the X11 _ICC_PROFILE atom / VCGT state before measuring",
}


@dataclass(frozen=True)
class Finding:
    category: str
    message: str
    severity: str = "warning"  # "info" (tool present) | "warning" (needs attention)

    def line(self) -> str:
        tag = "OK" if self.severity == "info" else "!!"
        return f"  [{tag}] {self.message}"


@dataclass
class DoctorReport:
    findings: list = field(default_factory=list)

    def add(self, category: str, message: str, severity: str = "warning") -> None:
        self.findings.append(Finding(category, message, severity))

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.warnings


# ---------------------------------------------------------------------------
# (a) external tool availability
# ---------------------------------------------------------------------------


def check_tools() -> list:
    findings = []
    for name in tools.KNOWN_TOOLS:
        purpose = TOOL_PURPOSE.get(name, "")
        if tools.which(name) is not None:
            findings.append(Finding("tools", f"{name}: present -- {purpose}", "info"))
        else:
            findings.append(Finding("tools", f"{name}: NOT FOUND -- needed for {purpose}", "warning"))
    return findings


# ---------------------------------------------------------------------------
# (b) records past their shelf life
# ---------------------------------------------------------------------------


def check_stale_records(st: storemod.Store) -> list:
    findings = []
    for record in st.all():
        if st.is_stale(record):
            max_age = storemod.SHELF_LIFE_DAYS.get(record.kind, storemod.DEFAULT_SHELF_LIFE_DAYS)
            findings.append(
                Finding(
                    "stale",
                    f"{record.device.get('id', '?')} {record.kind} ({record.created}) is older than its "
                    f"{max_age}-day shelf life",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# (c) camera firmware changed since an earlier record for the same body
# ---------------------------------------------------------------------------


def check_firmware_changes(st: storemod.Store) -> list:
    findings = []
    by_device: dict = {}
    for record in st.all():
        if record.device.get("kind") != "camera":
            continue
        by_device.setdefault(record.device["id"], []).append(record)
    for device_id, records in sorted(by_device.items()):
        firmwares = sorted({r.device.get("firmware") for r in records if r.device.get("firmware")})
        if len(firmwares) > 1:
            findings.append(
                Finding(
                    "firmware",
                    f"camera {device_id}: records span more than one firmware version {firmwares} -- "
                    "older records may not reflect the body's current behaviour",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# (d) a display's current EDID hash differs from its stored nominal record
# ---------------------------------------------------------------------------


def check_edid_changes(st: storemod.Store, edid_blobs: dict | None = None) -> list:
    """``edid_blobs``: ``{connector: bytes}``, as ``devices.list_linux_edids()``
    returns -- injectable for tests; defaults to reading the live machine."""
    findings = []
    if edid_blobs is None:
        try:
            edid_blobs = devicesmod.list_linux_edids()
        except Exception:
            return findings
    for connector, data in sorted(edid_blobs.items()):
        try:
            info = devicesmod.parse_edid(data)
        except ValueError:
            continue
        ref = devicesmod.display_ref(info)
        current_hash = devicesmod.edid_hash(data)
        record = st.latest("display.nominal", ref.id)
        stored_hash = record.result.get("edid_hash") if record is not None else None
        if stored_hash is not None and stored_hash != current_hash:
            findings.append(
                Finding(
                    "edid",
                    f"display {ref.id} ({connector}): current EDID hash differs from record "
                    f"{record.id}'s -- the panel behind this connector may have changed",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# (e) an installed profile with no passing validation newer than it
# ---------------------------------------------------------------------------


def check_profiles_without_recent_validation(st: storemod.Store) -> list:
    findings = []
    by_device: dict = {}
    for record in st.all(kind="display.profile"):
        by_device.setdefault(record.device["id"], []).append(record)
    for device_id, records in sorted(by_device.items()):
        latest_profile = max(records, key=lambda r: r.created)
        if latest_profile.status != "ok":
            continue
        newer_passing_validation = any(
            r.created > latest_profile.created and r.status == "ok"
            for r in st.all(kind="display.validation", device_id=device_id)
        )
        if not newer_passing_validation:
            findings.append(
                Finding(
                    "validation",
                    f"display {device_id}: profile {latest_profile.id} has no passing display.validation "
                    "record newer than it",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# (f) refused records never superseded by a passing one of the same kind
# ---------------------------------------------------------------------------


def check_unsuperseded_refusals(st: storemod.Store) -> list:
    findings = []
    by_key: dict = {}
    for record in st.all():
        by_key.setdefault((record.device.get("id", "?"), record.kind), []).append(record)
    for (device_id, kind), records in sorted(by_key.items()):
        latest = max(records, key=lambda r: r.created)
        if latest.status == "refused":
            findings.append(
                Finding(
                    "refused",
                    f"{device_id} {kind}: latest record ({latest.id}) is refused and has not been "
                    "superseded by a passing one",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

_CATEGORY_ORDER = ("tools", "stale", "firmware", "edid", "validation", "refused")
_CATEGORY_TITLE = {
    "tools": "External tools",
    "stale": "Records past their shelf life",
    "firmware": "Camera firmware changes",
    "edid": "Display EDID changes",
    "validation": "Installed profiles with no recent passing validation",
    "refused": "Refused records never superseded",
}


def run(records_dir: Path | str | None = None) -> DoctorReport:
    st = storemod.Store(Path(records_dir) if records_dir is not None else config.records_dir())
    report = DoctorReport()
    report.findings += check_tools()
    report.findings += check_stale_records(st)
    report.findings += check_firmware_changes(st)
    report.findings += check_edid_changes(st)
    report.findings += check_profiles_without_recent_validation(st)
    report.findings += check_unsuperseded_refusals(st)
    return report


def format_report(report: DoctorReport) -> str:
    by_category: dict = {}
    for f in report.findings:
        by_category.setdefault(f.category, []).append(f)

    lines = []
    for cat in _CATEGORY_ORDER:
        items = by_category.get(cat, [])
        if not items:
            continue
        lines.append(f"{_CATEGORY_TITLE[cat]}:")
        lines.extend(f.line() for f in items)
        lines.append("")

    n = len(report.warnings)
    lines.append(f"{n} finding(s) needing attention." if n else "No findings -- everything checked out clean.")
    return "\n".join(lines)

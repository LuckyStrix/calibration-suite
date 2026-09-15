"""Record store: schema v1 JSON files, plus sidecar ``.npz`` artifacts.

Layout: ``records/<device-id>/<kind>-<UTC stamp>-<rand>.json``, with any
array artifacts saved as ``records/<device-id>/<record-id>.<name>.npz`` next
to it. Refused records are saved too (house rule 3, docs/design.md: "a
refusal is a finding") -- ``status="refused"`` plus ``require_exportable()``
is what stops one being used downstream, not omission from the store.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from calsuite import provenance as prov
from calsuite.fit import Analysis, Refusal
from calsuite.raw import sha256_file

SCHEMA_VERSION = 1

# How long a record of each kind stays trustworthy before doctor.py flags it
# stale, in days. Every value has a *reason*, not just a source -- same
# convention as hydrationTracker's model/constants.py.
SHELF_LIFE_DAYS = {
    # Sensor electronics (gain, read noise, dark current shape) drift with
    # the silicon and with firmware, not with the seasons -- a year is
    # conservative against both, and doctor.py's separate firmware-change
    # check catches the faster-moving case independently of elapsed time.
    "camera.bias": 365,
    "camera.ptc": 365,
    "camera.linearity": 365,
    "camera.darks": 180,  # dark current is temperature-dependent and ambient temperature drifts seasonally
    "camera.fixed_pattern": 365,
    "camera.iso": 365,
    "camera.shutter": 365,
    "camera.color": 180,  # sensor aging plus the reference chart/light source it depended on
    "camera.ssf": 730,  # a physical property of the filter stack; changes only if the sensor itself is replaced
    "camera.dcp": 180,  # derived from camera.color -- inherits its shelf life rather than getting a longer one of its own
    # Lens optics are mechanically stable but can shift after a drop, a
    # service, or (on zooms) sample variation at a different focus-breathing
    # state -- a year balances "rarely changes" against "trust a five-year-
    # old distortion map on a lens that's been dropped since".
    "lens.distortion": 365,
    "lens.tca": 365,
    "lens.flats": 365,
    "lens.mtf": 365,
    "lens.psf": 365,
    # Displays age (backlight dims, primaries drift) faster than camera
    # sensors or lens glass -- three months matches the commonly-cited
    # "recalibrate quarterly" guidance for LCDs used for critical work.
    "display.measurement": 90,
    "display.profile": 90,
    "display.validation": 90,
    "display.nominal": 3650,  # an EDID reading; "stale" only if the panel changes (doctor's EDID-hash check), not with time
}
DEFAULT_SHELF_LIFE_DAYS = 180
# Applied to any kind not listed above: a conservative middle ground rather
# than silently treating an unlisted kind as eternal.


def utcnow_stamp() -> str:
    """UTC timestamp, filesystem- and JSON-safe, sortable as plain text."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_id(kind: str) -> str:
    """kind + UTC stamp + a few random hex chars, so two records of the same
    kind created in the same second (a batch analysis script) never
    collide."""
    return f"{kind}-{utcnow_stamp()}-{secrets.token_hex(3)}"


@dataclass
class Record:
    """Schema v1 -- every field docs/implementation-plan.md's "Core
    contracts" section lists, in the order it lists them. ``to_dict``/
    ``from_dict`` are the only (de)serialization path; ``Store`` uses them
    and tests building a ``Record`` from a fixture dict should too, so a
    schema change only has to happen in one place.
    """

    schema: int
    id: str
    kind: str
    device: dict  # DeviceRef.to_dict() of the record's primary device
    devices: list = field(default_factory=list)  # other involved devices, e.g. the body a lens record was shot on
    created: str = ""  # UTC stamp, utcnow_stamp() format; filled in by __post_init__ if omitted
    provenance: str = "nominal"
    status: str = "ok"  # "ok" | "refused"
    refusals: list = field(default_factory=list)  # list of dicts (Refusal.to_dict())
    conditions: dict = field(default_factory=dict)
    method: dict = field(default_factory=dict)  # {"name":..., "calsuite_version":..., "params": {...}}
    inputs: list = field(default_factory=list)  # [{"name":..., "sha256":...}, ...]
    derived_from: list = field(default_factory=list)  # [record id, ...]
    result: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)  # [{"name":..., "sha256":...}, ...] filled in by Store.save()

    def __post_init__(self):
        if self.provenance not in prov.PROVENANCE:
            raise ValueError(f"unknown provenance {self.provenance!r}; must be one of {prov.PROVENANCE}")
        if self.status not in ("ok", "refused"):
            raise ValueError(f"status must be 'ok' or 'refused', got {self.status!r}")
        if not self.created:
            self.created = utcnow_stamp()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Record:
        schema = d.get("schema")
        if schema is not None and not isinstance(schema, int):
            # A record with "schema": "2" or 2.0 used to slip past the guard
            # below entirely and load as if it were schema 1.
            raise RecordCorruptError(f"record schema must be an integer, got {schema!r}")
        if isinstance(schema, int) and schema > SCHEMA_VERSION:
            raise UnsupportedSchemaError(
                f"record schema {schema} is newer than this calsuite understands (schema {SCHEMA_VERSION}) "
                "-- upgrade calsuite before reading it, don't guess at the missing fields"
            )
        try:
            return cls(**d)
        except TypeError as exc:
            raise RecordCorruptError(f"record does not match schema {SCHEMA_VERSION}: {exc}") from exc

    @classmethod
    def from_analysis(
        cls,
        *,
        kind: str,
        device: dict,
        analysis: Analysis,
        provenance: str,
        method: dict,
        inputs: list | None = None,
        conditions: dict | None = None,
        devices: list | None = None,
        derived_from: list | None = None,
    ) -> Record:
        """Build a ``Record`` from a pure ``fit.Analysis`` result -- the one
        place that translates "arrays in, Analysis out" into the stored
        schema, so every ``commands.py`` doesn't reinvent this mapping."""
        return cls(
            schema=SCHEMA_VERSION,
            id=new_id(kind),
            kind=kind,
            device=device,
            devices=devices or [],
            provenance=provenance,
            status="ok" if analysis.ok else "refused",
            refusals=[r.to_dict() if isinstance(r, Refusal) else r for r in analysis.refusals],
            conditions=conditions or {},
            method=method,
            inputs=inputs or [],
            derived_from=derived_from or [],
            result=analysis.result,
            residuals=analysis.residuals,
            uncertainty=analysis.uncertainty,
        )


class ExportRefused(RuntimeError):
    """Raised by ``require_exportable`` -- a record that failed its checks,
    or carries weak provenance, is not fit to hand to lensfun/ICC/DCP
    export."""


class RecordCorruptError(RuntimeError):
    """A record file on disk isn't valid JSON, or doesn't match the current
    schema shape (missing/extra fields ``Record`` doesn't accept) -- e.g. a
    truncated write from a killed process. Raised with the path (for
    ``Store.load``/``Store.all``) so the failure is diagnosable without
    re-deriving which of possibly hundreds of files under ``records/`` is
    the bad one. Deliberately *not* swallowed and skipped by ``Store.all``:
    a store is exactly the thing every downstream check (``doctor.py``,
    every report) trusts, so silently dropping one unreadable record would
    hide the one failure mode -- a corrupted file -- that most needs
    surfacing, not the one safe to ignore."""


class ArtifactTamperedError(RuntimeError):
    """Raised by ``Store.load_artifact`` when an ``.npz`` sidecar's current
    SHA-256 doesn't match the hash its record was saved with -- the file
    was altered (or replaced) after ``Store.save`` wrote it. The whole
    point of recording an artifact's hash (``save``'s docstring) is that a
    later reader can tell; returning the array data anyway without checking
    would make that recorded hash decorative."""


class UnsupportedSchemaError(RuntimeError):
    """Raised by ``Record.from_dict`` when a record's ``schema`` is newer
    than this ``calsuite``'s ``SCHEMA_VERSION`` understands -- reading it
    with an older schema's field set would silently drop or misinterpret
    whatever the newer schema added, which is worse than refusing outright
    (house rule 3's logic applied to the store itself, not just an
    analysis)."""


def require_exportable(record: Record) -> None:
    """Refuse to let a bad or under-evidenced record leave the suite.

    A record may be exported (written into lensfun XML, installed as a
    system ICC profile, baked into a DCP, ...) only if its analysis passed
    (``status == "ok"``) **and** its provenance is strong enough to stand
    behind (``measured`` or ``derived`` -- see ``provenance.EXPORTABLE``).
    Every export path is expected to call this before writing anything.
    """
    if record.status != "ok":
        messages = [r.get("message", r) if isinstance(r, dict) else r for r in record.refusals]
        raise ExportRefused(f"record {record.id} has status={record.status!r}: {messages}")
    if not prov.is_exportable(record.provenance):
        raise ExportRefused(
            f"record {record.id} has provenance={record.provenance!r}, which is not one of {prov.EXPORTABLE}"
        )


class Store:
    """``records/<device-id>/<record-id>.json`` (+ ``.npz`` sidecars) on disk."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # -- paths -----------------------------------------------------------

    def _device_dir(self, device_id: str) -> Path:
        return self.root / device_id

    def _record_path(self, record: Record) -> Path:
        return self._device_dir(record.device["id"]) / f"{record.id}.json"

    def _artifact_path(self, record: Record, name: str) -> Path:
        return self._device_dir(record.device["id"]) / f"{record.id}.{name}.npz"

    # -- writing -----------------------------------------------------------

    def save(self, record: Record, artifacts: dict | None = None) -> Path:
        """Write ``<id>.json``, and (if given) one ``.npz`` sidecar per
        artifact.

        ``artifacts`` maps a name (e.g. ``"prnu_map"``) to a dict of array
        name -> ``ndarray``, written with ``np.savez_compressed``. Each
        sidecar's SHA-256 is recorded on ``record.artifacts`` so the JSON
        alone documents everything that was written, and a later reader can
        verify a sidecar hasn't been altered without re-deriving it.
        """
        device_dir = self._device_dir(record.device["id"])
        device_dir.mkdir(parents=True, exist_ok=True)

        record.artifacts = []
        for name, arrays in (artifacts or {}).items():
            path = self._artifact_path(record, name)
            np.savez_compressed(path, **arrays)
            record.artifacts.append({"name": name, "sha256": sha256_file(path)})

        # Serialized before anything is committed, and with `allow_nan=False`.
        # Two reasons:
        #
        # - json.dumps raising here (an ndarray or np.float64 left in
        #   `result`, the case fit.Refusal's docstring warns about) used to
        #   happen *after* the sidecars were on disk, leaving a .npz with no
        #   .json at all -- an artifact no `Store.all` can ever find.
        # - Python's json writes bare `NaN`/`Infinity` by default, which is
        #   not valid JSON. `records/` is committed to a public repo and
        #   advertised as the suite's output, and `devices.parse_edid`
        #   deliberately yields a NaN gamma for an EDID that defers gamma to
        #   an extension block, so `calsuite devices` really did write
        #   `"gamma": NaN` into a committed file that jq and every strict
        #   parser reject. Python's own json.loads accepts it, which is why
        #   nothing here noticed. Fail loudly instead; a non-finite number is
        #   a missing measurement and belongs in the record as null.
        try:
            payload = json.dumps(record.to_dict(), indent=2, allow_nan=False)
        except ValueError as exc:
            raise ValueError(
                f"record {record.id} contains a non-finite number (NaN/Infinity), which is not valid JSON: "
                f"{exc} -- write null for a value that wasn't measured"
            ) from exc

        path = self._record_path(record)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

        # Sidecars from a previous save of this same record id that the
        # current artifact set doesn't cover: leaving them on disk leaves
        # files nothing references and nothing verifies.
        keep = {self._artifact_path(record, a["name"]) for a in record.artifacts}
        for stale in device_dir.glob(f"{record.id}.*.npz"):
            if stale not in keep:
                stale.unlink()
        return path

    # -- reading -----------------------------------------------------------

    def load(self, path: Path | str) -> Record:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecordCorruptError(f"{path}: not valid JSON ({exc})") from exc
        try:
            return Record.from_dict(data)
        except RecordCorruptError as exc:
            raise RecordCorruptError(f"{path}: {exc}") from exc
        except UnsupportedSchemaError as exc:
            raise UnsupportedSchemaError(f"{path}: {exc}") from exc

    def load_artifact(self, record: Record, name: str) -> dict:
        """Load the ``name`` artifact sidecar for ``record``, verifying its
        current SHA-256 against the one ``save()`` recorded on
        ``record.artifacts`` -- see ``ArtifactTamperedError``. A record with
        no matching ``artifacts`` entry (e.g. one built by hand in a test,
        or from an older schema that didn't record one) can't be verified;
        it's loaded as-is rather than refused, since "unverifiable" and
        "verified tampered" are different findings.
        """
        path = self._artifact_path(record, name)
        expected = next((a["sha256"] for a in record.artifacts if a.get("name") == name), None)
        if expected is not None:
            actual = sha256_file(path)
            if actual != expected:
                raise ArtifactTamperedError(
                    f"{path}: sha256 {actual} does not match record {record.id}'s recorded "
                    f"sha256 {expected} for artifact {name!r} -- the sidecar was altered after save()"
                )
        with np.load(path) as npz:
            return {k: npz[k] for k in npz.files}

    def all(self, kind: str | None = None, device_id: str | None = None):
        """Yield every ``Record`` under ``root``, optionally filtered by
        ``kind`` and/or ``device_id``."""
        if not self.root.is_dir():
            return
        dirs = [self._device_dir(device_id)] if device_id else sorted(self.root.glob("*"))
        for d in dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                record = self.load(p)
                if kind is not None and record.kind != kind:
                    continue
                yield record

    # "iterate" is named explicitly alongside "all" in the design doc; both
    # exist because `for r in store.iterate():` reads better at a call site
    # that wants a generator, while `list(store.all())` reads better where a
    # caller wants the whole list immediately. Same generator either way.
    iterate = all

    def latest(self, kind: str, device_id: str) -> Record | None:
        """Most recent record of ``kind`` for ``device_id`` by ``created``,
        or ``None``. Refused records count -- doctor.py and staleness
        checks want the most recent *attempt*, not just the most recent
        success; a caller that wants only the last good one filters
        ``status == "ok"`` itself."""
        records = list(self.all(kind=kind, device_id=device_id))
        if not records:
            return None
        # `created` has one-second resolution, so two records of the same
        # kind saved in the same second tie. Breaking the tie on `id` keeps
        # the answer *reproducible* (it used to fall out of the glob order,
        # i.e. out of new_id's random hex suffix, so the same store could
        # answer differently on different machines); it can't make it
        # meaningful, because a one-second stamp genuinely doesn't say which
        # came first. A caller that needs "the latest *passing* one" must
        # filter on status rather than rely on this -- see
        # doctor.check_unsuperseded_refusals, which compares timestamps with
        # >= for exactly this reason.
        return max(records, key=lambda r: (r.created, r.id))

    def is_stale(self, record: Record, *, now: datetime | None = None) -> bool:
        """True if ``record`` is older than its kind's shelf life."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            # A naive `now` used to raise TypeError ("can't subtract
            # offset-naive and offset-aware datetimes") from inside a
            # staleness check. Every timestamp in a record is UTC by
            # construction; read a naive one the same way.
            now = now.replace(tzinfo=timezone.utc)
        try:
            created = datetime.strptime(record.created, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise RecordCorruptError(
                f"record {record.id}: created timestamp {record.created!r} is not the stored "
                "%Y%m%dT%H%M%SZ format"
            ) from exc
        max_age = SHELF_LIFE_DAYS.get(record.kind, DEFAULT_SHELF_LIFE_DAYS)
        return (now - created).days > max_age

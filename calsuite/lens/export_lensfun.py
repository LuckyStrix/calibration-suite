"""lensfun XML export: a ``<camera>`` block for the Canon EOS R100 (absent
from the locally-installed ``mil-canon.xml`` -- docs/implementation-plan.md's
"Facts checked") and a ``<lens>`` block with whatever distortion/tca/
vignetting calibrations are available, at the measured focal/aperture/
distance. Every input record goes through ``store.require_exportable``
first (house rule 2/3: only ``ok`` + ``measured``/``derived`` records ever
leave the suite).

See ``lens/constants.py`` for the XML attribute names and user-database
path citations this module writes against.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from calsuite import store
from calsuite.lens import constants as C


def build_xml(
    *,
    lens_model: str,
    lens_maker: str = C.DEFAULT_LENS_MAKER,
    lens_mount: str = C.DEFAULT_LENS_MOUNT,
    lens_cropfactor: float = 1.0,
    camera_model: str = C.DEFAULT_CAMERA_MODEL,
    camera_maker: str = C.DEFAULT_CAMERA_MAKER,
    camera_mount: str = C.DEFAULT_LENS_MOUNT,
    camera_cropfactor: float = C.R100_CROP_FACTOR,
    include_camera: bool = True,
    distortion: dict | None = None,
    tca: dict | None = None,
    vignetting: list | None = None,
) -> str:
    """Build the lensfun XML document as a string.

    ``distortion``: ``{"model": "ptlens"|"poly3", "focal": mm, "params": {...}}``.
    ``tca``: ``{"focal": mm, "vr": float, "vb": float}`` (always written as
    the ``poly3`` element per lens/constants.py's citation of the vendor
    entry's own convention).
    ``vignetting``: a list of ``{"focal": mm, "aperture": f, "distance": m,
    "k1":, "k2":, "k3":}``, one ``<vignetting>`` element each.
    """
    root = ET.Element("lensdatabase", version="1")

    if include_camera:
        camera_el = ET.SubElement(root, "camera")
        ET.SubElement(camera_el, "maker").text = camera_maker
        ET.SubElement(camera_el, "model").text = camera_model
        ET.SubElement(camera_el, "mount").text = camera_mount
        ET.SubElement(camera_el, "cropfactor").text = f"{camera_cropfactor:.3f}"

    lens_el = ET.SubElement(root, "lens")
    ET.SubElement(lens_el, "maker").text = lens_maker
    ET.SubElement(lens_el, "model").text = lens_model
    ET.SubElement(lens_el, "mount").text = lens_mount
    ET.SubElement(lens_el, "cropfactor").text = f"{lens_cropfactor:.3f}"
    calibration_el = ET.SubElement(lens_el, "calibration")

    if distortion is not None:
        attrs = {"model": distortion["model"], "focal": f"{distortion['focal']:.1f}"}
        params = distortion["params"]
        if distortion["model"] == "ptlens":
            for term in C.LENSFUN_DISTORTION_PTLENS_TERMS:
                attrs[term] = f"{params[term]:.6f}"
        elif distortion["model"] == "poly3":
            attrs[C.LENSFUN_DISTORTION_POLY3_TERM] = f"{params[C.LENSFUN_DISTORTION_POLY3_TERM]:.6f}"
        ET.SubElement(calibration_el, "distortion", attrs)

    if tca is not None:
        attrs = {"model": "poly3", "focal": f"{tca['focal']:.1f}", "vr": f"{tca['vr']:.7f}", "vb": f"{tca['vb']:.7f}"}
        ET.SubElement(calibration_el, "tca", attrs)

    for entry in vignetting or []:
        attrs = {
            "model": "pa",
            "focal": f"{entry['focal']:.1f}",
            "aperture": f"{entry['aperture']:.1f}",
            "distance": f"{entry['distance']:.2f}",
            "k1": f"{entry['k1']:.6f}",
            "k2": f"{entry['k2']:.6f}",
            "k3": f"{entry['k3']:.6f}",
        }
        ET.SubElement(calibration_el, "vignetting", attrs)

    ET.indent(root, space="    ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def export_records(
    *,
    lens_model: str,
    distortion_record=None,
    tca_record=None,
    flats_records: list | None = None,
    lens_cropfactor: float = 1.0,
    camera_model: str = C.DEFAULT_CAMERA_MODEL,
    camera_cropfactor: float = C.R100_CROP_FACTOR,
) -> str:
    """The commands.py entry point: pull focal/aperture/distance out of
    each ``Record``'s ``conditions`` (set there by the corresponding
    ``lens <kind>`` command) and the fitted params out of its ``result``,
    after checking every record with ``store.require_exportable`` --
    a refused or under-evidenced record raises ``store.ExportRefused``
    before anything is written, per house rule 2/3."""
    for record in filter(None, [distortion_record, tca_record, *(flats_records or [])]):
        store.require_exportable(record)

    distortion = None
    if distortion_record is not None:
        focal = float(distortion_record.conditions.get("focal_mm") or 0.0)
        distortion = {"model": "ptlens", "focal": focal, "params": distortion_record.result["ptlens"]}

    tca = None
    if tca_record is not None:
        focal = float(tca_record.conditions.get("focal_mm") or 0.0)
        tca = {"focal": focal, "vr": tca_record.result["vr"], "vb": tca_record.result["vb"]}

    vignetting = []
    for record in flats_records or []:
        pa = record.result.get("pa")
        if pa is None:
            continue
        vignetting.append(
            {
                "focal": float(record.conditions.get("focal_mm") or 0.0),
                "aperture": float(record.conditions.get("aperture") or 0.0),
                "distance": float(record.conditions.get("focus_distance_m") or 0.0),
                **pa,
            }
        )

    return build_xml(
        lens_model=lens_model,
        lens_cropfactor=lens_cropfactor,
        camera_model=camera_model,
        camera_cropfactor=camera_cropfactor,
        distortion=distortion,
        tca=tca,
        vignetting=vignetting,
    )


def user_data_dir() -> Path:
    """Where lensfun's own ``lfDatabase::Load()`` reads the *user*
    database from, unconditionally, in addition to the system one (see
    lens/constants.py's citation of ``database.cpp``): Linux
    ``~/.local/share/lensfun``; Windows ``%LOCALAPPDATA%\\lensfun``
    (GLib's ``g_get_user_data_dir()`` on Windows resolves to
    ``FOLDERID_LocalAppData``, corroborated by darktable users' reports of
    that exact path -- see the module docstring's citations)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise RuntimeError("%LOCALAPPDATA% is not set -- pass --out explicitly")
        return Path(base) / C.LENSFUN_USER_DIR_WINDOWS_RELATIVE
    return C.LENSFUN_USER_DIR_LINUX


def write_lensfun(xml_text: str, *, out: Path | str | None = None, filename: str = "calsuite-r100.xml") -> Path:
    """Write ``xml_text`` to ``out`` if given (a directory, or a full
    ``.xml`` path), else to ``user_data_dir()``."""
    if out is not None and Path(out).suffix.lower() == ".xml":
        target_path = Path(out)
    else:
        target_dir = Path(out) if out is not None else user_data_dir()
        target_path = target_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(xml_text, encoding="utf-8")
    return target_path


def parse_vendor_lens(xml_path: Path | str, lens_model: str) -> dict | None:
    """Every ``<distortion>``/``<tca>``/``<vignetting>`` element (as plain
    attribute dicts) for the ``<lens>`` block whose ``<model>`` matches
    ``lens_model`` in a lensfun database file -- ``None`` if no such lens
    is in the file."""
    tree = ET.parse(xml_path)
    for lens_el in tree.getroot().findall("lens"):
        model_el = lens_el.find("model")
        if model_el is None or model_el.text != lens_model:
            continue
        out = {"distortion": [], "tca": [], "vignetting": []}
        calibration_el = lens_el.find("calibration")
        if calibration_el is not None:
            for tag in ("distortion", "tca", "vignetting"):
                out[tag] = [dict(el.attrib) for el in calibration_el.findall(tag)]
        return out
    return None


def compare_with_vendor(
    our_ptlens: dict,
    our_tca: dict | None = None,
    *,
    system_xml_path: Path | str | None = None,
    lens_model: str = C.VENDOR_COMPARISON_LENS_MODEL,
) -> dict:
    """Compare our distortion/TCA fit against the vendor entry already in
    the system lensfun database (provenance ``"vendor"``,
    docs/implementation-plan.md's "Facts checked": the RF 50mm entry is
    already there, the R100 body isn't). Never raises: an absent database
    file or lens entry is reported in the result, not an exception, since
    "no vendor data to compare against" is an expected outcome on a
    machine without lensfun installed, not a bug."""
    system_xml_path = Path(system_xml_path) if system_xml_path is not None else C.SYSTEM_LENSFUN_DB_LINUX / "mil-canon.xml"
    if not system_xml_path.exists():
        return {"available": False, "reason": f"{system_xml_path} does not exist"}

    vendor = parse_vendor_lens(system_xml_path, lens_model)
    if vendor is None:
        return {"available": False, "reason": f"{lens_model!r} not found in {system_xml_path}"}

    comparison = {"available": True, "provenance": "vendor", "lens_model": lens_model, "vendor": vendor}
    if vendor["distortion"]:
        vd = vendor["distortion"][0]
        if vd.get("model") == "ptlens":
            comparison["distortion_diff"] = {
                term: our_ptlens[term] - float(vd[term]) for term in C.LENSFUN_DISTORTION_PTLENS_TERMS if term in vd
            }
    if our_tca is not None and vendor["tca"]:
        vt = vendor["tca"][0]
        comparison["tca_diff"] = {
            "vr": our_tca["vr"] - float(vt.get("vr", 1.0)),
            "vb": our_tca["vb"] - float(vt.get("vb", 1.0)),
        }
    return comparison

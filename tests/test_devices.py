"""devices.py: device id / slug rules and the pure EDID parser.

The EDID fixture (tests/fixtures/edid_csot_mng007ja1.bin) is this laptop's
real eDP-1 EDID with the serial-number field (bytes 12-15) zeroed -- see
docs/implementation-plan.md's privacy note and the Wave 1 build notes for
how it was produced. It must decode to the primaries/white/gamma/size the
design doc's §0 table states for this panel.
"""

from pathlib import Path

import pytest

from calsuite import devices

FIXTURE = Path(__file__).parent / "fixtures" / "edid_csot_mng007ja1.bin"


def test_slug():
    assert devices.slug("Canon EOS R100") == "canon-eos-r100"
    assert devices.slug("  Weird!!  Name??  ") == "weird-name"


def test_device_id_with_serial_is_stable_and_one_way():
    id1 = devices.device_id("Canon EOS R100", "0123456789")
    id2 = devices.device_id("Canon EOS R100", "0123456789")
    assert id1 == id2
    assert id1.startswith("canon-eos-r100-")
    assert "0123456789" not in id1  # raw serial never appears in the id


def test_device_id_without_serial_is_unknown_suffixed():
    assert devices.device_id("RF 50mm F1.8 STM", None) == "rf-50mm-f1-8-stm-unknown"
    assert devices.device_id("RF 50mm F1.8 STM", "") == "rf-50mm-f1-8-stm-unknown"


def test_device_id_different_serials_differ():
    a = devices.device_id("Canon EOS R100", "AAAA")
    b = devices.device_id("Canon EOS R100", "BBBB")
    assert a != b


def test_edid_fixture_exists():
    assert FIXTURE.exists(), "EDID fixture missing -- see tests/fixtures/"


def test_edid_checksum_and_header_valid():
    data = FIXTURE.read_bytes()
    devices.verify_edid(data)  # must not raise


def test_edid_decodes_to_design_doc_values():
    data = FIXTURE.read_bytes()
    info = devices.parse_edid(data)

    assert info.name == "CSOT T3"
    assert info.physical_size_mm == (344.0, 215.0)
    assert info.gamma == pytest.approx(2.2, abs=0.01)

    expected = {
        "r": (0.638, 0.334),
        "g": (0.300, 0.596),
        "b": (0.141, 0.058),
        "w": (0.312, 0.329),
    }
    for channel, (ex, ey) in expected.items():
        ax, ay = info.chromaticity[channel]
        assert ax == pytest.approx(ex, abs=0.001), channel
        assert ay == pytest.approx(ey, abs=0.001), channel


def test_edid_serial_is_zeroed_in_fixture():
    data = FIXTURE.read_bytes()
    info = devices.parse_edid(data)
    assert info.serial_number == 0
    assert info.serial_string == ""


def test_verify_edid_rejects_garbage():
    with pytest.raises(ValueError):
        devices.verify_edid(b"not an edid" * 20)


def test_verify_edid_rejects_short_input():
    with pytest.raises(ValueError):
        devices.verify_edid(b"\x00" * 10)


def test_display_ref_from_edid():
    data = FIXTURE.read_bytes()
    info = devices.parse_edid(data)
    ref = devices.display_ref(info)
    assert ref.kind == "display"
    assert ref.model == "CSOT T3"
    assert ref.id.startswith("csot-t3-")


def test_edid_hash_changes_with_content():
    data = bytearray(FIXTURE.read_bytes())
    h1 = devices.edid_hash(bytes(data))
    data[21] = (data[21] + 1) % 256  # perturb the physical size byte
    h2 = devices.edid_hash(bytes(data))
    assert h1 != h2


def test_list_linux_edids_skips_empty_files(tmp_path):
    drm = tmp_path / "card0-FAKE-1"
    drm.mkdir()
    (drm / "edid").write_bytes(b"")  # disconnected output: 0 bytes, not an error
    connected_dir = tmp_path / "card0-eDP-1"
    connected_dir.mkdir()
    (connected_dir / "edid").write_bytes(FIXTURE.read_bytes())

    found = devices.list_linux_edids(tmp_path)
    assert set(found) == {"card0-eDP-1"}


def test_read_edid_windows_raises_off_windows():
    import sys

    if sys.platform == "win32":
        pytest.skip("this test only checks the non-Windows guard")
    with pytest.raises(RuntimeError):
        devices.read_edid_windows()

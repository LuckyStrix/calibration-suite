from calsuite import doctor, store as storemod


def _record(kind, device, *, created, status="ok", provenance="measured", result=None):
    return storemod.Record(
        schema=1,
        id=f"{kind}-{created}",
        kind=kind,
        device=device,
        created=created,
        status=status,
        provenance=provenance,
        result=result or {},
    )


CAMERA = {"kind": "camera", "model": "Canon EOS R100", "id": "canon-eos-r100-abcd1234", "firmware": "1.0.0"}
DISPLAY = {"kind": "display", "model": "CSOT T3", "id": "csot-t3-deadbeef", "firmware": ""}


def test_check_tools_reports_present_and_missing(monkeypatch):
    from calsuite import tools as toolsmod

    monkeypatch.setattr(toolsmod, "which", lambda name: "/usr/bin/" + name if name == "dcraw" else None)
    findings = doctor.check_tools()
    by_name = {f.message.split(":")[0]: f for f in findings}
    assert by_name["dcraw"].severity == "info"
    assert by_name["exiftool"].severity == "warning"
    assert len(findings) == len(toolsmod.KNOWN_TOOLS)


def test_check_stale_records_fires_on_old_record(tmp_path):
    st = storemod.Store(tmp_path)
    old = _record("camera.bias", CAMERA, created="20200101T000000Z")
    st.save(old)
    findings = doctor.check_stale_records(st)
    assert len(findings) == 1
    assert "camera.bias" in findings[0].message


def test_check_stale_records_silent_on_fresh_record(tmp_path):
    st = storemod.Store(tmp_path)
    fresh = _record("camera.bias", CAMERA, created=storemod.utcnow_stamp())
    st.save(fresh)
    assert doctor.check_stale_records(st) == []


def test_check_firmware_changes_fires_when_records_disagree(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("camera.bias", CAMERA, created="20250101T000000Z"))
    st.save(_record("camera.bias", {**CAMERA, "firmware": "2.0.0"}, created=storemod.utcnow_stamp()))
    findings = doctor.check_firmware_changes(st)
    assert len(findings) == 1
    assert CAMERA["id"] in findings[0].message


def test_check_firmware_changes_silent_with_one_firmware(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("camera.bias", CAMERA, created="20250101T000000Z"))
    st.save(_record("camera.ptc", CAMERA, created=storemod.utcnow_stamp()))
    assert doctor.check_firmware_changes(st) == []


def test_check_edid_changes_fires_when_hash_differs(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("display.nominal", DISPLAY, created=storemod.utcnow_stamp(), result={"edid_hash": "old-hash"}))
    findings = doctor.check_edid_changes(st, edid_blobs={"card0-DP-1": b"\x00" * 200})
    # A fake 200-byte blob won't parse as valid EDID -- this only proves the
    # function handles that (skips, no crash); the real assertion is below.
    assert findings == []


def test_check_edid_changes_fires_with_real_edid(tmp_path):
    from calsuite import devices as devicesmod

    fixtures = __import__("pathlib").Path(__file__).parent / "fixtures"
    edid_files = list(fixtures.glob("*edid*")) if fixtures.is_dir() else []
    if not edid_files:
        import pytest

        pytest.skip("no EDID fixture available")
    data = edid_files[0].read_bytes()
    info = devicesmod.parse_edid(data)
    ref = devicesmod.display_ref(info)

    st = storemod.Store(tmp_path)
    st.save(_record("display.nominal", ref.to_dict(), created=storemod.utcnow_stamp(), result={"edid_hash": "not-the-real-hash"}))
    findings = doctor.check_edid_changes(st, edid_blobs={"card0-eDP-1": data})
    assert len(findings) == 1
    assert ref.id in findings[0].message


def test_check_profiles_without_recent_validation_fires(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("display.profile", DISPLAY, created="20250101T000000Z"))
    findings = doctor.check_profiles_without_recent_validation(st)
    assert len(findings) == 1
    assert DISPLAY["id"] in findings[0].message


def test_check_profiles_without_recent_validation_silent_when_validated_after(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("display.profile", DISPLAY, created="20250101T000000Z"))
    st.save(_record("display.validation", DISPLAY, created="20250102T000000Z"))
    assert doctor.check_profiles_without_recent_validation(st) == []


def test_check_profiles_without_recent_validation_fires_when_later_attempt_was_refused(tmp_path):
    # A later re-profile attempt that failed must not mask the fact that
    # the last *good* profile (still the one actually installed) has no
    # recent validation -- picking the literal latest record regardless of
    # status used to skip this check entirely once any newer refused
    # attempt existed.
    st = storemod.Store(tmp_path)
    st.save(_record("display.profile", DISPLAY, created="20250101T000000Z", status="ok"))
    st.save(_record("display.profile", DISPLAY, created="20250201T000000Z", status="refused"))
    findings = doctor.check_profiles_without_recent_validation(st)
    assert len(findings) == 1
    assert DISPLAY["id"] in findings[0].message


def test_check_profiles_without_recent_validation_silent_when_validated_in_the_same_second(tmp_path):
    # ``created`` is second-resolution (store.utcnow_stamp) -- a validation
    # that lands in the exact same second as the profile it validates (a
    # fast automated pipeline, or just two quick commands) is not provably
    # "before" it, so strict "newer than" must not treat this as unvalidated.
    st = storemod.Store(tmp_path)
    st.save(_record("display.profile", DISPLAY, created="20250101T000000Z"))
    st.save(_record("display.validation", DISPLAY, created="20250101T000000Z"))
    assert doctor.check_profiles_without_recent_validation(st) == []


def test_check_unsuperseded_refusals_fires(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("camera.ptc", CAMERA, created=storemod.utcnow_stamp(), status="refused"))
    findings = doctor.check_unsuperseded_refusals(st)
    assert len(findings) == 1
    assert "refused" in findings[0].message


def test_check_unsuperseded_refusals_silent_when_superseded(tmp_path):
    st = storemod.Store(tmp_path)
    st.save(_record("camera.ptc", CAMERA, created="20250101T000000Z", status="refused"))
    st.save(_record("camera.ptc", CAMERA, created="20250102T000000Z", status="ok"))
    assert doctor.check_unsuperseded_refusals(st) == []


def test_run_and_format_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    report = doctor.run(tmp_path)
    text = doctor.format_report(report)
    assert "External tools" in text
    assert isinstance(report.ok, bool)


def test_doctor_report_ok_false_when_any_warning():
    report = doctor.DoctorReport()
    report.add("tools", "missing thing")
    assert report.ok is False
    report2 = doctor.DoctorReport()
    report2.add("tools", "present thing", severity="info")
    assert report2.ok is True

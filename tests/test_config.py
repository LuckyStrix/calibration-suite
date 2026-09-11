from pathlib import Path

from calsuite import config


def test_records_dir_default():
    assert config.records_dir() == config.REPO_ROOT / "records"


def test_records_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CALSUITE_RECORDS", str(tmp_path))
    assert config.records_dir() == tmp_path


def test_captures_dir_default():
    assert config.captures_dir() == Path.home() / "calsuite-captures"


def test_captures_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CALSUITE_CAPTURES", str(tmp_path))
    assert config.captures_dir() == tmp_path

import os

import pytest

from calsuite import tools


def _make_script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def test_which_finds_fake_on_path(tmp_path, monkeypatch):
    _make_script(tmp_path, "fakecmd", "#!/bin/sh\necho hi\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    assert tools.which("fakecmd") is not None


def test_which_missing_returns_none():
    assert tools.which("definitely-not-a-real-binary-xyz") is None


def test_run_captures_stdout(tmp_path, monkeypatch):
    _make_script(tmp_path, "fakecmd", "#!/bin/sh\necho hello\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    result = tools.run(["fakecmd"])
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_raises_on_missing_binary():
    with pytest.raises(tools.ToolError):
        tools.run(["definitely-not-a-real-binary-xyz"])


def test_run_raises_on_nonzero_exit(tmp_path, monkeypatch):
    _make_script(tmp_path, "failcmd", "#!/bin/sh\necho oops 1>&2\nexit 3\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with pytest.raises(tools.ToolError) as excinfo:
        tools.run(["failcmd"])
    assert "oops" in str(excinfo.value)


def test_run_check_false_does_not_raise(tmp_path, monkeypatch):
    _make_script(tmp_path, "failcmd", "#!/bin/sh\nexit 3\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    result = tools.run(["failcmd"], check=False)
    assert result.returncode == 3


def test_run_times_out(tmp_path, monkeypatch):
    _make_script(tmp_path, "slowcmd", "#!/bin/sh\nsleep 5\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with pytest.raises(tools.ToolError):
        tools.run(["slowcmd"], timeout=0.2)

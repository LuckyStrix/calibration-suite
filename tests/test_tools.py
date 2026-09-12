import pytest

from calsuite import tools


def test_which_finds_fake_on_path(fake_bin):
    fake_bin("fakecmd", "print('hi')\n")
    assert tools.which("fakecmd") is not None


def test_which_missing_returns_none():
    assert tools.which("definitely-not-a-real-binary-xyz") is None


def test_run_captures_stdout(fake_bin):
    fake_bin("fakecmd", "print('hello')\n")
    result = tools.run(["fakecmd"])
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_run_raises_on_missing_binary():
    with pytest.raises(tools.ToolError):
        tools.run(["definitely-not-a-real-binary-xyz"])


def test_run_raises_on_nonzero_exit(fake_bin):
    fake_bin("failcmd", "import sys\nprint('oops', file=sys.stderr)\nsys.exit(3)\n")
    with pytest.raises(tools.ToolError) as excinfo:
        tools.run(["failcmd"])
    assert "oops" in str(excinfo.value)


def test_run_check_false_does_not_raise(fake_bin):
    fake_bin("failcmd", "import sys\nsys.exit(3)\n")
    result = tools.run(["failcmd"], check=False)
    assert result.returncode == 3


def test_run_times_out(fake_bin):
    fake_bin("slowcmd", "import time\ntime.sleep(5)\n")
    with pytest.raises(tools.ToolError):
        tools.run(["slowcmd"], timeout=0.2)

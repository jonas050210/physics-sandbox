"""Tests for the stdlib launcher helpers in main.py (loaded by path; the repo root is not a package)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("launcher_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_version_gate(launcher):
    with pytest.raises(launcher.LauncherError, match="3.11"):
        launcher.check_python((3, 8, 10))
    assert launcher.check_python((3, 12, 0)) == []
    assert launcher.check_python((3, 15, 0))  # warns, does not fail


def test_fingerprint_changes_with_requirements(launcher, tmp_path):
    a = tmp_path / "requirements.txt"
    a.write_text("numpy==1\n")
    f1 = launcher.requirements_fingerprint([a])
    assert f1 == launcher.requirements_fingerprint([a])
    a.write_text("numpy==2\n")
    assert launcher.requirements_fingerprint([a]) != f1


def test_stamp_round_trip(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "STAMP_DIR", tmp_path)
    assert launcher.read_stamp("runtime") is None
    launcher.write_stamp("runtime", "abc")
    assert launcher.read_stamp("runtime") == "abc"
    launcher.clear_stamp("runtime")
    assert launcher.read_stamp("runtime") is None


def test_venv_python_path_is_inside_project(launcher):
    assert str(launcher.venv_python_path()).startswith(str(launcher.VENV_DIR))


def test_pinned_requirements_are_exact(launcher):
    lines = [l.strip() for l in launcher.REQUIREMENTS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    assert all("==" in line for line in lines), lines
    names = {line.split("==")[0] for line in lines}
    assert {"mujoco", "panda3d", "numpy"} <= names


def test_help_flag_exits_zero(launcher, capsys):
    assert launcher.main(["--help-app"]) == 0
    assert "--headless" in capsys.readouterr().out


def test_check_command_runs(launcher, capsys):
    code = launcher.main(["--check"])
    out = capsys.readouterr().out
    assert "Physics Sandbox prerequisite check" in out
    assert code in (0, 1)

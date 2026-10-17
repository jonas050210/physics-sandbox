import json
import subprocess
import sys
from pathlib import Path

import pytest

from physics_sandbox.cli import main
from physics_sandbox.core.demo import tower_scene
from physics_sandbox.core.serialization import save_scene

ROOT = Path(__file__).resolve().parents[1]


def test_headless_cli_writes_report(tmp_path, capsys):
    report_path = tmp_path / "r.json"
    code = main(["--headless", "--duration", "0.5", "--report", str(report_path)])
    assert code == 0
    data = json.loads(report_path.read_text())
    assert data["format"] == "physics-sandbox.headless-report"
    assert data["steps"] == 500
    assert data["bodies"] == 7
    assert data["finite"] and data["ok"]
    assert "headless run ok=True" in capsys.readouterr().out


def test_headless_cli_prints_json_without_report(capsys):
    assert main(["--headless", "--duration", "0.1"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["simulated_seconds"] == pytest.approx(0.1, abs=1e-6)


def test_headless_cli_loads_scene_file(tmp_path, capsys):
    path = save_scene(tower_scene(), tmp_path / "scene.json")
    assert main(["--headless", "--duration", "0.1", "--scene", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["scene"] == "Turm"


def test_invalid_scene_file_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "nope"}')
    assert main(["--headless", "--scene", str(bad)]) == 2
    assert "not a Physics Sandbox scene" in capsys.readouterr().err


def test_module_entry_point_runs_headless(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "physics_sandbox", "--headless", "--duration", "0.05"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True

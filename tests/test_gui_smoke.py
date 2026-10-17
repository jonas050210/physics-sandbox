"""Real-window smoke test. Runs the application in a subprocess so that the Panda3D singleton
does not clash with the windowless tests. Skipped automatically without a display or OpenGL."""

import ctypes.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) or sys.platform != "linux"
HAS_GL = sys.platform != "linux" or ctypes.util.find_library("GL") is not None


@pytest.mark.gui
@pytest.mark.skipif(not (HAS_DISPLAY and HAS_GL), reason="needs a display and libGL")
def test_window_starts_renders_frames_and_exits():
    result = subprocess.run(
        [sys.executable, "-m", "physics_sandbox", "--exit-after-frames", "30", "--lang", "en"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "gui smoke ok: 30 frames" in result.stdout

#!/usr/bin/env python3
"""Physics Sandbox launcher.

Usage (from the repository root)::

    python3 main.py                    # start the 3D application (first run prepares everything)
    python3 main.py --headless         # headless simulation smoke run (no window)
    python3 main.py --test             # install dev tools if needed and run the test suite
    python3 main.py --check            # report prerequisites without changing anything
    python3 main.py --setup-only       # only prepare .venv and dependencies
    python3 main.py --reinstall        # force reinstalling the pinned dependencies

Any other arguments are forwarded to the application (see ``python3 main.py --help-app``).

This file deliberately uses only the Python standard library, because it has to run
before the project's own dependencies exist. Everything else lives in the project-local
``.venv`` directory, which can be deleted at any time; the next launch recreates it.
"""

from __future__ import annotations

import ctypes.util
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"
STAMP_DIR = VENV_DIR
MIN_PYTHON = (3, 11)
TESTED_MAX_PYTHON = (3, 14)  # newest minor version with verified wheels at the time of writing

LAUNCHER_FLAGS = {"--setup-only", "--check", "--reinstall", "--test"}
HELP_APP = """Application options (forwarded to `python -m physics_sandbox`):
  --lang {de,en}          interface language (default: de)
  --scene PATH            load a scene file (JSON) on start (default: built-in tower demo)
  --empty                 start with an empty baseplate
  --width N / --height N  window size in pixels (default 1440 x 900)
  --no-vsync              disable vertical sync
  --headless              run without a window (see --duration, --report)
  --duration SECONDS      headless: simulated seconds (default 3)
  --timestep SECONDS      physics timestep (default 0.001)
  --integrator {implicitfast,RK4}
  --report PATH           headless: write the JSON report to PATH (default: print to stdout)
"""


class LauncherError(RuntimeError):
    """A problem the user must fix; printed without a traceback."""


# ----------------------------------------------------------------------------- output helpers

def info(message: str) -> None:
    print(f"[physics-sandbox] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[physics-sandbox] WARNING: {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- prerequisites

def check_python(version_info: Sequence[int] = sys.version_info) -> List[str]:
    """Return warnings; raise LauncherError if the interpreter is too old."""
    current = tuple(version_info[:2])
    if current < MIN_PYTHON:
        raise LauncherError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required (found {current[0]}.{current[1]}).\n"
            "Install a newer Python, e.g. on Ubuntu: sudo apt install python3.12 python3.12-venv"
        )
    warnings = []
    if current > TESTED_MAX_PYTHON:
        warnings.append(
            f"Python {current[0]}.{current[1]} is newer than the last tested version "
            f"{TESTED_MAX_PYTHON[0]}.{TESTED_MAX_PYTHON[1]}; wheels may be missing."
        )
    return warnings


def venv_python_path(venv_dir: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def display_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def gl_library_available() -> bool:
    if not sys.platform.startswith("linux"):
        return True
    return ctypes.util.find_library("GL") is not None


def requirements_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{platform.python_implementation()}-{sys.version_info[:2]}-{platform.machine()}".encode())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_stamp(key: str) -> Optional[str]:
    path = STAMP_DIR / f".deps-{key}.sha256"
    return path.read_text().strip() if path.exists() else None


def write_stamp(key: str, value: str) -> None:
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    (STAMP_DIR / f".deps-{key}.sha256").write_text(value + "\n")


def clear_stamp(key: str) -> None:
    path = STAMP_DIR / f".deps-{key}.sha256"
    if path.exists():
        path.unlink()


def venv_version(python: Path) -> Optional[tuple]:
    """Interpreter (major, minor) of an existing venv python, or None if it cannot run."""
    try:
        out = subprocess.run(
            [str(python), "-c", "import sys, json; print(json.dumps(list(sys.version_info[:2])))"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return tuple(json.loads(out.stdout.strip()))
    except json.JSONDecodeError:
        return None


def ensure_venv() -> Path:
    python = venv_python_path()
    if python.exists() and venv_version(python) == tuple(sys.version_info[:2]):
        return python
    if VENV_DIR.exists():
        info("Existing .venv does not match this Python; recreating it.")
        shutil.rmtree(VENV_DIR)
    info(f"Creating project-local virtual environment in {VENV_DIR.name}/ ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR, ignore_errors=True)
        raise LauncherError(
            "Could not create the virtual environment.\n"
            "On Ubuntu, install the venv support package (once, with administrator rights):\n"
            f"    sudo apt install python{sys.version_info[0]}.{sys.version_info[1]}-venv   (or: sudo apt install python3-venv)\n"
            f"Details: {exc}"
        ) from exc
    if not python.exists():
        raise LauncherError(f"virtual environment was created but {python} is missing")
    return python


def ensure_requirements(python: Path, key: str, files: Sequence[Path], force: bool = False) -> bool:
    """Install requirement files when the fingerprint changed. Returns True if pip ran."""
    fingerprint = requirements_fingerprint(files)
    if not force and read_stamp(key) == fingerprint:
        info(f"Dependencies ({key}) already installed - reusing the existing environment.")
        return False
    info(f"Installing dependencies ({key}) into .venv - this can take a few minutes on first run ...")
    cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    for f in files:
        cmd += ["-r", str(f)]
    clear_stamp(key)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise LauncherError(
            "Dependency installation failed. Check your internet connection (pypi.org and "
            "files.pythonhosted.org must be reachable) and run `python3 main.py` again."
        )
    write_stamp(key, fingerprint)
    return True


def verify_runtime(python: Path) -> None:
    code = "import numpy, mujoco, panda3d.core; print(mujoco.__version__, panda3d.core.PandaSystem.getVersionString())"
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise LauncherError(
            "The installed dependencies could not be imported. Re-run with `python3 main.py --reinstall`.\n"
            f"{result.stderr.strip()}"
        )


def run_checks(gui: bool) -> List[str]:
    """Return a list of human-readable problems; empty means OK."""
    problems = []
    if gui and not display_available():
        problems.append(
            "No graphical display detected (DISPLAY/WAYLAND_DISPLAY is not set). Start the app from a desktop "
            "session, or use `python3 main.py --headless` for a window-less simulation."
        )
    if gui and not gl_library_available():
        problems.append(
            "OpenGL library libGL.so.1 was not found. Install it (once, with administrator rights): "
            "sudo apt install libgl1 libgl1-mesa-dri"
        )
    return problems


# ----------------------------------------------------------------------------- commands

def cmd_check() -> int:
    print("Physics Sandbox prerequisite check")
    print(f"  project root : {ROOT}")
    print(f"  python       : {platform.python_version()} ({sys.executable})")
    for w in check_python():
        print(f"  note         : {w}")
    print(f"  venv         : {'present' if venv_python_path().exists() else 'not created yet'}")
    print(f"  display      : {'available' if display_available() else 'not available (GUI cannot start)'}")
    print(f"  libGL        : {'found' if gl_library_available() else 'NOT FOUND'}")
    problems = run_checks(gui=True)
    for p in problems:
        print(f"  problem      : {p}")
    return 0 if not problems else 1


def prepare(force: bool = False, dev: bool = False) -> Path:
    for w in check_python():
        warn(w)
    python = ensure_venv()
    ensure_requirements(python, "runtime", [REQUIREMENTS], force=force)
    if force or read_stamp("runtime-verified") is None:
        verify_runtime(python)
        write_stamp("runtime-verified", requirements_fingerprint([REQUIREMENTS]))
    if dev:
        ensure_requirements(python, "dev", [REQUIREMENTS, REQUIREMENTS_DEV], force=force)
    return python


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help-app" in argv:
        print(HELP_APP)
        return 0
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        print(HELP_APP)
        return 0
    try:
        if "--check" in argv:
            return cmd_check()

        force = "--reinstall" in argv
        app_args = [a for a in argv if a not in LAUNCHER_FLAGS]
        is_test = "--test" in argv
        headless = "--headless" in app_args

        python = prepare(force=force, dev=is_test)
        if "--setup-only" in argv:
            info(f"Environment ready: {python}")
            return 0

        if is_test:
            info("Running the test suite ...")
            return subprocess.run([str(python), "-m", "pytest", *app_args], cwd=str(ROOT)).returncode

        problems = run_checks(gui=not headless)
        if problems:
            raise LauncherError("\n".join(problems))
        info("Starting Physics Sandbox ..." if not headless else "Starting headless simulation ...")
        return subprocess.run([str(python), "-m", "physics_sandbox", *app_args], cwd=str(ROOT)).returncode
    except LauncherError as exc:
        print(f"\n[physics-sandbox] ERROR: {exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

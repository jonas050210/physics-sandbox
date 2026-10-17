"""Application configuration. Defaults live here; the CLI overrides them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    language: str = "de"
    width: int = 1440
    height: int = 900
    vsync: bool = True
    scene_path: Optional[str] = None
    start_empty: bool = False
    physics_timestep: float = 0.001
    integrator: str = "implicitfast"
    saves_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "saves")
    examples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "examples")
    baseplate_size: tuple = (24.0, 24.0)
    grid: float = 0.125
    vertical_fov_deg: float = 42.0

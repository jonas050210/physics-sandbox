"""Headless simulation: advance the physics without a window and report the result.

Used by ``python3 main.py --headless`` and by the automated smoke tests. The report is
plain JSON so that batch experiments (Phase 5) can aggregate many runs.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .core.components import DEFAULT_REGISTRY
from .core.demo import tower_scene
from .core.scene import Scene
from .core.serialization import load_scene
from .sim.simulation import Simulation

REPORT_FORMAT = "physics-sandbox.headless-report"
REPORT_VERSION = 1


def run_headless(scene: Scene, seconds: float = 3.0, timestep: float = 0.001,
                 integrator: str = "implicitfast") -> Dict[str, Any]:
    if seconds < 0:
        raise ValueError("duration must be non-negative")
    sim = Simulation(timestep=timestep, integrator=integrator)
    sim.start(scene)
    wall0 = time.perf_counter()
    sim.advance_seconds(seconds)
    wall = time.perf_counter() - wall0
    state = sim.state
    assert state is not None
    finite = all(all(math.isfinite(v) for v in b.position) and all(math.isfinite(v) for v in b.linear_velocity)
                 for b in state.bodies.values())
    bodies = {
        str(bid): {
            "type": scene.get(bid).type_key,
            "position": [round(v, 6) for v in b.position],
            "orientation": [round(v, 6) for v in b.orientation],
            "linear_velocity": [round(v, 6) for v in b.linear_velocity],
            "angular_velocity": [round(v, 6) for v in b.angular_velocity],
        }
        for bid, b in sorted(state.bodies.items())
    }
    report = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "scene": scene.name,
        "backend": sim.backend.name if sim.backend else None,
        "integrator": integrator,
        "timestep": timestep,
        "steps": state.steps,
        "simulated_seconds": round(state.time, 6),
        "wall_seconds": round(wall, 4),
        "realtime_factor": round(state.time / wall, 2) if wall > 0 else None,
        "bodies": len(state.bodies),
        "contacts": state.contacts,
        "max_penetration_m": round(state.max_penetration, 6),
        "max_speed_mps": round(state.max_speed(), 6),
        "finite": finite,
        "settled": state.max_speed() < 0.05,
        "states": bodies,
    }
    report["ok"] = bool(finite and report["max_penetration_m"] < 0.02)
    return report


def scene_for_headless(scene_path: Optional[str], language: str = "de") -> Scene:
    if scene_path:
        return load_scene(scene_path, DEFAULT_REGISTRY)
    return tower_scene(language=language)


def write_report(report: Dict[str, Any], path: Optional[str]) -> str:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    return text

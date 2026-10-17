"""Built-in demo scenes used by the GUI, the headless runner and the examples."""

from __future__ import annotations

from .components import DEFAULT_REGISTRY, ComponentRegistry
from .geometry import quat_from_yaw_degrees
from .scene import Scene


def tower_scene(registry: ComponentRegistry = DEFAULT_REGISTRY, language: str = "de") -> Scene:
    """A plate with a four-cube tower, a beam on the baseplate and a ball released above the tower.

    Geometry check: the plate (height 0.25) rests on z=0, so its centre is at z=0.125 and its top at
    z=0.25. Each 0.5 m cube sits on the one below, so the cube centres are at z = 0.5, 1.0, 1.5, 2.0.
    """
    scene = Scene(registry=registry, name="Turm" if language == "de" else "Tower")
    scene.add("plate", (0.0, 0.0, 0.125))
    for level in range(4):
        scene.add("cube", (0.0, 0.0, 0.5 + 0.5 * level))
    scene.add("beam", (1.25, -1.0, 0.125), quat_from_yaw_degrees(90.0))
    scene.add("ball", (0.05, 0.0, 3.0))
    return scene

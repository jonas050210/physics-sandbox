"""Simulation controller: edit / running / paused modes and fixed-timestep stepping.

The controller is independent of any window. The GUI calls :meth:`Simulation.update`
once per frame with the wall-clock delta; the headless runner calls :meth:`advance_seconds`.
"""

from __future__ import annotations

import enum
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..core.scene import Scene
from .backend import BodySpec, PhysicsBackend, WorldSpec, WorldState

SPEED_STEPS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
MAX_FRAME_DT = 0.05  # cap per frame to avoid a "spiral of death" after stalls


class SimMode(str, enum.Enum):
    EDIT = "edit"
    RUNNING = "running"
    PAUSED = "paused"


def world_spec_from_scene(scene: Scene, gravity=(0.0, 0.0, -9.81), timestep: float = 0.001,
                          integrator: str = "implicitfast") -> WorldSpec:
    bodies = []
    for c in scene:
        ctype = scene.component_type(c)
        bodies.append(
            BodySpec(
                id=c.id,
                shape=ctype.shape,
                size=tuple(float(v) for v in ctype.extents),  # type: ignore[arg-type]
                mass_kg=ctype.mass_kg,
                friction=ctype.friction,
                position=tuple(float(v) for v in c.position),  # type: ignore[arg-type]
                orientation=tuple(float(v) for v in c.orientation),  # type: ignore[arg-type]
                color=ctype.color,
                linear_velocity=tuple(float(v) for v in c.attributes.get("linear_velocity", (0, 0, 0))),  # type: ignore[arg-type]
                angular_velocity=tuple(float(v) for v in c.attributes.get("angular_velocity", (0, 0, 0))),  # type: ignore[arg-type]
            )
        )
    return WorldSpec(bodies=tuple(bodies), baseplate_size=scene.baseplate_size, gravity=gravity,
                     timestep=timestep, integrator=integrator)


def _default_backend() -> PhysicsBackend:
    from .mujoco_backend import MujocoBackend

    return MujocoBackend()


class Simulation:
    def __init__(self, backend_factory: Callable[[], PhysicsBackend] = _default_backend,
                 timestep: float = 0.001, gravity=(0.0, 0.0, -9.81), integrator: str = "implicitfast") -> None:
        self._factory = backend_factory
        self._timestep = timestep
        self._integrator = integrator
        self.gravity = tuple(gravity)
        self.backend: Optional[PhysicsBackend] = None
        self.mode = SimMode.EDIT
        self.speed_index = SPEED_STEPS.index(1.0)
        self._accumulator = 0.0
        self._state: Optional[WorldState] = None
        self._steps = 0

    # ------------------------------------------------------------------ queries
    @property
    def speed(self) -> float:
        return SPEED_STEPS[self.speed_index]

    @property
    def is_active(self) -> bool:
        return self.mode != SimMode.EDIT

    @property
    def state(self) -> Optional[WorldState]:
        return self._state

    @property
    def time(self) -> float:
        return 0.0 if self._state is None else self._state.time

    # ------------------------------------------------------------------ control
    def start(self, scene: Scene) -> WorldState:
        """Snapshot ``scene`` into a fresh backend and start running (paused=False)."""
        world = world_spec_from_scene(scene, gravity=self.gravity, timestep=self._timestep,
                                      integrator=self._integrator)
        self.backend = self._factory()
        self.backend.load(world)
        self._accumulator = 0.0
        self._steps = 0
        self.mode = SimMode.RUNNING
        self._state = self.backend.state()
        return self._state

    def pause(self) -> None:
        if self.mode == SimMode.RUNNING:
            self.mode = SimMode.PAUSED

    def resume(self) -> None:
        if self.mode == SimMode.PAUSED:
            self.mode = SimMode.RUNNING

    def toggle_pause(self) -> None:
        if self.mode == SimMode.RUNNING:
            self.pause()
        elif self.mode == SimMode.PAUSED:
            self.resume()

    def reset(self) -> None:
        """Discard the physics world; the editor's scene is untouched."""
        self.backend = None
        self._state = None
        self._accumulator = 0.0
        self._steps = 0
        self.mode = SimMode.EDIT

    def restart(self) -> None:
        """Return to the initial state of the running world without leaving simulation mode."""
        if self.backend is None:
            return
        self.backend.reset()
        self._accumulator = 0.0
        self._steps = 0
        self._state = self.backend.state()

    def faster(self) -> None:
        self.speed_index = min(self.speed_index + 1, len(SPEED_STEPS) - 1)

    def slower(self) -> None:
        self.speed_index = max(self.speed_index - 1, 0)

    # ------------------------------------------------------------------ stepping
    def update(self, wall_dt: float) -> Optional[WorldState]:
        """Advance by ``wall_dt`` seconds of wall-clock time (scaled by the speed factor)."""
        if self.mode != SimMode.RUNNING or self.backend is None:
            return None
        dt = self.backend.timestep
        self._accumulator += min(max(wall_dt, 0.0), MAX_FRAME_DT) * self.speed
        steps = int(math.floor(self._accumulator / dt + 1e-9))
        if steps > 0:
            self._advance_steps(steps)
            self._accumulator -= steps * dt
        return self._state

    def step(self, steps: int = 1) -> Optional[WorldState]:
        """Manually advance while paused (single-step debugging)."""
        if self.backend is None or self.mode == SimMode.EDIT:
            return None
        self._advance_steps(int(steps))
        return self._state

    def advance_seconds(self, seconds: float) -> WorldState:
        """Deterministic advance used by headless runs. Requires an active backend."""
        if self.backend is None:
            raise RuntimeError("simulation has not been started")
        steps = int(round(seconds / self.backend.timestep))
        self._advance_steps(steps)
        return self._state  # type: ignore[return-value]

    def _advance_steps(self, steps: int) -> None:
        assert self.backend is not None
        self.backend.advance(steps)
        self._steps += steps
        self._state = self.backend.state()

    # ------------------------------------------------------------------ poses for rendering
    def poses(self) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        if self._state is None:
            return None
        return {
            bid: (np.array(b.position), np.array(b.orientation)) for bid, b in self._state.bodies.items()
        }

"""Engine-neutral physics interface.

The editor produces a :class:`WorldSpec` (plain data). A :class:`PhysicsBackend`
turns it into a running world and reports :class:`WorldState` snapshots. Future
mechanisms (joints, motors, sensors, aerodynamic forces) extend these dataclasses
and the backend, not the editor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

from ..core.components import Shape

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BodySpec:
    """One dynamic rigid body (a placed component)."""

    id: int
    shape: Shape
    size: Vec3  # full extents for BOX; (diameter, diameter, diameter) for SPHERE; (diameter, diameter, height) for CYLINDER
    mass_kg: float
    friction: float
    position: Vec3
    orientation: Quat = (1.0, 0.0, 0.0, 0.0)
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class WorldSpec:
    bodies: Tuple[BodySpec, ...] = ()
    baseplate_size: Tuple[float, float] = (24.0, 24.0)  # static plate, top surface at z = 0
    baseplate_thickness: float = 0.1
    floor_depth: float = 1.0  # static floor plane at z = -floor_depth (visual + collision)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    timestep: float = 0.001  # seconds per physics step (1 kHz)
    friction: float = 0.8
    contact_timeconst: float = 0.002  # soft-contact time constant (s); MuJoCo clamps it to >= 2*timestep
    integrator: str = "implicitfast"  # "implicitfast" (default, fast) or "RK4" (more accurate, ~2.5x cost)


@dataclass(frozen=True)
class BodyState:
    id: int
    position: Vec3
    orientation: Quat
    linear_velocity: Vec3
    angular_velocity: Vec3


@dataclass(frozen=True)
class WorldState:
    time: float
    steps: int
    bodies: Mapping[int, BodyState]
    contacts: int = 0
    max_penetration: float = 0.0  # metres; 0 when no contact penetrates
    extra: Dict[str, float] = field(default_factory=dict)

    def position(self, body_id: int) -> Vec3:
        return self.bodies[body_id].position

    def max_speed(self) -> float:
        best = 0.0
        for b in self.bodies.values():
            v = b.linear_velocity
            best = max(best, (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5)
        return best


class PhysicsBackend(ABC):
    """Minimal contract every physics engine adapter must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def load(self, world: WorldSpec) -> None:
        """Build the engine model and reset to the initial state of ``world``."""

    @abstractmethod
    def advance(self, steps: int) -> None:
        """Advance the world by ``steps`` fixed timesteps."""

    @abstractmethod
    def state(self) -> WorldState:
        """Return a snapshot of the current state."""

    @abstractmethod
    def reset(self) -> None:
        """Return to the initial state passed to :meth:`load`."""

    @property
    @abstractmethod
    def timestep(self) -> float:
        """Fixed timestep in seconds."""

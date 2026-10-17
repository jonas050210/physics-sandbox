"""Component types and component instances.

A *component type* (``ComponentType``) describes what a part is: its geometric
primitive, default size, mass, colour and localised names. It is static data.

A *component instance* (``Component``) is one placed part: an id, a type key,
a pose and free-form ``attributes``. Instances are immutable value objects;
the :class:`~physics_sandbox.core.scene.Scene` replaces them on change, which
makes undo/redo and snapshotting trivial.

Future component families (wheels, motors, joints, sensors, custom parts) are
added by registering more ``ComponentType`` objects and by putting their
parameters into ``Component.attributes``. The simulation layer maps attributes
to physics; the editor never needs to know the physics details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .geometry import IDENTITY_QUAT, Obb, Quat, Vec3, quat_to_matrix, vec3


class Shape(str, Enum):
    """Collision/visual primitives supported in Phase 1."""

    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"  # axis along local Z


class Language(str, Enum):
    DE = "de"
    EN = "en"


@dataclass(frozen=True)
class ComponentType:
    """Static description of a kind of part.

    ``extents`` are FULL sizes in metres in the local frame:
    * BOX: (size_x, size_y, size_z)
    * SPHERE: (diameter, diameter, diameter)
    * CYLINDER: (diameter, diameter, height)
    Keeping extents as multiples of the placement grid keeps every part on-grid.
    """

    key: str
    shape: Shape
    extents: Tuple[float, float, float]
    mass_kg: float
    color: Tuple[float, float, float]
    name_de: str
    name_en: str
    friction: float = 0.8
    category: str = "building_block"

    def __post_init__(self) -> None:
        if any(e <= 0 for e in self.extents):
            raise ValueError(f"component type {self.key!r}: extents must be positive")
        if self.mass_kg <= 0:
            raise ValueError(f"component type {self.key!r}: mass must be positive")
        if self.shape == Shape.SPHERE and not (self.extents[0] == self.extents[1] == self.extents[2]):
            raise ValueError(f"component type {self.key!r}: sphere extents must be equal")

    @property
    def half_extents(self) -> np.ndarray:
        return np.array(self.extents, dtype=float) / 2.0

    def name(self, language: str) -> str:
        return self.name_de if language == Language.DE.value else self.name_en


BUILTIN_TYPES: Tuple[ComponentType, ...] = (
    ComponentType("cube", Shape.BOX, (0.5, 0.5, 0.5), 2.0, (0.91, 0.51, 0.23), "Würfel", "Cube"),
    ComponentType("plate", Shape.BOX, (1.0, 1.0, 0.25), 3.0, (0.31, 0.56, 0.84), "Platte", "Plate"),
    ComponentType("beam", Shape.BOX, (2.0, 0.25, 0.25), 1.5, (0.85, 0.76, 0.35), "Balken", "Beam"),
    ComponentType("ball", Shape.SPHERE, (0.5, 0.5, 0.5), 1.0, (0.88, 0.33, 0.33), "Kugel", "Ball", friction=0.6),
    ComponentType("cylinder", Shape.CYLINDER, (0.5, 0.5, 0.5), 1.5, (0.45, 0.78, 0.46), "Zylinder", "Cylinder"),
)


class ComponentRegistry:
    """Lookup table for component types. Extensible via :meth:`register`."""

    def __init__(self, types: Iterable[ComponentType] = BUILTIN_TYPES) -> None:
        self._types: Dict[str, ComponentType] = {}
        for t in types:
            self.register(t)

    def register(self, component_type: ComponentType) -> None:
        if component_type.key in self._types:
            raise ValueError(f"duplicate component type key: {component_type.key!r}")
        self._types[component_type.key] = component_type

    def get(self, key: str) -> ComponentType:
        try:
            return self._types[key]
        except KeyError as exc:
            raise KeyError(f"unknown component type: {key!r}") from exc

    def __contains__(self, key: object) -> bool:
        return key in self._types

    def __iter__(self):
        return iter(self._types.values())

    def __len__(self) -> int:
        return len(self._types)

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._types.keys())


DEFAULT_REGISTRY = ComponentRegistry()


@dataclass(frozen=True)
class Component:
    """One placed part. ``position`` is the centre of the part in metres."""

    id: int
    type_key: str
    position: Vec3
    orientation: Quat = IDENTITY_QUAT
    attributes: Mapping[str, object] = field(default_factory=dict)

    def with_pose(self, position: Vec3, orientation: Optional[Quat] = None) -> "Component":
        return Component(
            self.id,
            self.type_key,
            tuple(float(v) for v in position),  # type: ignore[arg-type]
            self.orientation if orientation is None else orientation,
            dict(self.attributes),
        )


def component_obb(component: Component, component_type: ComponentType, position: Optional[np.ndarray] = None,
                  orientation: Optional[Quat] = None) -> Obb:
    """Oriented bounding box of a placed component, optionally at an override pose."""
    center = vec3(component.position if position is None else position)
    rot = quat_to_matrix(component.orientation if orientation is None else orientation)
    return Obb(center=center, rotation=rot, half=component_type.half_extents)

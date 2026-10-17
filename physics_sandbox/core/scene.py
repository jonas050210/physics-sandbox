"""The editable scene model.

``Scene`` is the single source of truth for *what was built*: which components
exist, where they are, and what is selected. It knows nothing about rendering
or physics. The simulation layer receives an immutable snapshot of it, and the
renderer reads it (or simulation poses) to draw.

All mutations bump ``Scene.version`` so that observers can cheaply detect change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .components import Component, ComponentRegistry, DEFAULT_REGISTRY, Shape, component_obb
from .geometry import Hit, Quat, Ray, ray_obb, ray_sphere, vec3, IDENTITY_QUAT
from .geometry import aabb_overlap, quat_to_matrix

DEFAULT_BASEPLATE_SIZE: Tuple[float, float] = (24.0, 24.0)
DEFAULT_GRID = 0.125
_PLACEMENT_TOL = 1e-4

REASON_OVERLAP = "overlap"
REASON_BELOW_GROUND = "below_ground"
REASON_OUTSIDE = "outside_baseplate"
REASON_UNKNOWN_TYPE = "unknown_type"


@dataclass(frozen=True)
class PlacementCheck:
    valid: bool
    reason: Optional[str] = None
    blocking_id: Optional[int] = None


class Scene:
    def __init__(
        self,
        registry: ComponentRegistry = DEFAULT_REGISTRY,
        baseplate_size: Tuple[float, float] = DEFAULT_BASEPLATE_SIZE,
        grid: float = DEFAULT_GRID,
        name: str = "Unbenannt",
    ) -> None:
        if baseplate_size[0] <= 0 or baseplate_size[1] <= 0:
            raise ValueError("baseplate size must be positive")
        if grid <= 0:
            raise ValueError("grid must be positive")
        self.registry = registry
        self.baseplate_size = (float(baseplate_size[0]), float(baseplate_size[1]))
        self.grid = float(grid)
        self.name = name
        self._components: Dict[int, Component] = {}
        self._selection: Set[int] = set()
        self._next_id = 1
        self.version = 0

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self._components)

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._components

    def __iter__(self) -> Iterator[Component]:
        return iter(sorted(self._components.values(), key=lambda c: c.id))

    def get(self, component_id: int) -> Component:
        return self._components[component_id]

    def components(self) -> List[Component]:
        return list(self)

    def component_type(self, component: Component):
        return self.registry.get(component.type_key)

    def snapshot(self) -> Tuple[Component, ...]:
        """Immutable copy of the current components, ordered by id."""
        return tuple(self)

    @property
    def next_id(self) -> int:
        return self._next_id

    @property
    def selection(self) -> Tuple[int, ...]:
        return tuple(sorted(self._selection))

    def is_selected(self, component_id: int) -> bool:
        return component_id in self._selection

    # ------------------------------------------------------------------ mutation
    def _touch(self) -> None:
        self.version += 1

    def _validate_type(self, type_key: str) -> None:
        if type_key not in self.registry:
            raise KeyError(f"unknown component type: {type_key!r}")

    def add(self, type_key: str, position: Sequence[float], orientation: Quat = IDENTITY_QUAT,
            attributes: Optional[Mapping[str, object]] = None) -> Component:
        """Create a new component with a fresh id. Does not validate placement."""
        self._validate_type(type_key)
        component = Component(self._next_id, type_key, tuple(float(v) for v in vec3(position)),
                              tuple(orientation), dict(attributes or {}))  # type: ignore[arg-type]
        self.insert(component)
        return component

    def insert(self, component: Component) -> None:
        """Insert an existing component (keeps its id). Used by undo/redo and loading."""
        self._validate_type(component.type_key)
        if component.id in self._components:
            raise ValueError(f"component id {component.id} already exists")
        self._components[component.id] = component
        self._next_id = max(self._next_id, component.id + 1)
        self._touch()

    def remove(self, component_id: int) -> Component:
        component = self._components.pop(component_id)
        self._selection.discard(component_id)
        self._touch()
        return component

    def set_pose(self, component_id: int, position: Sequence[float], orientation: Optional[Quat] = None) -> Component:
        old = self._components[component_id]
        new = old.with_pose(tuple(vec3(position)), orientation if orientation is not None else old.orientation)  # type: ignore[arg-type]
        self._components[component_id] = new
        self._touch()
        return old

    def clear(self) -> None:
        self._components.clear()
        self._selection.clear()
        self._next_id = 1
        self._touch()

    # ------------------------------------------------------------------ selection
    def select(self, ids: Iterable[int], additive: bool = False) -> None:
        ids = [i for i in ids if i in self._components]
        if not additive:
            self._selection.clear()
        self._selection.update(ids)

    def toggle_selection(self, component_id: int) -> None:
        if component_id not in self._components:
            return
        if component_id in self._selection:
            self._selection.remove(component_id)
        else:
            self._selection.add(component_id)

    def clear_selection(self) -> None:
        self._selection.clear()

    # ------------------------------------------------------------------ geometry
    def obb_of(self, component: Component, position: Optional[np.ndarray] = None,
               orientation: Optional[Quat] = None):
        return component_obb(component, self.component_type(component), position, orientation)

    def bounds_of(self, component: Component, position: Optional[np.ndarray] = None,
                  orientation: Optional[Quat] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.obb_of(component, position, orientation).aabb()

    def half_height_along(self, type_key: str, orientation: Quat, axis: np.ndarray) -> float:
        """Half extent of a type (in the given orientation) measured along a world axis."""
        ctype = self.registry.get(type_key)
        rot = quat_to_matrix(orientation)
        ext = np.abs(rot) @ ctype.half_extents
        return float(np.abs(axis) @ ext)

    def check_placement(self, component: Component, ignore_ids: Iterable[int] = ()) -> PlacementCheck:
        """Validate that ``component`` may exist at its pose.

        Rules: stays on the baseplate footprint, does not go below the ground, and
        does not overlap any other component (AABB test, slightly conservative for
        round parts). Components listed in ``ignore_ids`` are excluded from the
        overlap test; use this when moving a group that is checked as a whole.
        """
        if component.type_key not in self.registry:
            return PlacementCheck(False, REASON_UNKNOWN_TYPE)
        mn, mx = self.bounds_of(component)
        half_x, half_y = self.baseplate_size[0] / 2.0, self.baseplate_size[1] / 2.0
        if mn[2] < -_PLACEMENT_TOL:
            return PlacementCheck(False, REASON_BELOW_GROUND)
        if mn[0] < -half_x - _PLACEMENT_TOL or mx[0] > half_x + _PLACEMENT_TOL or \
                mn[1] < -half_y - _PLACEMENT_TOL or mx[1] > half_y + _PLACEMENT_TOL:
            return PlacementCheck(False, REASON_OUTSIDE)
        ignored = set(ignore_ids)
        for other in self._components.values():
            if other.id == component.id or other.id in ignored:
                continue
            omn, omx = self.bounds_of(other)
            if aabb_overlap(mn, mx, omn, omx, _PLACEMENT_TOL):
                return PlacementCheck(False, REASON_OVERLAP, other.id)
        return PlacementCheck(True)

    def find_at_ray(self, ray: Ray, poses: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None
                    ) -> Optional[Tuple[int, Hit]]:
        """Nearest component hit by ``ray``.

        ``poses`` optionally overrides component poses (e.g. live simulation poses).
        Spheres are tested exactly, all other shapes against their oriented box.
        """
        best: Optional[Tuple[int, Hit]] = None
        for component in self._components.values():
            ctype = self.component_type(component)
            if poses is not None and component.id in poses:
                pos, ori = poses[component.id]
            else:
                pos, ori = np.asarray(component.position), component.orientation
            if ctype.shape == Shape.SPHERE:
                hit = ray_sphere(ray, np.asarray(pos, dtype=float), ctype.extents[0] / 2.0)
            else:
                hit = ray_obb(ray, component_obb(component, ctype, np.asarray(pos, dtype=float), ori))
            if hit is not None and (best is None or hit.t < best[1].t):
                best = (component.id, hit)
        return best

    def bounding_box_all(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._components:
            return None
        mins, maxs = zip(*(self.bounds_of(c) for c in self._components.values()))
        return np.min(mins, axis=0), np.max(maxs, axis=0)

    def total_mass(self) -> float:
        return float(sum(self.component_type(c).mass_kg for c in self._components.values()))

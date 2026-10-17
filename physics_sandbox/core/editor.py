"""Editor interaction logic: tools, placement candidates, dragging, rotation and undo.

This module turns pointer rays and key presses into validated scene edits. It has no
dependency on the window system: the GUI converts mouse positions to :class:`Ray`
objects and forwards events here. Every method that changes the scene goes through
``CommandHistory`` so that it can be undone.

User-facing messages are returned as message *keys* (see ``ui/i18n.py``) plus format
arguments, so the core stays language-neutral.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .commands import AddComponent, ChangePoses, CommandHistory, PoseChange, RemoveComponents
from .components import Component
from .geometry import Quat, Ray, quat_from_yaw_degrees, quat_multiply, ray_plane_z, snap, snap_vec
from .scene import REASON_BELOW_GROUND, REASON_OUTSIDE, REASON_OVERLAP, Scene


class Tool(str, enum.Enum):
    SELECT = "select"
    PLACE = "place"
    MOVE = "move"
    DELETE = "delete"


@dataclass(frozen=True)
class Ghost:
    """Preview of a component about to be placed."""

    type_key: str
    position: np.ndarray
    orientation: Quat
    valid: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class EditResult:
    ok: bool
    message: str = ""  # i18n key, empty when there is nothing to report
    args: Mapping[str, object] = field(default_factory=dict)


@dataclass
class _Drag:
    ids: Tuple[int, ...]
    start_positions: Dict[int, np.ndarray]
    start_orientations: Dict[int, Quat]
    plane_z: float
    grab_xy: np.ndarray
    delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moved: bool = False


_PLACEMENT_MESSAGES = {
    REASON_OVERLAP: "msg.blocked_overlap",
    REASON_BELOW_GROUND: "msg.blocked_ground",
    REASON_OUTSIDE: "msg.blocked_outside",
}


def placement_candidate(scene: Scene, type_key: str, ray: Ray, yaw_deg: float = 0.0,
                        poses: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None) -> Optional[Ghost]:
    """Where a new part of ``type_key`` would go if placed from ``ray``.

    The hit surface is either the baseplate (z = 0) or the nearest component face.
    The new part's centre is pushed out along the surface normal by its half extent
    and snapped to the grid. Returns None when the ray hits nothing.
    """
    orientation = quat_from_yaw_degrees(yaw_deg)
    plane_hit = ray_plane_z(ray, 0.0)
    comp_hit = scene.find_at_ray(ray, poses)
    if comp_hit is not None and (plane_hit is None or comp_hit[1].t < plane_hit.t):
        point = comp_hit[1].point
        normal = comp_hit[1].normal
    elif plane_hit is not None:
        point = plane_hit.point
        normal = np.array([0.0, 0.0, 1.0])
    else:
        return None
    if np.linalg.norm(normal) < 1e-9:
        normal = np.array([0.0, 0.0, 1.0])
    axis = int(np.argmax(np.abs(normal)))
    unit = np.zeros(3)
    unit[axis] = 1.0 if normal[axis] > 0 else -1.0
    half_along = scene.half_height_along(type_key, orientation, unit)
    raw = point + unit * half_along
    center = snap_vec(raw, scene.grid)
    half_z = scene.half_height_along(type_key, orientation, np.array([0.0, 0.0, 1.0]))
    center[2] = max(center[2], half_z)  # never below the ground
    candidate = Component(0, type_key, tuple(float(v) for v in center), orientation)  # type: ignore[arg-type]
    check = scene.check_placement(candidate)
    return Ghost(type_key, center, orientation, check.valid, check.reason)


class Editor:
    def __init__(self, scene: Scene, history: CommandHistory,
                 locked: Callable[[], bool] = lambda: False, palette_type: str = "cube") -> None:
        self.scene = scene
        self.history = history
        self._locked = locked
        self.tool = Tool.SELECT
        self.palette_type = palette_type
        self.place_yaw_deg = 0.0
        self.hover_id: Optional[int] = None
        self.ghost: Optional[Ghost] = None
        self._drag: Optional[_Drag] = None

    # ------------------------------------------------------------------ state
    @property
    def locked(self) -> bool:
        return bool(self._locked())

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    def set_tool(self, tool: Tool) -> None:
        self._drag = None
        self.tool = Tool(tool)
        self.ghost = None
        self.hover_id = None

    def select_palette(self, type_key: str) -> None:
        if type_key not in self.scene.registry:
            raise KeyError(f"unknown component type {type_key!r}")
        self.palette_type = type_key
        self.set_tool(Tool.PLACE)

    def cancel(self) -> None:
        """Escape: abort a drag, leave placement mode, or clear the selection."""
        if self._drag is not None:
            self._restore_drag()
            self._drag = None
        elif self.tool != Tool.SELECT:
            self.set_tool(Tool.SELECT)
        else:
            self.scene.clear_selection()
        self.ghost = None

    # ------------------------------------------------------------------ pointer
    def pointer_move(self, ray: Ray, poses: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None) -> None:
        if self.locked:
            self.hover_id = None
            self.ghost = None
            return
        if self._drag is not None:
            self._update_drag(ray)
            self.hover_id = None
            return
        if self.tool == Tool.PLACE:
            self.hover_id = None
            self.ghost = placement_candidate(self.scene, self.palette_type, ray, self.place_yaw_deg, poses)
            return
        self.ghost = None
        hit = self.scene.find_at_ray(ray, poses)
        self.hover_id = hit[0] if hit else None

    def pointer_leave(self) -> None:
        """The pointer left the 3D viewport: clear hover and placement preview (drags continue)."""
        if self._drag is None:
            self.hover_id = None
            self.ghost = None

    def pointer_down(self, ray: Ray, additive: bool = False,
                     poses: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None) -> EditResult:
        if self.tool == Tool.PLACE:
            return self._place(ray, poses)
        if self.tool == Tool.SELECT:
            hit = self.scene.find_at_ray(ray, poses)
            if hit is None:
                if not additive:
                    self.scene.clear_selection()
                return EditResult(True)
            cid = hit[0]
            if additive:
                self.scene.toggle_selection(cid)
            elif not self.scene.is_selected(cid):
                self.scene.select([cid])
            return EditResult(True)
        if self.tool == Tool.DELETE:
            if self.locked:
                return EditResult(False, "msg.locked")
            hit = self.scene.find_at_ray(ray, poses)
            if hit is None:
                return EditResult(True)
            comp = self.scene.get(hit[0])
            self.history.execute(RemoveComponents((comp,)))
            return EditResult(True, "msg.deleted", {"n": 1})
        if self.tool == Tool.MOVE:
            return self._begin_move(ray, poses)
        return EditResult(False)

    def pointer_up(self, ray: Optional[Ray] = None) -> EditResult:
        if self._drag is None:
            return EditResult(True)
        if ray is not None:
            self._update_drag(ray)
        drag = self._drag
        self._drag = None
        if drag is None or not drag.moved or not np.any(np.abs(drag.delta) > 1e-12):
            return EditResult(True)
        changes = []
        for cid in drag.ids:
            start = drag.start_positions[cid]
            end = start + drag.delta
            changes.append(PoseChange(cid, tuple(start), drag.start_orientations[cid],
                                      tuple(end), drag.start_orientations[cid]))
        self.history.record(ChangePoses(tuple(changes), label="move"))
        return EditResult(True, "msg.moved", {"n": len(drag.ids)})

    # ------------------------------------------------------------------ commands
    def delete_selected(self) -> EditResult:
        if self.locked:
            return EditResult(False, "msg.locked")
        ids = self.scene.selection
        if not ids:
            return EditResult(False, "msg.select_first")
        comps = tuple(self.scene.get(i) for i in ids)
        self.history.execute(RemoveComponents(comps))
        return EditResult(True, "msg.deleted", {"n": len(comps)})

    def rotate(self) -> EditResult:
        """R key: rotate the placement preview in placement mode, otherwise the selection by 90°."""
        if self.locked:
            return EditResult(False, "msg.locked")
        if self.tool == Tool.PLACE:
            self.place_yaw_deg = (self.place_yaw_deg + 90.0) % 360.0
            self.ghost = None  # recomputed on the next pointer_move
            return EditResult(True, "msg.rotated_preview", {"deg": int(self.place_yaw_deg)})
        ids = self.scene.selection
        if not ids:
            return EditResult(False, "msg.select_first")
        turn = quat_from_yaw_degrees(90.0)
        old = {i: self.scene.get(i) for i in ids}
        changes = []
        for i in ids:
            comp = old[i]
            changes.append(PoseChange(i, comp.position, comp.orientation, comp.position,
                                      quat_multiply(turn, comp.orientation)))
        for ch in changes:
            self.scene.set_pose(ch.component_id, ch.old_position, ch.new_orientation)
        blocked = any(not self.scene.check_placement(self.scene.get(i)).valid for i in ids)
        if blocked:
            for ch in changes:
                self.scene.set_pose(ch.component_id, ch.old_position, ch.old_orientation)
            return EditResult(False, "msg.blocked_rotate")
        self.history.record(ChangePoses(tuple(changes), label="rotate"))
        return EditResult(True, "msg.rotated", {"n": len(ids)})

    def undo(self) -> EditResult:
        if self.locked:
            return EditResult(False, "msg.locked")
        label = self.history.undo()
        return EditResult(True, "msg.undo", {"label": label}) if label else EditResult(False, "msg.nothing_undo")

    def redo(self) -> EditResult:
        if self.locked:
            return EditResult(False, "msg.locked")
        label = self.history.redo()
        return EditResult(True, "msg.redo", {"label": label}) if label else EditResult(False, "msg.nothing_redo")

    # ------------------------------------------------------------------ internals
    def _place(self, ray: Ray, poses) -> EditResult:
        if self.locked:
            return EditResult(False, "msg.locked")
        ghost = placement_candidate(self.scene, self.palette_type, ray, self.place_yaw_deg, poses)
        if ghost is None:
            return EditResult(False, "msg.no_surface")
        if not ghost.valid:
            return EditResult(False, _PLACEMENT_MESSAGES.get(ghost.reason or "", "msg.blocked_overlap"))
        component = Component(self.scene.next_id, ghost.type_key, tuple(float(v) for v in ghost.position),
                              ghost.orientation)  # type: ignore[arg-type]
        self.history.execute(AddComponent(component))
        self.scene.select([component.id])
        self.ghost = None
        return EditResult(True, "msg.placed", {"type": ghost.type_key})

    def _begin_move(self, ray: Ray, poses) -> EditResult:
        if self.locked:
            return EditResult(False, "msg.locked")
        hit = self.scene.find_at_ray(ray, poses)
        if hit is None:
            return EditResult(True)
        cid, hit_info = hit
        if not self.scene.is_selected(cid):
            self.scene.select([cid])
        ids = self.scene.selection
        if poses is not None and cid in poses:
            plane_z = float(poses[cid][0][2])
        else:
            plane_z = float(self.scene.get(cid).position[2])
        plane = ray_plane_z(ray, plane_z)
        if plane is None:
            return EditResult(False, "msg.no_surface")
        self._drag = _Drag(
            ids=ids,
            start_positions={i: np.array(self.scene.get(i).position, dtype=float) for i in ids},
            start_orientations={i: self.scene.get(i).orientation for i in ids},
            plane_z=plane_z,
            grab_xy=plane.point[:2].copy(),
        )
        return EditResult(True)

    def _update_drag(self, ray: Ray) -> None:
        drag = self._drag
        if drag is None:
            return
        plane = ray_plane_z(ray, drag.plane_z)
        if plane is None:
            return
        raw = plane.point[:2] - drag.grab_xy
        dx = snap(float(raw[0]), self.scene.grid)
        dy = snap(float(raw[1]), self.scene.grid)
        candidate = np.array([dx, dy, 0.0])
        if np.allclose(candidate, drag.delta):
            return
        moved_ids = set(drag.ids)
        for cid in drag.ids:
            pos = drag.start_positions[cid] + candidate
            check = self.scene.check_placement(
                Component(cid, self.scene.get(cid).type_key, tuple(float(v) for v in pos),
                          drag.start_orientations[cid]),  # type: ignore[arg-type]
                ignore_ids=moved_ids,
            )
            if not check.valid:
                return  # keep the last valid preview
        for cid in drag.ids:
            self.scene.set_pose(cid, drag.start_positions[cid] + candidate, drag.start_orientations[cid])
        drag.delta = candidate
        drag.moved = True

    def _restore_drag(self) -> None:
        drag = self._drag
        if drag is None:
            return
        for cid in drag.ids:
            if cid in self.scene:
                self.scene.set_pose(cid, drag.start_positions[cid], drag.start_orientations[cid])

    def selection_positions(self) -> List[Tuple[int, np.ndarray]]:
        return [(i, np.array(self.scene.get(i).position)) for i in self.scene.selection]

"""Undoable editing commands.

Every edit made by the user is expressed as a command with ``apply`` and
``revert``. ``CommandHistory`` keeps the undo/redo stacks. Commands are
created by tools *after* validation; ``CommandHistory.record`` stores a command
that has already been applied (used for interactive drags), while
``CommandHistory.execute`` applies and records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

from .components import Component
from .geometry import Quat, Vec3
from .scene import Scene


class Command(Protocol):
    label: str

    def apply(self, scene: Scene) -> None: ...

    def revert(self, scene: Scene) -> None: ...


@dataclass
class AddComponent:
    component: Component
    label: str = "add"

    def apply(self, scene: Scene) -> None:
        scene.insert(self.component)

    def revert(self, scene: Scene) -> None:
        scene.remove(self.component.id)


@dataclass
class RemoveComponents:
    components: Tuple[Component, ...]
    label: str = "remove"

    def apply(self, scene: Scene) -> None:
        for c in self.components:
            scene.remove(c.id)

    def revert(self, scene: Scene) -> None:
        for c in self.components:
            scene.insert(c)


@dataclass(frozen=True)
class PoseChange:
    component_id: int
    old_position: Vec3
    old_orientation: Quat
    new_position: Vec3
    new_orientation: Quat


@dataclass
class ChangePoses:
    changes: Tuple[PoseChange, ...]
    label: str = "move"

    def apply(self, scene: Scene) -> None:
        for ch in self.changes:
            scene.set_pose(ch.component_id, ch.new_position, ch.new_orientation)

    def revert(self, scene: Scene) -> None:
        for ch in self.changes:
            scene.set_pose(ch.component_id, ch.old_position, ch.old_orientation)


class CommandHistory:
    def __init__(self, scene: Scene, limit: int = 500) -> None:
        self.scene = scene
        self.limit = limit
        self._undo: List[Command] = []
        self._redo: List[Command] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def execute(self, command: Command) -> None:
        command.apply(self.scene)
        self.record(command)

    def record(self, command: Command) -> None:
        """Record a command that has already been applied to the scene."""
        self._undo.append(command)
        if len(self._undo) > self.limit:
            del self._undo[0]
        self._redo.clear()

    def undo(self) -> str | None:
        if not self._undo:
            return None
        command = self._undo.pop()
        command.revert(self.scene)
        self._redo.append(command)
        return command.label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        command = self._redo.pop()
        command.apply(self.scene)
        self._undo.append(command)
        return command.label

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

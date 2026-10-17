"""The 3D scene view: draws the workshop from *snapshots*.

``SceneView`` never mutates the scene or the simulation. Each frame the application
calls :meth:`SceneView.sync` with the component list, a pose map (editor poses while
editing, simulation poses while running), the selection and the placement ghost. The
view creates/destroys/moves Panda3D nodes to match.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    GeomNode,
    LPoint3,
    LQuaternion,
    NodePath,
    TransparencyAttrib,
)

from ..core.components import Component, ComponentRegistry, Shape
from ..core.editor import Ghost
from ..ui import theme
from .mesh_data import MeshData, build_mesh
from .panda_mesh import make_edges_node, make_geom, make_solid_node

GRID_MINOR = 0.5
GRID_MAJOR = 2.0
FLOOR_SIZE = 400.0


class _ComponentNode:
    __slots__ = ("root", "type_key", "selected_outline", "hover_outline", "shown_selected", "shown_hover")

    def __init__(self, root: NodePath, type_key: str, sel: NodePath, hov: NodePath) -> None:
        self.root = root
        self.type_key = type_key
        self.selected_outline = sel
        self.hover_outline = hov
        self.shown_selected = False
        self.shown_hover = False


def _outline_segments(mesh: MeshData, scale: float = 1.04):
    return [(np.asarray(a) * scale, np.asarray(b) * scale) for a, b in mesh.edges]


class SceneView:
    def __init__(self, parent: NodePath, registry: ComponentRegistry, baseplate_size: Tuple[float, float],
                 lights: bool = True, shadows: bool = True) -> None:
        self.registry = registry
        self.root = parent.attachNewNode("scene-root")
        self._geom_cache: Dict[str, MeshData] = {}
        self._geom_nodes: Dict[str, NodePath] = {}
        self._nodes: Dict[int, _ComponentNode] = {}
        self._ghost: Optional[NodePath] = None
        self._ghost_key: Optional[str] = None
        self.baseplate_size = baseplate_size

        self.world = self.root.attachNewNode("world")
        self._build_environment()
        if lights:
            self._build_lights(shadows)

    # ------------------------------------------------------------------ environment
    def _build_environment(self) -> None:
        hx, hy = self.baseplate_size[0] / 2.0, self.baseplate_size[1] / 2.0
        base_mesh = build_mesh(Shape.BOX, (2 * hx, 2 * hy, 0.1))
        plate = make_solid_node(base_mesh, "baseplate", theme.BASEPLATE)
        plate.setPos(0, 0, -0.05)
        plate.reparentTo(self.world)

        floor_mesh = build_mesh(Shape.BOX, (FLOOR_SIZE, FLOOR_SIZE, 0.1))
        floor = make_solid_node(floor_mesh, "floor", (0.085, 0.095, 0.11, 1.0))
        floor.setPos(0, 0, -1.05)
        floor.reparentTo(self.world)

        minor, major = [], []
        steps_x = int(round(hx / GRID_MINOR))
        steps_y = int(round(hy / GRID_MINOR))
        for i in range(-steps_x, steps_x + 1):
            x = i * GRID_MINOR
            seg = ((x, -hy, 0.003), (x, hy, 0.003))
            (major if abs(x / GRID_MAJOR - round(x / GRID_MAJOR)) < 1e-9 else minor).append(seg)
        for j in range(-steps_y, steps_y + 1):
            y = j * GRID_MINOR
            seg = ((-hx, y, 0.003), (hx, y, 0.003))
            (major if abs(y / GRID_MAJOR - round(y / GRID_MAJOR)) < 1e-9 else minor).append(seg)
        make_edges_node(minor, "grid-minor", theme.GRID_MINOR, 1.0).reparentTo(self.world)
        make_edges_node(major, "grid-major", theme.GRID_MAJOR, 1.5).reparentTo(self.world)
        make_edges_node([((0, 0, 0.004), (1.0, 0, 0.004))], "axis-x", theme.AXIS_X, 3.0).reparentTo(self.world)
        make_edges_node([((0, 0, 0.004), (0, 1.0, 0.004))], "axis-y", theme.AXIS_Y, 3.0).reparentTo(self.world)
        make_edges_node([((0, 0, 0.004), (0, 0, 1.0))], "axis-z", theme.AXIS_Z, 3.0).reparentTo(self.world)

        self.ghost_root = self.root.attachNewNode("ghost")
        self.ghost_root.hide()

    def _build_lights(self, shadows: bool) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.42, 0.45, 0.50, 1.0))
        key = DirectionalLight("key")
        key.setColor((0.95, 0.93, 0.88, 1.0))
        key.getLens().setFilmSize(40, 40)
        key.getLens().setNearFar(1.0, 120.0)
        if shadows:
            key.setShadowCaster(True, 2048, 2048)
        key_np = self.root.attachNewNode(key)
        key_np.setHpr(-35, -58, 0)
        key_np.setPos(-14, -14, 28)
        fill = DirectionalLight("fill")
        fill.setColor((0.25, 0.30, 0.38, 1.0))
        fill_np = self.root.attachNewNode(fill)
        fill_np.setHpr(140, -30, 0)
        amb_np = self.root.attachNewNode(ambient)
        self.root.setLight(amb_np)
        self.root.setLight(key_np)
        self.root.setLight(fill_np)
        self.root.setShaderAuto()

    # ------------------------------------------------------------------ meshes
    def _mesh_for(self, type_key: str) -> MeshData:
        if type_key not in self._geom_cache:
            ctype = self.registry.get(type_key)
            self._geom_cache[type_key] = build_mesh(ctype.shape, ctype.extents)
        return self._geom_cache[type_key]

    def _make_component_node(self, component: Component) -> _ComponentNode:
        ctype = self.registry.get(component.type_key)
        mesh = self._mesh_for(component.type_key)
        root = self.world.attachNewNode(f"component-{component.id}")
        body = root.attachNewNode(GeomNode(f"body-{component.id}"))
        body.node().addGeom(make_geom(mesh, f"geom-{component.type_key}"))
        body.setColor(*ctype.color, 1.0)
        edges = make_edges_node(mesh.edges, f"edges-{component.id}", theme.EDGE, 1.2)
        edges.reparentTo(root)
        sel = make_edges_node(_outline_segments(mesh, 1.06), f"sel-{component.id}", theme.SELECTION, 4.0)
        sel.reparentTo(root)
        sel.hide()
        hov = make_edges_node(_outline_segments(mesh, 1.02), f"hov-{component.id}", theme.HOVER, 2.0)
        hov.reparentTo(root)
        hov.hide()
        return _ComponentNode(root, component.type_key, sel, hov)

    def _ensure_ghost(self, type_key: str) -> NodePath:
        if self._ghost is None or self._ghost_key != type_key:
            if self._ghost is not None:
                self._ghost.removeNode()
            mesh = self._mesh_for(type_key)
            self._ghost = make_solid_node(mesh, "ghost", theme.GHOST_VALID)
            self._ghost.setTransparency(TransparencyAttrib.MAlpha)
            self._ghost.setDepthWrite(False)
            self._ghost.reparentTo(self.ghost_root)
            self._ghost_key = type_key
        return self._ghost

    # ------------------------------------------------------------------ per-frame sync
    def sync(
        self,
        components: Iterable[Component],
        poses: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]],
        selection: Sequence[int] = (),
        hover_id: Optional[int] = None,
        ghost: Optional[Ghost] = None,
    ) -> None:
        """Make the view match the given state. ``poses`` overrides component transforms when given."""
        selected = set(selection)
        seen = set()
        for component in components:
            seen.add(component.id)
            node = self._nodes.get(component.id)
            if node is None or node.type_key != component.type_key:
                if node is not None:
                    node.root.removeNode()
                node = self._make_component_node(component)
                self._nodes[component.id] = node
            if poses is not None and component.id in poses:
                pos, quat = poses[component.id]
            else:
                pos, quat = component.position, component.orientation
            node.root.setPosQuat(
                self.world,
                LPoint3(float(pos[0]), float(pos[1]), float(pos[2])),
                LQuaternion(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )
            want_sel = component.id in selected
            if want_sel != node.shown_selected:
                node.selected_outline.show() if want_sel else node.selected_outline.hide()
                node.shown_selected = want_sel
            want_hov = hover_id == component.id and not want_sel
            if want_hov != node.shown_hover:
                node.hover_outline.show() if want_hov else node.hover_outline.hide()
                node.shown_hover = want_hov
        for cid in [k for k in self._nodes if k not in seen]:
            self._nodes.pop(cid).root.removeNode()
        self._sync_ghost(ghost)

    def _sync_ghost(self, ghost: Optional[Ghost]) -> None:
        if ghost is None:
            self.ghost_root.hide()
            return
        node = self._ensure_ghost(ghost.type_key)
        color = theme.GHOST_VALID if ghost.valid else theme.GHOST_INVALID
        node.setColor(*color)
        pos, quat = ghost.position, ghost.orientation
        self.ghost_root.setPosQuat(
            self.root,
            LPoint3(float(pos[0]), float(pos[1]), float(pos[2])),
            LQuaternion(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        )
        self.ghost_root.show()

    @property
    def node_count(self) -> int:
        return len(self._nodes)

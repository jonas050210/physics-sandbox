"""Conversion of pure mesh data to Panda3D nodes (requires ``panda3d.core``)."""

from __future__ import annotations

from typing import Tuple

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
)

from .mesh_data import MeshData


def make_geom(mesh: MeshData, name: str = "mesh") -> Geom:
    fmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(len(mesh.vertices))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    for v, n in zip(mesh.vertices, mesh.normals):
        vw.addData3(float(v[0]), float(v[1]), float(v[2]))
        nw.addData3(float(n[0]), float(n[1]), float(n[2]))
    prim = GeomTriangles(Geom.UHStatic)
    for a, b, c in mesh.triangles:
        prim.addVertices(int(a), int(b), int(c))
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_solid_node(mesh: MeshData, name: str, color: Tuple[float, float, float, float]) -> NodePath:
    node = GeomNode(name)
    node.addGeom(make_geom(mesh, name))
    np_ = NodePath(node)
    np_.setColor(*color)
    return np_


def make_edges_node(segments, name: str, color, thickness: float = 1.5, lit: bool = False) -> NodePath:
    """Line segments given as an iterable of ``(p0, p1)`` 3-tuples/arrays."""
    ls = LineSegs(name)
    ls.setThickness(thickness)
    ls.setColor(*color)
    for p0, p1 in segments:
        ls.moveTo(float(p0[0]), float(p0[1]), float(p0[2]))
        ls.drawTo(float(p1[0]), float(p1[1]), float(p1[2]))
    np_ = NodePath(ls.create())
    if not lit:
        np_.setLightOff(1)
    return np_

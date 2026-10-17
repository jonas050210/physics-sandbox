"""Pure-numpy mesh generation for the primitive shapes (no Panda3D dependency).

Each mesh is returned as ``(vertices, normals, triangles)`` with vertices in the
shape's local frame (centred on the origin, Z-up).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..core.components import Shape


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray  # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64, unit length
    triangles: np.ndarray  # (M, 3) int64
    edges: np.ndarray  # (E, 2, 3) float64 line segments for the outline


def _box(half: np.ndarray) -> MeshData:
    hx, hy, hz = (float(v) for v in half)
    verts, norms, tris = [], [], []
    faces = [
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)]),
        ((0, 1, 0), [(hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)]),
    ]
    for normal, quad in faces:
        base = len(verts)
        verts.extend(quad)
        norms.extend([normal] * 4)
        tris.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    corners = np.array([(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            diff = np.abs(corners[a] - corners[b])
            if np.count_nonzero(np.isclose(diff, 0.0)) == 2:
                edges.append((corners[a], corners[b]))
    return MeshData(np.array(verts, float), np.array(norms, float), np.array(tris, np.int64),
                    np.array(edges, float))


def _sphere(radius: float, rings: int = 16, segments: int = 24) -> MeshData:
    verts, norms = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings  # 0 at north pole
        for j in range(segments + 1):
            theta = 2 * math.pi * j / segments
            n = np.array([math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)])
            verts.append(n * radius)
            norms.append(n)
    tris = []
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            tris.append((a, b, a + 1))
            tris.append((a + 1, b, b + 1))
    ring_pts = []
    for k in range(4):
        ang = k * math.pi / 2
        ring_pts.append((radius * math.cos(ang), radius * math.sin(ang), 0.0))
    edges = []
    for k in range(4):
        edges.append((ring_pts[k], ring_pts[(k + 1) % 4]))
    return MeshData(np.array(verts), np.array(norms), np.array(tris, np.int64), np.array(edges, float))


def _cylinder(radius: float, height: float, segments: int = 32) -> MeshData:
    hz = height / 2.0
    verts, norms, tris = [], [], []
    # side
    for j in range(segments + 1):
        theta = 2 * math.pi * j / segments
        c, s = math.cos(theta), math.sin(theta)
        verts.append((radius * c, radius * s, -hz))
        norms.append((c, s, 0.0))
        verts.append((radius * c, radius * s, hz))
        norms.append((c, s, 0.0))
    for j in range(segments):
        a = 2 * j
        tris.append((a, a + 2, a + 1))
        tris.append((a + 1, a + 2, a + 3))
    # caps
    for sign in (1.0, -1.0):
        centre = len(verts)
        verts.append((0.0, 0.0, sign * hz))
        norms.append((0.0, 0.0, sign))
        rim_start = len(verts)
        for j in range(segments + 1):
            theta = 2 * math.pi * j / segments
            verts.append((radius * math.cos(theta), radius * math.sin(theta), sign * hz))
            norms.append((0.0, 0.0, sign))
        for j in range(segments):
            if sign > 0:
                tris.append((centre, rim_start + j, rim_start + j + 1))
            else:
                tris.append((centre, rim_start + j + 1, rim_start + j))
    rim_top = [(radius * math.cos(2 * math.pi * k / 8), radius * math.sin(2 * math.pi * k / 8), hz) for k in range(8)]
    rim_bot = [(p[0], p[1], -hz) for p in rim_top]
    edges = [(rim_top[k], rim_top[(k + 1) % 8]) for k in range(8)]
    edges += [(rim_bot[k], rim_bot[(k + 1) % 8]) for k in range(8)]
    edges += [(rim_top[k], rim_bot[k]) for k in range(0, 8, 2)]
    return MeshData(np.array(verts, float), np.array(norms, float), np.array(tris, np.int64),
                    np.array(edges, float))


def build_mesh(shape: Shape, extents: Tuple[float, float, float]) -> MeshData:
    """Mesh for a primitive given its FULL extents (see ``ComponentType``)."""
    sx, sy, sz = (float(v) for v in extents)
    if shape == Shape.BOX:
        return _box(np.array([sx, sy, sz]) / 2.0)
    if shape == Shape.SPHERE:
        return _sphere(sx / 2.0)
    if shape == Shape.CYLINDER:
        return _cylinder(sx / 2.0, sz)
    raise ValueError(f"unsupported shape {shape}")

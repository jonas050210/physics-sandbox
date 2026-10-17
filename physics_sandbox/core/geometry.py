"""Pure geometry helpers used by the scene model, picking and placement.

Conventions (project-wide):
* Right-handed, Z-up world. Units are SI (metres, kilograms, seconds).
* Quaternions are ``(w, x, y, z)`` tuples, matching MuJoCo and Panda3D's ordering
  after conversion. Identity is ``(1, 0, 0, 0)``.
* Nothing in this module imports a renderer or a physics engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
UP: Vec3 = (0.0, 0.0, 1.0)
_EPS = 1e-12


def vec3(value: Sequence[float]) -> np.ndarray:
    """Return ``value`` as a float64 array of shape (3,)."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3-vector, got {len(arr)} values")
    return arr


def as_vec3_tuple(value: Sequence[float]) -> Vec3:
    x, y, z = (float(v) for v in vec3(value))
    return (x, y, z)


# --------------------------------------------------------------------------- quaternions

def quat_normalize(q: Sequence[float]) -> Quat:
    arr = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        raise ValueError("cannot normalise a zero quaternion")
    w, x, y, z = (float(v) for v in arr / norm)
    return (w, x, y, z)


def quat_from_axis_angle(axis: Sequence[float], angle_rad: float) -> Quat:
    a = vec3(axis)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        raise ValueError("rotation axis must be non-zero")
    a = a / n
    half = angle_rad / 2.0
    s = math.sin(half)
    return quat_normalize((math.cos(half), a[0] * s, a[1] * s, a[2] * s))


def quat_from_yaw_degrees(degrees: float) -> Quat:
    """Rotation about the world Z axis (heading in the ground plane)."""
    return quat_from_axis_angle(UP, math.radians(degrees))


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> Quat:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    w1, x1, y1, z1 = (float(v) for v in a)
    w2, x2, y2, z2 = (float(v) for v in b)
    return quat_normalize(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Return the 3x3 rotation matrix for a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------- grid

def snap(value: float, grid: float) -> float:
    """Round ``value`` to the nearest multiple of ``grid`` (``grid`` > 0)."""
    if grid <= 0:
        raise ValueError("grid size must be positive")
    return math.floor(value / grid + 0.5) * grid


def snap_vec(value: Sequence[float], grid: float) -> np.ndarray:
    return np.array([snap(float(v), grid) for v in vec3(value)])


# --------------------------------------------------------------------------- rays

@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray  # unit length

    @staticmethod
    def create(origin: Sequence[float], direction: Sequence[float]) -> "Ray":
        d = vec3(direction)
        n = float(np.linalg.norm(d))
        if n < _EPS:
            raise ValueError("ray direction must be non-zero")
        return Ray(vec3(origin), d / n)

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t


@dataclass(frozen=True)
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray  # world-space outward unit normal (zero if unknown)


@dataclass(frozen=True)
class Obb:
    """Oriented box. ``half`` are half-extents along the box's local axes."""

    center: np.ndarray
    rotation: np.ndarray  # 3x3, local -> world
    half: np.ndarray

    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        ext = np.abs(self.rotation) @ self.half
        return self.center - ext, self.center + ext


def ray_obb(ray: Ray, obb: Obb) -> Optional[Hit]:
    """Slab test of a ray against an oriented box. Returns the entry hit.

    If the origin lies inside the box the hit is reported at ``t = 0`` with a
    zero normal, which is enough for picking purposes.
    """
    o = obb.rotation.T @ (ray.origin - obb.center)
    d = obb.rotation.T @ ray.direction
    t_min, t_max = -math.inf, math.inf
    axis, sign = -1, 0.0
    for i in range(3):
        h = float(obb.half[i])
        if abs(d[i]) < _EPS:
            if o[i] < -h or o[i] > h:
                return None
            continue
        t1 = (-h - o[i]) / d[i]
        t2 = (h - o[i]) / d[i]
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        if lo > t_min:
            t_min = lo
            axis = i
            sign = -1.0 if d[i] > 0 else 1.0
        t_max = min(t_max, hi)
        if t_min > t_max:
            return None
    if t_max < 0:
        return None
    if t_min < 0:
        return Hit(0.0, ray.origin.copy(), np.zeros(3))
    normal = np.zeros(3)
    if axis >= 0:
        local = np.zeros(3)
        local[axis] = sign
        normal = obb.rotation @ local
    return Hit(t_min, ray.point_at(t_min), normal)


def ray_sphere(ray: Ray, center: np.ndarray, radius: float) -> Optional[Hit]:
    oc = ray.origin - center
    b = float(np.dot(oc, ray.direction))
    c = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    t = -b - root
    if t < 0:
        t = -b + root
        if t < 0:
            return None
    point = ray.point_at(t)
    normal = point - center
    n = float(np.linalg.norm(normal))
    normal = normal / n if n > _EPS else np.zeros(3)
    return Hit(t, point, normal)


def ray_plane_z(ray: Ray, z: float) -> Optional[Hit]:
    """Intersection with the horizontal plane ``z = const`` (t >= 0 only)."""
    dz = float(ray.direction[2])
    if abs(dz) < _EPS:
        return None
    t = (z - float(ray.origin[2])) / dz
    if t < 0:
        return None
    normal = np.array([0.0, 0.0, 1.0 if dz < 0 else -1.0])
    return Hit(t, ray.point_at(t), normal)


# --------------------------------------------------------------------------- overlap

def aabb_overlap(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray, tol: float = 1e-4
) -> bool:
    """True when two AABBs share volume (touching faces do not count)."""
    return bool(np.all(max_a - min_b > tol) and np.all(max_b - min_a > tol))

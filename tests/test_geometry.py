import math

import numpy as np
import pytest

from physics_sandbox.core.geometry import (
    Obb, Ray, aabb_overlap, quat_from_yaw_degrees, quat_multiply, quat_normalize, quat_to_matrix,
    ray_obb, ray_plane_z, ray_sphere, snap, snap_vec,
)


def test_quaternion_yaw_matrix_is_rotation_about_z():
    q = quat_from_yaw_degrees(90.0)
    m = quat_to_matrix(q)
    assert np.allclose(m @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-12)
    assert math.isclose(np.linalg.det(m), 1.0)


def test_quaternion_multiply_composes_rotations():
    a = quat_from_yaw_degrees(90.0)
    both = quat_multiply(a, a)
    assert np.allclose(both, quat_from_yaw_degrees(180.0), atol=1e-12) or np.allclose(
        both, -np.array(quat_from_yaw_degrees(180.0)), atol=1e-12)


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quat_normalize((0, 0, 0, 0))


def test_snap_rounds_to_grid():
    assert snap(0.19, 0.125) == pytest.approx(0.25)
    assert snap(0.12, 0.125) == pytest.approx(0.125)
    assert snap(-0.07, 0.125) == pytest.approx(-0.125)
    assert np.allclose(snap_vec((0.3, -0.3, 1.01), 0.25), [0.25, -0.25, 1.0])
    with pytest.raises(ValueError):
        snap(1.0, 0.0)


def test_ray_hits_axis_aligned_box_top_with_normal():
    box = Obb(center=np.array([0.0, 0.0, 1.0]), rotation=np.eye(3), half=np.array([0.5, 0.5, 0.5]))
    ray = Ray.create([0.1, 0.1, 10.0], [0.0, 0.0, -1.0])
    hit = ray_obb(ray, box)
    assert hit is not None
    assert hit.t == pytest.approx(8.5)
    assert np.allclose(hit.normal, [0, 0, 1])


def test_ray_misses_box_and_behind_origin():
    box = Obb(center=np.zeros(3), rotation=np.eye(3), half=np.ones(3))
    assert ray_obb(Ray.create([5, 5, 0], [0, 0, 1]), box) is None
    assert ray_obb(Ray.create([0, 0, 5], [0, 0, 1]), box) is None  # box is behind the ray


def test_ray_hits_rotated_box_side():
    rot = quat_to_matrix(quat_from_yaw_degrees(45.0))
    box = Obb(center=np.zeros(3), rotation=rot, half=np.array([1.0, 0.1, 1.0]))
    ray = Ray.create([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    hit = ray_obb(ray, box)
    assert hit is not None and hit.t > 0


def test_ray_sphere_and_plane():
    hit = ray_sphere(Ray.create([0, 0, 10], [0, 0, -1]), np.array([0, 0, 1.0]), 0.5)
    assert hit is not None and hit.t == pytest.approx(8.5)
    assert ray_sphere(Ray.create([0, 0, 10], [0, 1, 0]), np.array([0, 0, 1.0]), 0.5) is None
    plane = ray_plane_z(Ray.create([1, 2, 3], [0, 0, -1]), 0.0)
    assert plane is not None and np.allclose(plane.point, [1, 2, 0])
    assert ray_plane_z(Ray.create([1, 2, 3], [0, 0, 1]), 0.0) is None
    assert ray_plane_z(Ray.create([1, 2, 3], [1, 0, 0]), 0.0) is None


def test_aabb_overlap_ignores_touching_faces():
    a0, a1 = np.zeros(3), np.ones(3)
    assert not aabb_overlap(a0, a1, np.array([1.0, 0, 0]), np.array([2.0, 1, 1]))
    assert aabb_overlap(a0, a1, np.array([0.5, 0, 0]), np.array([2.0, 1, 1]))

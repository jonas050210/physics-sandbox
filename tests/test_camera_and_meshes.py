import numpy as np
import pytest

from physics_sandbox.core.components import Shape
from physics_sandbox.render.mesh_data import build_mesh
from physics_sandbox.render.orbit_camera import DISTANCE_MAX, DISTANCE_MIN, OrbitCamera


def test_camera_position_is_at_distance_and_looks_at_target():
    cam = OrbitCamera(target=np.array([1.0, 2.0, 0.5]), yaw_deg=30, pitch_deg=20, distance=10)
    assert np.linalg.norm(cam.position() - cam.target) == pytest.approx(10.0)
    assert np.allclose(cam.forward(), (cam.target - cam.position()) / 10.0)
    assert cam.position()[2] > cam.target[2]  # elevated


def test_pitch_and_distance_are_clamped():
    cam = OrbitCamera()
    cam.orbit(0, 500)
    assert cam.pitch_deg == pytest.approx(89.0)
    cam.orbit(0, -500)
    assert cam.pitch_deg == pytest.approx(2.0)
    cam.zoom(1e-6)
    assert cam.distance == pytest.approx(DISTANCE_MIN)
    cam.zoom(1e9)
    assert cam.distance == pytest.approx(DISTANCE_MAX)


def test_yaw_wraps_into_range():
    cam = OrbitCamera()
    cam.orbit(400, 0)
    assert -180 <= cam.yaw_deg <= 180


def test_presets_and_top_view_is_downward():
    cam = OrbitCamera()
    cam.set_view("top")
    assert cam.position()[2] > cam.target[2] + 0.9 * cam.distance
    with pytest.raises(KeyError):
        cam.set_view("bogus")


def test_pan_moves_target_on_ground_plane():
    cam = OrbitCamera(target=np.array([0.0, 0.0, 0.0]))
    cam.pan(0.1, 0.0)
    assert cam.target[2] == pytest.approx(0.0)
    assert np.linalg.norm(cam.target) > 0


def test_centre_ray_points_along_forward():
    cam = OrbitCamera()
    ray = cam.ray_through(0.0, 0.0, 16 / 9)
    assert np.allclose(ray.direction, cam.forward(), atol=1e-9)
    assert np.allclose(ray.origin, cam.position())


def test_focus_frames_sphere():
    cam = OrbitCamera()
    cam.focus(np.array([0.0, 0.0, 1.0]), radius=5.0)
    assert cam.distance > 5.0


@pytest.mark.parametrize("shape,extents", [
    (Shape.BOX, (1.0, 2.0, 3.0)),
    (Shape.SPHERE, (0.5, 0.5, 0.5)),
    (Shape.CYLINDER, (0.5, 0.5, 0.5)),
])
def test_meshes_are_well_formed(shape, extents):
    mesh = build_mesh(shape, extents)
    assert mesh.vertices.shape[1] == 3
    assert mesh.normals.shape == mesh.vertices.shape
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-9)
    assert mesh.triangles.max() < len(mesh.vertices)
    assert mesh.edges.shape[1:] == (2, 3)


def test_box_mesh_bounds_match_extents():
    mesh = build_mesh(Shape.BOX, (1.0, 2.0, 3.0))
    assert np.allclose(mesh.vertices.max(axis=0), [0.5, 1.0, 1.5])
    assert len(mesh.edges) == 12


def test_sphere_mesh_radius():
    mesh = build_mesh(Shape.SPHERE, (1.0, 1.0, 1.0))
    assert np.allclose(np.linalg.norm(mesh.vertices, axis=1), 0.5, atol=1e-9)

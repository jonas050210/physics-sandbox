import numpy as np
import pytest

from physics_sandbox.core.commands import AddComponent, RemoveComponents
from physics_sandbox.core.components import Component, ComponentRegistry, ComponentType, Shape
from physics_sandbox.core.scene import (
    REASON_BELOW_GROUND, REASON_OUTSIDE, REASON_OVERLAP, Scene,
)


def test_add_assigns_increasing_ids_and_version_changes(scene):
    v0 = scene.version
    a = scene.add("cube", (0, 0, 0.25))
    b = scene.add("cube", (2, 0, 0.25))
    assert (a.id, b.id) == (1, 2)
    assert scene.version > v0
    assert len(scene) == 2


def test_unknown_type_is_rejected(scene):
    with pytest.raises(KeyError):
        scene.add("does-not-exist", (0, 0, 0))


def test_placement_rules(scene):
    scene.add("cube", (0, 0, 0.25))
    ok = Component(99, "cube", (2.0, 0.0, 0.25))
    assert scene.check_placement(ok).valid
    overlap = Component(99, "cube", (0.1, 0.0, 0.25))
    res = scene.check_placement(overlap)
    assert not res.valid and res.reason == REASON_OVERLAP and res.blocking_id == 1
    below = Component(99, "cube", (2.0, 0.0, 0.0))
    assert scene.check_placement(below).reason == REASON_BELOW_GROUND
    outside = Component(99, "cube", (13.0, 0.0, 0.25))
    assert scene.check_placement(outside).reason == REASON_OUTSIDE


def test_touching_stack_is_valid(scene):
    scene.add("cube", (0, 0, 0.25))
    stacked = Component(99, "cube", (0, 0, 0.75))
    assert scene.check_placement(stacked).valid


def test_ignore_ids_allows_moving_into_own_old_space(scene):
    c = scene.add("cube", (0, 0, 0.25))
    moved = Component(c.id, "cube", (0.1, 0, 0.25))
    assert scene.check_placement(moved, ignore_ids=[c.id]).valid


def test_selection_and_removal(scene):
    a = scene.add("cube", (0, 0, 0.25))
    b = scene.add("ball", (3, 0, 0.25))
    scene.select([a.id])
    scene.toggle_selection(b.id)
    assert scene.selection == (a.id, b.id)
    scene.remove(a.id)
    assert scene.selection == (b.id,)
    scene.clear_selection()
    assert scene.selection == ()


def test_find_at_ray_returns_nearest(scene):
    near = scene.add("cube", (0, 0, 0.25))
    scene.add("cube", (0, 2, 0.25))
    from physics_sandbox.core.geometry import Ray
    hit = scene.find_at_ray(Ray.create([0, -5, 0.25], [0, 1, 0]))
    assert hit is not None and hit[0] == near.id
    assert scene.find_at_ray(Ray.create([0, -5, 3], [0, 1, 0])) is None


def test_find_at_ray_uses_pose_overrides(scene):
    c = scene.add("cube", (0, 0, 0.25))
    from physics_sandbox.core.geometry import Ray
    ray = Ray.create([5, 0, 0.25], [-1, 0, 0])
    assert scene.find_at_ray(ray) is not None
    assert scene.find_at_ray(ray, {c.id: (np.array([20.0, 0, 0.25]), np.array([1.0, 0, 0, 0]))}) is None


def test_undo_redo_add_remove(scene, history):
    comp = Component(scene.next_id, "cube", (0, 0, 0.25))
    history.execute(AddComponent(comp))
    assert comp.id in scene
    assert history.undo() == "add"
    assert comp.id not in scene
    history.redo()
    assert comp.id in scene
    history.execute(RemoveComponents((comp,)))
    assert comp.id not in scene
    history.undo()
    assert scene.get(comp.id) == comp  # identical component restored, same id


def test_new_command_clears_redo_stack(scene, history):
    history.execute(AddComponent(Component(1, "cube", (0, 0, 0.25))))
    history.undo()
    assert history.can_redo
    history.execute(AddComponent(Component(2, "ball", (3, 0, 0.25))))
    assert not history.can_redo


def test_custom_registry_is_extensible():
    reg = ComponentRegistry([])
    reg.register(ComponentType("wheel", Shape.CYLINDER, (0.5, 0.5, 0.1), 0.5, (0.1, 0.1, 0.1), "Rad", "Wheel"))
    s = Scene(registry=reg)
    s.add("wheel", (0, 0, 0.25))
    with pytest.raises(ValueError):
        reg.register(ComponentType("wheel", Shape.BOX, (1, 1, 1), 1, (0, 0, 0), "a", "b"))


def test_component_type_validation():
    with pytest.raises(ValueError):
        ComponentType("bad", Shape.BOX, (0.0, 1, 1), 1.0, (0, 0, 0), "a", "b")
    with pytest.raises(ValueError):
        ComponentType("bad", Shape.SPHERE, (1.0, 1.0, 2.0), 1.0, (0, 0, 0), "a", "b")

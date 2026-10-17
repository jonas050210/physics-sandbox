import numpy as np
import pytest

from physics_sandbox.core.commands import CommandHistory
from physics_sandbox.core.editor import Editor, Tool, placement_candidate
from physics_sandbox.core.geometry import Ray
from physics_sandbox.core.scene import Scene


def down_ray(x, y, z=10.0):
    return Ray.create([x, y, z], [0.0, 0.0, -1.0])


@pytest.fixture
def editor():
    scene = Scene()
    return Editor(scene, CommandHistory(scene))


def test_place_on_baseplate_snaps_and_sits_on_ground(editor):
    editor.select_palette("cube")
    editor.pointer_move(down_ray(0.30, 0.10))
    assert editor.ghost is not None and editor.ghost.valid
    res = editor.pointer_down(down_ray(0.30, 0.10))
    assert res.ok and res.message == "msg.placed"
    comp = editor.scene.get(editor.scene.selection[0])
    assert comp.position == pytest.approx((0.25, 0.125, 0.25))


def test_place_on_top_of_existing_block_stacks(editor):
    editor.select_palette("cube")
    editor.pointer_down(down_ray(0.0, 0.0))
    editor.pointer_move(down_ray(0.0, 0.0))  # ray now hits the top face of the first cube (z = 0.5)
    assert editor.ghost is not None and editor.ghost.valid
    assert editor.ghost.position[2] == pytest.approx(0.75)
    res = editor.pointer_down(down_ray(0.0, 0.0))
    assert res.ok and len(editor.scene) == 2


def test_side_face_placement(editor):
    editor.scene.add("cube", (0.0, 0.0, 0.25))
    ray = Ray.create([-10.0, 0.0, 0.25], [1.0, 0.0, 0.0])
    ghost = placement_candidate(editor.scene, "cube", ray)
    assert ghost is not None and ghost.valid
    assert ghost.position[0] == pytest.approx(-0.5)


def test_placement_blocked_outside_baseplate(editor):
    editor.select_palette("cube")
    res = editor.pointer_down(down_ray(20.0, 0.0))
    assert not res.ok and res.message == "msg.blocked_outside"


def test_select_click_and_ctrl_multiselect(editor):
    s = editor.scene
    a = s.add("cube", (0, 0, 0.25))
    b = s.add("cube", (3, 0, 0.25))
    editor.set_tool(Tool.SELECT)
    editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    assert s.selection == (a.id,)
    editor.pointer_down(down_ray(3.0, 0.0, 5.0), additive=True)
    assert s.selection == (a.id, b.id)
    editor.pointer_down(down_ray(20.0, 20.0, 5.0))
    assert s.selection == ()


def test_delete_tool_and_delete_key_are_undoable(editor):
    s = editor.scene
    a = s.add("cube", (0, 0, 0.25))
    editor.set_tool(Tool.DELETE)
    editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    assert a.id not in s
    assert editor.history.undo() == "remove"
    assert a.id in s
    editor.set_tool(Tool.SELECT)
    editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    res = editor.delete_selected()
    assert res.ok and a.id not in s


def test_delete_without_selection_reports_message(editor):
    res = editor.delete_selected()
    assert not res.ok and res.message == "msg.select_first"


def test_move_drag_snaps_and_is_one_undo_step(editor):
    s = editor.scene
    a = s.add("cube", (0.25, 0.25, 0.25))
    editor.set_tool(Tool.MOVE)
    editor.pointer_down(down_ray(0.25, 0.25, 5.0))
    editor.pointer_move(down_ray(0.55, 0.25, 5.0))
    editor.pointer_move(down_ray(0.80, 0.25, 5.0))
    editor.pointer_up(down_ray(0.80, 0.25, 5.0))
    assert s.get(a.id).position[0] == pytest.approx(0.75)
    assert s.get(a.id).position[1] == pytest.approx(0.25)
    assert editor.history.undo() == "move"
    assert s.get(a.id).position == pytest.approx((0.25, 0.25, 0.25))


def test_move_is_blocked_by_collision_and_keeps_last_valid(editor):
    s = editor.scene
    a = s.add("cube", (0.0, 0.0, 0.25))
    s.add("cube", (1.0, 0.0, 0.25))
    editor.set_tool(Tool.MOVE)
    editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    editor.pointer_move(down_ray(0.25, 0.0, 5.0))  # 0.25 shift is fine
    editor.pointer_move(down_ray(0.75, 0.0, 5.0))  # would overlap the neighbour
    editor.pointer_up()
    assert s.get(a.id).position[0] == pytest.approx(0.25)


def test_move_cancel_restores_start(editor):
    s = editor.scene
    a = s.add("cube", (0.0, 0.0, 0.25))
    editor.set_tool(Tool.MOVE)
    editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    editor.pointer_move(down_ray(1.0, 0.0, 5.0))
    editor.cancel()
    assert s.get(a.id).position == pytest.approx((0.0, 0.0, 0.25))


def test_rotate_selection_turns_about_own_centre_and_is_undoable(editor):
    s = editor.scene
    a = s.add("beam", (0.0, 0.0, 0.125))
    s.select([a.id])
    res = editor.rotate()
    assert res.ok
    from physics_sandbox.core.geometry import quat_to_matrix
    m = quat_to_matrix(s.get(a.id).orientation)
    assert np.allclose(m @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert s.get(a.id).position == pytest.approx((0.0, 0.0, 0.125))
    editor.history.undo()
    assert s.get(a.id).orientation == pytest.approx((1, 0, 0, 0))


def test_rotate_blocked_when_it_would_collide(editor):
    s = editor.scene
    a = s.add("beam", (0.0, 0.0, 0.125))
    s.add("cube", (0.0, 0.9, 0.25))  # lies in the path of the beam once it is turned along Y
    s.select([a.id])
    res = editor.rotate()
    assert not res.ok and res.message == "msg.blocked_rotate"
    assert s.get(a.id).orientation == pytest.approx((1, 0, 0, 0))


def test_placement_preview_rotates(editor):
    editor.select_palette("beam")
    editor.pointer_move(down_ray(0.0, 0.0))
    before = editor.ghost.orientation
    editor.rotate()
    editor.pointer_move(down_ray(0.0, 0.0))
    assert editor.ghost.orientation != before


def test_locked_editor_refuses_edits(editor):
    locked = {"v": True}
    editor.scene.add("cube", (0, 0, 0.25))
    editor._locked = lambda: locked["v"]
    editor.set_tool(Tool.DELETE)
    res = editor.pointer_down(down_ray(0.0, 0.0, 5.0))
    assert not res.ok and res.message == "msg.locked"
    assert len(editor.scene) == 1
    editor.pointer_move(down_ray(0.0, 0.0, 5.0))
    assert editor.hover_id is None

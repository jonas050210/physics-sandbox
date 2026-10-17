"""Application logic driven without a window (Panda3D ``window-type none``).

ShowBase is a per-process singleton, so one application instance is shared by this module.
Each test resets the state it touches.
"""


import pytest

from physics_sandbox.config import AppConfig
from physics_sandbox.core.editor import Tool
from physics_sandbox.core.geometry import Ray
from physics_sandbox.core.serialization import save_scene
from physics_sandbox.core.demo import tower_scene
from physics_sandbox.sim.simulation import SimMode


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    from physics_sandbox.app import PhysicsSandboxApp

    saves = tmp_path_factory.mktemp("saves")
    application = PhysicsSandboxApp(AppConfig(saves_dir=saves), window=False)
    application._test_saves = saves
    yield application


@pytest.fixture
def fresh(app):
    app.sim.reset()
    app.act_new()
    app.editor.set_tool(Tool.SELECT)
    app.history.clear()
    app.language = "de"
    yield app
    app.sim.reset()


def test_starts_with_demo_tower_and_hud(fresh):
    fresh.act_new()
    fresh.pump(2)
    assert len(fresh.scene) == 0
    assert len(fresh.hud._items) > 50
    assert fresh.build_hud_state().scene_count == 0


def test_palette_button_places_component_via_ray(fresh):
    fresh.hud._buttons["palette.cube"]["command"]()
    assert fresh.editor.tool.value == "place"
    ray = Ray.create([3.0, 3.0, 10.0], [0.0, 0.0, -1.0])
    fresh.editor.pointer_move(ray)
    assert fresh.editor.ghost is not None and fresh.editor.ghost.valid
    fresh.editor.pointer_down(ray)
    assert len(fresh.scene) == 1
    fresh.pump(1)
    assert fresh.scene.selection == (1,)
    lines = fresh.inspector_lines("en")
    assert any("Cube" in line for line in lines)


def test_undo_redo_via_actions(fresh):
    fresh.scene.add("ball", (0.0, 0.0, 0.25))
    fresh.act_delete_selection()  # nothing selected -> message, no change
    assert len(fresh.scene) == 1
    fresh.scene.select([1])
    fresh.act_delete_selection()
    assert len(fresh.scene) == 0
    fresh.act_undo()
    assert len(fresh.scene) == 1
    fresh.act_redo()
    assert len(fresh.scene) == 0


def test_simulation_lifecycle_locks_editing(fresh):
    fresh.scene.add("cube", (0.0, 0.0, 0.25))
    fresh.act_sim_toggle()
    assert fresh.sim.mode == SimMode.RUNNING
    fresh.pump(60)
    assert fresh.sim.time == pytest.approx(1.0, abs=0.02)
    fresh.act_palette("ball")  # refused while running
    assert fresh.editor.tool.value != "place"
    fresh.act_sim_toggle()
    assert fresh.sim.mode == SimMode.PAUSED
    fresh.act_sim_toggle()
    assert fresh.sim.mode == SimMode.RUNNING
    fresh.act_sim_reset()
    assert fresh.sim.mode == SimMode.EDIT


def test_running_simulation_moves_components_in_view_and_inspector(fresh):
    ball = fresh.scene.add("ball", (0.0, 0.0, 3.0))
    fresh.scene.select([ball.id])
    fresh.act_sim_toggle()
    start = fresh.sim.poses()[ball.id][0][2]
    fresh.pump(30)
    assert fresh.sim.poses()[ball.id][0][2] < start
    assert any("Geschwindigkeit" in line for line in fresh.inspector_lines("de"))


def test_save_and_load_round_trip(fresh):
    fresh.scene.add("plate", (0.0, 0.0, 0.125))
    fresh.act_save()
    saved = sorted(fresh._test_saves.glob("*.json"))
    assert saved, "act_save should write a file"
    fresh.act_new()
    assert len(fresh.scene) == 0
    fresh.act_load()
    assert len(fresh.scene) == 1


def test_load_reports_invalid_file(fresh, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    fresh.config_saves = fresh.app_config.saves_dir
    fresh.app_config.saves_dir = tmp_path
    fresh.act_load()
    fresh.app_config.saves_dir = fresh._test_saves
    assert fresh._message[0] == "msg.error"


def test_language_toggle_updates_texts(fresh):
    fresh.act_language()
    fresh.pump(1)
    assert fresh.language == "en"
    assert fresh.hud._buttons["tool.select"]["text"] == "Select (V)"
    fresh.act_language()
    fresh.pump(1)
    assert fresh.hud._buttons["tool.select"]["text"] == "Auswahl (V)"


def test_resize_rebuilds_layout_and_keeps_viewport_valid(fresh):
    fresh._resize(1920, 1080)
    l, r, b, t = fresh.hud.viewport_fractions()
    assert 0 < l < r < 1 and 0 < b < t < 1
    assert fresh._aspect > 1.0
    fresh._resize(1440, 900)


def test_view_presets_change_camera(fresh):
    fresh.act_view("top")
    assert fresh.orbit.pitch_deg > 85
    fresh.act_view("iso")
    assert fresh.orbit.pitch_deg < 40


def test_inspector_for_multiple_selection(fresh):
    fresh.scene.add("cube", (0, 0, 0.25))
    fresh.scene.add("cube", (2, 0, 0.25))
    fresh.scene.select([1, 2])
    lines = fresh.inspector_lines("en")
    assert lines[0].startswith("2 components")


def test_scene_view_nodes_follow_scene(fresh):
    fresh.act_new()
    fresh.scene.add("cube", (0, 0, 0.25))
    fresh.scene.add("ball", (2, 0, 0.25))
    fresh.pump(1)
    assert fresh.view.node_count == 2
    fresh.scene.remove(1)
    fresh.pump(1)
    assert fresh.view.node_count == 1


def test_loaded_example_scene_is_valid_and_stable(tmp_path):
    from physics_sandbox.headless import run_headless
    scene = tower_scene()
    path = save_scene(scene, tmp_path / "t.json")
    from physics_sandbox.core.serialization import load_scene
    report = run_headless(load_scene(path), seconds=1.0)
    assert report["ok"] and report["finite"]


def test_app_does_not_mutate_scene_during_simulation(fresh):
    fresh.scene.add("cube", (0, 0, 0.25))
    before = fresh.scene.snapshot()
    fresh.act_sim_toggle()
    fresh.pump(30)
    assert fresh.scene.snapshot() == before  # simulation state is separate from the editable scene

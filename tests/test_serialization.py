import json

import pytest

from physics_sandbox.core.demo import tower_scene
from physics_sandbox.core.serialization import (
    FORMAT_ID, SceneFormatError, load_scene, save_scene, scene_from_dict, scene_to_dict,
)


def test_round_trip_preserves_components(tmp_path):
    scene = tower_scene()
    path = save_scene(scene, tmp_path / "tower.json")
    loaded = load_scene(path)
    assert [c.id for c in loaded] == [c.id for c in scene]
    for a, b in zip(scene, loaded):
        assert a.type_key == b.type_key
        assert a.position == pytest.approx(b.position)
        assert a.orientation == pytest.approx(b.orientation)
    assert loaded.name == scene.name
    assert loaded.baseplate_size == scene.baseplate_size


def test_saved_file_is_versioned_json(tmp_path):
    data = scene_to_dict(tower_scene())
    assert data["format"] == FORMAT_ID and data["version"] == 1
    path = save_scene(tower_scene(), tmp_path / "s.json")
    assert json.loads(path.read_text())["components"]


def test_rejects_wrong_format_and_version():
    with pytest.raises(SceneFormatError):
        scene_from_dict({"format": "other", "version": 1, "components": []})
    with pytest.raises(SceneFormatError):
        scene_from_dict({"format": FORMAT_ID, "version": 99, "components": []})


def test_rejects_unknown_type_and_bad_numbers():
    base = scene_to_dict(tower_scene())
    bad_type = json.loads(json.dumps(base))
    bad_type["components"][0]["type"] = "nope"
    with pytest.raises(SceneFormatError, match="unknown component type"):
        scene_from_dict(bad_type)
    bad_number = json.loads(json.dumps(base))
    bad_number["components"][0]["position"] = [0, "x", 1]
    with pytest.raises(SceneFormatError):
        scene_from_dict(bad_number)


def test_rejects_duplicate_ids_and_invalid_placement():
    base = scene_to_dict(tower_scene())
    dup = json.loads(json.dumps(base))
    dup["components"][1]["id"] = dup["components"][0]["id"]
    with pytest.raises(SceneFormatError, match="duplicate"):
        scene_from_dict(dup)
    overlap = json.loads(json.dumps(base))
    overlap["components"].append({"id": 999, "type": "cube", "position": [0, 0, 0.5]})
    with pytest.raises(SceneFormatError, match="invalid"):
        scene_from_dict(overlap)


def test_invalid_json_file_is_reported(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(SceneFormatError, match="invalid JSON"):
        load_scene(p)


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    save_scene(tower_scene(), tmp_path / "a.json")
    save_scene(tower_scene(), tmp_path / "a.json")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]

"""Versioned JSON persistence for scenes.

File format (``physics-sandbox.scene`` version 1)::

    {
      "format": "physics-sandbox.scene",
      "version": 1,
      "name": "Turm",
      "baseplate": [24.0, 24.0],
      "grid": 0.125,
      "components": [
        {"id": 1, "type": "plate", "position": [0, 0, 0.125],
         "orientation": [1, 0, 0, 0], "attributes": {}}
      ]
    }

Loading is strict: unknown component types, malformed numbers, overlapping or
out-of-bounds parts are rejected with :class:`SceneFormatError` so that a broken
file can never produce an unstable simulation silently.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

from .components import Component, ComponentRegistry, DEFAULT_REGISTRY
from .geometry import quat_normalize
from .scene import Scene

FORMAT_ID = "physics-sandbox.scene"
FORMAT_VERSION = 1


class SceneFormatError(ValueError):
    """Raised when a scene file is malformed or fails validation."""


def scene_to_dict(scene: Scene) -> Dict[str, Any]:
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "name": scene.name,
        "baseplate": [scene.baseplate_size[0], scene.baseplate_size[1]],
        "grid": scene.grid,
        "components": [
            {
                "id": c.id,
                "type": c.type_key,
                "position": [float(v) for v in c.position],
                "orientation": [float(v) for v in c.orientation],
                "attributes": dict(c.attributes),
            }
            for c in scene
        ],
    }


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneFormatError(f"{where}: expected a number, got {value!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise SceneFormatError(f"{where}: number must be finite")
    return float(value)


def _vector(value: Any, length: int, where: str) -> list:
    if not isinstance(value, list) or len(value) != length:
        raise SceneFormatError(f"{where}: expected a list of {length} numbers")
    return [_number(v, where) for v in value]


def scene_from_dict(data: Dict[str, Any], registry: ComponentRegistry = DEFAULT_REGISTRY,
                    validate: bool = True) -> Scene:
    if not isinstance(data, dict):
        raise SceneFormatError("scene file must contain a JSON object")
    if data.get("format") != FORMAT_ID:
        raise SceneFormatError(f"not a Physics Sandbox scene (format={data.get('format')!r})")
    version = data.get("version")
    if version != FORMAT_VERSION:
        raise SceneFormatError(f"unsupported scene version {version!r} (supported: {FORMAT_VERSION})")

    baseplate = _vector(data.get("baseplate", [24.0, 24.0]), 2, "baseplate")
    grid = _number(data.get("grid", 0.125), "grid")
    name = data.get("name", "Unbenannt")
    if not isinstance(name, str):
        raise SceneFormatError("name must be a string")
    try:
        scene = Scene(registry=registry, baseplate_size=(baseplate[0], baseplate[1]), grid=grid, name=name)
    except ValueError as exc:
        raise SceneFormatError(str(exc)) from exc

    components = data.get("components", [])
    if not isinstance(components, list):
        raise SceneFormatError("components must be a list")
    seen_ids = set()
    for index, entry in enumerate(components):
        where = f"components[{index}]"
        if not isinstance(entry, dict):
            raise SceneFormatError(f"{where}: must be an object")
        cid = entry.get("id")
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 1:
            raise SceneFormatError(f"{where}: id must be a positive integer")
        if cid in seen_ids:
            raise SceneFormatError(f"{where}: duplicate id {cid}")
        seen_ids.add(cid)
        type_key = entry.get("type")
        if type_key not in registry:
            raise SceneFormatError(f"{where}: unknown component type {type_key!r}")
        position = _vector(entry.get("position"), 3, f"{where}.position")
        try:
            orientation = quat_normalize(_vector(entry.get("orientation", [1, 0, 0, 0]), 4, f"{where}.orientation"))
        except ValueError as exc:
            raise SceneFormatError(f"{where}.orientation: {exc}") from exc
        attributes = entry.get("attributes", {})
        if not isinstance(attributes, dict):
            raise SceneFormatError(f"{where}.attributes must be an object")
        scene.insert(Component(cid, type_key, tuple(position), orientation, dict(attributes)))  # type: ignore[arg-type]

    if validate:
        for component in list(scene):
            check = scene.check_placement(component)
            if not check.valid:
                raise SceneFormatError(
                    f"component {component.id} ({component.type_key}) is invalid: {check.reason}"
                )
    scene.clear_selection()
    return scene


def save_scene(scene: Scene, path: Union[str, Path]) -> Path:
    """Write the scene atomically (temporary file + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(scene_to_dict(scene), indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        # mkstemp creates 0600 files; give the saved scene normal, umask-respecting permissions.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def load_scene(path: Union[str, Path], registry: ComponentRegistry = DEFAULT_REGISTRY) -> Scene:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SceneFormatError(f"{source}: invalid JSON ({exc})") from exc
    return scene_from_dict(data, registry)

"""Shared fixtures. The test-suite never opens a window unless a test is marked ``gui``."""

from __future__ import annotations

import pytest

from physics_sandbox.core.commands import CommandHistory
from physics_sandbox.core.scene import Scene


@pytest.fixture
def scene() -> Scene:
    return Scene()


@pytest.fixture
def history(scene: Scene) -> CommandHistory:
    return CommandHistory(scene)

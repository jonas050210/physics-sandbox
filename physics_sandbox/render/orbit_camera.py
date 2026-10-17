"""Orbit camera maths (pure Python/numpy).

The camera orbits a target point. Z is up. ``yaw`` rotates around Z, ``pitch`` is the
elevation above the ground plane. The Panda3D camera simply mirrors these values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..core.geometry import Ray

PITCH_MIN_DEG = 2.0
PITCH_MAX_DEG = 89.0
DISTANCE_MIN = 0.6
DISTANCE_MAX = 90.0

VIEW_PRESETS = {
    "iso": (-45.0, 32.0),
    "front": (-90.0, 8.0),
    "side": (0.0, 8.0),
    "top": (-90.0, 89.0),
}


@dataclass
class OrbitCamera:
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    yaw_deg: float = -45.0
    pitch_deg: float = 32.0
    distance: float = 14.0
    fov_v_deg: float = 42.0

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(3)
        self.clamp()

    def clamp(self) -> None:
        self.pitch_deg = min(max(self.pitch_deg, PITCH_MIN_DEG), PITCH_MAX_DEG)
        self.distance = min(max(self.distance, DISTANCE_MIN), DISTANCE_MAX)
        self.yaw_deg = ((self.yaw_deg + 180.0) % 360.0) - 180.0

    # ------------------------------------------------------------------ basis
    def position(self) -> np.ndarray:
        yaw, pitch = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        offset = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
        return self.target + offset * self.distance

    def forward(self) -> np.ndarray:
        f = self.target - self.position()
        return f / np.linalg.norm(f)

    def right(self) -> np.ndarray:
        r = np.cross(self.forward(), np.array([0.0, 0.0, 1.0]))
        return r / np.linalg.norm(r)

    def up(self) -> np.ndarray:
        return np.cross(self.right(), self.forward())

    def ground_forward(self) -> np.ndarray:
        """Forward direction projected onto the ground plane (used for panning)."""
        f = np.array([self.forward()[0], self.forward()[1], 0.0])
        n = np.linalg.norm(f)
        return f / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def ground_right(self) -> np.ndarray:
        r = self.right()
        r = np.array([r[0], r[1], 0.0])
        return r / np.linalg.norm(r)

    # ------------------------------------------------------------------ controls
    def orbit(self, delta_yaw_deg: float, delta_pitch_deg: float) -> None:
        self.yaw_deg += delta_yaw_deg
        self.pitch_deg += delta_pitch_deg
        self.clamp()

    def zoom(self, factor: float) -> None:
        """Multiply the distance by ``factor`` (< 1 zooms in)."""
        if factor <= 0:
            raise ValueError("zoom factor must be positive")
        self.distance *= factor
        self.clamp()

    def pan(self, dx_units: float, dy_units: float) -> None:
        """Move the target across the ground. Units are fractions of the view distance."""
        scale = self.distance
        self.target = self.target + self.ground_right() * dx_units * scale + self.ground_forward() * dy_units * scale
        self.target[2] = max(self.target[2], 0.0)

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            raise KeyError(f"unknown view preset {name!r}")
        self.yaw_deg, self.pitch_deg = VIEW_PRESETS[name]
        self.clamp()

    def focus(self, center: np.ndarray, radius: float) -> None:
        """Frame a sphere of ``radius`` around ``center``."""
        self.target = np.asarray(center, dtype=float).reshape(3)
        half_fov = math.radians(self.fov_v_deg) / 2.0
        self.distance = max(radius / math.sin(half_fov) * 1.1, DISTANCE_MIN)
        self.clamp()

    # ------------------------------------------------------------------ picking
    def ray_through(self, ndc_x: float, ndc_y: float, aspect: float) -> Ray:
        """Ray through normalised device coordinates in [-1, 1] (x right, y up)."""
        tan_y = math.tan(math.radians(self.fov_v_deg) / 2.0)
        tan_x = tan_y * aspect
        direction = self.forward() + self.right() * (ndc_x * tan_x) + self.up() * (ndc_y * tan_y)
        return Ray.create(self.position(), direction)

    def state(self) -> Tuple[float, float, float]:
        return (self.yaw_deg, self.pitch_deg, self.distance)

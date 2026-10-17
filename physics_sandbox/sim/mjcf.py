"""Builds a MuJoCo MJCF XML document from a :class:`WorldSpec` (pure string generation)."""

from __future__ import annotations

from typing import List

from ..core.components import Shape
from .backend import BodySpec, WorldSpec


def body_name(body_id: int) -> str:
    return f"c{body_id}"


def geom_size(body: BodySpec) -> str:
    sx, sy, sz = body.size
    if body.shape == Shape.BOX:
        return f"{sx / 2:.6g} {sy / 2:.6g} {sz / 2:.6g}"
    if body.shape == Shape.SPHERE:
        return f"{sx / 2:.6g}"
    if body.shape == Shape.CYLINDER:
        return f"{sx / 2:.6g} {sz / 2:.6g}"
    raise ValueError(f"unsupported shape: {body.shape}")


def _mj_shape(shape: Shape) -> str:
    return {"box": "box", "sphere": "sphere", "cylinder": "cylinder"}[shape.value]


def build_mjcf(world: WorldSpec) -> str:
    gx, gy, gz = world.gravity
    if world.integrator not in ("implicitfast", "RK4", "Euler", "implicit"):
        raise ValueError(f"unsupported integrator {world.integrator!r}")
    hx, hy = world.baseplate_size[0] / 2.0, world.baseplate_size[1] / 2.0
    t = world.baseplate_thickness
    lines: List[str] = [
        '<mujoco model="physics-sandbox">',
        '  <compiler angle="degree"/>',
        f'  <option timestep="{world.timestep:.6g}" gravity="{gx:.6g} {gy:.6g} {gz:.6g}" integrator="{world.integrator}"/>',
        "  <default>",
        f'    <geom condim="3" friction="{world.friction:.6g} 0.005 0.0001" solref="{world.contact_timeconst:.6g} 1"/>',
        "  </default>",
        "  <worldbody>",
        f'    <geom name="floor" type="plane" pos="0 0 {-world.floor_depth:.6g}" size="200 200 0.1" '
        'rgba="0.085 0.095 0.11 1"/>',
        f'    <geom name="baseplate" type="box" pos="0 0 {-t / 2:.6g}" size="{hx:.6g} {hy:.6g} {t / 2:.6g}" '
        'rgba="0.17 0.19 0.23 1"/>',
    ]
    for body in world.bodies:
        r, g, b = body.color
        px, py, pz = body.position
        qw, qx, qy, qz = body.orientation
        lines += [
            f'    <body name="{body_name(body.id)}" pos="{px:.9g} {py:.9g} {pz:.9g}" '
            f'quat="{qw:.9g} {qx:.9g} {qy:.9g} {qz:.9g}">',
            "      <freejoint/>",
            f'      <geom name="g{body.id}" type="{_mj_shape(body.shape)}" size="{geom_size(body)}" '
            f'mass="{body.mass_kg:.6g}" friction="{body.friction:.6g} 0.005 0.0001" '
            f'rgba="{r:.4g} {g:.4g} {b:.4g} 1"/>',
            "    </body>",
        ]
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)

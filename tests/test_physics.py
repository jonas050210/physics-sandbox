"""Physics acceptance tests: gravity, rest, stacking, collisions, momentum, reset, determinism."""

import math

import numpy as np
import pytest

from physics_sandbox.core.components import Shape
from physics_sandbox.core.demo import tower_scene
from physics_sandbox.sim.backend import BodySpec, WorldSpec
from physics_sandbox.sim.mjcf import build_mjcf
from physics_sandbox.sim.mujoco_backend import MujocoBackend
from physics_sandbox.sim.simulation import Simulation, world_spec_from_scene

G = 9.81


def ball(bid, pos, vel=(0, 0, 0), mass=1.0, d=0.5):
    return BodySpec(bid, Shape.SPHERE, (d, d, d), mass, 0.6, pos, linear_velocity=vel)


def cube(bid, pos, vel=(0, 0, 0), mass=2.0, ang=(0, 0, 0)):
    return BodySpec(bid, Shape.BOX, (0.5, 0.5, 0.5), mass, 0.8, pos, angular_velocity=ang, linear_velocity=vel)


def test_mjcf_contains_every_body_and_gravity():
    xml = build_mjcf(WorldSpec(bodies=(cube(1, (0, 0, 1)), ball(2, (1, 0, 1)))))
    assert 'name="c1"' in xml and 'name="c2"' in xml
    assert 'gravity="0 0 -9.81"' in xml
    assert 'type="sphere"' in xml and 'type="box"' in xml


def test_free_fall_matches_analytic_solution():
    be = MujocoBackend()
    be.load(WorldSpec(bodies=(ball(1, (0, 0, 20.0)),)))
    be.advance(int(round(1.0 / be.timestep)))
    s = be.state()
    z_expected = 20.0 - 0.5 * G * s.time ** 2
    # semi-implicit Euler ("implicitfast") is first-order: error ~ g*dt*t/2 = 0.5 cm after 1 s at dt=1 ms
    assert s.bodies[1].position[2] == pytest.approx(z_expected, abs=0.01)
    assert s.bodies[1].linear_velocity[2] == pytest.approx(-G * s.time, abs=0.02)


def test_rk4_integrator_is_exact_for_free_fall():
    be = MujocoBackend()
    be.load(WorldSpec(bodies=(ball(1, (0, 0, 20.0)),), integrator="RK4"))
    be.advance(int(round(1.0 / be.timestep)))
    s = be.state()
    assert s.bodies[1].position[2] == pytest.approx(20.0 - 0.5 * G * s.time ** 2, abs=1e-3)


def test_box_at_rest_stays_at_rest():
    be = MujocoBackend()
    be.load(WorldSpec(bodies=(cube(1, (0, 0, 0.25)),)))
    be.advance(int(3.0 / be.timestep))
    b = be.state().bodies[1]
    assert b.position == pytest.approx((0.0, 0.0, 0.25), abs=2e-3)
    assert np.linalg.norm(b.linear_velocity) < 1e-3


def test_stacked_cubes_settle_without_sinking():
    bodies = tuple(cube(i + 1, (0, 0, 0.25 + 0.5 * i + 0.02)) for i in range(4))
    be = MujocoBackend()
    be.load(WorldSpec(bodies=bodies))
    be.advance(int(4.0 / be.timestep))
    s = be.state()
    for i in range(4):
        assert s.bodies[i + 1].position[2] == pytest.approx(0.25 + 0.5 * i, abs=5e-3)
    assert s.max_penetration < 5e-3


def test_ball_lands_on_baseplate_and_comes_to_rest_without_sinking():
    be = MujocoBackend()
    be.load(WorldSpec(bodies=(ball(1, (0, 0, 1.0)),)))
    lowest = math.inf
    impact_speed = 0.0
    for _ in range(int(3.0 / be.timestep)):
        be.advance(1)
        s = be.state()
        z, vz = s.bodies[1].position[2], s.bodies[1].linear_velocity[2]
        lowest = min(lowest, z)
        impact_speed = max(impact_speed, -vz)
    final = be.state().bodies[1]
    assert impact_speed == pytest.approx(math.sqrt(2 * G * 0.75), rel=0.05)  # 0.75 m drop to the surface
    assert lowest > 0.25 - 0.01  # at most 1 cm of contact penetration during impact
    assert final.position[2] == pytest.approx(0.25, abs=2e-3)  # resting on the plate (radius 0.25)
    assert np.linalg.norm(final.linear_velocity) < 1e-2


def test_head_on_spheres_conserve_momentum_and_do_not_pass_through():
    world = WorldSpec(
        bodies=(ball(1, (-1.0, 0, 5.0), vel=(2.0, 0, 0)), ball(2, (1.0, 0, 5.0), vel=(-2.0, 0, 0))),
        gravity=(0.0, 0.0, 0.0),
    )
    be = MujocoBackend()
    be.load(world)
    p0 = np.array(be.state().bodies[1].linear_velocity) + np.array(be.state().bodies[2].linear_velocity)
    min_gap = math.inf
    for _ in range(int(2.0 / be.timestep)):
        be.advance(1)
        s = be.state()
        gap = s.bodies[2].position[0] - s.bodies[1].position[0]
        min_gap = min(min_gap, gap)
    s = be.state()
    p1 = np.array(s.bodies[1].linear_velocity) + np.array(s.bodies[2].linear_velocity)
    assert np.allclose(p0, p1, atol=2e-2)
    assert min_gap > 0.5 - 2e-3  # centres never closer than the sum of radii (2 mm tolerance)


def test_free_body_keeps_angular_velocity_in_zero_gravity():
    world = WorldSpec(bodies=(cube(1, (0, 0, 5.0), ang=(0.0, 0.0, 3.0)),), gravity=(0.0, 0.0, 0.0))
    be = MujocoBackend()
    be.load(world)
    be.advance(int(1.0 / be.timestep))
    assert np.linalg.norm(be.state().bodies[1].angular_velocity) == pytest.approx(3.0, rel=1e-3)


def test_reset_restores_initial_state_exactly():
    be = MujocoBackend()
    be.load(WorldSpec(bodies=(ball(1, (0, 0, 3.0)),)))
    start = be.state().bodies[1].position
    be.advance(500)
    assert be.state().bodies[1].position != start
    be.reset()
    assert be.state().bodies[1].position == pytest.approx(start)
    assert be.state().time == 0.0


def test_simulation_is_deterministic():
    def run():
        sim = Simulation()
        sim.start(tower_scene())
        sim.advance_seconds(2.0)
        return [b.position for b in sim.state.bodies.values()]

    assert np.allclose(np.array(run()), np.array(run()), atol=0.0)


def test_tower_demo_is_stable_and_ball_lands_on_top():
    sim = Simulation()
    sim.start(tower_scene())
    sim.advance_seconds(6.0)
    st = sim.state
    ball_id = max(st.bodies)
    cube_ids = [i for i in st.bodies if i not in (1, 6, ball_id)]
    assert st.max_penetration < 1e-2
    assert st.bodies[ball_id].position[2] > 2.2  # ball rests on the tower top (~2.5 m)
    assert st.max_speed() < 0.5  # settled
    assert len(cube_ids) == 4


def test_world_spec_carries_scene_geometry():
    spec = world_spec_from_scene(tower_scene())
    assert len(spec.bodies) == 7
    assert all(b.mass_kg > 0 for b in spec.bodies)

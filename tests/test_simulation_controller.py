from physics_sandbox.core.demo import tower_scene
from physics_sandbox.sim.backend import PhysicsBackend, WorldState
from physics_sandbox.sim.simulation import SimMode, Simulation


class CountingBackend(PhysicsBackend):
    """Fake backend: counts steps, no physics. Lets controller logic be tested in isolation."""

    name = "fake"

    def __init__(self):
        self.loaded = 0
        self.steps = 0
        self.resets = 0

    def load(self, world):
        self.loaded += 1
        self.world = world

    def advance(self, steps):
        self.steps += steps

    def state(self):
        return WorldState(time=self.steps * self.timestep, steps=self.steps, bodies={})

    def reset(self):
        self.resets += 1
        self.steps = 0

    @property
    def timestep(self):
        return 0.002


def make_sim():
    return Simulation(backend_factory=CountingBackend)


def test_mode_transitions():
    sim = make_sim()
    assert sim.mode == SimMode.EDIT and not sim.is_active
    sim.start(tower_scene())
    assert sim.mode == SimMode.RUNNING
    sim.toggle_pause()
    assert sim.mode == SimMode.PAUSED
    sim.toggle_pause()
    assert sim.mode == SimMode.RUNNING
    sim.reset()
    assert sim.mode == SimMode.EDIT and sim.backend is None and sim.state is None


def test_update_uses_fixed_timesteps_and_speed():
    sim = make_sim()
    sim.start(tower_scene())
    sim.update(0.05)  # 0.05 s at speed 1 -> 25 steps of 0.002 s (fake backend timestep)
    assert sim.backend.steps == 25
    sim.faster()  # speed 2
    sim.update(0.05)  # 0.1 s of simulated time -> 50 more steps
    assert sim.backend.steps == 75


def test_frame_time_is_capped_to_avoid_spiral_of_death():
    sim = make_sim()
    sim.start(tower_scene())
    sim.update(10.0)  # e.g. after a long stall
    assert sim.backend.steps == 25  # capped at MAX_FRAME_DT=0.05 s


def test_paused_simulation_does_not_advance_but_single_step_does():
    sim = make_sim()
    sim.start(tower_scene())
    sim.pause()
    sim.update(1.0)
    assert sim.backend.steps == 0
    sim.step(3)
    assert sim.backend.steps == 3


def test_restart_keeps_mode_and_resets_backend():
    sim = make_sim()
    sim.start(tower_scene())
    sim.update(0.5)
    sim.restart()
    assert sim.mode == SimMode.RUNNING
    assert sim.backend.resets == 1 and sim.backend.steps == 0


def test_speed_steps_are_bounded():
    sim = make_sim()
    for _ in range(10):
        sim.faster()
    assert sim.speed == 2.0
    for _ in range(10):
        sim.slower()
    assert sim.speed == 0.25


def test_start_snapshots_scene_so_later_edits_do_not_leak():
    sim = make_sim()
    scene = tower_scene()
    sim.start(scene)
    scene.add("cube", (5, 5, 0.25))
    assert len(sim.backend.world.bodies) == 7  # snapshot taken at start (tower demo has 7 bodies)

"""MuJoCo implementation of :class:`~physics_sandbox.sim.backend.PhysicsBackend`."""

from __future__ import annotations

import numpy as np

from .backend import BodyState, PhysicsBackend, WorldSpec, WorldState
from .mjcf import body_name, build_mjcf


class MujocoBackend(PhysicsBackend):
    name = "mujoco"

    def __init__(self) -> None:
        import mujoco  # local import: keep the rest of the package importable without MuJoCo

        self._mj = mujoco
        self._model = None
        self._data = None
        self._body_ids: dict[int, int] = {}
        self._initial_qpos: np.ndarray | None = None
        self._initial_qvel: np.ndarray | None = None
        self._world: WorldSpec | None = None

    # ------------------------------------------------------------------ setup
    def load(self, world: WorldSpec) -> None:
        mj = self._mj
        xml = build_mjcf(world)
        self._model = mj.MjModel.from_xml_string(xml)
        self._data = mj.MjData(self._model)
        self._body_ids = {
            b.id: mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, body_name(b.id)) for b in world.bodies
        }
        self._world = world
        # Initial velocities are applied on top of the MJCF pose (free joints: qvel layout is [lin, ang]).
        for body in world.bodies:
            if any(body.linear_velocity) or any(body.angular_velocity):
                dof = self._model.body_dofadr[self._body_ids[body.id]]
                self._data.qvel[dof:dof + 3] = body.linear_velocity
                self._data.qvel[dof + 3:dof + 6] = body.angular_velocity
        mj.mj_forward(self._model, self._data)
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()

    def reset(self) -> None:
        if self._model is None or self._initial_qpos is None:
            raise RuntimeError("backend has not been loaded")
        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = self._initial_qvel
        self._data.time = 0.0
        self._mj.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------ stepping
    @property
    def timestep(self) -> float:
        if self._model is None:
            raise RuntimeError("backend has not been loaded")
        return float(self._model.opt.timestep)

    def advance(self, steps: int) -> None:
        if self._model is None:
            raise RuntimeError("backend has not been loaded")
        for _ in range(int(steps)):
            self._mj.mj_step(self._model, self._data)

    # ------------------------------------------------------------------ observation
    def state(self) -> WorldState:
        if self._model is None or self._data is None:
            raise RuntimeError("backend has not been loaded")
        mj, model, data = self._mj, self._model, self._data
        bodies = {}
        res = np.zeros(6)
        for body_id_key, bid in self._body_ids.items():
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, bid, res, 0)  # world frame: [ang, lin]
            bodies[body_id_key] = BodyState(
                id=body_id_key,
                position=tuple(float(v) for v in data.xpos[bid]),
                orientation=tuple(float(v) for v in data.xquat[bid]),
                linear_velocity=tuple(float(v) for v in res[3:6]),
                angular_velocity=tuple(float(v) for v in res[0:3]),
            )
        ncon = int(data.ncon)
        max_pen = float(max((-data.contact[i].dist for i in range(ncon)), default=0.0))
        return WorldState(
            time=float(data.time),
            steps=int(round(data.time / model.opt.timestep)),
            bodies=bodies,
            contacts=ncon,
            max_penetration=max(0.0, max_pen),
        )

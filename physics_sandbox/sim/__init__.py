"""Simulation layer: physics backends behind a stable interface.

``backend.py`` defines the engine-neutral data types and the abstract ``PhysicsBackend``.
``mujoco_backend.py`` is the Phase 1 implementation. Nothing above this package should
import ``mujoco`` directly.
"""

# Physics Sandbox - roadmap

Status key: **done** (implemented and verified), **partial** (implemented with known gaps),
**planned** (designed, not built). `project.md` is the authority on what exists today.

Each phase lists its dependencies and acceptance criteria. A phase is complete only when all of
its acceptance criteria are verified and recorded in `project.md` §12.

---

## Phase 0 - Launcher and environment: **done**

Dependencies: none.

Delivered: `python3 main.py` with a project-local `.venv`, pinned dependencies, reuse of an
existing environment, dependency fingerprinting, prerequisite checks with clear messages,
`--check`, `--setup-only`, `--reinstall`, `--test`.

Acceptance criteria:
1. First run creates `.venv/` and installs pinned dependencies without system changes. **Met**
   (verified in the build environment).
2. Second run reuses the environment without reinstalling. **Met** (stamp check).
3. Missing display or libGL produces a message naming the fix, and does not install anything.
   **Met** (`--check` output verified).
4. Ubuntu 26.04 on the target hardware. **Not yet verified.**

## Phase 1 - Foundation: **partial** (functionality done; GPU verification open)

Dependencies: Phase 0.

Delivered:
- Real 3D window with baseplate, grid, orbit camera, dark engineering UI, German and English.
- Place, select (with amber outline and inspector), move, rotate, delete, undo/redo.
- Component registry with five primitive types; extensible.
- Rigid-body physics with gravity, collisions and friction via MuJoCo; clean separation of
  editable scene and simulation state.
- Save and load (JSON, strict validation).
- Headless simulation with a JSON report and a smoke test.
- 104 automated tests plus one GUI smoke test (skipped without a display).

Acceptance criteria:
1. `python3 main.py` opens a window with the baseplate, grid and visible UI. **Software-rendered
   screenshot verified; GPU window not verified.**
2. Place, select, move and remove blocks with visible feedback. **Verified by application-level
   tests and the software renderer; not by real mouse input.**
3. Physics: free fall within 1 cm after 1 s; a 4-block tower settles without penetration;
   head-on balls do not pass through each other. **Met** (tests).
4. Save and load a scene, round-trip equal. **Met** (tests).
5. Headless run with a smoke test. **Met** (`--headless`, `test_cli.py`).
6. Documentation separates implemented from planned. **Met** (`project.md`, this file).
7. Frame rate at or above 60 fps on the target GPU. **Not yet measured.**

Open items before closing Phase 1:
- Run the GUI on the target machine and record the real frame rate and screenshot.
- Check the real mouse and keyboard input path (the test suite drives application logic only).
- Verify on Ubuntu 26.04.

## Phase 2 - Joints, motors, wheels and sensors: **planned**

Dependencies: Phase 1 (component system, backend interface).

Scope:
- Joints: fixed, hinge, slider, with limits. Built on MuJoCo equality or joint constraints.
- Motors and actuators: torque or velocity control with a controller interface.
- Wheels: a wheel component with friction and a drive model, connected by hinge joints.
- Sensors: at least IMU (acceleration, angular rate), distance (ray), contact.
- Scene format version 2 with a migration from version 1.

Acceptance criteria:
1. A hinged pendulum matches the analytic period within 1 %.
2. A motor-driven wheel on a plane reaches a commanded angular speed within 2 % at steady state.
3. A sensor reading is available in headless mode and in the report.
4. Version 1 scene files still load.
5. All new types are registered through the component registry, with German and English names.

## Phase 3 - Aircraft and vehicles: **planned**

Dependencies: Phase 2 (joints, actuators, sensors).

Scope:
- Aerodynamic force and moment model for simple wings and control surfaces (lift, drag,
  pitching moment), with documented coefficients.
- Flight dynamics integration with the rigid-body simulation.
- Vehicle templates (car on wheels) and a simple controller.

Acceptance criteria:
1. A reference wing reproduces the expected lift coefficient within 5 % in a documented test.
2. A trimmed aircraft in steady level flight holds altitude within 1 m over 60 s of simulated time.
3. A vehicle template drives in a straight line and turns with the commanded steering.
4. Limits and assumptions are documented in `project.md`.

## Phase 4 - Custom meshes (GLB/glTF import): **planned**

Dependencies: Phase 1 (render layer, component system); Phase 2 for joints on imported parts.

Scope:
- Import GLB files as component types, with a convex or compound collision approximation.
- Asset storage outside Git (large binaries are not committed).

Acceptance criteria:
1. A GLB part loads, renders and collides; scene save and load keep it by reference.
2. Import fails with a clear error for unsupported files.
3. Collision approximation error is reported.

## Phase 5 - Reinforcement learning with PyTorch: **planned**

Dependencies: Phase 1 headless mode; Phase 2 actuators and sensors recommended.

Scope:
- Environment wrapper around the headless simulation (reset, step, observation, reward hook).
- Training entry point, checkpoints in a non-versioned folder, and a seeded reproducibility check.
- PyTorch added as a pinned dependency only in this phase, installed by the launcher as an
  optional group.

Acceptance criteria:
1. Environment API is deterministic under a fixed seed.
2. A reference task (e.g. balancing a plate) improves over a random policy in a documented run.
3. The GUI is not required for training; training runs headless.

## Phase 6 - Rocket-League-style mode: **planned**

Dependencies: Phases 2 and 3 (vehicles, joints); Phase 1 physics.

Scope: arena, ball, cars, scoring rules, match timer and reset logic.

Acceptance criteria: a scored goal is detected deterministically in a scripted test; a match
runs for its full duration without numerical instability.

## Phase 7 - Localisation beyond German and English: **planned**

Dependencies: Phase 1 i18n tables.

Acceptance criteria: a new language is added by adding one table; a parity test fails if keys are
missing.

## Cross-cutting, not scheduled

- Release packaging and CI: none exist. Nothing is published, and no release or pull request is
  opened unless requested.
- Documentation: `project.md` must be updated with each phase; implemented and planned items
  must stay separate.

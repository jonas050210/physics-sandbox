# Physics Sandbox - project knowledge base

This file is the authoritative reference for future contributors and coding agents.
Every statement is tagged **[implemented]** (in the repository and verified), **[partial]**
(exists but with known gaps) or **[planned]** (designed for, not built). Do not treat a
**[planned]** item as available. See `ROADMAP.md` for phases and acceptance criteria.

---

## 1. Purpose

A realistic, extensible 3D engineering sandbox: place building blocks, combine them with
physics, and later add vehicles, aircraft and AI. Phase 1 is the foundation: scene model,
editor, physics, rendering, UI, persistence and a headless mode.

## 2. Target environment

| Item | Value |
|---|---|
| Target OS | Ubuntu 26.04 LTS (not yet verified on that OS, see §12) |
| Target hardware | Intel i7-12700F, NVIDIA RTX 4060 Ti 8 GB, 32 GB RAM |
| Python | 3.11 to 3.14 (wheels verified for cp311-cp314 for the pinned versions) |
| Runtime libraries | MuJoCo 3.16.0, Panda3D 1.10.16, numpy 2.4.6 (pinned in `requirements.txt`) |
| Dev library | pytest 8.4.2 (`requirements-dev.txt`) |
| System (GUI only) | a graphical session (`DISPLAY` or `WAYLAND_DISPLAY`) and `libGL.so.1` |

**[implemented]** Pins are exact (`==`). Upgrading a pin changes the launcher's dependency
fingerprint and triggers a reinstall on the next launch.

## 3. How to run

```bash
python3 main.py                       # GUI; first run prepares .venv and dependencies
python3 main.py --headless            # simulation without a window, JSON report to stdout
python3 main.py --headless --scene examples/tower.json --duration 5 --report reports/tower.json
python3 main.py --test                # installs dev tools if needed, runs pytest
python3 main.py --check               # reports prerequisites, changes nothing
python3 main.py --setup-only          # only prepares .venv and dependencies
python3 main.py --reinstall           # forces reinstalling the pinned dependencies
python3 main.py --help-app            # application options
```

**[implemented]** All of the above, verified in the build environment (except the GUI window,
see §12).

### 3.1 Launcher behaviour ([implemented], `main.py`)

- Standard library only, so it runs before any project dependency exists.
- Creates `.venv/` inside the project root. Recreates it only if the interpreter's major.minor
  version differs. Nothing is installed system-wide and no `sudo` is used.
- Stamps `.venv/.deps-runtime.sha256` and `.venv/.deps-dev.sha256` are keyed on Python
  implementation and version, machine architecture, and the requirement file contents. If
  the stamp matches, installation is skipped ("reusing the existing environment").
- After installing, `.venv/.deps-runtime-verified.sha256` is written only after `numpy`, `mujoco` and
  `panda3d.core` import successfully.
- Before starting the GUI it checks `DISPLAY`/`WAYLAND_DISPLAY` and `libGL.so.1`. Missing
  prerequisites produce a clear message with the one command that fixes them; the launcher
  never runs that command itself.
- Headless runs skip the display and GL checks.
- Exit codes: 0 success, 2 launcher error, 130 interrupt; the GUI/headless child's exit code is
  passed through.

## 4. Architecture

Layers, from bottom to top. Dependencies may only point downwards.

```
physics_sandbox/
  core/     pure Python (no Panda3D, no MuJoCo): geometry, components, scene,
            commands (undo/redo), editor (tools and placement rules), serialization, demo
  sim/      physics: backend interface, MJCF builder, MuJoCo backend, Simulation controller
  render/   Panda3D presentation: SceneView (reads snapshots), orbit camera, meshes
  ui/       Panda3D HUD (DirectGUI), theme, fonts, widgets, i18n strings (DE/EN)
  app.py    ShowBase application: wires the layers, input handling, frame loop
  headless.py  run a scene without a window; JSON report
  cli.py    argument parsing for `python -m physics_sandbox`
tests/      pytest suite (headless tests, plus one GUI smoke test that is skipped without a display)
tools/      developer utilities (screenshot.py)
main.py     launcher (see §3.1)
```

### 4.1 Key decisions

1. **Visual scene is separate from simulation state.** `core.Scene` is the editable model
   (positions the user placed). `sim.Simulation` owns body states while running. `render.SceneView`
   only reads snapshots (`scene.snapshot()` and `sim.poses()`) and never mutates state.
   Consequence: stopping the simulation restores the edited layout.
2. **Physics runs in MuJoCo; the editor never imports it.** `sim.PhysicsBackend` is an
   abstract interface (`load`, `advance`, `state`, `reset`). `MujocoBackend` imports `mujoco`
   lazily. This keeps a future second backend possible without touching the editor.
3. **Core is testable without a display.** `core/`, `sim/` and most of `render/` import no
   Panda3D code at module load, so the 100+ tests run on any machine.
4. **Components are data plus a registry.** `ComponentType` (name per language, shape,
   extents, mass, colour, friction, optional attributes) lives in a `ComponentRegistry`.
   New parts are registered; the editor, serialization and simulation read the registry rather
   than hard-coding types.
5. **Command history.** Every edit that changes the scene goes through `CommandHistory`
   (`AddComponent`, `RemoveComponents`, `ChangePoses`). Undo/redo works on those commands. A
   drag records one `ChangePoses` on release.
6. **Coordinates and units.** Z-up, SI units (m, kg, s), quaternions `(w, x, y, z)`. Placement
   grid 0.125 m. Baseplate 24 x 24 m, top surface at z = 0. Physics floor plane at z = -1.0.
7. **Fixed-step physics with an accumulator.** `Simulation.update(wall_dt)` caps frame time at
   0.05 s and steps in multiples of the timestep (default 0.001 s). Headless runs call
   `advance_seconds`.
8. **Physics configuration (chosen values).** Timestep 0.001 s; integrator `implicitfast`;
   contact time constant `solref` 0.002. Measured in the build environment:
   - free fall after 1 s: error 4.9 mm (about 0.5 %, first-order bias of implicitfast);
   - `RK4` at the same timestep is exact for free fall but about 3x slower (about 640 ms per
     4 s of simulated time, against about 205 ms for implicitfast);
   - stacked-tower demo: no penetration after settling (max penetration 0.0 mm at rest;
     0.004 mm in the headless report);
   - head-on balls: minimum gap 0.500 m (no tunnelling in the test case).
   The `dt 0.002` / solref 0.01 variant was also tried (fall error 9.8 mm, tower penetration
   0.11 mm); it was rejected for accuracy.
9. **Persistence is plain JSON.** Format `physics-sandbox.scene` version 1. Saves are atomic
   (temporary file then `os.replace`), with normal file permissions. Loading is strict:
   types, finite numbers, unique ids, known component types and valid placement are checked.
10. **Internationalisation.** All user-visible text goes through `ui.i18n.t(key, lang)`. German
    is the default, English is the alternative. `missing_keys()` and a test enforce parity.
11. **Software-render path for screenshots.** `PhysicsSandboxApp(window_type="offscreen")`
    uses Panda3D's built-in software renderer (tinydisplay). It is used only by
    `tools/screenshot.py` and is not the user-facing path.

### 4.2 Panda3D window notes

- ShowBase is a process singleton: only one `PhysicsSandboxApp` can exist per Python process.
  Tests share one instance and reset state; the real-window smoke test runs in a subprocess.
- Without a window, `base.cam`, `base.camLens`, `base.mouseWatcherNode` and `base.win` are
  `None`. All window-dependent code is guarded (`has_window`, `None` checks).
- `NodePath.node()` drops NodePath-level state such as `setLightOff`; keep NodePaths and use
  `reparentTo`.
- `ShowBase` uses `self.config` internally, so the application's own config is `self.app_config`.
- `Lens` has no `setVfov`; use `setFov(hfov, vfov)` with `hfov` derived from the aspect ratio.

## 5. Implemented functionality (Phase 1)

### 5.1 Scene and editing [implemented]
- Component types: **Cube** (Würfel, 0.5 m, 2 kg), **Plate** (Platte, 1 x 1 x 0.25 m, 3 kg),
  **Beam** (Balken, 2 x 0.25 x 0.25 m, 1.5 kg), **Ball** (Kugel, Ø 0.5 m, 1 kg, friction 0.6),
  **Cylinder** (Zylinder, Ø 0.5 x 0.5 m, 1.5 kg).
- Tools: Select (V), Place (B), Move (G), Delete (X). Rotate by 90° about the component's own
  vertical axis (R). Undo/redo (Ctrl+Z, Ctrl+Y or Ctrl+Shift+Z).
- Placement snaps to the grid, stacks on the face under the cursor, and is rejected if it
  overlaps another component, lies outside the baseplate or penetrates the ground. The ghost
  preview shows validity.
- Selection with Ctrl+click for multiple components. Selected components show an amber outline;
  hover shows a light-blue outline. The inspector shows id, type, position, rotation, mass,
  size and (while simulating) speed and angular rate.
- Escape cancels a drag, leaves placement mode, or clears the selection.

### 5.2 Physics [implemented]
- Gravity, rigid bodies, contacts and friction via MuJoCo.
- Simulation modes: Edit, Running, Paused. Speed steps 0.25x, 0.5x, 1x, 2x. Start/pause (P),
  reset to the edited layout, speed buttons.
- While running, editing is locked (the palette and placement are refused, with a message).
- Initial velocities: components may carry `linear_velocity` and `angular_velocity` attributes
  (the scene file supports them; the GUI does not yet set them, see §6).

### 5.3 Rendering and camera [implemented]
- Large viewport between left (components, view, file) and right (simulation, properties, scene)
  panels; top toolbar; bottom status bar.
- Baseplate, grid (0.125 m minor, 1 m major), axes, key and fill lights, shadow map (on GPU).
- Orbit camera: right-drag or arrow keys orbit; middle-drag or WASD pan; wheel zooms; Home
  frames the scene; presets Iso, Top, Front, Side.
- Dark engineering theme; UI text in German or English (L or the toolbar button switches).

### 5.4 Component system [implemented, foundational]
- `ComponentType` and `ComponentRegistry` allow new types to be registered. `Component`
  instances hold id, type key, position, orientation and an attributes map.
- `component_obb` and `Shape` (BOX, SPHERE, CYLINDER) are the geometry primitives.
- **Not implemented:** joints, motors, wheels, sensors, custom meshes (see §7 and ROADMAP).

### 5.5 Persistence [implemented]
- F5 saves to `saves/scene-YYYYMMDD-HHMMSS.json` (personal saves, gitignored).
- F9 loads the newest JSON file in `saves/`.
- `examples/tower.json` is a versioned example. Load it with `--scene examples/tower.json`.
- The file-open dialog, scene browser, and named saves are **[planned]**.

### 5.6 Headless mode [implemented]
- `python3 main.py --headless [--scene PATH] [--duration S] [--timestep S] [--integrator I] [--report PATH]`.
- Writes JSON format `physics-sandbox.headless-report` version 1: scene name, backend, integrator,
  timestep, steps, simulated/wall seconds, realtime factor, body and contact counts, maximum
  penetration, maximum speed, finiteness, settled flag and `ok`.
- `ok` is true when all values are finite and maximum penetration is below 0.02 m. The
  process exits 0 if ok, 1 otherwise, 2 on a scene error.
- Measured: the 7-body tower runs about 19x faster than real time on the build machine (2 vCPU).

### 5.7 Tests [implemented]
- `python3 main.py --test` → **104 passed, 1 skipped** (the skipped test is the real-window
  smoke test, which needs a display and libGL). Run time about 1.5 s.
- Coverage by file: geometry, scene and commands, editor, serialization, physics, simulation
  controller, camera and meshes, i18n (including dynamically built keys), launcher, CLI, headless,
  application logic without a window (`test_app_headless.py`).
- Known test gap: rendering output is verified only by the screenshot tool, not by assertions.

## 6. Known limitations (Phase 1)

- **Not verified on a real GPU window.** The build environment has no OpenGL and no display.
  The window path was exercised through Panda3D's software renderer (offscreen, tinydisplay)
  with screenshots (`docs/screenshots/overview-de.png`), but shadows, anti-aliasing and
  real-time frame rate on the RTX 4060 Ti are unverified.
- Ubuntu 26.04 was not tested. The launcher targets it, but the default Python there and the
  wheel availability must be confirmed on first setup.
- Panda3D 1.10.16 needs system `libGL.so.1` (`sudo apt install libgl1 libgl1-mesa-dri`) for the
  window. The launcher detects this and stops with that message.
- Font: the HUD uses DejaVu Sans from `/usr/share/fonts/truetype/dejavu/` when available;
  otherwise Panda3D's default font is used.
- Editing is limited to the five component types and 90° yaw rotation. No free-angle rotation.
- Contacts use MuJoCo's default model, which has no restitution (bounce). Contact parameters
  are tuned only against the free-fall, head-on and tower checks in §4.1, not against reference data.
- Undo history is cleared by New and Load.
- No undo of physics state changes (only editing).
- F9 loads the newest save only; there is no file picker yet.

## 7. Planned (not implemented)

Each item is detailed with dependencies and acceptance criteria in `ROADMAP.md`.

- Joints (hinge, slider, fixed), motors and actuators, wheels with a drive model.
- Sensors (IMU, distance, contact).
- Aircraft flight dynamics (aerodynamic forces, control surfaces).
- Vehicle templates and vehicle controllers.
- GLB/glTF import of custom meshes with collision generation.
- Reinforcement learning with PyTorch (environment wrapper over the headless simulation,
  trainer, checkpoint management). PyTorch is **not** a dependency yet.
- Rocket-League-style mode (ball, cars, scoring rules, match loop).
- Localisation beyond German and English.
- Installable packaging, release tooling and CI (none exist; this repository has no release).

## 8. Conventions for contributors

- Keep `core/`, `sim/` (except `mujoco_backend.py`) and `scene` logic free of Panda3D imports.
- Add user-visible strings to both `ui/i18n.py` language tables; run `pytest -q tests/test_i18n.py`.
- Every new scene-changing operation goes through a command so undo/redo keeps working.
- Physics changes must keep: free fall within the documented tolerance, no penetration in the
  tower demo, and `headless` ok for `examples/tower.json`.
- Do not commit `saves/*.json`, `reports/`, `.venv/`, screenshots from personal scenes, or large
  binary assets. Versioned examples belong in `examples/`.
- Do not create releases or pull requests unless the user asks.

## 9. File map

| Path | Purpose |
|---|---|
| `main.py` | launcher (stdlib only) |
| `requirements.txt`, `requirements-dev.txt` | pinned dependencies |
| `pytest.ini` | test paths, `gui` marker |
| `physics_sandbox/config.py` | `AppConfig` (window size, language, physics timestep, paths) |
| `physics_sandbox/app.py` | Panda3D application, input, frame loop |
| `physics_sandbox/cli.py`, `__main__.py` | command line |
| `physics_sandbox/headless.py` | window-less run and report |
| `physics_sandbox/core/*` | scene model, components, editor, commands, serialization, demo |
| `physics_sandbox/sim/*` | backend interface, MJCF, MuJoCo backend, simulation controller |
| `physics_sandbox/render/*` | scene view, orbit camera, mesh and node builders |
| `physics_sandbox/ui/*` | HUD, widgets, theme, fonts, i18n |
| `examples/tower.json` | example scene (the demo tower) |
| `tools/screenshot.py` | offscreen screenshot via software renderer |
| `docs/screenshots/` | screenshots shipped with the docs |
| `saves/` | personal saves (gitignored, `.gitkeep` kept) |
| `reports/` | headless reports (gitignored) |
| `README.md`, `ROADMAP.md`, `project.md` | user guide, roadmap, knowledge base |

## 10. Scene file format (version 1)

```json
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
```

Attributes `linear_velocity` and `angular_velocity` (3-vectors) are read by the simulation.
Unknown component types, non-finite numbers, duplicate ids and invalid placement are rejected.

## 11. Decisions log

| # | Decision | Reason |
|---|---|---|
| D1 | MuJoCo for physics, Panda3D for rendering and UI | MuJoCo is fast and accurate for rigid bodies; Panda3D gives a scene graph, picking and DirectGUI without a large engine |
| D2 | Pure-Python core, tested without a display | Fast, deterministic tests on any machine |
| D3 | Project-local `.venv`, pinned versions, stdlib launcher | Reproducible; no system changes; works before dependencies exist |
| D4 | JSON scene format, version 1, strict validation | Human-readable, diffable, safe to load |
| D5 | Timestep 0.001 s, implicitfast, solref 0.002 | Best measured trade-off of accuracy and speed (§4.1) |
| D6 | German default, English second, key-parity test | Stated product requirement |
| D7 | No release, no PR in Phase 1 | Explicit user instruction |

## 12. Verification record (build environment)

- Environment: Debian 12, kernel 6.1, Python 3.11.2, 2 vCPU, about 3 GB RAM, no display, no
  GPU, no `libGL`. (Target hardware and OS are not available here.)
- Installation: `python3 main.py --setup-only` succeeded (about 12 s).
- Tests: 104 passed, 1 skipped (GUI smoke test) in about 1.5 s.
- Headless: `examples/tower.json`, 3 s simulated: ok, 25 contacts, max penetration 0.004 mm.
- Application logic without a window: placement, simulation start/pause/reset, language switch,
  save/load, undo/redo, view presets, resize: exercised in `tests/test_app_headless.py`.
- Window path: exercised through the software renderer (`tools/screenshot.py`). Bugs found and
  fixed this way: transparent background (alpha 0) displayed as white, and clipped title and
  speed labels.
- Not verified: GPU rendering, the real mouse and keyboard input path, the 60 fps target.

"""The graphical application: Panda3D window, input, camera, and the frame loop.

Data flow each frame::

    input -> Editor (core)  -> Scene / CommandHistory      (what was built)
    Simulation (sim)        -> WorldState poses            (what physics does)
    SceneView (render)      <- scene or simulation poses   (how it looks)
    Hud (ui)                <- HudState                    (what the user reads)
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ClockObject,
    KeyboardButton,
    LPoint3,
    LVector3,
    MouseButton,
    loadPrcFileData,
)

from .config import AppConfig
from .core.commands import CommandHistory
from .core.components import ComponentRegistry, DEFAULT_REGISTRY
from .core.demo import tower_scene
from .core.editor import EditResult, Editor, Tool
from .core.geometry import Ray
from .core.scene import Scene
from .core.serialization import SceneFormatError, load_scene, save_scene
from .render.orbit_camera import OrbitCamera
from .render.viewport import SceneView
from .sim.simulation import SimMode, Simulation
from .ui import theme
from .ui.fonts import load_ui_font
from .ui.hud import Hud, HudCallbacks, HudState
from .ui.i18n import normalize_language, t

_MOUSE_ORBIT_DEG_PER_NDC = 140.0
_MOUSE_PAN_PER_NDC = 0.55  # fraction of the view distance per NDC unit
_KEY_ORBIT_DEG_PER_S = 90.0
_KEY_PAN_PER_S = 0.45
_ZOOM_STEP = 0.88


def configure_panda(config: AppConfig, window: bool, software: bool = False) -> None:
    """Set Panda3D config before ShowBase starts. ``software`` selects Panda's built-in
    tinydisplay renderer (no OpenGL needed); used only for offscreen screenshots and tests."""
    lines = [
        "window-title Physics Sandbox",
        f"win-size {config.width} {config.height}",
        f"sync-video {'#t' if config.vsync else '#f'}",
        "framebuffer-multisample 1",
        "multisamples 4",
        "audio-library-name null",
        "show-frame-rate-meter #f",
        "notify-level-gobj error",
    ]
    if software:
        lines.append("load-display p3tinydisplay")
    if not window:
        lines.append("window-type none")
    loadPrcFileData("", "\n".join(lines) + "\n")


class PhysicsSandboxApp(ShowBase):
    def __init__(self, config: Optional[AppConfig] = None, scene: Optional[Scene] = None,
                 window: bool = True, registry: ComponentRegistry = DEFAULT_REGISTRY,
                 window_type: str = "onscreen") -> None:
        """``window_type``: "onscreen" (normal), "offscreen" (render to a buffer, software
        renderer; for screenshots) or "none" (no graphics at all; tests and headless)."""
        self.app_config = config or AppConfig()
        self.language = normalize_language(self.app_config.language)
        if window_type not in ("onscreen", "offscreen", "none"):
            raise ValueError(f"unknown window_type {window_type!r}")
        self.has_window = window and window_type != "none"
        self.registry = registry
        software = window_type == "offscreen"
        configure_panda(self.app_config, window=self.has_window, software=software)
        if not self.has_window:
            base_window_type = "none"
        elif software:
            base_window_type = "offscreen"
        else:
            base_window_type = None  # ShowBase default: onscreen window
        ShowBase.__init__(self, windowType=base_window_type)
        if self.has_window:
            self.disableMouse()  # we own the camera
        self.setBackgroundColor(*theme.BACKGROUND)  # full RGBA: alpha 0 would show as transparent
        # Without a window ShowBase has no camera; use a plain node so the same code path runs.
        self.camera_np = self.cam if self.cam is not None else self.render.attachNewNode("virtual-camera")
        if self.camLens is not None:  # None when running without a window
            self.camLens.setNearFar(0.05, 800.0)

        self._win_size = (self.app_config.width, self.app_config.height)
        self._region = (0.0, 1.0, 0.0, 1.0)
        self._aspect = self.app_config.width / float(self.app_config.height)

        if scene is None:
            scene = self._initial_scene()
        self.scene = scene
        self.history = CommandHistory(self.scene)
        self.sim = Simulation(timestep=self.app_config.physics_timestep, integrator=self.app_config.integrator)
        self.editor = Editor(self.scene, self.history, locked=lambda: self.sim.is_active)

        self.orbit = OrbitCamera(target=np.array([0.0, 0.0, 0.4]), fov_v_deg=self.app_config.vertical_fov_deg)
        self.orbit.set_view("iso")
        self.orbit.focus(np.array([0.0, 0.0, 0.5]), radius=4.0)

        self.view = SceneView(self.render, self.registry, self.app_config.baseplate_size)
        self._font = load_ui_font(self.loader) if self.has_window else None
        self.hud = Hud(self.a2dTopLeft, self.registry, self._hud_callbacks(), font=self._font)

        self._last_mouse: Optional[Tuple[float, float]] = None
        self._message: Tuple[str, Dict[str, object]] = ("msg.ready", {})
        self._hint_key = "hint.select"
        self._clock = ClockObject.getGlobalClock()

        self._bind_events()
        self._resize(*self._win_size)
        self.taskMgr.add(self._on_frame, "physics-sandbox-frame")

    # ------------------------------------------------------------------ setup helpers
    def _initial_scene(self) -> Scene:
        if self.app_config.scene_path:
            return load_scene(self.app_config.scene_path, self.registry)
        if self.app_config.start_empty:
            return Scene(registry=self.registry, baseplate_size=self.app_config.baseplate_size, grid=self.app_config.grid)
        return tower_scene(self.registry, self.language)

    def _hud_callbacks(self) -> HudCallbacks:
        return HudCallbacks(
            on_tool=self.act_tool,
            on_palette=self.act_palette,
            on_undo=self.act_undo,
            on_redo=self.act_redo,
            on_rotate=self.act_rotate,
            on_view=self.act_view,
            on_language=self.act_language,
            on_sim_toggle=self.act_sim_toggle,
            on_sim_reset=self.act_sim_reset,
            on_speed_up=self.act_speed_up,
            on_speed_down=self.act_speed_down,
            on_new=self.act_new,
            on_save=self.act_save,
            on_load=self.act_load,
        )

    def _bind_events(self) -> None:
        self.accept("mouse1", self._on_press)
        self.accept("mouse1-up", self._on_release)
        self.accept("wheel_up", self._on_wheel, [1])
        self.accept("wheel_down", self._on_wheel, [-1])
        keys = {
            "v": lambda: self.act_tool("select"),
            "b": lambda: self.act_tool("place"),
            "g": lambda: self.act_tool("move"),
            "x": lambda: self.act_tool("delete"),
            "delete": self.act_delete_selection,
            "backspace": self.act_delete_selection,
            "escape": self.act_escape,
            "r": self.act_rotate,
            "p": self.act_sim_toggle,
            "control-z": self.act_undo,
            "control-y": self.act_redo,
            "control-shift-z": self.act_redo,
            "shift-control-z": self.act_redo,  # Panda3D modifier order is not guaranteed
            "control-q": self.userExit,
            "f5": self.act_save,
            "f9": self.act_load,
            "home": self.act_frame_all,
            "l": self.act_language,
        }
        for event, handler in keys.items():
            self.accept(event, handler)

    # ------------------------------------------------------------------ window / layout
    def windowEvent(self, win):  # noqa: N802 - Panda3D API name
        ShowBase.windowEvent(self, win)
        if getattr(self, "hud", None) is None:  # may fire while ShowBase is still initialising
            return
        if self.has_window and self.win is not None:
            props = self.win.getProperties()
            size = (props.getXSize(), props.getYSize())
            if size != self._win_size and size[0] > 0 and size[1] > 0:
                self._resize(*size)

    def _resize(self, width: int, height: int) -> None:
        self._win_size = (int(width), int(height))
        self.hud.rebuild(*self._win_size)
        l, r, b, t_ = self.hud.viewport_fractions()
        self._region = (l, r, b, t_)
        w_px = (r - l) * self._win_size[0]
        h_px = (t_ - b) * self._win_size[1]
        self._aspect = max(w_px / max(h_px, 1.0), 0.1)
        if self.has_window and self.win is not None:
            dr = self.cam.node().getDisplayRegion(0)
            dr.setDimensions(l, r, b, t_)
        if self.camLens is not None:  # None when running without a window
            self.camLens.setAspectRatio(self._aspect)
            vfov = self.app_config.vertical_fov_deg
            hfov = math.degrees(2.0 * math.atan(math.tan(math.radians(vfov) / 2.0) * self._aspect))
            self.camLens.setFov(hfov, vfov)

    # ------------------------------------------------------------------ pointer helpers
    def _pointer_raw(self) -> Optional[Tuple[float, float]]:
        """Pointer as normalised device coordinates of the *viewport* (may lie outside [-1, 1])."""
        if not self.has_window or self.mouseWatcherNode is None or not self.mouseWatcherNode.hasMouse():
            return None
        mx, my = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
        u, v = (mx + 1) / 2.0, (my + 1) / 2.0
        l, r, b, t_ = self._region
        return ((u - l) / (r - l)) * 2 - 1, ((v - b) / (t_ - b)) * 2 - 1

    def _pointer_in_viewport(self) -> Optional[Tuple[float, float]]:
        ndc = self._pointer_raw()
        if ndc is None or abs(ndc[0]) > 1 or abs(ndc[1]) > 1:
            return None
        return ndc

    def _ray(self, ndc: Tuple[float, float]) -> Ray:
        return self.orbit.ray_through(ndc[0], ndc[1], self._aspect)

    def _poses(self):
        return self.sim.poses()

    def _ctrl_down(self) -> bool:
        if not self.has_window or self.mouseWatcherNode is None:
            return False
        return bool(self.mouseWatcherNode.isButtonDown(KeyboardButton.control()))

    # ------------------------------------------------------------------ mouse/keyboard events
    def _on_press(self) -> None:
        ndc = self._pointer_in_viewport()
        if ndc is None:
            return
        result = self.editor.pointer_down(self._ray(ndc), additive=self._ctrl_down(), poses=self._poses())
        self._report(result)

    def _on_release(self) -> None:
        ndc = self._pointer_raw()
        ray = self._ray(ndc) if ndc is not None else None
        result = self.editor.pointer_up(ray)
        self._report(result)

    def _on_wheel(self, direction: int) -> None:
        if self._pointer_in_viewport() is None:
            return
        self.orbit.zoom(_ZOOM_STEP if direction > 0 else 1.0 / _ZOOM_STEP)

    # ------------------------------------------------------------------ actions
    def _report(self, result: EditResult) -> None:
        if result.message:
            self._message = (result.message, dict(result.args))

    def _say(self, key: str, **args) -> None:
        self._message = (key, args)

    def act_tool(self, name: str) -> None:
        if name == "place" and self.sim.is_active:
            self._say("msg.locked")
            return
        self.editor.set_tool(Tool(name))
        self._hint_key = {"select": "hint.select", "place": "hint.place", "move": "hint.move",
                          "delete": "hint.delete"}[name]

    def act_palette(self, type_key: str) -> None:
        if self.sim.is_active:
            self._say("msg.locked")
            return
        self.editor.select_palette(type_key)
        self._hint_key = "hint.place"

    def act_undo(self) -> None:
        self._report(self.editor.undo())

    def act_redo(self) -> None:
        self._report(self.editor.redo())

    def act_rotate(self) -> None:
        self._report(self.editor.rotate())

    def act_delete_selection(self) -> None:
        self._report(self.editor.delete_selected())

    def act_escape(self) -> None:
        self.editor.cancel()
        self._hint_key = "hint.select" if self.editor.tool == Tool.SELECT else self._hint_key

    def act_view(self, name: str) -> None:
        self.orbit.set_view(name)

    def act_frame_all(self) -> None:
        bounds = self.scene.bounding_box_all()
        if bounds is None:
            self.orbit.focus(np.array([0.0, 0.0, 0.5]), radius=4.0)
            return
        mn, mx = bounds
        center = (mn + mx) / 2.0
        self.orbit.focus(center, radius=float(np.linalg.norm(mx - mn) / 2.0 + 0.5))

    def act_language(self) -> None:
        self.language = "en" if self.language == "de" else "de"

    def act_sim_toggle(self) -> None:
        if self.sim.mode == SimMode.EDIT:
            self.sim.start(self.scene)
            self.editor.set_tool(Tool.SELECT)  # leave placement mode but keep the selection
            self._say("msg.sim_started")
        elif self.sim.mode == SimMode.RUNNING:
            self.sim.pause()
            self._say("msg.sim_paused")
        else:
            self.sim.resume()
            self._say("msg.sim_resumed")

    def act_sim_reset(self) -> None:
        self.sim.reset()
        self._say("msg.sim_reset")

    def act_speed_up(self) -> None:
        self.sim.faster()

    def act_speed_down(self) -> None:
        self.sim.slower()

    def act_new(self) -> None:
        self.sim.reset()
        self._set_scene(Scene(registry=self.registry, baseplate_size=self.app_config.baseplate_size, grid=self.app_config.grid))
        self._say("msg.new")

    def act_save(self) -> None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = Path(self.app_config.saves_dir) / f"scene-{stamp}.json"
        try:
            save_scene(self.scene, target)
        except OSError as exc:
            self._say("msg.error", error=str(exc))
            return
        self._say("msg.saved", name=target.name)

    def act_load(self) -> None:
        folder = Path(self.app_config.saves_dir)
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime) if folder.exists() else []
        if not files:
            self._say("msg.no_save")
            return
        newest = files[-1]
        try:
            scene = load_scene(newest, self.registry)
        except SceneFormatError as exc:
            self._say("msg.error", error=str(exc))
            return
        self.sim.reset()
        self._set_scene(scene)
        self._say("msg.loaded", name=newest.name)

    def _set_scene(self, scene: Scene) -> None:
        tool, palette = self.editor.tool, self.editor.palette_type
        self.scene = scene
        self.history = CommandHistory(scene)
        self.editor = Editor(scene, self.history, locked=lambda: self.sim.is_active, palette_type=palette)
        self.editor.tool = tool

    # ------------------------------------------------------------------ per-frame
    def _update_camera_input(self, dt: float) -> None:
        mw = self.mouseWatcherNode
        if mw is None:  # offscreen or no window: there is no pointer or keyboard
            return
        ndc = self._pointer_raw()
        in_view = self._pointer_in_viewport() is not None
        if ndc is not None and self._last_mouse is not None and in_view:
            dx = ndc[0] - self._last_mouse[0]
            dy = ndc[1] - self._last_mouse[1]
            if mw.isButtonDown(MouseButton.three()):
                self.orbit.orbit(-dx * _MOUSE_ORBIT_DEG_PER_NDC, dy * _MOUSE_ORBIT_DEG_PER_NDC)
            if mw.isButtonDown(MouseButton.two()):
                self.orbit.pan(-dx * _MOUSE_PAN_PER_NDC * self._aspect, -dy * _MOUSE_PAN_PER_NDC)
        self._last_mouse = ndc if in_view else None
        step_deg = _KEY_ORBIT_DEG_PER_S * dt
        step_pan = _KEY_PAN_PER_S * dt
        if mw.isButtonDown(KeyboardButton.left()):
            self.orbit.orbit(step_deg, 0)
        if mw.isButtonDown(KeyboardButton.right()):
            self.orbit.orbit(-step_deg, 0)
        if mw.isButtonDown(KeyboardButton.up()):
            self.orbit.orbit(0, step_deg)
        if mw.isButtonDown(KeyboardButton.down()):
            self.orbit.orbit(0, -step_deg)
        if mw.isButtonDown(KeyboardButton.asciiKey("w")):
            self.orbit.pan(0, step_pan)
        if mw.isButtonDown(KeyboardButton.asciiKey("s")):
            self.orbit.pan(0, -step_pan)
        if mw.isButtonDown(KeyboardButton.asciiKey("a")):
            self.orbit.pan(-step_pan, 0)
        if mw.isButtonDown(KeyboardButton.asciiKey("d")):
            self.orbit.pan(step_pan, 0)

    def _apply_camera(self) -> None:
        pos = self.orbit.position()
        self.camera_np.setPos(LPoint3(float(pos[0]), float(pos[1]), float(pos[2])))
        tgt = self.orbit.target
        self.camera_np.lookAt(LPoint3(float(tgt[0]), float(tgt[1]), float(tgt[2])), LVector3(0, 0, 1))

    def frame_update(self, dt: float) -> None:
        """One frame of logic. Public so that tests can drive the application without a clock."""
        if self.has_window:
            self._update_camera_input(dt)
        self._apply_camera()
        self.sim.update(dt)
        poses = self._poses()

        ndc = self._pointer_raw()
        if ndc is not None and (self._pointer_in_viewport() is not None or self.editor.dragging):
            self.editor.pointer_move(self._ray(ndc), poses)
        else:
            self.editor.pointer_leave()

        self.view.sync(self.scene.snapshot(), poses, self.scene.selection,
                       self.editor.hover_id, self.editor.ghost)
        self.hud.update(self.build_hud_state())

    def _on_frame(self, task):
        self.frame_update(self._clock.getDt())
        return task.cont

    def build_hud_state(self) -> HudState:
        lang = self.language
        state = self.sim.state
        locked = self.sim.is_active
        hint = t("hint.sim", lang) if locked else t(self._hint_key, lang)
        key, args = self._message
        message = t(key, lang, **self._format_args(args, lang))
        return HudState(
            language=lang,
            tool=self.editor.tool.value,
            palette_type=self.editor.palette_type,
            sim_mode=self.sim.mode.value,
            sim_speed=self.sim.speed,
            sim_time=self.sim.time,
            fps=self._clock.getAverageFrameRate() if self.has_window else 0.0,
            message=message,
            hint=hint,
            can_undo=self.history.can_undo,
            can_redo=self.history.can_redo,
            scene_count=len(self.scene),
            scene_mass=self.scene.total_mass(),
            contacts=state.contacts if state is not None else 0,
            inspector=self.inspector_lines(lang),
        )

    def _format_args(self, args: Dict[str, object], lang: str) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for k, v in args.items():
            if k == "type" and isinstance(v, str) and v in self.registry:
                out[k] = self.registry.get(v).name(lang)
            elif k == "label" and isinstance(v, str):
                out[k] = t(f"cmd.{v}", lang)
            elif k == "error":
                out[k] = str(v)
            else:
                out[k] = v
        out.setdefault("n", 0)
        out.setdefault("name", "")
        out.setdefault("deg", 0)
        return out

    def inspector_lines(self, lang: str) -> List[str]:
        ids = self.scene.selection
        if not ids:
            return [t("inspector.none", lang)]
        if len(ids) > 1:
            mass = sum(self.registry.get(self.scene.get(i).type_key).mass_kg for i in ids)
            return [t("inspector.multi", lang, n=len(ids)), f"{t('field.mass', lang)}: {mass:.2f}"]
        cid = ids[0]
        comp = self.scene.get(cid)
        ctype = self.registry.get(comp.type_key)
        poses = self._poses()
        if poses is not None and cid in poses:
            pos, quat = poses[cid]
        else:
            pos, quat = np.array(comp.position), np.array(comp.orientation)
        w, x, y, z = (float(v) for v in quat)
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        sx, sy, sz = ctype.extents
        lines = [
            f"{t('field.id', lang)}: {cid}",
            f"{t('field.type', lang)}: {ctype.name(lang)}",
            f"{t('field.position', lang)}:",
            f"   x {pos[0]: .3f}   y {pos[1]: .3f}   z {pos[2]: .3f}",
            f"{t('field.rotation', lang)}: {yaw:.0f}",
            f"{t('field.mass', lang)}: {ctype.mass_kg:.2f}",
            f"{t('field.size', lang)}: {sx:g} × {sy:g} × {sz:g}",
        ]
        if self.sim.state is not None and cid in self.sim.state.bodies:
            body = self.sim.state.bodies[cid]
            speed = math.sqrt(sum(v * v for v in body.linear_velocity))
            lines.append(f"{t('field.speed', lang)}: {speed:.3f}")
            omega = math.sqrt(sum(v * v for v in body.angular_velocity))
            lines.append(f"{t('field.omega', lang)}: {omega:.3f}")
        return lines

    # ------------------------------------------------------------------ test/debug helpers
    def pump(self, frames: int = 1, dt: float = 1.0 / 60.0) -> None:
        """Run frame logic without the Panda3D loop (works with or without a window)."""
        for _ in range(frames):
            self.frame_update(dt)

    def run_frames(self, frames: int) -> int:
        """Run the real Panda3D frame loop for ``frames`` frames, then exit (used by smoke tests)."""
        for _ in range(frames):
            self.taskMgr.step()
        return frames


def run_gui(config: AppConfig) -> int:
    app = PhysicsSandboxApp(config)
    app.run()
    return 0

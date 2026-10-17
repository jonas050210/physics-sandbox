"""The HUD: toolbar, component palette, view/file/simulation panels, inspector and status bar.

The HUD is a *view*. The application produces a :class:`HudState` every frame and calls
:meth:`Hud.update`; button presses are forwarded through :class:`HudCallbacks`.

Coordinates: everything is parented to ``aspect2d``'s top-left anchor and positioned in
window pixels measured from the top-left corner. The HUD is rebuilt on window resize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from direct.gui.DirectGui import DirectButton, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from ..core.components import ComponentRegistry
from . import theme
from .i18n import t
from .widgets import Scale, button, destroy_all, label, rect, set_button_active

TOP_H = 56
BOTTOM_H = 30
LEFT_W = 236
RIGHT_W = 304
PAD = 14
INSPECTOR_LINES = 13
INSPECTOR_LINE_H = 19

TOOL_KEYS = ("select", "place", "move", "delete")
VIEW_KEYS = ("iso", "top", "front", "side")
FILE_KEYS = ("file.new", "file.save", "file.load")
HEADERS = ("panel.palette", "panel.view", "panel.file", "panel.sim", "panel.inspector", "panel.scene")


@dataclass
class HudCallbacks:
    on_tool: Callable[[str], None] = lambda name: None
    on_palette: Callable[[str], None] = lambda key: None
    on_undo: Callable[[], None] = lambda: None
    on_redo: Callable[[], None] = lambda: None
    on_rotate: Callable[[], None] = lambda: None
    on_view: Callable[[str], None] = lambda name: None
    on_language: Callable[[], None] = lambda: None
    on_sim_toggle: Callable[[], None] = lambda: None
    on_sim_reset: Callable[[], None] = lambda: None
    on_speed_up: Callable[[], None] = lambda: None
    on_speed_down: Callable[[], None] = lambda: None
    on_new: Callable[[], None] = lambda: None
    on_save: Callable[[], None] = lambda: None
    on_load: Callable[[], None] = lambda: None


@dataclass
class HudState:
    language: str
    tool: str
    palette_type: str
    sim_mode: str  # "edit" | "running" | "paused"
    sim_speed: float
    sim_time: float
    fps: float
    message: str
    hint: str
    can_undo: bool
    can_redo: bool
    scene_count: int
    scene_mass: float
    contacts: int
    inspector: List[str] = field(default_factory=list)


class Hud:
    def __init__(self, top_left_root, registry: ComponentRegistry, callbacks: HudCallbacks, font=None) -> None:
        self.root = top_left_root
        self.registry = registry
        self.cb = callbacks
        self.font = font
        self.width_px = 1440
        self.height_px = 900
        self._items: list = []
        self._buttons: Dict[str, DirectButton] = {}
        self._labels: Dict[str, OnscreenText] = {}
        self._inspector: List[OnscreenText] = []
        self._swatches: Dict[str, DirectFrame] = {}
        self._language: Optional[str] = None
        self._scale = Scale(900)
        self._cache: Dict[int, object] = {}

    # ------------------------------------------------------------------ caching setters
    def _set(self, node, text: str) -> None:
        """Set text only when it changed; regenerating Panda text geometry every frame is wasteful."""
        key = id(node)
        if self._cache.get(key) != text:
            self._cache[key] = text
            node.setText(text)

    def _set_button_text(self, key: str, text: str) -> None:
        cache_key = ("btn", key)
        if self._cache.get(cache_key) != text:
            self._cache[cache_key] = text
            self._buttons[key]["text"] = text

    def _set_fg(self, node, color) -> None:
        cache_key = ("fg", id(node))
        if self._cache.get(cache_key) != tuple(color):
            self._cache[cache_key] = tuple(color)
            node.setFg(color)

    # ------------------------------------------------------------------ geometry
    def viewport_pixels(self) -> Tuple[float, float, float, float]:
        """(left, right, bottom, top) of the 3D viewport in window pixels, from the top-left corner."""
        return float(LEFT_W), float(self.width_px - RIGHT_W), float(self.height_px - BOTTOM_H), float(TOP_H)

    def viewport_fractions(self) -> Tuple[float, float, float, float]:
        """(left, right, bottom, top) as display-region fractions in [0, 1], origin bottom-left."""
        left, right, bottom_px, top_px = self.viewport_pixels()
        w, h = float(self.width_px), float(self.height_px)
        return left / w, right / w, (h - bottom_px) / h, (h - top_px) / h

    # ------------------------------------------------------------------ build
    def rebuild(self, width_px: int, height_px: int) -> None:
        self.width_px, self.height_px = int(width_px), int(height_px)
        destroy_all(self._items)
        self._items = []
        self._buttons.clear()
        self._labels.clear()
        self._inspector = []
        self._swatches.clear()
        self._cache.clear()
        self._scale = Scale(self.height_px)
        self._build()
        self._language = None  # force a static text refresh on the next update

    def _keep(self, item):
        self._items.append(item)
        return item

    def _text(self, x, y, px, color=theme.TEXT, align=TextNode.ALeft, name=None) -> OnscreenText:
        """Text node; ``name`` is accepted for readability at call sites and otherwise ignored."""
        return self._keep(label(self.root, self._scale, "", x, y, px, color, self.font, align=align))

    def _btn(self, key: str, x, y, w, h, command, text: str = "", text_px: float = 13.0, align="center",
             pad_px: float = 12.0) -> DirectButton:
        b = button(self.root, self._scale, text, command, x, y, w, h, self.font, text_px=text_px,
                   name=key, align=align, pad_px=pad_px)
        self._buttons[key] = b
        return self._keep(b)

    def _rect(self, x, y, w, h, color) -> DirectFrame:
        return self._keep(rect(self.root, self._scale, x, y, w, h, color))

    def _header(self, x, y, key: str) -> None:
        self._labels[key] = self._text(x, y, 11, theme.TEXT_FAINT)

    def _build(self) -> None:
        W, H = self.width_px, self.height_px

        # toolbar (full width)
        self._rect(0, 0, W, TOP_H, theme.PANEL_HEADER)
        self._rect(0, TOP_H - 1, W, 1, theme.PANEL_BORDER)
        self._labels["title"] = self._text(PAD, 25, 16, theme.ACCENT)  # y = text baseline
        self._labels["subtitle"] = self._text(PAD, 44, 10.5, theme.TEXT_DIM)
        x = LEFT_W + 6
        for key in TOOL_KEYS:
            self._btn(f"tool.{key}", x, 11, 116, 34, lambda k=key: self.cb.on_tool(k))
            x += 122
        self._rect(x + 2, 14, 1, 28, theme.PANEL_BORDER)
        x += 14
        self._btn("action.undo", x, 11, 118, 34, self.cb.on_undo)
        self._btn("action.redo", x + 124, 11, 118, 34, self.cb.on_redo)
        self._btn("action.rotate", x + 248, 11, 118, 34, self.cb.on_rotate)
        self._btn("lang", W - 128, 11, 112, 34, self.cb.on_language)

        # left panel
        body_h = H - TOP_H - BOTTOM_H
        self._rect(0, TOP_H, LEFT_W, body_h, theme.PANEL)
        self._rect(LEFT_W - 1, TOP_H, 1, body_h, theme.PANEL_BORDER)
        inner = LEFT_W - 2 * PAD
        y = TOP_H + PAD
        self._header(PAD, y, "panel.palette")
        y += 26
        for key in self.registry.keys():
            self._palette_button(key, PAD, y, inner, 40)
            y += 46
        self._labels["palette.hint"] = self._text(PAD, y + 12, 10.5, theme.TEXT_DIM)
        y += 34
        self._header(PAD, y, "panel.view")
        y += 26
        half = (inner - 8) / 2
        for i, key in enumerate(VIEW_KEYS):
            self._btn(f"view.{key}", PAD + (i % 2) * (half + 8), y + (i // 2) * 40, half, 34,
                      lambda k=key: self.cb.on_view(k))
        y += 88
        self._header(PAD, y, "panel.file")
        y += 26
        for key, cb in (("file.new", self.cb.on_new), ("file.save", self.cb.on_save), ("file.load", self.cb.on_load)):
            self._btn(key, PAD, y, inner, 34, cb)
            y += 40

        # right panel
        rx = W - RIGHT_W
        self._rect(rx, TOP_H, RIGHT_W, body_h, theme.PANEL)
        self._rect(rx, TOP_H, 1, body_h, theme.PANEL_BORDER)
        inner_r = RIGHT_W - 2 * PAD
        x0 = rx + PAD
        y = TOP_H + PAD
        self._header(x0, y, "panel.sim")
        y += 26
        half = (inner_r - 8) / 2
        self._btn("sim.toggle", x0, y, half, 36, self.cb.on_sim_toggle)
        self._btn("sim.reset", x0 + half + 8, y, half, 36, self.cb.on_sim_reset)
        y += 46
        self._labels["sim.speed"] = self._text(x0, y + 20, 12, theme.TEXT_DIM)
        self._btn("speed.down", x0 + inner_r - 128, y, 36, 30, self.cb.on_speed_down, text="−")
        self._labels["speed.value"] = self._text(x0 + inner_r - 84, y + 20, 13, theme.TEXT, align=TextNode.ACenter)
        self._btn("speed.up", x0 + inner_r - 36, y, 36, 30, self.cb.on_speed_up, text="+")
        y += 40
        self._labels["sim.status"] = self._text(x0, y, 12, theme.TEXT_DIM)
        y += 24
        self._rect(x0, y, inner_r, 1, theme.PANEL_BORDER)
        y += 10
        self._header(x0, y, "panel.inspector")
        y += 26
        for i in range(INSPECTOR_LINES):
            node = self._text(x0, y + i * INSPECTOR_LINE_H, 12.5, theme.TEXT)
            self._inspector.append(node)
        y += INSPECTOR_LINES * INSPECTOR_LINE_H + 12
        self._rect(x0, y, inner_r, 1, theme.PANEL_BORDER)
        y += 10
        self._header(x0, y, "panel.scene")
        y += 26
        for key in ("scene.count", "scene.mass", "scene.contacts"):
            self._labels[key] = self._text(x0, y, 12.5, theme.TEXT)
            y += 20

        # status bar (bottom)
        by = H - BOTTOM_H
        self._rect(0, by, W, BOTTOM_H, theme.PANEL_HEADER)
        self._rect(0, by, W, 1, theme.PANEL_BORDER)
        self._labels["status.mode"] = self._text(PAD, by + 9, 12, theme.TEXT)
        self._labels["status.tool"] = self._text(230, by + 9, 12, theme.TEXT_DIM)
        self._labels["status.message"] = self._text(420, by + 9, 12.5, theme.TEXT)
        self._labels["status.time"] = self._text(W - RIGHT_W + PAD, by + 9, 12, theme.TEXT_DIM)
        self._labels["status.fps"] = self._text(W - PAD, by + 9, 12, theme.TEXT_DIM, align=TextNode.ARight)

        # viewport overlays: plain text only, so clicks fall through to the 3D view
        left = LEFT_W + PAD
        self._labels["vp.hint"] = self._text(left, TOP_H + PAD, 11.5, theme.TEXT_DIM)
        self._labels["vp.message"] = self._text(left, H - BOTTOM_H - 30, 13.5, theme.ACCENT)


    def _palette_button(self, key: str, x: float, y: float, w: float, h: float) -> None:
        self._btn(f"palette.{key}", x, y, w, h, lambda k=key: self.cb.on_palette(k), align="left", pad_px=40.0)
        self._swatches[key] = self._rect(x + 12, y + (h - 16) / 2, 16, 16, (0.5, 0.5, 0.5, 1.0))

    # ------------------------------------------------------------------ update
    def _apply_static_texts(self, lang: str) -> None:
        self._labels["title"].setText(t("app.title", lang))
        self._labels["subtitle"].setText(t("app.subtitle", lang))
        self._labels["palette.hint"].setText(t("palette.hint", lang))
        for key in HEADERS:
            if key in self._labels:
                self._labels[key].setText(t(key, lang))
        for key in TOOL_KEYS:
            self._buttons[f"tool.{key}"]["text"] = t(f"tool.{key}", lang)
        for key in ("undo", "redo", "rotate"):
            self._buttons[f"action.{key}"]["text"] = t(f"action.{key}", lang)
        self._buttons["lang"]["text"] = t("lang.switch", lang)
        for key in VIEW_KEYS:
            self._buttons[f"view.{key}"]["text"] = t(f"view.{key}", lang)
        for key in FILE_KEYS:
            self._buttons[key]["text"] = t(key, lang)
        for key in self.registry.keys():
            ctype = self.registry.get(key)
            self._buttons[f"palette.{key}"]["text"] = ctype.name(lang)
            self._swatches[key]["frameColor"] = tuple(ctype.color) + (1.0,)
        self._buttons["speed.down"]["text"] = "−"
        self._buttons["speed.up"]["text"] = "+"
        self._language = lang

    def update(self, state: HudState) -> None:
        lang = state.language
        if self._language != lang:
            self._apply_static_texts(lang)

        for tool in TOOL_KEYS:
            set_button_active(self._buttons[f"tool.{tool}"], state.tool == tool)
        for key in self.registry.keys():
            set_button_active(self._buttons[f"palette.{key}"], state.tool == "place" and state.palette_type == key)
        set_button_active(self._buttons["action.undo"], False, enabled=state.can_undo)
        set_button_active(self._buttons["action.redo"], False, enabled=state.can_redo)
        set_button_active(self._buttons["action.rotate"], False)

        running = state.sim_mode == "running"
        paused = state.sim_mode == "paused"
        toggle_key = "sim.resume" if paused else ("sim.pause" if running else "sim.start")
        self._set_button_text("sim.toggle", t(toggle_key, lang))
        set_button_active(self._buttons["sim.toggle"], running)
        set_button_active(self._buttons["sim.reset"], False, enabled=state.sim_mode != "edit")

        self._set(self._labels["speed.value"], t("status.speed", lang, x=state.sim_speed))
        self._set(self._labels["sim.speed"], t("sim.speed", lang))
        status_key = {"edit": "sim.status.edit", "running": "sim.status.running",
                      "paused": "sim.status.paused"}[state.sim_mode]
        self._set(self._labels["sim.status"], t(status_key, lang))
        status_color = theme.SUCCESS if running else (theme.WARNING if paused else theme.TEXT_DIM)
        self._set_fg(self._labels["sim.status"], status_color)

        for i, node in enumerate(self._inspector):
            self._set(node, state.inspector[i] if i < len(state.inspector) else "")
        self._set(self._labels["scene.count"], t("scene.count", lang, n=state.scene_count))
        self._set(self._labels["scene.mass"], t("scene.mass", lang, m=state.scene_mass))
        self._set(self._labels["scene.contacts"], t("scene.contacts", lang, n=state.contacts))

        self._set(self._labels["status.mode"], f"{t('status.mode', lang)}: {t(status_key, lang)}")
        self._set_fg(self._labels["status.mode"], status_color)
        self._set(self._labels["status.tool"], f"{t('status.tool', lang)}: {t(f'tool.{state.tool}', lang)}")
        self._set(self._labels["status.time"], t("status.time", lang, t=state.sim_time))
        self._set(self._labels["status.fps"], t("status.fps", lang, fps=state.fps))
        self._set(self._labels["status.message"], state.message)
        self._set(self._labels["vp.hint"], state.hint)
        self._set(self._labels["vp.message"], state.message)

"""Small helpers that turn pixel-based layout into Panda3D GUI objects.

All layout is specified in window pixels (origin at the top-left). ``Scale`` converts
pixels to aspect2d units for the current window height, so the GUI stays crisp and
correctly proportioned when the window is resized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from direct.gui.DirectGui import DGG, DirectButton, DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

from . import theme


@dataclass(frozen=True)
class Scale:
    """Pixel -> aspect2d unit conversion for a window of height ``window_h`` pixels."""

    window_h: float

    @property
    def u(self) -> float:
        return 2.0 / max(self.window_h, 1.0)

    def __call__(self, px: float) -> float:
        return px * self.u


def panel(parent, scale: Scale, x: float, y: float, w: float, h: float, color=theme.PANEL) -> DirectFrame:
    """A rectangular panel whose top-left corner is at (x, y) pixels relative to ``parent``."""
    u = scale.u
    return DirectFrame(
        parent=parent,
        pos=(x * u, 0, -y * u),
        frameColor=color,
        frameSize=(0, w * u, -h * u, 0),
        relief=DGG.FLAT,
    )


def rect(parent, scale: Scale, x: float, y: float, w: float, h: float, color) -> DirectFrame:
    return panel(parent, scale, x, y, w, h, color=color)


def button(parent, scale: Scale, text: str, command: Callable[[], None], x: float, y: float, w: float,
           h: float, font=None, text_px: float = 13.0, active: bool = False, enabled: bool = True,
           name: Optional[str] = None, align: str = "center", pad_px: float = 12.0) -> DirectButton:
    u = scale.u
    if not enabled:
        colors = (theme.BUTTON_DISABLED,) * 4
    elif active:
        colors = (theme.ACCENT_DIM, theme.ACCENT, theme.ACCENT, theme.ACCENT_DIM)
    else:
        colors = (theme.BUTTON, theme.BUTTON_HOVER, theme.BUTTON_PRESSED, theme.BUTTON_DISABLED)
    ts = scale(text_px)
    fg = theme.TEXT if enabled else theme.TEXT_FAINT
    kwargs = dict(
        parent=parent,
        text=text,
        text_scale=ts,
        text_fg=fg,
        text_align=TextNode.ACenter if align == "center" else TextNode.ALeft,
        text_pos=((w * u / 2.0, -(h * u / 2.0) - ts * 0.36) if align == "center"
                  else (pad_px * u, -(h * u / 2.0) - ts * 0.36)),
        pos=(x * u, 0, -y * u),
        frameColor=colors,
        frameSize=(0, w * u, -h * u, 0),
        relief=DGG.FLAT,
        borderWidth=(0, 0),
        state=DGG.NORMAL if enabled else DGG.DISABLED,
        command=command if enabled else None,
        pressEffect=False,
    )
    if font is not None:
        kwargs["text_font"] = font
    btn = DirectButton(**kwargs)
    if name:
        btn.setName(name)
    return btn


def label(parent, scale: Scale, text: str, x: float, y: float, px: float = 13.0,
          color=theme.TEXT, font=None, align=TextNode.ALeft) -> OnscreenText:
    kwargs = dict(
        text=text,
        pos=(x * scale.u, -y * scale.u),
        scale=scale(px),
        fg=color,
        align=align,
        parent=parent,
        mayChange=True,
    )
    if font is not None:
        kwargs["font"] = font
    return OnscreenText(**kwargs)


def set_button_active(btn: DirectButton, active: bool, enabled: bool = True) -> None:
    if not enabled:
        btn["frameColor"] = (theme.BUTTON_DISABLED,) * 4
    elif active:
        btn["frameColor"] = (theme.ACCENT_DIM, theme.ACCENT, theme.ACCENT, theme.ACCENT_DIM)
    else:
        btn["frameColor"] = (theme.BUTTON, theme.BUTTON_HOVER, theme.BUTTON_PRESSED, theme.BUTTON_DISABLED)


def destroy_all(items: Sequence) -> None:
    for item in items:
        try:
            item.destroy()
        except Exception:  # noqa: BLE001 - widget already gone; nothing to do
            pass

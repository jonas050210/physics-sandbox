"""Font selection: a system UI font when one is installed, Panda3D's default otherwise."""

from __future__ import annotations

import os
from typing import Optional

CANDIDATE_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)


def find_ui_font_path(candidates=CANDIDATE_FONTS) -> Optional[str]:
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_ui_font(loader):
    """Return a Panda3D font object (or None to use the engine default)."""
    path = find_ui_font_path()
    if path is None:
        return None
    font = loader.loadFont(path)
    font.setPixelsPerUnit(60)
    font.setPageSize(512, 512)
    return font

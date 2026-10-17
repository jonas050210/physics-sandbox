"""Render the application to a PNG without a display.

Uses Panda3D's built-in software renderer (tinydisplay) in an offscreen buffer, so it works on
machines without OpenGL. The output is a faithful picture of the UI layout and 3D geometry, but
software rendering does not produce the GPU shadow maps and anti-aliasing of the real window.

    .venv/bin/python tools/screenshot.py --out docs/screenshots/overview-de.png --lang de
    .venv/bin/python tools/screenshot.py --out /tmp/x.png --lang en --select 2 --sim 1.5

Run from the repository root, after `python3 main.py --setup-only`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .png path")
    p.add_argument("--lang", choices=["de", "en"], default="de")
    p.add_argument("--size", default="1440x900", help="WIDTHxHEIGHT")
    p.add_argument("--scene", help="scene JSON (default: built-in tower demo)")
    p.add_argument("--select", type=int, action="append", default=[], help="component id to select (repeatable)")
    p.add_argument("--sim", type=float, default=0.0, help="simulate this many seconds before the capture")
    p.add_argument("--view", default="iso", help="camera preset: iso, front, top, side")
    args = p.parse_args(argv)

    from physics_sandbox.config import AppConfig
    from physics_sandbox.app import PhysicsSandboxApp
    from panda3d.core import Filename

    width, height = (int(v) for v in args.size.lower().split("x"))
    config = AppConfig(language=args.lang, width=width, height=height, scene_path=args.scene,
                       saves_dir=ROOT / "saves")
    app = PhysicsSandboxApp(config, window_type="offscreen")
    app.act_view(args.view)
    if args.select:
        app.scene.select(args.select)
    if args.sim > 0:
        app.act_sim_toggle()
        app.pump(int(args.sim * 60))
    app.pump(2)
    for _ in range(3):  # let Panda3D render a few frames
        app.taskMgr.step()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = app.win.saveScreenshot(Filename.fromOsSpecific(str(out.resolve())))
    app.destroy()
    if not ok:
        print("screenshot failed", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point: GUI by default, ``--headless`` for window-less simulation."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import AppConfig
from .core.serialization import SceneFormatError
from .headless import run_headless, scene_for_headless, write_report
from .ui.i18n import SUPPORTED_LANGUAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="physics-sandbox", description="Physics Sandbox - 3D engineering sandbox")
    p.add_argument("--lang", choices=SUPPORTED_LANGUAGES, default="de", help="interface language (default: de)")
    p.add_argument("--scene", metavar="PATH", help="scene file (JSON) to load on start")
    p.add_argument("--empty", action="store_true", help="start with an empty baseplate instead of the demo")
    p.add_argument("--width", type=int, default=1440, help="window width in pixels")
    p.add_argument("--height", type=int, default=900, help="window height in pixels")
    p.add_argument("--no-vsync", action="store_true", help="disable vertical sync")
    p.add_argument("--headless", action="store_true", help="run the simulation without a window")
    p.add_argument("--duration", type=float, default=3.0, help="headless: simulated seconds (default 3)")
    p.add_argument("--timestep", type=float, default=0.001, help="physics timestep in seconds (default 0.001)")
    p.add_argument("--integrator", choices=["implicitfast", "RK4"], default="implicitfast")
    p.add_argument("--report", metavar="PATH", help="headless: write the JSON report to PATH")
    p.add_argument("--exit-after-frames", type=int, metavar="N", help=argparse.SUPPRESS)  # smoke tests
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.headless:
        try:
            scene = scene_for_headless(args.scene, args.lang)
        except (SceneFormatError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        report = run_headless(scene, seconds=args.duration, timestep=args.timestep, integrator=args.integrator)
        text = write_report(report, args.report)
        if args.report:
            print(f"headless run ok={report['ok']} steps={report['steps']} "
                  f"t={report['simulated_seconds']:.3f}s contacts={report['contacts']} "
                  f"max_penetration={report['max_penetration_m'] * 1000:.3f}mm "
                  f"max_speed={report['max_speed_mps']:.4f}m/s -> {args.report}")
        else:
            print(text)
        return 0 if report["ok"] else 1

    config = AppConfig(
        language=args.lang,
        width=args.width,
        height=args.height,
        vsync=not args.no_vsync,
        scene_path=args.scene,
        start_empty=args.empty,
        integrator=args.integrator,
        physics_timestep=args.timestep,
    )
    try:
        from .app import PhysicsSandboxApp
        app = PhysicsSandboxApp(config)
    except (SceneFormatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.exit_after_frames is not None:
        frames = app.run_frames(args.exit_after_frames)
        print(f"gui smoke ok: {frames} frames, {len(app.scene)} components, "
              f"tool={app.editor.tool.value}, hud_items={len(app.hud._items)}")
        app.destroy()
        return 0
    app.run()
    return 0

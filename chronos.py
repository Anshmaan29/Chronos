#!/usr/bin/env python3
"""CHRONOS console.

    python chronos.py --video data/real/clip.mp4
    python chronos.py                       # synthetic corner, car runs wide
    python chronos.py --path one_wheel_on_line
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser(description="CHRONOS -- reference integrity console")
    p.add_argument("--video", default=None, help="path to a video file")
    p.add_argument("--path", default="violation",
                   choices=["clean", "brush", "one_wheel_on_line", "violation",
                            "straight_wide"],
                   help="synthetic scenario, when no --video is given")
    p.add_argument("--detector", default="auto",
                   choices=["auto", "real", "synthetic"],
                   help="boundary detector for real footage")
    p.add_argument("--theme", default="f1", choices=["f1", "instrument", "brutal"],
                   help="console look; f1 is the broadcast livery, "
                        "instrument the original")
    p.add_argument("--kind", default="rubber", help="degradation the slider applies")
    p.add_argument("--frames", type=int, default=70, help="synthetic clip length")
    args = p.parse_args()

    try:
        from chronos.ui.app import run
    except ImportError as exc:
        print(f"the console needs PyQt6: {exc}\n"
              f"  uv pip install --python .venv/bin/python PyQt6", file=sys.stderr)
        return 2
    return run(args.video, args.path, args.kind, args.frames, args.detector,
               args.theme)


if __name__ == "__main__":
    raise SystemExit(main())

"""Regression checks for frame analysis on the real test footage.

Expected outcomes are human judgements made by looking at each image, and
they are deliberately loose where the call is genuinely marginal (a tyre on
the line can fairly read BORDERLINE or VIOLATION).  What they pin down is the
thing that must not silently regress: the right car found, and no confident
verdict on the wrong object.

Run:  python tests/real_regression.py
"""

from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.analyze import AnalyzeConfig, analyze_frame, iter_frames  # noqa: E402

D = "data/real"
IMG = "WhatsApp_Image_2026-09-12_at_22.56.47_{}.jpeg"
GIF = os.path.join(D, "WhatsApp_GIF_2026-09-12_at_23.06.02.gif")

# name -> (source, frame, expected verdicts for the main car, (xmin,xmax) of
# the main car's centre as a fraction of width, minimum width fraction)
CASES = {
    "haas":      (IMG.format("1_"), 0, {"VIOLATION", "BORDERLINE"}, (0.10, 0.55), 0.18),
    "redbull":   (IMG.format("2_"), 0, {"BORDERLINE", "VIOLATION"}, (0.10, 0.40), 0.12),
    "alpine":    (IMG.format("3_"), 0, {"BORDERLINE", "VIOLATION"}, (0.20, 0.60), 0.20),
    "ferrari":   (IMG.format("4_"), 0, {"BORDERLINE", "VIOLATION"}, (0.30, 0.70), 0.28),
    "gif_f1":    (GIF, 1, {"REVIEW REQUIRED", "CLEAR", "BORDERLINE"}, (0.35, 0.60), 0.03),
    "gif_f36":   (GIF, 36, {"CLEAR", "BORDERLINE"}, (0.35, 0.65), 0.03),
    "gif_f44":   (GIF, 44, {"CLEAR", "BORDERLINE"}, (0.35, 0.65), 0.03),
}


def load(src: str, frame: int):
    path = src if os.path.isabs(src) or src.startswith(D) else os.path.join(D, src)
    for idx, _, im, _ in iter_frames(path):
        if idx == frame:
            return im
    raise FileNotFoundError(f"{path} frame {frame}")


def run(cfg: AnalyzeConfig | None = None, verbose: bool = True) -> int:
    cfg = cfg or AnalyzeConfig()
    passed = 0
    for name, (src, frame, ok_verdicts, (cx0, cx1), min_w) in CASES.items():
        im = load(src, frame)
        r = analyze_frame(im, frame, 0.0, cfg, annotate=False)
        w = r.width
        problems = []
        if not r.cars:
            problems.append("no car found")
        else:
            # the main car is the one whose centre is where the car really is
            def score(c):
                cx = (c.box[0] + c.box[2]) / 2 / w
                return -abs(cx - (cx0 + cx1) / 2)
            main = max(r.cars, key=score)
            cx = (main.box[0] + main.box[2]) / 2 / w
            bw = (main.box[2] - main.box[0]) / w
            if not cx0 <= cx <= cx1:
                problems.append(f"main car centre at {cx:.2f}, expected {cx0}-{cx1}")
            if bw < min_w:
                problems.append(f"main car box {bw:.2f} wide, expected >= {min_w}")
            if main.verdict not in ok_verdicts:
                problems.append(f"verdict {main.verdict}, expected {sorted(ok_verdicts)}")
            others = [c for c in r.cars if c is not main and c.verdict == "VIOLATION"]
            if others:
                problems.append(f"{len(others)} extra VIOLATION on another object")
        good = not problems
        passed += good
        if verbose:
            print(f"{'PASS' if good else 'FAIL'}  {name:9s} frame={r.verdict:16s} "
                  f"cars={len(r.cars)}  {'; '.join(problems)}")
    if verbose:
        print(f"\n{passed}/{len(CASES)} real-footage checks passed")
    return passed


if __name__ == "__main__":
    n = run()
    sys.exit(0 if n == len(CASES) else 1)

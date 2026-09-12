"""Run CHRONOS Module 1 and the YOLO detector on REAL footage, and report.

Everything in this repo so far is measured against synthetic ground truth.
Real frames have no ground truth, so this tool cannot score accuracy -- and it
does not pretend to.  It answers two narrower questions honestly:

  1. Does ``boundary.py`` still RETURN a boundary, and what does the overlay
     look like?  "ok" here means the pipeline produced a polyline, nothing
     more.  Whether that polyline is on the actual white line is decided by
     eye, from the debug panels this writes.  The per-stage diagnostics
     (drivable-mask area, kerb-mask area, channel agreement) are printed so a
     failure points at a stage rather than at "it didn't work".

  2. Does YOLO fire on real cars?  The README's honest limit says YOLO reports
     "tennis racket" on the synthetic renders.  So this reports the RAW
     detections -- every class, with its confidence, at a low floor -- and then
     separately how many survive ``CarConfig``'s vehicle-class filter and
     confidence threshold.  A class histogram is the evidence; "found cars" is
     the conclusion.

Both are reported independently.  Boundary working while YOLO does not is a
real and useful outcome, not a half-failure: Modules 1-3 are the wow moment
and have no ML dependency.

Run::

    python tools/real_test.py                          # whatever is in data/real
    python tools/real_test.py --path data/real/clip.mp4 --frames 12
    python tools/real_test.py --width 1280             # rescale to the scale
                                                       # boundary.py is tuned at
    python tools/real_test.py --outer-side left --no-yolo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.boundary import (BoundaryConfig, detect_boundary,  # noqa: E402
                              save_debug_image)
from chronos.car import CarConfig  # noqa: E402

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg")

# The scale every pixel tunable in BoundaryConfig was calibrated at.  Reported
# so a native-resolution failure can be told apart from a scale mismatch.
CALIBRATION_WIDTH = 1280


@dataclass
class FrameSource:
    """One sampled frame and where it came from."""

    name: str
    frame: np.ndarray
    origin: str
    frame_index: Optional[int] = None


# --------------------------------------------------------------------------
# input discovery
# --------------------------------------------------------------------------


def sample_video(path: str, n: int) -> list[FrameSource]:
    """Take ``n`` frames spread evenly across a clip."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    stem = os.path.splitext(os.path.basename(path))[0]
    out: list[FrameSource] = []
    if total > 0:
        idx = np.unique(np.linspace(0, total - 1, n).round().astype(int))
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                out.append(FrameSource(f"{stem}_f{int(i):05d}", frame,
                                       f"{os.path.basename(path)} "
                                       f"({total} frames, {fps:.1f} fps)", int(i)))
    else:
        # some containers report no frame count; walk it instead
        i = 0
        while len(out) < n:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 10 == 0:
                out.append(FrameSource(f"{stem}_f{i:05d}", frame,
                                       f"{os.path.basename(path)} (streamed)", i))
            i += 1
    cap.release()
    if not out:
        raise RuntimeError(f"read no frames from {path}")
    return out


def collect(path: str, n_frames: int) -> list[FrameSource]:
    """Gather frames from a file or a directory of images and/or clips."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Put a clip or some frames in data/real/ "
            f"-- any of {', '.join(IMAGE_EXT + VIDEO_EXT)}.")

    if os.path.isfile(path):
        files = [path]
    else:
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if not f.startswith("."))

    images = [f for f in files if f.lower().endswith(IMAGE_EXT)]
    videos = [f for f in files if f.lower().endswith(VIDEO_EXT)]
    if not images and not videos:
        raise FileNotFoundError(
            f"no images or videos in {path}. Found: "
            f"{[os.path.basename(f) for f in files] or 'nothing'}")

    out: list[FrameSource] = []
    for f in images:
        frame = cv2.imread(f)
        if frame is None:
            print(f"  ! cannot decode {f}, skipped")
            continue
        out.append(FrameSource(os.path.splitext(os.path.basename(f))[0],
                               frame, os.path.basename(f)))
    per_video = max(1, n_frames // max(len(videos), 1)) if videos else 0
    for f in videos:
        out.extend(sample_video(f, per_video))
    return out


def rescale(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    h = int(round(frame.shape[0] * width / frame.shape[1]))
    interp = cv2.INTER_AREA if width < frame.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, h), interpolation=interp)


# --------------------------------------------------------------------------
# stage 1 -- boundary
# --------------------------------------------------------------------------


def run_boundary(src: FrameSource, cfg: BoundaryConfig, out_dir: str) -> dict:
    """Detect the boundary, save the 2x2 debug panel, return diagnostics."""
    cfg = BoundaryConfig(**{**asdict(cfg),
                            "debug_dir": out_dir,
                            "debug_name": f"{src.name}_boundary.jpg"})
    h, w = src.frame.shape[:2]
    t0 = time.perf_counter()
    res = detect_boundary(src.frame, cfg, save_debug=True)
    ms = (time.perf_counter() - t0) * 1e3

    rec: dict = {
        "frame": src.name, "origin": src.origin, "size": [w, h],
        "ok": bool(res.ok), "reason": res.reason, "ms": round(ms, 1),
        "debug": res.debug_path,
    }
    for key, mask in (("drivable_frac", res.drivable_mask),
                      ("kerb_frac", res.kerb_mask)):
        rec[key] = (round(float((mask > 0).mean()), 4)
                    if mask is not None else None)
    if res.ok:
        poly = res.polyline
        rec.update(
            points=int(len(poly)),
            # how much of the frame the boundary actually spans -- a 40-pixel
            # stub and a full-width boundary are both "ok" without this
            span_x_frac=round(float(np.ptp(poly[:, 0]) / w), 3),
            span_y_frac=round(float(np.ptp(poly[:, 1]) / h), 3),
            length_px=round(float(np.hypot(*np.diff(poly, axis=0).T).sum()), 1),
            agreement_px=(None if res.agreement_px is None
                          else round(float(res.agreement_px), 2)),
            agreement_span=(None if res.agreement_span is None
                            else round(float(res.agreement_span), 3)),
        )
    return rec


# --------------------------------------------------------------------------
# stage 2 -- YOLO, reported raw
# --------------------------------------------------------------------------


class RawYolo:
    """Thin wrapper that keeps every class, not just vehicles.

    ``chronos.car.YoloDetector`` filters to COCO vehicle classes, which is
    correct for the pipeline and useless for a diagnosis: if YOLO is calling a
    Formula 1 car a "tennis racket", a vehicle-filtered detector reports an
    empty frame and the reason is invisible.  So detection happens once at a
    low floor, and the filter is applied afterwards, in the report.
    """

    def __init__(self, cfg: CarConfig, floor: float):
        self.cfg = cfg
        self.floor = floor
        from ultralytics import YOLO
        self.model = YOLO(cfg.model)
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> list[dict]:
        res = self.model.predict(frame, conf=self.floor, imgsz=self.cfg.imgsz,
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        order = np.argsort(-conf)
        return [{"cls": int(cls[i]), "name": str(self.names[int(cls[i])]),
                 "conf": round(float(conf[i]), 3),
                 "box": [round(float(v), 1) for v in xyxy[i]],
                 "vehicle": int(cls[i]) in self.cfg.coco_vehicle_classes,
                 "passes": (int(cls[i]) in self.cfg.coco_vehicle_classes
                            and float(conf[i]) >= self.cfg.conf_threshold)}
                for i in order]


def draw_yolo(frame: np.ndarray, dets: list[dict], cfg: CarConfig) -> np.ndarray:
    """Green = counted as a car.  Amber = vehicle class, below threshold.
    Grey = some other class entirely."""
    vis = frame.copy()
    for d in dets:
        if d["passes"]:
            color, tag = (80, 230, 80), "CAR"
        elif d["vehicle"]:
            color, tag = (60, 190, 250), "low"
        else:
            color, tag = (150, 150, 150), "other"
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(vis, f"{d['name']} {d['conf']:.2f} [{tag}]",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    banner = (f"conf floor {min([d['conf'] for d in dets], default=0):.2f}  "
              f"raw {len(dets)}  vehicles>={cfg.conf_threshold:.2f} "
              f"{sum(d['passes'] for d in dets)}")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _fmt(v, spec="", dash="--"):
    return dash if v is None else format(v, spec)


def report(rows: list[dict], yolo_rows: Optional[list[dict]],
           cfg: CarConfig, out_dir: str, width: int) -> None:
    n = len(rows)
    print()
    print("=" * 78)
    print("BOUNDARY (Module 1) on real frames")
    print("=" * 78)
    print(f"{'frame':<26}{'ok':<4}{'span':>7}{'agree':>8}{'span%':>7}"
          f"{'drive':>7}{'kerb':>7}{'ms':>7}")
    for r in rows:
        print(f"{r['frame'][:25]:<26}"
              f"{'yes' if r['ok'] else 'NO':<4}"
              f"{_fmt(r.get('span_x_frac'), '>7.2f')}"
              f"{_fmt(r.get('agreement_px'), '>8.1f')}"
              f"{_fmt(r.get('agreement_span'), '>7.0%')}"
              f"{_fmt(r.get('drivable_frac'), '>7.2f')}"
              f"{_fmt(r.get('kerb_frac'), '>7.3f')}"
              f"{r['ms']:>7.0f}")
    ok = [r for r in rows if r["ok"]]
    print(f"\n  returned a boundary : {len(ok)}/{n}")
    if ok:
        spans = [r["span_x_frac"] for r in ok]
        print(f"  horizontal span     : {min(spans):.2f} - {max(spans):.2f} "
              f"of frame width")
        ag = [r["agreement_px"] for r in ok if r["agreement_px"] is not None]
        if ag:
            print(f"  channel agreement   : {min(ag):.1f} - {max(ag):.1f} px "
                  f"({len(ag)}/{len(ok)} frames; fallback silent in "
                  f"{len(ok) - len(ag)})")
        else:
            print("  channel agreement   : fallback channel silent on every frame")
    for r in rows:
        if not r["ok"]:
            print(f"  ! {r['frame']}: {r['reason']}")
    print(f"\n  NOTE: 'ok' means a polyline came back, NOT that it is on the "
          f"white line.\n        There is no ground truth on real footage. "
          f"Check the overlays in\n        {out_dir}/ by eye -- green is the "
          f"detection, magenta the classical\n        fallback.")
    if width > 0 and width != CALIBRATION_WIDTH:
        print(f"  NOTE: frames rescaled to {width} px wide; BoundaryConfig's "
              f"pixel tunables\n        are calibrated at {CALIBRATION_WIDTH} px.")

    if yolo_rows is None:
        print("\nYOLO: skipped (--no-yolo)")
        return

    print()
    print("=" * 78)
    print(f"YOLO ({cfg.model}) on real frames")
    print("=" * 78)
    print(f"{'frame':<26}{'raw':>5}{'cars':>6}{'best car':>10}  top raw classes")
    hist: dict[str, int] = {}
    best_car = []
    for r in yolo_rows:
        dets = r["detections"]
        cars = [d for d in dets if d["passes"]]
        if cars:
            best_car.append(max(d["conf"] for d in cars))
        for d in dets:
            hist[d["name"]] = hist.get(d["name"], 0) + 1
        top = ", ".join(f"{d['name']} {d['conf']:.2f}" for d in dets[:3]) or "nothing"
        best = f"{max(d['conf'] for d in cars):.2f}" if cars else "--"
        print(f"{r['frame'][:25]:<26}{len(dets):>5}{len(cars):>6}{best:>10}"
              f"  {top}")
    n_with = sum(1 for r in yolo_rows if any(d["passes"] for d in r["detections"]))
    print(f"\n  frames with >=1 vehicle at conf>={cfg.conf_threshold:.2f}"
          f" : {n_with}/{len(yolo_rows)}")
    if best_car:
        print(f"  best vehicle confidence per such frame : "
              f"{min(best_car):.2f} - {max(best_car):.2f}")
    print(f"  vehicle classes filtered for : {list(cfg.coco_vehicle_classes)}"
          f" = car, motorcycle, bus, truck")
    if hist:
        print("  every class YOLO returned (count over all frames):")
        for name, c in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"      {name:<20} {c}")
    else:
        print("  YOLO returned NOTHING at all, at any class, at the conf floor.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="data/real",
                    help="clip, frame, or directory of them (default data/real)")
    ap.add_argument("--frames", type=int, default=8,
                    help="frames to sample per video (default 8)")
    ap.add_argument("--width", type=int, default=0,
                    help=f"rescale frames to this width; 0 = native. "
                         f"boundary.py is tuned at {CALIBRATION_WIDTH}")
    ap.add_argument("--out", default="debug/real", help="where overlays go")
    ap.add_argument("--outer-side", default="auto",
                    choices=["auto", "left", "right"])
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="low floor for the RAW YOLO report (default 0.05)")
    ap.add_argument("--no-yolo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sources = collect(args.path, args.frames)
    sources = [FrameSource(s.name, rescale(s.frame, args.width), s.origin,
                           s.frame_index) for s in sources]
    print(f"{len(sources)} frame(s) from {args.path}")
    for s in sources:
        print(f"  {s.name:<26} {s.frame.shape[1]}x{s.frame.shape[0]}  {s.origin}")

    bcfg = BoundaryConfig(outer_side=args.outer_side)
    rows = [run_boundary(s, bcfg, args.out) for s in sources]

    ccfg = CarConfig()
    yolo_rows: Optional[list[dict]] = None
    if not args.no_yolo:
        try:
            det = RawYolo(ccfg, args.conf_floor)
        except Exception as exc:                       # noqa: BLE001
            print(f"\nYOLO could not be loaded: {type(exc).__name__}: {exc}")
        else:
            yolo_rows = []
            for s in sources:
                dets = det.detect(s.frame)
                path = os.path.join(args.out, f"{s.name}_yolo.jpg")
                cv2.imwrite(path, draw_yolo(s.frame, dets, ccfg))
                yolo_rows.append({"frame": s.name, "detections": dets,
                                  "debug": path})

    report(rows, yolo_rows, ccfg, args.out, args.width)

    out_json = os.path.join(args.out, "real_test_report.json")
    with open(out_json, "w") as fh:
        json.dump({"path": args.path, "width": args.width,
                   "outer_side": args.outer_side,
                   "calibration_width": CALIBRATION_WIDTH,
                   "boundary": rows, "yolo": yolo_rows}, fh, indent=2)
    print(f"\noverlays + JSON in {args.out}/")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        # a missing folder is a normal state before the footage lands, not a
        # crash -- say what to do and leave
        print(f"\n{exc}\n\nPut frames or a clip in data/real/ and run this again.",
              file=sys.stderr)
        raise SystemExit(2)

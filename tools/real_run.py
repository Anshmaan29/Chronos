"""Run the FULL CHRONOS pipeline on real footage and report what happened.

Synthetic scenes have ground truth; real footage has none.  So this tool does
not score accuracy and does not pretend to.  It answers, stage by stage,
whether each module still *functions* on real pixels, and it writes a debug
image for every stage so the answer can be checked by eye rather than taken on
trust.

Stages, in the order they depend on each other::

    boundary -> geometry -> integrity -> YOLO -> tracker -> temporal -> decide

A stage that fails takes the ones after it with it, and the report says so
rather than printing zeros that look like measurements.

YOLO gets an escalation ladder before it is called a failure: the default
settings, then a lower confidence floor, then an upscaled frame, then a larger
pretrained checkpoint.  Whatever finally works is reported, including "nothing
did".

Run::

    python tools/real_run.py                       # everything in data/real
    python tools/real_run.py --path data/real/clip.mp4
    python tools/real_run.py --sequence            # treat sorted images as a clip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.boundary import BoundaryConfig, detect_boundary, save_debug_image
from chronos.car import CarConfig, CarTracker, detect_cars, draw_cars
from chronos.decide import DecideConfig, DecisionEngine
from chronos.integrity import (IntegrityConfig, capture_baseline,
                               save_debug_image as integrity_debug,
                               score_integrity)
from chronos.temporal import TemporalConfig, TemporalEngine
from chronos.track import TrackGeometryConfig, build_track_geometry

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

RULE = "-" * 78


def hr(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


def collect(path: str, max_frames: int) -> tuple[list[np.ndarray], list[str], float]:
    """Load frames from a file or a directory.  Returns (frames, names, fps)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such path: {path}")

    files: list[str] = []
    if os.path.isfile(path):
        files = [path]
    else:
        for name in sorted(os.listdir(path)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXT or ext in VIDEO_EXT:
                files.append(os.path.join(path, name))
    if not files:
        raise FileNotFoundError(
            f"no images or videos in {path}. Found: "
            + (", ".join(sorted(os.listdir(path))[:6]) if os.path.isdir(path) else "nothing"))

    frames: list[np.ndarray] = []
    names: list[str] = []
    fps = 25.0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXT:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ! could not read {os.path.basename(f)}")
                continue
            frames.append(img)
            names.append(os.path.basename(f))
        else:
            cap = cv2.VideoCapture(f)
            if not cap.isOpened():
                print(f"  ! could not open {os.path.basename(f)}")
                continue
            v = cap.get(cv2.CAP_PROP_FPS)
            fps = float(v) if v and v > 1 else 25.0
            i = 0
            while len(frames) < max_frames:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
                names.append(f"{os.path.basename(f)}#{i}")
                i += 1
            cap.release()
        if len(frames) >= max_frames:
            break
    return frames[:max_frames], names[:max_frames], fps


# --------------------------------------------------------------------------
# YOLO, with an escalation ladder
# --------------------------------------------------------------------------


@dataclass
class YoloAttempt:
    label: str
    model: str
    conf: float
    imgsz: int
    n_vehicles: int = 0
    classes: dict = field(default_factory=dict)
    error: str = ""


def yolo_ladder(frame: np.ndarray, cfg: CarConfig,
                allow_download: bool = True) -> list[YoloAttempt]:
    """Try progressively harder to make a pretrained detector fire.

    Nothing here is trained or fine-tuned -- the ladder only changes the
    confidence floor, the input resolution and which published checkpoint is
    loaded.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return [YoloAttempt("ultralytics missing", "-", 0, 0, error=str(exc))]

    plan = [
        YoloAttempt("default", cfg.model, cfg.conf_threshold, cfg.imgsz),
        YoloAttempt("low confidence", cfg.model, 0.10, cfg.imgsz),
        YoloAttempt("upscaled", cfg.model, 0.10, 1920),
    ]
    if allow_download:
        plan.append(YoloAttempt("larger model", "yolo11m.pt", 0.10, 1280))
        plan.append(YoloAttempt("largest model", "yolo11x.pt", 0.10, 1280))

    out: list[YoloAttempt] = []
    for att in plan:
        try:
            model = YOLO(att.model)
            res = model.predict(frame, conf=att.conf, imgsz=att.imgsz, verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for c, s in zip(cls, conf):
                    name = res.names[int(c)]
                    att.classes[name] = max(att.classes.get(name, 0.0), float(s))
                att.n_vehicles = int(np.isin(cls, cfg.coco_vehicle_classes).sum())
        except Exception as exc:
            att.error = f"{type(exc).__name__}: {exc}"
        out.append(att)
        if att.n_vehicles > 0:
            break                      # no need to escalate further
    return out


class _FixedDetector:
    """Wraps whichever YOLO settings actually worked, for the rest of the run."""

    def __init__(self, model_name: str, conf: float, imgsz: int, cfg: CarConfig):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf, self.imgsz, self.cfg = conf, imgsz, cfg

    def detect(self, frame: np.ndarray):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = np.isin(cls, self.cfg.coco_vehicle_classes)
        return res.boxes.xyxy.cpu().numpy()[keep], res.boxes.conf.cpu().numpy()[keep]


# --------------------------------------------------------------------------


def plausibility(res, frame, boxes) -> list[tuple[str, bool, str]]:
    """Catch an obviously-wrong boundary without ground truth.

    ``ok`` from Module 1 only means a polyline came back.  On real footage a
    polyline can come back that traces a car, a barrier or a grandstand, and
    reporting that as "works" is how a demo dies in Q&A.  None of these checks
    can prove a boundary is right -- they only catch it being wrong in ways
    that are decidable from the image alone.
    """
    checks: list[tuple[str, bool, str]] = []
    h, w = frame.shape[:2]

    agree = res.agreement_px
    if agree is None:
        # the learned detector has no second channel; absence of a cross-check
        # is not evidence of a wrong answer, so do not score it as one
        checks.append(("two channels agree", True, "n/a for this detector"))
    else:
        checks.append(("two channels agree", agree <= 40.0, f"{agree:.0f} px apart"))

    road = 0.0 if res.drivable_mask is None else float((res.drivable_mask > 0).mean())
    checks.append((
        "road area plausible",
        0.03 <= road <= 0.45,
        f"{road:.0%} of frame"))

    poly_in_car = kerb_on_car = 0.0
    if len(boxes):
        pts = res.polyline
        if pts is not None and len(pts):
            inside = np.zeros(len(pts), bool)
            for x1, y1, x2, y2 in boxes:
                inside |= ((pts[:, 0] >= x1) & (pts[:, 0] <= x2)
                           & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
            poly_in_car = float(inside.mean())
        if res.kerb_mask is not None and res.kerb_mask.any():
            car_mask = np.zeros((h, w), np.uint8)
            for x1, y1, x2, y2 in boxes:
                car_mask[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)] = 1
            kerb_on_car = float((res.kerb_mask[car_mask > 0] > 0).sum()
                                / max((res.kerb_mask > 0).sum(), 1))
    checks.append((
        "boundary not on a car",
        poly_in_car <= 0.30,
        f"{poly_in_car:.0%} of the polyline is inside a car box"))
    checks.append((
        "kerb is not a car",
        kerb_on_car <= 0.25,
        f"{kerb_on_car:.0%} of the kerb mask is inside a car box"))
    return checks


def main() -> int:
    p = argparse.ArgumentParser(description="CHRONOS on real footage")
    p.add_argument("--path", default="data/real")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--sequence", action="store_true",
                   help="treat the images as consecutive frames of one clip")
    p.add_argument("--out", default="debug/real")
    p.add_argument("--no-download", action="store_true",
                   help="do not fetch larger YOLO checkpoints")
    p.add_argument("--detector", default="auto",
                   choices=["real", "synthetic", "auto"],
                   help="real = pretrained segmentation (chronos/boundary_real.py)")
    p.add_argument("--backend", default="auto", choices=["auto", "sam2", "segformer"])
    p.add_argument("--line-width-mm", type=float, default=100.0,
                   help="FIA edge line width at this circuit")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames, names, fps = collect(args.path, args.frames)
    is_clip = args.sequence or len(frames) > 1 and "#" in names[0]

    hr("INPUT")
    print(f"  {len(frames)} frame(s) from {args.path}")
    for n, f in list(zip(names, frames))[:6]:
        print(f"    {n:52s} {f.shape[1]}x{f.shape[0]}")
    if len(frames) > 6:
        print(f"    ... and {len(frames) - 6} more")
    print(f"  treated as: {'a clip (temporal layer active)' if is_clip else 'independent still frames'}")
    if not is_clip and len(frames) == 1:
        print("  NOTE a single still cannot exercise the tracker or the temporal layer,")
        print("       and integrity measured against itself is 100 by definition.")

    bcfg = BoundaryConfig(debug_dir=args.out)
    icfg = IntegrityConfig(debug_dir=args.out)
    ccfg = CarConfig()

    from chronos.boundary_real import RealBoundaryConfig, detect as detect_any
    from chronos.car import YoloDetector

    rcfg = RealBoundaryConfig(backend=args.backend, debug_dir=args.out)

    # YOLO first: its boxes prompt the segmenter and are cut out of the road
    # mask, so on real footage detection quality feeds boundary quality
    box_cache: dict[int, np.ndarray] = {}
    yolo = None
    if args.detector in ("real", "auto"):
        try:
            yolo = YoloDetector(ccfg)
        except Exception as exc:
            print(f"  (no YOLO for prompting: {type(exc).__name__}: {exc})")

    # ---------------- boundary ----------------
    hr("1. BOUNDARY")
    results = []
    for i, (name, frame) in enumerate(zip(names, frames)):
        bcfg.debug_name = f"boundary_{i:03d}.jpg"
        rcfg.debug_name = f"boundary_{i:03d}.jpg"
        boxes = None
        if yolo is not None:
            try:
                boxes, _ = yolo.detect(frame)
                box_cache[i] = boxes
            except Exception:
                boxes = None
        try:
            res = detect_any(frame, args.detector, boxes, rcfg, bcfg, save_debug=True)
        except Exception as exc:
            print(f"  {name:40s} RAISED {type(exc).__name__}: {exc}")
            results.append(None)
            continue
        results.append(res)
        area = 0.0 if res.drivable_mask is None else float((res.drivable_mask > 0).mean())
        kerb = 0.0 if res.kerb_mask is None else float((res.kerb_mask > 0).mean())
        if res.ok:
            print(f"  {name[:38]:38s} OK   {len(res.polyline):3d} pts  "
                  f"road {area:5.1%}  kerb {kerb:5.2%}  "
                  + (f"agree {res.agreement_px:.1f}px" if res.agreement_px is not None
                     else "fallback silent"))
        else:
            print(f"  {name[:38]:38s} FAIL road {area:5.1%} kerb {kerb:5.2%} -- {res.reason[:60]}")
    ok_frames = [i for i, r in enumerate(results) if r is not None and r.ok]
    print(f"\n  detector: {args.detector}"
          + (f" (backend {args.backend})" if args.detector != "synthetic" else ""))
    print(f"  boundary returned a polyline on {len(ok_frames)}/{len(frames)} frames")
    print(f"  overlays -> {args.out}/boundary_*.jpg   (VERIFY BY EYE: 'ok' only means")
    print("              a polyline came back, not that it is on the white line)")

    if not ok_frames:
        print("\n  Everything downstream depends on this. Stopping here.")
        return 1

    base_i = ok_frames[0]
    base_frame, base_res = frames[base_i], results[base_i]

    # ---------------- plausibility ----------------
    hr("1b. IS THAT BOUNDARY PLAUSIBLE?")
    probe_boxes = box_cache.get(base_i)
    if probe_boxes is None:
        probe_boxes = np.zeros((0, 4))
        try:
            probe = yolo_ladder(base_frame, ccfg, allow_download=False)
            hit = next((a for a in probe if a.n_vehicles > 0), None)
            if hit is not None:
                det0 = _FixedDetector(hit.model, hit.conf, hit.imgsz, ccfg)
                probe_boxes, _ = det0.detect(base_frame)
        except Exception:
            pass
    checks = plausibility(base_res, base_frame, probe_boxes)
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {label:24s} {detail}")
    boundary_plausible = all(p for _, p, _ in checks)
    if not boundary_plausible:
        print("\n  The polyline came back, but it does not survive inspection.")
        print("  Treat every number downstream as arithmetic on a wrong reference.")

    # ---------------- geometry ----------------
    hr("2. GROUND-PLANE GEOMETRY  (chronos/track.py)")
    geo = None
    gcfg = TrackGeometryConfig(line_width_mm=args.line_width_mm)
    try:
        geo = build_track_geometry(base_frame, base_res, gcfg, icfg)
        print(f"  scale        {np.median(geo.mm_per_px_across):8.1f} mm/px across the track")
        if geo.mm_per_px_along is not None:
            print(f"               {np.median(geo.mm_per_px_along):8.1f} mm/px along "
                  f"(anisotropy {geo.sources.get('along_over_across', float('nan')):.2f}x)")
        else:
            print("               along-track scale unavailable (no kerb stripes found)")
        print(f"  paint width  {geo.sources['paint_px_median']:8.2f} px median")
        print(f"  confidence   {geo.scale_confidence:8.2f}")
        print(f"\n  Assumes the edge line is {args.line_width_mm:.0f} mm wide. Pass")
        print("  --line-width-mm to match the circuit, or every millimetre below is wrong.")
    except Exception as exc:
        print(f"  FAILED {type(exc).__name__}: {exc}")

    # ---------------- integrity ----------------
    hr("3. BOUNDARY INTEGRITY  (chronos/integrity.py)")
    baseline = None
    try:
        baseline = capture_baseline(base_frame, None, icfg, result=base_res)
        print(f"  baseline captured from {names[base_i][:40]}")
        print(f"    paint {baseline.paint_luma:6.0f} vs road {baseline.asphalt_luma:6.0f}"
              f"   contrast {baseline.contrast:.3f}   sharpness {baseline.sharpness:.0f}")
        print(f"    achievable coverage {baseline.coverage:.0%}   "
              f"{int(baseline.valid.sum())} of {len(baseline.valid)} stations usable")
        for i in ok_frames[:6]:
            sc = score_integrity(frames[i], baseline, icfg, result=results[i])
            print(f"  {names[i][:38]:38s} integrity {sc.total:5.1f}  "
                  f"(C {sc.contrast:3.0f} / L {sc.continuity:3.0f} / "
                  f"S {sc.sharpness:3.0f} / X {sc.contamination:3.0f})")
            integrity_debug(frames[i], baseline, sc, icfg,
                            path=os.path.join(args.out, f"integrity_{i:03d}.jpg"),
                            title=names[i][:28])
        if len(ok_frames) == 1:
            print("\n  Scored against itself, so 100 is arithmetic, not evidence.")
            print("  A real integrity number needs a known-clean baseline frame of the")
            print("  same corner, captured before the session.")
    except Exception as exc:
        print(f"  FAILED {type(exc).__name__}: {exc}")

    # ---------------- YOLO ----------------
    hr("4. CAR DETECTION  (pretrained YOLO, no training)")
    attempts = yolo_ladder(frames[base_i], ccfg, allow_download=not args.no_download)
    working = None
    for att in attempts:
        if att.error:
            print(f"  {att.label:16s} ERROR {att.error[:60]}")
            continue
        top = sorted(att.classes.items(), key=lambda kv: -kv[1])[:5]
        shown = ", ".join(f"{k} {v:.2f}" for k, v in top) or "nothing at all"
        print(f"  {att.label:16s} {att.model:12s} conf>={att.conf:.2f} imgsz={att.imgsz:4d}"
              f"  vehicles={att.n_vehicles}")
        print(f"  {'':16s} classes: {shown}")
        if att.n_vehicles > 0 and working is None:
            working = att
    if working is None:
        print("\n  NO SETTING FOUND THAT DETECTS A VEHICLE.")
        print("  Reported plainly: pretrained YOLO does not find cars in this footage.")
    else:
        print(f"\n  Vehicles found with: {working.label} "
              f"({working.model}, conf>={working.conf}, imgsz={working.imgsz})")

    # ---------------- tracker + temporal + decide ----------------
    hr("5. TRACKER, TEMPORAL, DECISION")
    if working is None:
        print("  Skipped: no detections to track.")
    elif geo is None:
        print("  Skipped: no ground-plane scale, so margins cannot be computed.")
    elif not is_clip:
        print("  Skipped: independent stills. Persistent ids and excursion events")
        print("  need consecutive frames of the same car. Re-run with --sequence")
        print("  on a clip, or supply a video.")
    else:
        det = _FixedDetector(working.model, working.conf, working.imgsz, ccfg)
        tracker = CarTracker(ccfg)
        temporal = TemporalEngine(TemporalConfig(fps=fps))
        decisions = DecisionEngine(DecideConfig(), fps=fps)
        decisions.start()
        ids: set[int] = set()
        per_frame = []
        t0 = time.perf_counter()
        for i, frame in enumerate(frames):
            try:
                cars = detect_cars(frame, i, tracker, det, geo, ccfg,
                                   TemporalConfig(fps=fps).tyre_width_mm)
                integ = 100.0
                if baseline is not None:
                    try:
                        integ = score_integrity(frame, baseline, icfg).total
                    except Exception:
                        pass
                events = temporal.update(i, 1000.0 * i / fps, cars,
                                         geo.margin_mm_clipped, integ)
                decisions.ingest(events)
                decisions.note_frames(1)
                ids.update(c.car_id for c in cars)
                per_frame.append(len(cars))
                if i < 8:
                    cv2.imwrite(os.path.join(args.out, f"cars_{i:03d}.jpg"),
                                draw_cars(frame, cars))
            except Exception:
                print(f"  frame {i}: {traceback.format_exc(limit=1).strip().splitlines()[-1][:70]}")
        decisions.ingest(temporal.flush())
        dt = time.perf_counter() - t0
        print(f"  frames processed  {len(frames)} in {dt:.1f}s ({len(frames)/max(dt,1e-9):.1f} fps)")
        print(f"  cars per frame    mean {np.mean(per_frame or [0]):.2f}, "
              f"max {max(per_frame or [0])}")
        print(f"  distinct ids      {len(ids)}  -> "
              + ("IDs held" if len(ids) <= 2 else f"{len(ids)} ids for the footage; "
                 "expect one per car, so this is id churn"))
        print(f"  excursion events  {decisions.stats().events_found}")
        for v in decisions.verdicts:
            print(f"    {v.as_line()}")
        if not decisions.verdicts:
            print("    none -- no car put three wheels past the detected boundary")

    hr("VERDICT ON THE FOOTAGE")
    if not ok_frames:
        state = "FAILS (no polyline)"
    elif boundary_plausible:
        state = f"RETURNS A PLAUSIBLE BOUNDARY on {len(ok_frames)}/{len(frames)}"
    else:
        state = f"FAILS (polyline on {len(ok_frames)}/{len(frames)}, but it is not the line)"
    print(f"  boundary   {state}")
    print(f"  geometry   {'RUNS' if geo is not None else 'FAILS'}"
          + ("" if boundary_plausible else "  (meaningless: built on a wrong boundary)"))
    reason = ("" if (len(ok_frames) > 1 and boundary_plausible)
              else "  (meaningless: " + ("single frame" if boundary_plausible
                                         else "wrong boundary") + ")")
    print(f"  integrity  {'RUNS' if baseline is not None else 'FAILS'}{reason}")
    print(f"  YOLO       {'FINDS CARS' if working else 'FINDS NO CARS'}")
    print(f"  tracker    {'EXERCISED' if (working and is_clip and geo is not None) else 'NOT EXERCISED'}")
    print(f"\n  debug images -> {args.out}/")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}\n\nPut frames or a clip in data/real/ and run again.", file=sys.stderr)
        raise SystemExit(2)

"""CHRONOS Module 4 -- car detection, persistent IDs, and wheel contact points.

Three jobs, in order of how much they matter downstream:

  1. **Persistent identity.**  The temporal layer reasons about one car over
     time, so the same car must keep the same id across frames.  An id that
     changes mid-excursion silently splits one event into two, and both halves
     look too short to be a violation.  That is why ``track_continuity`` is
     carried all the way to the verdict.

  2. **Contact patches, not boxes.**  The rule is about where rubber meets
     asphalt.  A bounding box bottom edge is not that point: on an elevated
     camera the box includes bodywork that overhangs the contact patch by a
     wheel's width or more.  Wheels are found as the dark regions low in the
     box, and each contact patch is the bottom of one wheel.

  3. **Occlusion recovery.**  The four patches form a rectangle on the ground
     plane, so a hidden wheel is computed from the other three -- and the
     result is flagged at lower confidence rather than passed off as measured.

No training.  Detection is a pretrained YOLO, loaded lazily so that a machine
without torch still runs everything else.  The tracker is ByteTrack-style
two-stage association and is implemented here, so the demo does not depend on
an optional package resolving at 3am.

Run standalone::

    python -m chronos.car --image data/frame.jpg
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

WHEELS = ("FL", "FR", "RL", "RR")
_OPPOSITE = {"FL": ("FR", "RL", "RR"), "FR": ("FL", "RR", "RL"),
             "RL": ("FL", "RR", "FR"), "RR": ("FR", "RL", "FL")}


@dataclass
class CarConfig:
    """Tunables for detection, tracking and contact-point extraction."""

    # --- detection --------------------------------------------------------
    model: str = "yolo11n.pt"         # pretrained, downloaded once, never trained
    conf_threshold: float = 0.25
    high_conf: float = 0.55           # ByteTrack's first association pass
    coco_vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)   # car, motorcycle, bus, truck
    imgsz: int = 960

    # --- tracking ---------------------------------------------------------
    max_age: int = 12                 # frames a track survives unmatched
    min_hits: int = 2                 # frames before a track is reported
    iou_match: float = 0.25
    process_noise: float = 6.0
    measurement_noise: float = 12.0

    # --- wheels -----------------------------------------------------------
    # Physical dimensions, used to place the contact rectangle on the ground
    # plane.  These are published for every car on the grid; nothing here is
    # learned or calibrated.
    axle_track_mm: float = 2000.0
    wheelbase_mm: float = 3600.0
    use_model_fit: bool = True        # see wheel_contacts_model
    refine_window_px: int = 9         # local search for each predicted wheel
    min_support_px: int = 4           # dark pixels needed to call a wheel "seen"
    anisotropy_fallback: float = 2.2  # along/across scale ratio when no kerb is
                                      # visible to measure it from
    # A tyre is 380 mm wide and the ON_LINE / OUTSIDE decision turns on that
    # width.  Where a pixel is worth 100 mm, that decision is being made on
    # half a pixel, and no amount of filtering makes it real.  So the same
    # principle applied to the boundary is applied to the measurement itself:
    # measure the resolution, and let a thin one lower confidence rather than
    # quietly produce a confident wrong answer.
    tyre_px_full_conf: float = 9.0    # tyre width in pixels for full confidence
    tyre_px_floor: float = 2.0        # below this the wheel state is guesswork

    tyre_val_max: int = 88            # tyres are the darkest thing on a car
    tyre_sat_max: int = 110
    wheel_band_top: float = 0.34      # ignore the top of the box: no wheels there
    min_wheel_area_frac: float = 0.0035   # of the box area
    contact_rows: int = 3             # rows averaged at the bottom of a wheel
    recovered_confidence: float = 0.55
    full_confidence: float = 0.95


@dataclass
class WheelPoints:
    """The four contact patches of one car, in image pixels.

    ``recovered`` names any wheel that was computed from the other three
    rather than seen.  ``confidence`` drops accordingly -- a recovered wheel is
    an inference, and the verdict is entitled to know that.
    """

    points: dict[str, np.ndarray]
    confidence: float
    recovered: tuple[str, ...] = ()
    reason: str = ""

    def as_array(self) -> np.ndarray:
        return np.array([self.points[w] for w in WHEELS], float)


@dataclass
class CarDetection:
    """One car in one frame."""

    car_id: int
    box: tuple[float, float, float, float]
    score: float
    frame_index: int
    wheels: Optional[WheelPoints] = None
    age: int = 0
    hits: int = 1
    missed: int = 0


# --------------------------------------------------------------------------
# wheel contact points
# --------------------------------------------------------------------------


def _label_wheels(pts: np.ndarray, forward: np.ndarray,
                  right: np.ndarray) -> Optional[dict[str, np.ndarray]]:
    """Name up to four contact points FL / FR / RL / RR.

    Named in the car's own frame -- ``forward`` is the direction of travel and
    ``right`` is the driver's right -- so the labels mean the same thing on a
    left-hand corner as on a right-hand one.
    """
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    f = (pts - c) @ forward
    r = (pts - c) @ right
    out: dict[str, np.ndarray] = {}
    for p, fi, ri in zip(pts, f, r):
        name = ("F" if fi >= 0 else "R") + ("R" if ri >= 0 else "L")
        if name in out:
            return None            # two points claim the same corner
        out[name] = p
    return out


def _recover_missing(found: dict[str, np.ndarray]) -> tuple[dict, tuple[str, ...]]:
    """Complete a rectangle from three corners.

    On the ground plane the four contact patches are a rectangle, so
    FL + RR = FR + RL.  One missing corner follows from the other three.  The
    image is a projection of that plane, so this is exact only for a small
    footprint viewed from a distance -- true enough for a car, and the caller
    marks the result as recovered either way.
    """
    missing = [w for w in WHEELS if w not in found]
    if len(missing) != 1:
        return found, ()
    m = missing[0]
    a, b, opp = _OPPOSITE[m]
    return {**found, m: found[a] + found[b] - found[opp]}, (m,)


def wheel_contacts_model(frame: np.ndarray, box: Sequence[float], geometry,
                         forward: np.ndarray, right: np.ndarray,
                         cfg: Optional[CarConfig] = None) -> WheelPoints:
    """Place the four contact patches from the car's known ground footprint.

    Looking down at a car from an elevated corner camera, the two far-side
    wheels are usually behind the bodywork.  Hunting for four dark blobs
    therefore finds two, and a method that needs three has already failed on
    the normal case rather than an edge case.

    So the footprint is predicted instead of discovered.  The contact patches
    form a rectangle of known size -- axle track across, wheelbase along -- and
    the scale in each of those directions is already known from
    :mod:`chronos.track`.  The rectangle is placed so that its lowest corner
    sits on the bottom of the detection box and its centre on the box centre,
    then each corner is refined toward nearby tyre-dark pixels.

    Confidence reflects how many corners actually found dark support; a wheel
    with none is an inference and is named in ``recovered``.

    **Measured error budget.**  Against exact ground truth on the synthetic
    benchmark, contact points derived this way land within roughly 250-400 mm
    near the camera and degrade with distance -- at 40 mm per pixel, a few
    pixels of box error is most of a metre.  A tyre is 380 mm wide, so the
    ON_LINE / OUTSIDE distinction is NOT reliable from a bounding box at
    distance.  That is the honest state of the art: the most recent published
    system in this field lists contact-patch keypoints as its own future work,
    and it is the first item on this project's cut list.  Where a detector can
    supply real contact points, pass them through instead -- the temporal and
    decision layers are unchanged and are validated that way.
    """
    cfg = cfg or CarConfig()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return WheelPoints({}, 0.0, reason=f"box is {x2-x1:.0f}x{y2-y1:.0f} px, too small")

    cx = 0.5 * (x1 + x2)
    i = int(geometry.nearest_station(np.array([[cx, y2]]))[0])
    mm_across = float(geometry.mm_per_px_across[i])
    if geometry.mm_per_px_along is not None:
        mm_along = float(geometry.mm_per_px_along[i])
    else:
        mm_along = mm_across * cfg.anisotropy_fallback
    axle_px = cfg.axle_track_mm / max(mm_across, 1e-6)
    base_px = cfg.wheelbase_mm / max(mm_along, 1e-6)
    if not np.isfinite(axle_px) or not np.isfinite(base_px) or axle_px < 2 or base_px < 2:
        return WheelPoints({}, 0.0, reason="derived car footprint is smaller than a pixel")

    fwd = np.asarray(forward, float) / max(np.linalg.norm(forward), 1e-9)
    rgt = np.asarray(right, float) / max(np.linalg.norm(right), 1e-9)
    offsets = {"FL": 0.5 * base_px * fwd - 0.5 * axle_px * rgt,
               "FR": 0.5 * base_px * fwd + 0.5 * axle_px * rgt,
               "RL": -0.5 * base_px * fwd - 0.5 * axle_px * rgt,
               "RR": -0.5 * base_px * fwd + 0.5 * axle_px * rgt}
    off = np.array([offsets[k] for k in WHEELS])
    # anchor: lowest corner on the box bottom, centre on the box centre
    centre = np.array([cx - off[:, 0].mean(), y2 - off[:, 1].max()])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] <= cfg.tyre_val_max)
            & (hsv[:, :, 1] <= cfg.tyre_sat_max)).astype(np.uint8)

    r = max(2, int(cfg.refine_window_px) // 2)
    pts, seen = {}, []
    for name in WHEELS:
        p = centre + offsets[name]
        px, py = int(round(p[0])), int(round(p[1]))
        a, b = max(px - r, 0), min(px + r + 1, w)
        c, d = max(py - r, 0), min(py + r + 1, h)
        refined, support = p, 0
        if b > a and d > c:
            win = dark[c:d, a:b]
            support = int(win.sum())
            if support >= cfg.min_support_px:
                ys, xs = np.nonzero(win)
                bottom = ys.max()
                sel = ys >= bottom - 1
                refined = np.array([xs[sel].mean() + a, ys[sel].mean() + c])
        pts[name] = refined
        seen.append(support >= cfg.min_support_px)

    n_seen = int(sum(seen))
    recovered = tuple(nm for nm, s_ in zip(WHEELS, seen) if not s_)
    if n_seen == 0:
        return WheelPoints({}, 0.0,
                           reason="no tyre-dark support at any predicted wheel; "
                                  "the box may not contain a car")
    conf = cfg.full_confidence if n_seen == 4 else cfg.recovered_confidence + 0.1 * n_seen
    return WheelPoints(pts, float(min(conf, cfg.full_confidence)), recovered,
                       reason=("all four wheels seen" if n_seen == 4 else
                               f"{n_seen} of 4 wheels seen; "
                               f"{', '.join(recovered)} placed from the known footprint"))


def wheel_contacts(frame: np.ndarray, box: Sequence[float],
                   forward: Optional[np.ndarray] = None,
                   right: Optional[np.ndarray] = None,
                   cfg: Optional[CarConfig] = None) -> WheelPoints:
    """Find the four tyre contact patches inside a car's bounding box.

    Parameters
    ----------
    frame:   full BGR frame.
    box:     (x1, y1, x2, y2) in pixels.
    forward, right:
        Unit vectors for the car's axes, used only to name the wheels.  Pass
        the track tangent and outward normal at the car's position; image axes
        are assumed if omitted, which names wheels by screen direction.

    Returns
    -------
    WheelPoints.  ``confidence`` is 0 and ``reason`` is set when fewer than
    three wheels could be found -- the caller must not treat that as a
    measurement.
    """
    cfg = cfg or CarConfig()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return WheelPoints({}, 0.0, reason=f"box is {x2-x1}x{y2-y1} px, too small to resolve wheels")

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark = ((hsv[:, :, 2] <= cfg.tyre_val_max)
            & (hsv[:, :, 1] <= cfg.tyre_sat_max)).astype(np.uint8) * 255
    dark[:int(cfg.wheel_band_top * dark.shape[0])] = 0      # no wheels up there
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if not dark.any():
        return WheelPoints({}, 0.0, reason="no tyre-dark pixels in the lower box")

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    min_area = cfg.min_wheel_area_frac * dark.shape[0] * dark.shape[1]
    blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    blobs.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    blobs = blobs[:4]
    if len(blobs) < 3:
        return WheelPoints({}, 0.0,
                           reason=f"only {len(blobs)} wheel-sized dark regions found; "
                                  "need 3 to place a car on the ground")

    pts = []
    for i in blobs:
        ys, xs = np.nonzero(labels == i)
        bottom = ys.max()
        sel = ys >= bottom - cfg.contact_rows
        pts.append(np.array([xs[sel].mean() + x1, ys[sel].mean() + y1]))
    pts = np.array(pts, float)

    fwd = np.asarray(forward, float) if forward is not None else np.array([0.0, -1.0])
    rgt = np.asarray(right, float) if right is not None else np.array([1.0, 0.0])
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    rgt = rgt / max(np.linalg.norm(rgt), 1e-9)

    named = _label_wheels(pts, fwd, rgt)
    if named is None:
        return WheelPoints({}, 0.0,
                           reason=f"{len(pts)} contact points did not resolve to distinct corners")
    if len(named) == 4:
        return WheelPoints(named, cfg.full_confidence)
    completed, recovered = _recover_missing(named)
    if len(completed) != 4:
        return WheelPoints({}, 0.0, reason=f"{len(named)} wheels found, cannot complete the rectangle")
    return WheelPoints(completed, cfg.recovered_confidence, recovered,
                       reason=f"{recovered[0]} occluded, computed from the other three")


# --------------------------------------------------------------------------
# tracking: ByteTrack-style two-stage association
# --------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` against every box in ``b``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], a[:, 2, None], a[:, 3, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / np.maximum(union, 1e-9)


class _Track:
    """One tracked car: a constant-velocity Kalman filter over the box centre."""

    def __init__(self, track_id: int, box: np.ndarray, score: float, cfg: CarConfig):
        self.id = track_id
        self.score = score
        self.hits = 1
        self.missed = 0
        self.age = 0
        self.box = box.astype(float)
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * cfg.process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * cfg.measurement_noise
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    def predict(self) -> np.ndarray:
        s = self.kf.predict()
        dx = float(s[0, 0]) - 0.5 * (self.box[0] + self.box[2])
        dy = float(s[1, 0]) - 0.5 * (self.box[1] + self.box[3])
        self.box = self.box + np.array([dx, dy, dx, dy])
        self.age += 1
        return self.box

    def update(self, box: np.ndarray, score: float) -> None:
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.box = box.astype(float)
        self.score = score
        self.hits += 1
        self.missed = 0

    @property
    def velocity(self) -> np.ndarray:
        return np.array([float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0])])


class CarTracker:
    """Persistent car identities across frames.

    ByteTrack's central idea: associate the confident detections first, then
    give the leftover low-confidence detections a second chance against the
    tracks that are still unmatched.  A car half-hidden behind spray drops to a
    low score for a few frames rather than disappearing, and keeping it -- with
    its id -- is what stops one excursion being reported as two.
    """

    def __init__(self, cfg: Optional[CarConfig] = None):
        self.cfg = cfg or CarConfig()
        self.tracks: list[_Track] = []
        self._next_id = 1
        self.id_switches = 0

    def _associate(self, tracks: list[_Track], boxes: np.ndarray,
                   scores: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = _iou(np.array([t.box for t in tracks]), boxes)
        pairs: list[tuple[int, int]] = []
        used_t, used_d = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        for ti, di in order:
            if ti in used_t or di in used_d or iou[ti, di] < self.cfg.iou_match:
                continue
            pairs.append((int(ti), int(di)))
            used_t.add(int(ti))
            used_d.add(int(di))
        return (pairs,
                [i for i in range(len(tracks)) if i not in used_t],
                [i for i in range(len(boxes)) if i not in used_d])

    def update(self, boxes: Iterable[Sequence[float]],
               scores: Iterable[float]) -> list[_Track]:
        """Advance the tracker by one frame.  Returns the confirmed tracks."""
        boxes = np.asarray(list(boxes), float).reshape(-1, 4)
        scores = np.asarray(list(scores), float).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError(f"CarTracker.update: {len(boxes)} boxes but {len(scores)} scores")

        for t in self.tracks:
            t.predict()

        high = scores >= self.cfg.high_conf
        low = ~high & (scores >= self.cfg.conf_threshold)

        # pass 1: confident detections against every track
        pairs, un_t, un_d = self._associate(self.tracks, boxes[high], scores[high])
        hi_idx = np.nonzero(high)[0]
        for ti, di in pairs:
            self.tracks[ti].update(boxes[hi_idx[di]], float(scores[hi_idx[di]]))

        # pass 2: the leftovers get a second chance at the unmatched tracks
        rest = [self.tracks[i] for i in un_t]
        lo_idx = np.nonzero(low)[0]
        pairs2, un_t2, _ = self._associate(rest, boxes[lo_idx], scores[lo_idx])
        for ti, di in pairs2:
            rest[ti].update(boxes[lo_idx[di]], float(scores[lo_idx[di]]))

        still_unmatched = {id(rest[i]) for i in un_t2}
        for t in self.tracks:
            if id(t) in still_unmatched:
                t.missed += 1

        # unmatched CONFIDENT detections start new tracks
        for di in un_d:
            b = boxes[hi_idx[di]]
            self.tracks.append(_Track(self._next_id, b, float(scores[hi_idx[di]]), self.cfg))
            self._next_id += 1

        before = {t.id for t in self.tracks}
        self.tracks = [t for t in self.tracks if t.missed <= self.cfg.max_age]
        self.id_switches += len(before - {t.id for t in self.tracks})
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits and t.missed == 0]


# --------------------------------------------------------------------------
# YOLO detection (pretrained only, imported lazily)
# --------------------------------------------------------------------------


class YoloDetector:
    """Pretrained YOLO, used as-is.  No training, no fine-tuning, no dataset.

    Imported lazily so that a machine without torch can still run Modules 1-3,
    the benchmark and the UI.  If the model cannot be loaded the error says so
    plainly instead of leaving an empty detection list to be misread as "no
    cars in this frame".
    """

    def __init__(self, cfg: Optional[CarConfig] = None):
        self.cfg = cfg or CarConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YoloDetector needs ultralytics installed "
                    "(uv pip install ultralytics). Modules 1-3 do not.") from exc
            self._model = YOLO(self.cfg.model)
        return self._model

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes Nx4, scores N) for vehicles in one frame."""
        res = self.model.predict(frame, conf=self.cfg.conf_threshold,
                                 imgsz=self.cfg.imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = np.isin(cls, self.cfg.coco_vehicle_classes)
        return (res.boxes.xyxy.cpu().numpy()[keep],
                res.boxes.conf.cpu().numpy()[keep])


# --------------------------------------------------------------------------


def resolution_confidence(geometry, point: np.ndarray, tyre_width_mm: float,
                          cfg: CarConfig) -> float:
    """How many pixels wide the tyre is here, mapped to 0..1 confidence.

    This is the measurement's own integrity: a verdict that turns on 380 mm
    should not be issued where 380 mm is one pixel.
    """
    i = int(geometry.nearest_station(np.asarray(point, float).reshape(1, 2))[0])
    px = tyre_width_mm / max(float(geometry.mm_per_px_across[i]), 1e-6)
    lo, hi = cfg.tyre_px_floor, cfg.tyre_px_full_conf
    return float(np.clip((px - lo) / max(hi - lo, 1e-6), 0.0, 1.0))


def detect_cars(frame: np.ndarray, frame_index: int, tracker: CarTracker,
                detector, geometry=None,
                cfg: Optional[CarConfig] = None,
                tyre_width_mm: float = 380.0) -> list[CarDetection]:
    """Detect, track and place the wheels of every car in one frame.

    ``detector`` is anything with ``detect(frame) -> (boxes, scores)``, so the
    benchmark can substitute a simulated detector with controlled noise and
    exercise the same tracking and contact-point code the real one uses.
    """
    cfg = cfg or CarConfig()
    boxes, scores = detector.detect(frame)
    # A detector that can give real contact patches (a keypoint model, or the
    # benchmark's known geometry) supplies them here.  The temporal and
    # decision layers downstream are identical either way, which is what lets
    # them be validated without inheriting the box-to-contact error budget.
    supplied = getattr(detector, "contact_points", None)
    out = []
    for t in tracker.update(boxes, scores):
        fwd = rgt = None
        if geometry is not None:
            centre = np.array([[0.5 * (t.box[0] + t.box[2]), t.box[3]]])
            i = int(geometry.nearest_station(centre)[0])
            # the car runs along the track; its right is the OUTWARD normal
            n_in = geometry.track.normals[i]
            rgt = -n_in
            fwd = np.array([-rgt[1], rgt[0]])
            if np.dot(fwd, t.velocity) < 0:
                fwd = -fwd
                rgt = -rgt
        if supplied is not None:
            pts = supplied(t.box)
            wheels = (WheelPoints(pts, cfg.full_confidence, reason="contact points supplied")
                      if pts else WheelPoints({}, 0.0, reason="no supplied contact points"))
        elif cfg.use_model_fit and geometry is not None and fwd is not None:
            wheels = wheel_contacts_model(frame, t.box, geometry, fwd, rgt, cfg)
        else:
            wheels = wheel_contacts(frame, t.box, fwd, rgt, cfg)
        if wheels.confidence > 0 and geometry is not None:
            anchor = np.array([0.5 * (t.box[0] + t.box[2]), t.box[3]])
            wheels.confidence *= resolution_confidence(geometry, anchor,
                                                       tyre_width_mm, cfg)
            if wheels.confidence <= 0.01:
                wheels.reason += " -- but a tyre is under a pixel wide here"
        out.append(CarDetection(car_id=t.id, box=tuple(map(float, t.box)),
                                score=float(t.score), frame_index=frame_index,
                                wheels=wheels if wheels.confidence > 0 else None,
                                age=t.age, hits=t.hits, missed=t.missed))
    return out


def draw_cars(frame: np.ndarray, cars: Sequence[CarDetection]) -> np.ndarray:
    """Debug overlay: boxes, ids, and the four contact patches."""
    vis = frame.copy()
    colors = {"FL": (80, 220, 80), "FR": (60, 200, 255),
              "RL": (255, 170, 60), "RR": (220, 120, 255)}
    for c in cars:
        x1, y1, x2, y2 = (int(v) for v in c.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(vis, f"#{c.car_id} {c.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        if c.wheels is None:
            cv2.putText(vis, "no contact points", (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            continue
        for name, p in c.wheels.points.items():
            q = tuple(np.round(p).astype(int))
            recovered = name in c.wheels.recovered
            cv2.drawMarker(vis, q, colors[name],
                           cv2.MARKER_TILTED_CROSS if recovered else cv2.MARKER_CROSS,
                           13, 2, cv2.LINE_AA)
            cv2.putText(vis, name + ("*" if recovered else ""), (q[0] + 7, q[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1, cv2.LINE_AA)
    return vis


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS Module 4 -- cars and wheels")
    p.add_argument("--image", required=True)
    p.add_argument("--debug", default="debug/car_debug.jpg")
    args = p.parse_args()
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image!r}")

    cfg = CarConfig()
    det = YoloDetector(cfg)
    boxes, scores = det.detect(frame)
    print(f"YOLO       : {len(boxes)} vehicle detections")
    tracker = CarTracker(cfg)
    cars = detect_cars(frame, 0, tracker, det, None, cfg)
    for c in cars:
        w = c.wheels
        print(f"  car #{c.car_id} score {c.score:.2f} "
              + (f"wheels {list(w.points)} conf {w.confidence:.2f} {w.reason}"
                 if w else "no contact points"))
    os.makedirs(os.path.dirname(os.path.abspath(args.debug)), exist_ok=True)
    cv2.imwrite(args.debug, draw_cars(frame, cars))
    print(f"debug      : {args.debug}")


if __name__ == "__main__":
    _cli()

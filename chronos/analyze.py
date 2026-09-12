"""CHRONOS frame analysis -- verdicts on ANY image or video, frame by frame.

The session engine (:mod:`chronos.pipeline`) is built for the camera the
system is ultimately for: one fixed camera at one corner, a clean baseline
captured before the session, and integrity measured as *change since then*.
It is rigorous and it refuses on everything else -- stills, broadcast pans,
drone shots -- because none of those have a session baseline to be relative
to.

That refusal is correct for a session, and useless for the question a judge,
a steward or a team actually brings to a single photo or clip: **is this car
over the line, by how much, and can the picture be trusted?**  This module
answers that question per frame, with no baseline, and says so:

  * the **track surface** is found as unsaturated grey asphalt -- kerbs,
    painted run-off and astroturf are strongly coloured, so the colour
    contrast that defeats region-growing detectors on real footage is the
    thing this uses;
  * the **track limit** is the edge where that surface meets the painted zone,
    fitted as one smooth curve through every visible edge point (so it runs
    straight through the stretch a car is hiding) and snapped outward to the
    **outer edge of the white line** -- the line is part of the track;
  * **cars are found without COCO semantics.**  Pretrained YOLO labels F1 cars
    seen from above as "suitcase", "kite" and "boat" (measured on this
    footage), so car candidates come from three independent physical cues --
    near-black neutral tyre/floor mass, car-sized holes the car punches in the
    asphalt surface, and YOLO boxes of any class -- and a candidate needs
    physical evidence, not a label, to count;
  * **tyres are judged, not boxes.**  Each tyre's ground contact is measured as
    a signed distance from the outer edge of the line.  A tyre with any part
    on the track side is still on the track.  A car is a violation only when
    every measured tyre is fully beyond the line -- the FIA wording, "no part
    of the car in contact with the track";
  * **reference quality is measured per frame, absolutely** -- contrast,
    continuity, edge sharpness and contamination of the line in this frame --
    and a violation on a line that cannot be trusted is escalated to REVIEW
    REQUIRED rather than called.  That gating is the CHRONOS contribution, and
    it survives the move from sessions to single frames.

Millimetres come from the car itself: an F1 car is 2000 mm wide, so the pixel
span between the outer edges of an axle's tyres fixes the local scale.  That is
a first-order estimate and is labelled as one.

Run::

    python -m chronos.analyze data/real/frame.jpeg
    python -m chronos.analyze data/real/clip.mp4 --every 1
    python -m chronos.analyze data/real            # every file in a folder
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional

import cv2
import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".gif"}

# verdicts, strongest last
CLEAR = "CLEAR"
BORDERLINE = "BORDERLINE"
VIOLATION = "VIOLATION"
REVIEW = "REVIEW REQUIRED"
NO_REFERENCE = "NO REFERENCE"

# wheel states
INSIDE = "INSIDE"
ON_LINE = "ON LINE"
OUTSIDE = "OUTSIDE"
UNKNOWN = "UNKNOWN"

# BGR drawing colours
_C = {CLEAR: (80, 220, 80), BORDERLINE: (0, 200, 255), VIOLATION: (40, 40, 255),
      REVIEW: (0, 170, 255), NO_REFERENCE: (160, 160, 160),
      INSIDE: (80, 220, 80), ON_LINE: (0, 200, 255), OUTSIDE: (40, 40, 255),
      UNKNOWN: (200, 200, 200)}


# --------------------------------------------------------------------------
# CONFIG -- every tunable lives here.
# --------------------------------------------------------------------------


@dataclass
class AnalyzeConfig:
    # --- working resolution ------------------------------------------------
    work_width: int = 1100            # analyse at this width, report in full px

    # --- track surface: neutral grey asphalt (Lab) -------------------------
    asphalt_chroma_max: float = 8.5   # measured: asphalt < 8, painted run-off > 12
    asphalt_l_min: float = 25.0       # below this: tyre, carbon, OR deep shade --
                                      # which is why a tyre in shade is UNKNOWN,
                                      # never OUTSIDE (see judge_car)
    asphalt_l_max: float = 88.0       # above this is white paint
    min_track_frac: float = 0.05      # smaller than this is not a track in shot

    # --- the painted line --------------------------------------------------
    paint_v_min: int = 175            # white paint, relative floor
    paint_s_max: int = 75
    paint_band_px: float = 0.030      # of work width: how far out to look
    min_paint_frac: float = 0.20      # stations that must find paint
    edge_outside_chroma: float = 14.0  # the far side of a limit edge is coloured
    edge_outside_l: float = 82.0       # ...or bright paint
    limit_join_frac: float = 0.28      # rejoin the line across a car this wide
    limit_extend_frac: float = 0.12    # extend traced ends along their tangent

    # --- the fitted limit curve --------------------------------------------
    fit_degree: int = 2
    fit_iterations: int = 4
    fit_reject_mad: float = 3.0
    min_edge_points: int = 25
    n_stations: int = 240

    # --- cars ----------------------------------------------------------------
    dark_l_max: float = 30.0          # tyres, floor, diffuser (Lab lightness)
    dark_chroma_max: float = 14.0     # ...and NEUTRAL, which navy run-off is not
    black_l_max: float = 16.0         # this dark is car regardless of hue
    min_black_core: float = 0.04      # share of a car box that must be black
    tyres_only_core: float = 0.10     # ...if nothing but darkness found it
    tyre_blob_min_frac: float = 0.0006  # a tyre-shaped black blob, of the frame
    min_rel_car_area: float = 0.18    # of the biggest car in the frame
    max_vs_vehicle: float = 3.0       # no "car" this many times a labelled one
    max_cars: int = 6
    car_track_dist_frac: float = 0.06  # a car's box must come this close to asphalt
    car_track_dist_cap: float = 0.12   # ...but never further than this
    min_car_frac: float = 0.004
    max_car_frac: float = 0.25
    tyre_min_frac: float = 0.0010
    use_yolo: bool = True
    yolo_model: str = "yolo11n.pt"
    yolo_conf: float = 0.20
    yolo_imgsz: int = 1280            # small, distant cars need the resolution
    yolo_strong_conf: float = 0.30    # may stand without a black core
    vehicle_classes: tuple = (2, 3, 5, 7)   # COCO car, motorcycle, bus, truck
    vehicle_conf: float = 0.20
    min_vehicle_frac: float = 0.0008
    car_aspect: tuple = (0.45, 3.6)   # w / h: barriers and shadow bands are not
    car_width_mm: float = 2000.0      # F1 overall width: the scale reference
    tyre_width_mm: float = 380.0      # mean of 305 front / 405 rear

    # --- judgement -----------------------------------------------------------
    borderline_mm: float = 150.0      # closer than this to the edge: flag it
    integrity_review: float = 50.0    # below this a violation goes to REVIEW
    min_tyres: int = 2                # fewer measured than this: REVIEW

    # --- video -----------------------------------------------------------------
    event_gap_frames: int = 2         # violation frames this close are one event
    track_iou: float = 0.20


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------


@dataclass
class Wheel:
    name: str
    x: float
    y: float
    margin_px: float                  # + on the track side of the outer edge
    margin_mm: Optional[float]
    state: str


@dataclass
class CarResult:
    car_id: int
    box: tuple                        # x1, y1, x2, y2 full-frame px
    evidence: list                    # which cues found it
    wheels: list                      # list[Wheel]
    verdict: str
    reason: str
    worst_margin_mm: Optional[float]  # the car's best-placed tyre: how far in/out
    tyres_out: int
    tyres_measured: int
    mm_per_px: Optional[float]
    trust: float


@dataclass
class FrameResult:
    index: int
    t_s: float
    width: int
    height: int
    line_found: bool
    line_reason: str
    integrity: Optional[float]
    contrast: Optional[float]
    continuity: Optional[float]
    sharpness: Optional[float]
    contamination: Optional[float]
    line_width_px: Optional[float]
    cars: list                        # list[CarResult]
    verdict: str                      # the frame's strongest car verdict
    ms: float
    annotated: Optional[np.ndarray] = field(default=None, repr=False)
    polyline: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("annotated", "polyline", "cars")}
        d["cars"] = [asdict(c) for c in self.cars]
        return d


class _Score:
    """Adapter so the console's component widget can read a frame's scores."""

    def __init__(self, fr: FrameResult):
        self.contrast = fr.contrast
        self.continuity = fr.continuity
        self.sharpness = fr.sharpness
        self.contamination = fr.contamination
        self.total = fr.integrity


# --------------------------------------------------------------------------
# 1. track surface
# --------------------------------------------------------------------------


def _kernel(px: int) -> np.ndarray:
    px = max(1, int(px) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def lab_parts(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lightness 0-100, chroma) -- colourfulness that does not blow up in shade.

    HSV saturation is unstable on dark pixels: dark grey with a faint blue cast
    reads as saturated, and navy paint in shadow reads as grey.  On this
    footage that let shaded blue run-off pass as asphalt and moved the track.
    Lab chroma is a distance from neutral, so grey asphalt stays near zero in
    sun and in shade while any painted surface stays well above it.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a, b = lab[..., 1] - 128.0, lab[..., 2] - 128.0
    return L, np.sqrt(a * a + b * b)


def track_surface(bgr: np.ndarray, cfg: AnalyzeConfig) -> np.ndarray:
    """0/255 mask of neutral grey asphalt, cleaned.  White paint is excluded."""
    L, chroma = lab_parts(bgr)
    m = ((chroma <= cfg.asphalt_chroma_max) & (L >= cfg.asphalt_l_min)
         & (L <= cfg.asphalt_l_max)).astype(np.uint8) * 255
    w = bgr.shape[1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(w // 220))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(w // 80))
    return m


def main_track(surface: np.ndarray, near: Optional[list], cfg: AnalyzeConfig
               ) -> tuple[Optional[np.ndarray], str]:
    """The asphalt component the cars are on (or the largest one)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(surface, 8)
    if n <= 1:
        return None, "no unsaturated asphalt surface in frame"
    h, w = surface.shape
    area = h * w
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= cfg.min_track_frac * area]
    if not comps:
        big = st[1:, cv2.CC_STAT_AREA].max() / area
        return None, (f"largest asphalt region is {big:.1%} of the frame, below "
                      f"{cfg.min_track_frac:.0%} -- no track surface in shot")
    # The LARGEST surface is the track.  Preferring the surface a car is
    # touching sounds sensible and is exactly backwards: a car that has left
    # the track is, by definition, not touching it -- on the Red Bull frame that
    # rule picked a shaded patch beside the car and called a violation CLEAR.
    best = max(comps, key=lambda i: st[i, cv2.CC_STAT_AREA])
    return (lab == best).astype(np.uint8) * 255, ""


# --------------------------------------------------------------------------
# 2. cars, from physical evidence
# --------------------------------------------------------------------------


_YOLO = {}


def _yolo(cfg: AnalyzeConfig):
    if cfg.yolo_model not in _YOLO:
        try:
            from ultralytics import YOLO
            _YOLO[cfg.yolo_model] = YOLO(cfg.yolo_model)
        except Exception:
            _YOLO[cfg.yolo_model] = None
    return _YOLO[cfg.yolo_model]


_CUE_WEIGHT = {"vehicle": 5.0, "tyres": 3.0, "yolo": 2.0, "yolo_strong": 2.0, "occlusion": 1.0}


def _merge_boxes(cands: list, w: int, h: int) -> list:
    """Suppress duplicates of the same car; never grow a box into a blob.

    Unioning every box that nearly touches another sounded harmless and was
    not: on real frames a car's box chained into the run-off beside it, the
    kerb strip behind it and a stray detector box, and the union -- 37-43% of
    the frame -- was then thrown away as too big to be a car.  So the strongest
    candidate keeps its own box, and weaker ones that overlap it only donate
    their evidence.
    """
    def area(c):
        return max(1, (c[2] - c[0]) * (c[3] - c[1]))

    def score(c):
        return sum(_CUE_WEIGHT.get(e, 0) for e in c[4]) + 1e-6 * area(c)

    pool = sorted(([int(c[0]), int(c[1]), int(c[2]), int(c[3]), list(c[4])]
                   for c in cands), key=score, reverse=True)
    kept: list = []
    for c in pool:
        absorbed = False
        for k in kept:
            ix = max(0, min(k[2], c[2]) - max(k[0], c[0]))
            iy = max(0, min(k[3], c[3]) - max(k[1], c[1]))
            inter = ix * iy
            iou = inter / float(area(k) + area(c) - inter)
            ccx, ccy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
            centre_inside = k[0] <= ccx <= k[2] and k[1] <= ccy <= k[3]
            # a small box touching or overlapping a bigger car is that car's
            # own tyre or wing, not a second car
            gapx = max(0, max(k[0], c[0]) - min(k[2], c[2]))
            gapy = max(0, max(k[1], c[1]) - min(k[3], c[3]))
            touching = (gapx <= 0.10 * (k[2] - k[0]) and gapy <= 0.10 * (k[3] - k[1]))
            part_of = touching and area(c) < 0.45 * area(k)
            if (iou >= 0.25 or inter / float(area(c)) >= 0.35
                    or inter / float(area(k)) >= 0.6 or centre_inside or part_of):
                k[4] = sorted(set(k[4]) | set(c[4]))
                absorbed = True
                break
        if not absorbed:
            kept.append(c)
    return [c for c in kept
            if 0 <= c[0] < c[2] <= w and 0 <= c[1] < c[3] <= h]


def find_cars(frame: np.ndarray, hsv: np.ndarray, surface: np.ndarray,
              cfg: AnalyzeConfig) -> tuple[list, np.ndarray]:
    """Car candidates as (x1, y1, x2, y2, evidence) plus the car-pixel mask."""
    h, w = hsv.shape[:2]
    area = float(h * w)
    L, chroma = lab_parts(frame)

    # dark AND neutral: rubber, carbon floor, diffuser.  Navy paint in shade is
    # dark but not neutral, which is the distinction HSV could not make.
    carpix = (((L <= cfg.dark_l_max) & (chroma <= cfg.dark_chroma_max))
              | (L <= cfg.black_l_max)).astype(np.uint8) * 255
    carpix = cv2.morphologyEx(carpix, cv2.MORPH_OPEN, _kernel(w // 360))
    carpix = cv2.morphologyEx(carpix, cv2.MORPH_CLOSE, _kernel(w // 110))

    cands: list = []

    # cue 1: near-black neutral mass -- tyres, floor, diffuser.  A car's dark
    # mass often fuses with a shadow or dark run-off beside it, so each blob is
    # tightened to its BLACK core: rubber and carbon are far darker than any
    # painted surface, which is what keeps the box on the car.
    black = (L <= cfg.black_l_max + 4) & (chroma <= cfg.dark_chroma_max)
    n, lab, st, _ = cv2.connectedComponentsWithStats(carpix, 8)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < cfg.tyre_min_frac * area or min(bw, bh) < 0.012 * min(w, h):
            continue
        comp = lab[y:y + bh, x:x + bw] == i
        core = comp & black[y:y + bh, x:x + bw]
        if core.sum() < max(0.25 * a, cfg.tyre_min_frac * area):
            # Too little black for a core -- unless the blob is huge, in which
            # case a car may be buried in it (fused with shade and run-off) and
            # its black core is still worth extracting below.
            if core.sum() < cfg.tyre_min_frac * area:
                if a > cfg.max_car_frac * area or a / float(bw * bh) < 0.30:
                    continue
                cands.append((x, y, x + bw, y + bh, ["tyres"]))
                continue
        core_u8 = core.astype(np.uint8) * 255
        core_u8 = cv2.morphologyEx(core_u8, cv2.MORPH_CLOSE, _kernel(w // 80))
        m2, lab2, st2, _ = cv2.connectedComponentsWithStats(core_u8, 8)
        if m2 <= 1:
            continue
        big = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        # Keep the biggest black piece and only the pieces NEAR it.  A car's
        # tyres and floor sit within about a car-width of each other; black
        # shading on a kerb two car-lengths away fused into the same blob and
        # stretched the box until it was rejected as too big to be a car.
        bx, by, bw2, bh2, _ = st2[big]
        reach = 0.9 * max(bw2, bh2)
        keepm = np.zeros_like(core_u8)
        for j in range(1, m2):
            jx, jy, jw, jh, ja = st2[j]
            if ja < 0.08 * st2[big, cv2.CC_STAT_AREA]:
                continue
            gx = max(0, max(bx, jx) - min(bx + bw2, jx + jw))
            gy = max(0, max(by, jy) - min(by + bh2, jy + jh))
            if j == big or (gx <= reach and gy <= reach):
                keepm[lab2 == j] = 255
        ys, xs = np.nonzero(keepm)
        if len(xs) == 0:
            continue
        bx1, by1, bx2, by2 = x + xs.min(), y + ys.min(), x + xs.max() + 1, y + ys.max() + 1
        ca = (bx2 - bx1) * (by2 - by1)
        if ca > cfg.max_car_frac * area:
            # Dark streaks on the run-off are joined to the car by thin
            # strands, so even the biggest black piece spans half the frame.
            # A car is THICK and a streak is thin: open the core at growing
            # strength until the strands break, and keep the thick mass.
            solved = False
            for kk in (w // 70, w // 45, w // 30, w // 22):
                opened = cv2.morphologyEx(core_u8, cv2.MORPH_OPEN, _kernel(max(3, kk)))
                m3, lab3, st3, _ = cv2.connectedComponentsWithStats(opened, 8)
                if m3 <= 1:
                    break
                b3 = 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))
                ox, oy, ow, oh, oa = st3[b3]
                if ow * oh <= cfg.max_car_frac * area and oa >= cfg.tyre_min_frac * area:
                    pad = kk // 2
                    bx1, by1 = max(0, x + int(ox) - pad), max(0, y + int(oy) - pad)
                    bx2, by2 = min(w, x + int(ox + ow) + pad), min(h, y + int(oy + oh) + pad)
                    ca = (bx2 - bx1) * (by2 - by1)
                    solved = True
                    break
            if not solved:
                bx1, by1 = x + int(bx), y + int(by)
                bx2, by2 = bx1 + int(bw2), by1 + int(bh2)
                ca = (bx2 - bx1) * (by2 - by1)
        if ca < cfg.min_car_frac * area or ca > cfg.max_car_frac * area:
            continue
        cands.append((bx1, by1, bx2, by2, ["tyres"]))

    # cue 2: car-sized holes and bites the car makes in the asphalt surface
    if surface is not None and surface.any():
        closed = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, _kernel(w // 9))
        filled = closed.copy()
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, ff, (0, 0), 255) if closed[0, 0] == 0 else None
        occl = cv2.bitwise_and(closed, cv2.bitwise_not(surface))
        occl = cv2.morphologyEx(occl, cv2.MORPH_OPEN, _kernel(w // 120))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(occl, 8)
        for i in range(1, n2):
            x, y, bw, bh, a = st2[i]
            if a < cfg.min_car_frac * area or a > cfg.max_car_frac * area:
                continue
            if a / float(bw * bh) < 0.45:
                continue
            # a band running most of the way across the frame is a kerb or a
            # strip of run-off between two surfaces, not a car
            if bw > 0.70 * w or bh > 0.70 * h:
                continue
            aspect = bw / float(max(bh, 1))
            if not 0.5 <= aspect <= 4.0:
                continue
            # a hole needs something car-like inside it, or it is a shadow
            sub = carpix[y:y + bh, x:x + bw]
            colourful = chroma[y:y + bh, x:x + bw]
            if (sub > 0).mean() < 0.04 and (colourful > 25).mean() < 0.35:
                continue
            cands.append((x, y, x + bw, y + bh, ["occlusion"]))

    # cue 3: YOLO, any class -- only ever corroborating
    if cfg.use_yolo:
        model = _yolo(cfg)
        if model is not None:
            try:
                # Detect at the frame's own resolution.  Upscaling a 547-px
                # photo to 1280 made the detector lose the car it found at
                # native size; downscaling a 1080p clip loses distant cars.
                side = max(frame.shape[:2])
                imgsz = int(np.clip(32 * round(side / 32), 640, cfg.yolo_imgsz))
                r = model.predict(frame, conf=min(cfg.yolo_conf, cfg.vehicle_conf),
                                  imgsz=imgsz, verbose=False)[0]
                if r.boxes is not None:
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    for b, s, cl in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), classes):
                        x1, y1, x2, y2 = b
                        a = (x2 - x1) * (y2 - y1)
                        if cl in cfg.vehicle_classes and float(s) >= cfg.vehicle_conf:
                            # a real vehicle label: trusted without a black core
                            # (touring cars are white and red, not black) and
                            # allowed to be small, because it is far away
                            if cfg.min_vehicle_frac * area <= a <= cfg.max_car_frac * area:
                                cands.append((int(x1), int(y1), int(x2), int(y2), ["vehicle"]))
                            continue
                        if a < cfg.min_car_frac * area or a > cfg.max_car_frac * area:
                            continue
                        sub = carpix[int(y1):int(y2), int(x1):int(x2)]
                        if float(s) < cfg.yolo_conf:
                            continue
                        if sub.size and (sub > 0).mean() >= 0.08:
                            cands.append((int(x1), int(y1), int(x2), int(y2), ["yolo"]))
                        elif float(s) >= cfg.yolo_strong_conf and float(s) >= cfg.yolo_conf:
                            cands.append((int(x1), int(y1), int(x2), int(y2), ["yolo_strong"]))
            except Exception:
                pass

    # compact, solid, tyre-shaped black blobs: the physical signature of a car
    black_u8 = cv2.morphologyEx(black.astype(np.uint8) * 255, cv2.MORPH_OPEN, _kernel(3))
    nb, labb, stb, _ = cv2.connectedComponentsWithStats(black_u8, 8)
    tyre_blobs = []
    for j in range(1, nb):
        bx, by, bbw, bbh, ba = stb[j]
        if ba < cfg.tyre_blob_min_frac * area:
            continue
        asp = bbh / float(max(bbw, 1))
        if not 0.45 <= asp <= 3.8:
            continue
        pts = np.column_stack(np.nonzero(labb[by:by + bbh, bx:bx + bbw] == j))
        if len(pts) < 5:
            continue
        hull = cv2.convexHull(pts[:, ::-1].astype(np.int32))
        solidity = ba / float(max(cv2.contourArea(hull), 1.0))
        if solidity < 0.55:
            continue
        tyre_blobs.append((bx, by, bx + bbw, by + bbh, ba))

    def blobs_in(x1, y1, x2, y2):
        return [b for b in tyre_blobs
                if b[0] >= x1 - 2 and b[2] <= x2 + 2 and b[1] >= y1 - 2 and b[3] <= y2 + 2]

    # A detector box often covers half a car (YOLO boxed only the Ferrari's
    # right side).  Extend a detector box to take in tyre blobs beside it at the
    # same height -- a car's other tyre, not a kerb two car-lengths away.
    grown = []
    for c in cands:
        x1, y1, x2, y2, ev = c
        if not ({"yolo", "vehicle", "yolo_strong"} & set(ev)):
            grown.append(c)
            continue
        bw_ = x2 - x1
        nx1, nx2, ny1, ny2 = x1, x2, y1, y2
        for b in tyre_blobs:
            cy = (b[1] + b[3]) / 2.0
            if not (y1 <= cy <= y2):
                continue
            gap = max(0, max(x1, b[0]) - min(x2, b[2]))
            if gap <= 1.0 * bw_:
                nx1, nx2 = min(nx1, b[0]), max(nx2, b[2])
                ny1, ny2 = min(ny1, b[1]), max(ny2, b[3])
        if (nx2 - nx1) <= 2.6 * bw_ and (nx2 - nx1) * (ny2 - ny1) <= cfg.max_car_frac * area:
            grown.append((nx1, ny1, nx2, ny2, ev))
        else:
            grown.append(c)
    cands = grown

    merged = _merge_boxes(cands, w, h)
    scored = []
    have_vehicle = any("vehicle" in ev for *_, ev in merged)
    for x1, y1, x2, y2, ev in merged:
        a = (x2 - x1) * (y2 - y1)
        floor = cfg.min_vehicle_frac if "vehicle" in ev else cfg.min_car_frac
        if a < floor * area or a > cfg.max_car_frac * area:
            continue
        aspect = (x2 - x1) / float(max(1, y2 - y1))
        if "vehicle" not in ev and not cfg.car_aspect[0] <= aspect <= cfg.car_aspect[1]:
            continue
        sub = carpix[y1:y2, x1:x2]
        core = black[y1:y2, x1:x2]
        dark_frac = float((sub > 0).mean()) if sub.size else 0.0
        core_frac = float(core.mean()) if core.size else 0.0
        # a car has a real black core -- tyres are the blackest thing in shot;
        # dark run-off, kerb shading and grass do not have one
        trusted = "vehicle" in ev or "yolo_strong" in ev
        # With a labelled vehicle in view, the dark masses left over are
        # grandstands, fences and shade -- keep one only if it is really black.
        need = cfg.min_black_core * (6.0 if have_vehicle else 1.0)
        if not trusted and "occlusion" not in ev and core_frac < need:
            continue
        if ev == ["tyres"] and core_frac < cfg.tyres_only_core:
            # dark mass alone, with no detector agreeing, must be unmistakably
            # black -- shaded kerb and run-off are dark but not that dark
            continue
        if not ({"yolo", "vehicle", "yolo_strong"} & set(ev)):
            # with no detector behind it, a car must SHOW two tyres side by
            # side.  Kerb shading and a barrier are one dark band, not two
            # separate solid blobs a car-width apart.
            inside = blobs_in(x1, y1, x2, y2)
            if len(inside) < 2:
                continue
            cxs = [(b[0] + b[2]) / 2.0 for b in inside]
            if max(cxs) - min(cxs) < 0.30 * (x2 - x1):
                continue
        if ev == ["occlusion"]:
            # a hole in the asphalt with no black core and no detector behind it
            # is a kerb bite or a shadow, not a car
            continue
        strength = sum(_CUE_WEIGHT.get(e, 0) for e in ev) + 10 * core_frac
        scored.append((strength, a, (x1, y1, x2, y2, ev)))
    if not scored:
        return [], carpix
    # Cars in one frame are the same kind of object at a similar scale.  A
    # labelled vehicle is never dropped for being small -- it is small because
    # it is far away -- and when one is present, anything several times its
    # size is a barrier or a stand, not another car.
    veh = [a for _, a, c in scored if "vehicle" in c[4]]
    out = []
    others = [a for _, a, c in scored if "vehicle" not in c[4]]
    biggest_other = max(others) if others else 0
    for sc, a, c in sorted(scored, key=lambda t: -t[0]):
        if "vehicle" in c[4]:
            out.append(c)
            continue
        if veh and a > cfg.max_vs_vehicle * max(veh):
            continue
        if a < cfg.min_rel_car_area * biggest_other:
            continue
        out.append(c)
    return out[:cfg.max_cars], carpix


# --------------------------------------------------------------------------
# 3. the track limit
# --------------------------------------------------------------------------


def _fit_curve(pts: np.ndarray, cfg: AnalyzeConfig):
    """Robust polynomial in the points' own principal frame."""
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    a = evecs[:, int(np.argmax(evals))]
    b = np.array([-a[1], a[0]])
    t = (pts - c) @ a
    u = (pts - c) @ b
    keep = np.ones(len(t), bool)
    deg = cfg.fit_degree if len(t) >= 3 * (cfg.fit_degree + 1) else 1
    coef = np.polyfit(t, u, deg)
    for _ in range(cfg.fit_iterations):
        r = u - np.polyval(coef, t)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
        keep = np.abs(r) <= cfg.fit_reject_mad * 1.4826 * mad + 1.0
        if keep.sum() < deg + 2:
            break
        coef = np.polyfit(t[keep], u[keep], deg)
    return c, a, b, coef, float(t[keep].min()), float(t[keep].max()), keep


def _sample_curve(c, a, b, coef, t0, t1, tmin, tmax, n) -> np.ndarray:
    """Polynomial between the data, straight tangent extensions beyond it."""
    t = np.linspace(tmin, tmax, n)
    d1 = np.polyder(coef)
    u = np.polyval(coef, np.clip(t, t0, t1))
    lo, hi = t < t0, t > t1
    u[lo] = np.polyval(coef, t0) + np.polyval(d1, t0) * (t[lo] - t0)
    u[hi] = np.polyval(coef, t1) + np.polyval(d1, t1) * (t[hi] - t1)
    return c[None, :] + t[:, None] * a[None, :] + u[:, None] * b[None, :]


def _runs_circular(flags: np.ndarray) -> list:
    """Contiguous True runs of a circular boolean array, as index arrays."""
    n = len(flags)
    if n == 0 or not flags.any():
        return []
    if flags.all():
        return [np.arange(n)]
    start = int(np.flatnonzero(~flags)[0])          # rotate so a run cannot wrap
    rot = np.roll(flags, -start)
    idx = np.flatnonzero(rot)
    pieces = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(pc + start) % n for pc in pieces if len(pc)]


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win) | 1)
    if win == 1 or len(arr) < win:
        return arr.copy()
    pad = win // 2
    padded = np.concatenate([np.repeat(arr[:1], pad, 0), arr, np.repeat(arr[-1:], pad, 0)])
    k = np.ones(win) / win
    if arr.ndim == 1:
        return np.convolve(padded, k, "valid")
    return np.stack([np.convolve(padded[:, j], k, "valid") for j in range(arr.shape[1])], 1)


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return poly
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


def _inward_normals(poly: np.ndarray, track: np.ndarray, w: int, h: int) -> np.ndarray:
    tang = np.gradient(poly, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    probe = poly + max(6.0, w / 60.0) * nrm
    pxi = np.clip(probe[:, 0].astype(int), 0, w - 1)
    pyi = np.clip(probe[:, 1].astype(int), 0, h - 1)
    if (track[pyi, pxi] > 0).mean() < 0.5:
        nrm = -nrm                      # normals point INTO the track
    return nrm


def find_limit(frame: np.ndarray, hsv: np.ndarray, track: np.ndarray,
               cars: list, cfg: AnalyzeConfig) -> dict:
    """Trace the track limit and measure the line's condition along it.

    The limit is TRACED along the real boundary of the asphalt, not fitted with
    one global curve.  A polynomial is smooth and wrong on exactly the corners
    that produce track-limit calls: the Red Bull Ring's kerbs are tight arcs,
    and a quadratic cut straight across them and through the cars.  So the
    boundary is followed point by point, broken where a car hides it and
    rejoined across the gap, and only then smoothed.
    """
    h, w = track.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    S, V = hsv[..., 1], hsv[..., 2]
    L, chroma = lab_parts(frame)
    res = {"ok": False, "reason": "", "poly": None, "normals": None}

    contours, _ = cv2.findContours(track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        res["reason"] = "track surface has no outline"
        return res
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    n = len(cnt)

    m = max(3, w // 200)
    ok = ((cnt[:, 0] > m) & (cnt[:, 0] < w - 1 - m)
          & (cnt[:, 1] > m) & (cnt[:, 1] < h - 1 - m))
    carmask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2, _ in cars:
        pad = int(0.08 * max(x2 - x1, y2 - y1))
        cv2.rectangle(carmask, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 255, -1)
    xi = np.clip(cnt[:, 0].astype(int), 0, w - 1)
    yi = np.clip(cnt[:, 1].astype(int), 0, h - 1)
    ok &= carmask[yi, xi] == 0

    # a track LIMIT edge has paint or coloured run-off on its far side; an edge
    # against a grey wall, a shadow or the frame does not
    dist = cv2.distanceTransform(track, cv2.DIST_L2, 3)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)
    step = max(5, w // 110)
    colored = np.zeros(n, bool)
    for i in np.flatnonzero(ok):
        vx, vy = -gx[yi[i], xi[i]], -gy[yi[i], xi[i]]
        nn = np.hypot(vx, vy)
        if nn < 1e-6:
            continue
        for k in (1.0, 2.0):
            px = int(np.clip(xi[i] + k * step * vx / nn, 0, w - 1))
            py = int(np.clip(yi[i] + k * step * vy / nn, 0, h - 1))
            if chroma[py, px] >= cfg.edge_outside_chroma or L[py, px] >= cfg.edge_outside_l:
                colored[i] = True
                break
    valid = ok & colored
    # bridge speckle-sized gaps so one noisy pixel does not cut the line
    gap = max(3, n // 250)
    if valid.any():
        closed = valid.copy()
        runs = _runs_circular(~valid)
        for r in runs:
            if len(r) <= gap:
                closed[r] = True
        valid = closed & (ok | colored)

    min_run = max(cfg.min_edge_points, n // 80)
    runs = [r for r in _runs_circular(valid) if len(r) >= min_run]
    if not runs:
        res["reason"] = (f"no stretch of the asphalt edge borders paint or coloured "
                         f"run-off for {min_run}+ points -- no track limit visible")
        return res

    # seed with the longest run, then rejoin neighbours across car-sized gaps
    runs.sort(key=lambda r: int(r[0]))
    seed = max(range(len(runs)), key=lambda k: len(runs[k]))
    chain = [runs[seed]]
    join_px = cfg.limit_join_frac * max(w, h)

    def direction(idx: np.ndarray, at_end: bool) -> np.ndarray:
        seg = cnt[idx[-min(len(idx), 12):]] if at_end else cnt[idx[:min(len(idx), 12)]]
        v = seg[-1] - seg[0]
        return v / max(np.linalg.norm(v), 1e-9)

    used = {seed}
    for sgn in (1, -1):
        k = seed
        while True:
            k2 = (k + sgn) % len(runs)
            if k2 in used or len(used) == len(runs):
                break
            cur = chain[-1] if sgn == 1 else chain[0]
            nxt = runs[k2]
            a_pt = cnt[cur[-1]] if sgn == 1 else cnt[cur[0]]
            b_pt = cnt[nxt[0]] if sgn == 1 else cnt[nxt[-1]]
            gapv = b_pt - a_pt
            dgap = float(np.linalg.norm(gapv))
            if dgap > join_px:
                break
            d_cur = direction(cur, at_end=(sgn == 1))
            if dgap > 3 and float(np.dot(gapv / dgap, d_cur if sgn == 1 else -d_cur)) < 0.2:
                break
            if sgn == 1:
                chain.append(nxt)
            else:
                chain.insert(0, nxt)
            used.add(k2)
            k = k2
    pts = np.concatenate([cnt[r] for r in chain], axis=0)
    if len(pts) < cfg.min_edge_points:
        res["reason"] = "track-limit edge too short to trace"
        return res

    poly = _resample(pts, cfg.n_stations)
    poly = _moving_average(poly, max(3, cfg.n_stations // 30))
    nrm = _inward_normals(poly, track, w, h)

    # snap outward across the white line to its outer edge
    band = max(4.0, cfg.paint_band_px * w)
    depths = np.arange(-0.4 * band, band, 1.0)
    offsets = np.full(len(poly), np.nan)
    widths, lp, la, sharp, dirty = [], [], [], [], []
    track_v = V[track > 0]
    vmin = max(cfg.paint_v_min, int(np.percentile(track_v, 92))) if track_v.size else cfg.paint_v_min
    for i in range(len(poly)):
        q = poly[i][None, :] - depths[:, None] * nrm[i][None, :]      # outward
        qx = np.clip(q[:, 0].astype(int), 0, w - 1)
        qy = np.clip(q[:, 1].astype(int), 0, h - 1)
        white = (V[qy, qx] >= vmin) & (S[qy, qx] <= cfg.paint_s_max)
        if not white.any():
            continue
        idx = np.flatnonzero(white)
        pieces = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        first = min(pieces, key=lambda r: abs(depths[r[0]]))
        if abs(depths[first[0]]) > 0.5 * band:
            continue
        wpx = float(depths[first[-1]] - depths[first[0]] + 1)
        widths.append(wpx)
        offsets[i] = depths[first[-1]]
        lp.append(float(gray[qy[first], qx[first]].mean()))
        ins = poly[i] + 2.5 * max(wpx, 2) * nrm[i]
        la.append(float(gray[int(np.clip(ins[1], 0, h - 1)), int(np.clip(ins[0], 0, w - 1))]))
        e0 = poly[i] - depths[first[-1]] * nrm[i]
        e_in, e_out = e0 + 2 * nrm[i], e0 - 2 * nrm[i]
        gi = float(gray[int(np.clip(e_in[1], 0, h - 1)), int(np.clip(e_in[0], 0, w - 1))])
        go = float(gray[int(np.clip(e_out[1], 0, h - 1)), int(np.clip(e_out[0], 0, w - 1))])
        sharp.append(abs(gi - go))
        dirty.append(float(np.mean(V[qy[first], qx[first]] < vmin - 25)))

    found = ~np.isnan(offsets)
    frac = float(found.mean())
    line_w = None
    if frac >= cfg.min_paint_frac and found.sum() >= 6:
        line_w = float(np.median(widths))
        cap = 1.8 * line_w
        o = np.clip(offsets, -cap, cap)
        ii = np.arange(len(o))
        o = np.interp(ii, ii[found], o[found])        # carry across unpainted stations
        o = _moving_average(o, max(3, cfg.n_stations // 20))
        poly = poly - o[:, None] * nrm
        poly = _moving_average(poly, max(5, cfg.n_stations // 16))
        nrm = _inward_normals(poly, track, w, h)

    # extend both ends along their tangents so a tyre just past the traced
    # stretch is still measured against the line, not against an endpoint
    ext = cfg.limit_extend_frac * max(w, h)
    t0 = poly[0] - poly[min(8, len(poly) - 1)]
    t1 = poly[-1] - poly[max(-9, -len(poly))]
    t0 /= max(np.linalg.norm(t0), 1e-9)
    t1 /= max(np.linalg.norm(t1), 1e-9)
    k = 12
    head = np.array([poly[0] + t0 * ext * (j / k) for j in range(k, 0, -1)])
    tail = np.array([poly[-1] + t1 * ext * (j / k) for j in range(1, k + 1)])
    full = np.concatenate([head, poly, tail], axis=0)
    full_n = np.concatenate([np.repeat(nrm[:1], k, 0), nrm, np.repeat(nrm[-1:], k, 0)], axis=0)
    inside = ((full[:, 0] >= -w * 0.05) & (full[:, 0] <= w * 1.05)
              & (full[:, 1] >= -h * 0.05) & (full[:, 1] <= h * 1.05))

    # absolute condition of the reference, in this frame
    if lp:
        Lp, La = float(np.mean(lp)), float(np.mean(la))
        michelson = (Lp - La) / max(Lp + La, 1e-6)
        contrast = 100.0 * float(np.clip(michelson / 0.30, 0, 1))
        sharpness = 100.0 * float(np.clip(np.mean(sharp) / 70.0, 0, 1))
        contamination = 100.0 * float(np.clip(1.0 - np.mean(dirty) / 0.5, 0, 1))
    else:
        contrast = sharpness = contamination = 0.0
    continuity = 100.0 * float(np.clip(frac / 0.70, 0, 1))
    parts = np.array([contrast, continuity, sharpness, contamination]) / 100.0
    wts = np.array([0.30, 0.25, 0.20, 0.25])
    integrity = float(np.exp(np.sum(wts * np.log(np.maximum(parts, 0.02)))) * 100.0)

    res.update({"ok": True, "poly": poly, "normals": nrm,
                "poly_ext": full[inside], "normals_ext": full_n[inside],
                "line_w": line_w, "contrast": contrast, "continuity": continuity,
                "sharpness": sharpness, "contamination": contamination,
                "integrity": integrity, "paint_frac": frac,
                "reason": (f"limit traced along {len(pts)} boundary points "
                           f"({len(chain)} segment{'s' if len(chain) != 1 else ''}); "
                           f"white line found at {frac:.0%} of stations"
                           + (f", {line_w:.1f} px wide" if line_w else
                              " -- using the surface edge"))})
    return res


def signed_margin(points: np.ndarray, poly: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Signed pixel distance to the limit. Positive = on the track side."""
    p = np.asarray(points, float).reshape(-1, 2)
    d = ((p[:, None, :] - poly[None, :, :]) ** 2).sum(axis=2)
    i = np.argmin(d, axis=1)
    return np.einsum("ij,ij->i", p - poly[i], normals[i])


# --------------------------------------------------------------------------
# 4. tyres against the limit
# --------------------------------------------------------------------------


def judge_car(car_id: int, box, evidence, carpix: np.ndarray, hsv: np.ndarray,
              limit: dict, cfg: AnalyzeConfig, scale: float) -> CarResult:
    """Locate a car's tyres and decide which side of the line each is on."""
    h, w = carpix.shape
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    sub = carpix[y1:y2, x1:x2] > 0
    full_box = tuple(float(v) / scale for v in box)

    def result(verdict, reason, wheels=(), worst=None, out=0, meas=0, mmpp=None, trust=0.0):
        return CarResult(car_id, full_box, list(evidence), list(wheels), verdict, reason,
                         worst, out, meas, mmpp, trust)

    if (sub.size == 0 or sub.sum() < 20) and limit.get("fields") is not None:
        # tyres recoloured (an annotated frame) or in deep glare: fall back to
        # the car's silhouette -- everything in the box that is not track
        asphalt = limit["fields"][2][y1:y2, x1:x2] > 0
        sub = ~asphalt
        sub = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_OPEN, _kernel(5)) > 0
    if sub.size == 0 or sub.sum() < 20:
        return result(REVIEW, "car found, but no tyre mass inside it to measure")

    cols = np.flatnonzero(sub.sum(axis=0) > max(2, 0.03 * bh))
    if len(cols) < 4:
        return result(REVIEW, "tyres could not be separated from the car body")

    # near axle: the lowest band of dark mass; far axle: the band above it
    rows = np.flatnonzero(sub.any(axis=1))
    ybot, ytop = rows.max(), rows.min()
    span = max(ybot - ytop, 1)
    tyres = []
    for name, (lo, hi) in (("near", (0.62, 1.00)), ("far", (0.30, 0.62))):
        r0, r1 = ytop + int(lo * span), ytop + int(hi * span)
        band = sub[r0:r1 + 1]
        cc = np.flatnonzero(band.sum(axis=0) > max(1, 0.15 * band.shape[0]))
        if len(cc) < 2:
            continue
        left, right = cc.min(), cc.max()
        tw = max(2, int(0.20 * (right - left)))
        for side, c0, c1 in (("L", left, left + tw), ("R", right - tw, right)):
            col = sub[r0:r1 + 1, c0:c1 + 1]
            ys = np.flatnonzero(col.any(axis=1))
            if len(ys) == 0:
                continue
            yb = r0 + ys.max()
            tyres.append((f"{name}-{side}", x1 + c0, x1 + c1, y1 + yb,
                          (right - left)))
    if len(tyres) < cfg.min_tyres:
        return result(REVIEW, f"only {len(tyres)} tyre contact(s) located -- need "
                              f"{cfg.min_tyres} to judge the car")

    near = [t for t in tyres if t[0].startswith("near")]
    axle_px = (near[0][4] if near else tyres[0][4]) / scale
    mmpp = cfg.car_width_mm / max(axle_px, 1.0)

    if not limit.get("ok"):
        return result(NO_REFERENCE, "track limit not established in this frame -- "
                                    "the car cannot be judged against it",
                      mmpp=mmpp)

    fields = limit.get("fields")
    if fields is None:
        return result(NO_REFERENCE, "no track surface to judge the car against", mmpp=mmpp)
    dist_track, dist_off, asphalt = fields[:3]
    colour_off, shade = (fields[3], fields[4]) if len(fields) > 4 else (None, None)
    # tyre width in WORK pixels: the unit the masks are in
    tyre_px = max(3.0, cfg.tyre_width_mm / mmpp * scale)
    tol = max(2.0, 0.25 * tyre_px)

    wheels, out = [], 0
    best = []
    for name, c0, c1, yb, _ in tyres:
        # Look at the ground right at and just below the contact patch -- the
        # tyre hides the surface it sits on, but not the surface beside it.
        xs = np.clip(np.linspace(c0, c1, 9).astype(int), 0, w - 1)
        ys = np.clip(np.array([yb, yb + 2, yb + 4, yb + 7]), 0, h - 1)
        grid = dist_track[np.ix_(ys, xs)]
        per_x = grid.min(axis=0)                 # best depth for each column
        within = per_x <= tol
        if within.all():
            state = INSIDE
        elif within.any():
            state = ON_LINE
        else:
            state = OUTSIDE
            out += 1
        if state == OUTSIDE and colour_off is not None:
            # "not asphalt" is not the same as "off the track": deep shade is
            # neither asphalt nor kerb.  Require the ground just beyond the
            # contact patch to be POSITIVELY off-track -- coloured kerb,
            # run-off, grass, white kerb stripe -- or the tyre is unmeasurable.
            # Ground is sampled BELOW the car's own box so the car's body can
            # never be mistaken for the surface.
            r = int(max(4, 0.6 * tyre_px))
            ya = min(h - 1, max(yb + 1, y2 + 1))
            yb2 = min(h, ya + 2 * r)
            xa, xb2 = max(0, c0 - r), min(w, c1 + r)
            if yb2 - ya >= 3 and xb2 - xa >= 3:
                kn = float(colour_off[ya:yb2, xa:xb2].sum())
                dk = float(shade[ya:yb2, xa:xb2].sum())
                if dk > kn:
                    state = UNKNOWN
                    out -= 1
        k = int(np.argmin(per_x))
        if state == OUTSIDE:
            m_px = -float(per_x.min())
        elif state == UNKNOWN:
            m_px = -float(per_x.min())
        else:
            m_px = float(dist_off[int(ys[0]), int(xs[k])])
        best.append(m_px)
        cx = float((c0 + c1) / 2) / scale
        wheels.append(Wheel(name, cx, float(yb) / scale, m_px / scale,
                            m_px / scale * mmpp, state))

    unknown = sum(1 for wl in wheels if wl.state == UNKNOWN)
    deepest = max(best)                          # the car's best-placed tyre
    deepest_mm = deepest / scale * mmpp
    measured = len(wheels)
    integ = limit.get("integrity", 0.0)
    trust = float(min(1.0, measured / 4.0 + 0.25) * integ / 100.0)

    if unknown and out + unknown == measured:
        v = REVIEW
        why = (f"{out}/{measured} tyres beyond the line and {unknown} sitting on "
               "ground that cannot be read (deep shade) -- cannot confirm a violation")
    elif out == measured:
        v = VIOLATION
        why = (f"all {measured} measured tyres fully beyond the line -- nearest "
               f"is {abs(deepest_mm):.0f} mm outside")
    elif any(wl.state == ON_LINE for wl in wheels) and not any(wl.state == INSIDE for wl in wheels):
        v = BORDERLINE
        why = (f"{out}/{measured} tyres beyond, but a tyre is still touching the "
               f"line ({deepest_mm:.0f} mm of it inside) -- legal, borderline")
    elif out > 0 and deepest_mm < cfg.borderline_mm:
        v = BORDERLINE
        why = (f"{out}/{measured} tyres beyond the line; best-placed tyre only "
               f"{deepest_mm:.0f} mm inside -- borderline")
    else:
        v = CLEAR
        why = (f"{measured - out}/{measured} tyres on the track side "
               f"(best {deepest_mm:.0f} mm inside)")

    if v in (VIOLATION, BORDERLINE) and integ < cfg.integrity_review:
        why = (f"{why}. But the line's integrity in this frame is {integ:.0f}/100, "
               "so the reference cannot be trusted -- escalated for review")
        v = REVIEW
    return result(v, why, wheels, deepest_mm, out, measured, mmpp, trust)


def surface_fields(track: np.ndarray, hsv: np.ndarray, line_w: Optional[float],
                   carpix: np.ndarray, cars: list, cfg: AnalyzeConfig):
    """Distance fields a tyre is judged against.

    ``on_track`` is the asphalt plus the painted line bordering it -- the line
    is part of the track, so a tyre on the paint is on the track.  Car pixels
    are treated as UNKNOWN rather than off-track: the surface under a car is
    hidden, not absent.

    Returns ``(dist_to_track, dist_to_known_off_track, asphalt)`` in work px.
    """
    h, w = track.shape
    S, V = hsv[..., 1], hsv[..., 2]
    on_track = track.copy()
    if line_w:
        track_v = V[track > 0]
        vmin = max(cfg.paint_v_min, int(np.percentile(track_v, 92))) if track_v.size else cfg.paint_v_min
        white = ((V >= vmin) & (S <= cfg.paint_s_max)).astype(np.uint8) * 255
        reach = int(max(3, round(1.25 * line_w)))
        ring = cv2.subtract(cv2.dilate(track, _kernel(2 * reach + 1)), track)
        on_track = cv2.bitwise_or(on_track, cv2.bitwise_and(white, ring))
    on_track = cv2.morphologyEx(on_track, cv2.MORPH_CLOSE, _kernel(5))
    dist_track = cv2.distanceTransform(cv2.bitwise_not(on_track), cv2.DIST_L2, 3)

    car_zone = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2, _ in cars:
        car_zone[max(y1, 0):y2, max(x1, 0):x2] = carpix[max(y1, 0):y2, max(x1, 0):x2]
    car_zone = cv2.dilate(car_zone, _kernel(7))
    off_known = cv2.bitwise_and(cv2.bitwise_not(on_track), cv2.bitwise_not(car_zone))
    dist_off = cv2.distanceTransform(cv2.bitwise_not(off_known), cv2.DIST_L2, 3)
    L, chroma = lab_parts(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    colour_off = (chroma >= cfg.edge_outside_chroma) | (L >= cfg.edge_outside_l)
    shade = (L < cfg.asphalt_l_min) & (chroma < cfg.asphalt_chroma_max * 1.5)
    return dist_track, dist_off, track, colour_off, shade


# --------------------------------------------------------------------------
# 5. one frame
# --------------------------------------------------------------------------


_RANK = {CLEAR: 0, NO_REFERENCE: 1, BORDERLINE: 2, REVIEW: 3, VIOLATION: 4}


def analyze_frame(frame: np.ndarray, index: int = 0, t_s: float = 0.0,
                  cfg: Optional[AnalyzeConfig] = None, annotate: bool = True,
                  id_map: Optional[dict] = None) -> FrameResult:
    """Everything about one frame: limit, line condition, cars, verdicts."""
    cfg = cfg or AnalyzeConfig()
    t0 = time.perf_counter()
    if frame is None or frame.ndim != 3:
        raise ValueError("analyze_frame: expected a BGR image")
    H, W = frame.shape[:2]
    scale = min(1.0, cfg.work_width / float(W))
    small = cv2.resize(frame, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    surface = track_surface(small, cfg)
    cars, carpix = find_cars(small, hsv, surface, cfg)
    track, why = main_track(surface, [c[:4] for c in cars], cfg)
    if track is not None and cars:
        # a car is on or beside the track; black lettering on an advertising
        # board has a black core too, and sits nowhere near the asphalt
        away = cv2.distanceTransform(cv2.bitwise_not(track), cv2.DIST_L2, 3)
        near = []
        for c in cars:
            x1, y1, x2, y2 = (int(v) for v in c[:4])
            # a car well over the limit can sit a full car-width off the
            # asphalt, so the allowance scales with the car, not the frame
            lim = min(max(cfg.car_track_dist_frac * small.shape[1],
                          1.3 * max(x2 - x1, y2 - y1)),
                      cfg.car_track_dist_cap * small.shape[1])
            yb0 = max(y1, y2 - max(2, (y2 - y1) // 4))
            sub = away[max(yb0, 0):max(y2, yb0 + 1), max(x1, 0):max(x2, x1 + 1)]
            if sub.size and float(sub.min()) <= lim:
                near.append(c)
        cars = near

    limit = {"ok": False, "reason": why}
    if track is not None:
        limit = find_limit(small, hsv, track, cars, cfg)
        limit["fields"] = surface_fields(track, hsv, limit.get("line_w"),
                                         carpix, cars, cfg)

    results = []
    for k, c in enumerate(sorted(cars, key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))):
        cid = id_map.get(k, k + 1) if id_map else k + 1
        results.append(judge_car(cid, c[:4], c[4], carpix, hsv, limit, cfg, scale))

    verdict = max((r.verdict for r in results), key=lambda v: _RANK[v], default=(
        CLEAR if limit.get("ok") else NO_REFERENCE))
    if not results and limit.get("ok"):
        verdict = CLEAR

    fr = FrameResult(
        index=index, t_s=t_s, width=W, height=H,
        line_found=bool(limit.get("ok")), line_reason=limit.get("reason", ""),
        integrity=limit.get("integrity"), contrast=limit.get("contrast"),
        continuity=limit.get("continuity"), sharpness=limit.get("sharpness"),
        contamination=limit.get("contamination"),
        line_width_px=(limit["line_w"] / scale if limit.get("line_w") else None),
        cars=results, verdict=verdict, ms=(time.perf_counter() - t0) * 1000.0,
        polyline=(limit["poly"] / scale if limit.get("ok") else None))
    if annotate:
        fr.annotated = draw(frame, fr, limit, scale, track)
    return fr


def draw(frame: np.ndarray, fr: FrameResult, limit: dict, scale: float,
         track: Optional[np.ndarray]) -> np.ndarray:
    """The frame as a steward would want it annotated."""
    H, W = frame.shape[:2]
    vis = frame.copy()
    th = max(2, W // 400)
    if track is not None:
        tm = cv2.resize(track, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        tint = vis.copy()
        tint[tm] = (0.6 * tint[tm] + 0.4 * np.array([120, 90, 30])).astype(np.uint8)
        vis = cv2.addWeighted(tint, 0.55, vis, 0.45, 0)
    if fr.polyline is not None:
        pts = np.round(fr.polyline).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), th * 4, cv2.LINE_AA)
        cv2.polylines(vis, [pts], False, (255, 255, 0), th * 2, cv2.LINE_AA)
    for c in fr.cars:
        col = _C[c.verdict]
        x1, y1, x2, y2 = (int(v) for v in c.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, th * 2)
        label = f"#{c.car_id} {c.verdict}"
        if c.worst_margin_mm is not None:
            label += f"  {c.worst_margin_mm:+.0f}mm"
        fs = max(0.5, W / 1600)
        (tw, tth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - tth - 12)), (x1 + tw + 10, y1), col, -1)
        cv2.putText(vis, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), 2, cv2.LINE_AA)
        for wl in c.wheels:
            p = (int(wl.x), int(wl.y))
            cv2.circle(vis, p, th * 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, p, th * 3, _C[wl.state], -1, cv2.LINE_AA)
            if wl.margin_mm is not None:
                cv2.putText(vis, f"{wl.margin_mm:+.0f}", (p[0] + 6, p[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.4, W / 2600),
                            _C[wl.state], 2, cv2.LINE_AA)
    # banner
    band = max(34, H // 18)
    cv2.rectangle(vis, (0, 0), (W, band), (0, 0, 0), -1)
    integ = "--" if fr.integrity is None else f"{fr.integrity:.0f}"
    txt = (f"FRAME {fr.index}   {fr.verdict}   INTEGRITY {integ}   "
           f"CARS {len(fr.cars)}")
    cv2.putText(vis, txt, (12, int(band * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.55, W / 1500), _C.get(fr.verdict, (255, 255, 255)), 2, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------
# 6. a whole video, frame by frame
# --------------------------------------------------------------------------


def iter_frames(path: str, every: int = 1) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """(index, t_s, frame, fps) for an image, a folder or a video."""
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path)
                       if os.path.splitext(f)[1].lower() in IMAGE_EXT)
        for i, f in enumerate(files):
            im = cv2.imread(os.path.join(path, f), cv2.IMREAD_COLOR)
            if im is not None:
                yield i, float(i), im, 1.0
        return
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"could not read image {path!r}")
        yield 0, 0.0, im, 1.0
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1 else 25.0
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % max(1, every) == 0:
            yield i, i / fps, im, fps
        i += 1
    cap.release()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class _Tracker:
    """Keeps car ids stable across frames by box overlap."""

    def __init__(self, iou: float):
        self.iou = iou
        self.tracks: dict[int, tuple] = {}
        self.next = 1

    def assign(self, boxes: list) -> dict:
        ids, used = {}, set()
        for k, b in enumerate(boxes):
            best, best_iou = None, self.iou
            for tid, tb in self.tracks.items():
                if tid in used:
                    continue
                v = _iou(b, tb)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = self.next
                self.next += 1
            ids[k] = best
            used.add(best)
            self.tracks[best] = b
        return ids


def analyze(path: str, cfg: Optional[AnalyzeConfig] = None, every: int = 1,
            out_dir: Optional[str] = "output", on_frame=None,
            keep_annotated: int = 400) -> dict:
    """Analyse every frame of an image, folder or video and write a report.

    Returns the report dict.  ``on_frame(FrameResult)`` is called as frames
    finish, for a live UI.
    """
    cfg = cfg or AnalyzeConfig()
    t_start = time.perf_counter()
    tracker = _Tracker(cfg.track_iou)
    frames: list[FrameResult] = []
    annotated: dict[int, np.ndarray] = {}
    fps_src = 1.0

    for idx, t_s, im, fps in iter_frames(path, every):
        fps_src = fps
        # first pass without ids to get boxes, then stable ids, then judge
        pre = analyze_frame(im, idx, t_s, cfg, annotate=False)
        order = sorted(range(len(pre.cars)), key=lambda k: -(
            (pre.cars[k].box[2] - pre.cars[k].box[0]) * (pre.cars[k].box[3] - pre.cars[k].box[1])))
        ids = tracker.assign([pre.cars[k].box for k in order])
        for pos, k in enumerate(order):
            pre.cars[k].car_id = ids[pos]
        limit_poly = pre.polyline
        pre.annotated = draw(im, pre, {}, 1.0, None) if limit_poly is not None or pre.cars else draw(im, pre, {}, 1.0, None)
        frames.append(pre)
        if len(annotated) < keep_annotated or pre.verdict in (VIOLATION, REVIEW, BORDERLINE):
            annotated[idx] = pre.annotated
        if on_frame is not None:
            on_frame(pre)

    wall = time.perf_counter() - t_start
    report = build_report(path, frames, fps_src, wall, cfg)
    report["_annotated"] = annotated
    if out_dir:
        report["output_dir"] = write_outputs(report, frames, annotated, out_dir, path)
    return report


def build_report(path: str, frames: list, fps: float, wall_s: float,
                 cfg: AnalyzeConfig) -> dict:
    """Every number a judge will ask for, computed from the frames."""
    n = len(frames)
    integ = [f.integrity for f in frames if f.integrity is not None]
    counts = {v: 0 for v in (CLEAR, BORDERLINE, VIOLATION, REVIEW, NO_REFERENCE)}
    car_counts = {v: 0 for v in counts}
    ids = set()
    margins = []
    for f in frames:
        counts[f.verdict] += 1
        for c in f.cars:
            car_counts[c.verdict] += 1
            ids.add(c.car_id)
            if c.worst_margin_mm is not None:
                margins.append(c.worst_margin_mm)

    # events: runs of VIOLATION / REVIEW frames per car
    per_car: dict[int, list] = {}
    for f in frames:
        for c in f.cars:
            per_car.setdefault(c.car_id, []).append((f, c))
    events = []
    for cid, seq in per_car.items():
        run = []
        for f, c in seq:
            hit = c.verdict in (VIOLATION, REVIEW)
            if hit and run and f.index - run[-1][0].index > cfg.event_gap_frames * max(1, 1):
                events.append(_event(cid, run, fps))
                run = []
            if hit:
                run.append((f, c))
            elif run and f.index - run[-1][0].index > cfg.event_gap_frames:
                events.append(_event(cid, run, fps))
                run = []
        if run:
            events.append(_event(cid, run, fps))
    events.sort(key=lambda e: e["entry_frame"])

    violations = sum(1 for e in events if e["verdict"] == VIOLATION)
    reviews = sum(1 for e in events if e["verdict"] == REVIEW)
    decided = counts[CLEAR] + counts[BORDERLINE] + counts[VIOLATION]
    judged = decided + counts[REVIEW]
    return {
        "source": os.path.abspath(path),
        "source_name": os.path.basename(os.path.normpath(path)),
        "analysed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frames_analysed": n,
        "source_fps": round(fps, 2),
        "duration_s": round(n / fps, 2) if fps > 1 else None,
        "wall_time_s": round(wall_s, 2),
        "throughput_fps": round(n / wall_s, 1) if wall_s > 0 else None,
        "mean_ms_per_frame": round(float(np.mean([f.ms for f in frames])), 1) if frames else None,
        "line_found_frames": sum(1 for f in frames if f.line_found),
        "line_found_pct": round(100.0 * sum(1 for f in frames if f.line_found) / max(n, 1), 1),
        "integrity_mean": round(float(np.mean(integ)), 1) if integ else None,
        "integrity_p05": round(float(np.percentile(integ, 5)), 1) if integ else None,
        "integrity_min": round(float(np.min(integ)), 1) if integ else None,
        "integrity_below_50_frames": sum(1 for v in integ if v < cfg.integrity_review),
        "cars_detected_total": sum(len(f.cars) for f in frames),
        "unique_cars": len(ids),
        "frame_verdicts": counts,
        "car_verdicts": car_counts,
        "violation_events": violations,
        "review_events": reviews,
        "auto_resolved_pct": round(100.0 * decided / judged, 1) if judged else None,
        "margin_mm_min": round(float(np.min(margins)), 0) if margins else None,
        "margin_mm_median": round(float(np.median(margins)), 0) if margins else None,
        "violation_frames": [f.index for f in frames if f.verdict == VIOLATION],
        "borderline_frames": [f.index for f in frames if f.verdict == BORDERLINE],
        "review_frames": [f.index for f in frames if f.verdict == REVIEW],
        "events": events,
        "per_frame": [f.to_dict() for f in frames],
        "method": {
            "track_limit": "outer edge of white line, fitted on asphalt/paint boundary",
            "cars": "tyre mass + surface occlusion + YOLO(any class) evidence",
            "rule": "violation only when every measured tyre is fully beyond the line",
            "integrity": "absolute, per frame: contrast, continuity, sharpness, contamination",
            "scale": f"from car width {cfg.car_width_mm:.0f} mm (first-order)",
            "gating": f"violation with integrity < {cfg.integrity_review:.0f} -> REVIEW REQUIRED",
        },
    }


def _event(cid: int, run: list, fps: float) -> dict:
    fr0, fr1 = run[0][0], run[-1][0]
    worst = min(run, key=lambda fc: fc[1].worst_margin_mm if fc[1].worst_margin_mm is not None else 1e9)
    verdict = VIOLATION if any(c.verdict == VIOLATION for _, c in run) else REVIEW
    ints = [f.integrity for f, _ in run if f.integrity is not None]
    return {
        "car_id": cid, "verdict": verdict,
        "entry_frame": fr0.index, "exit_frame": fr1.index,
        "frames": len(run),
        "duration_ms": round((fr1.index - fr0.index + 1) * 1000.0 / fps, 0) if fps > 1 else None,
        "peak_frame": worst[0].index,
        "peak_margin_mm": worst[1].worst_margin_mm,
        "tyres_out_max": max(c.tyres_out for _, c in run),
        "integrity_mean": round(float(np.mean(ints)), 1) if ints else None,
        "trust": round(float(np.mean([c.trust for _, c in run])), 2),
        "reason": worst[1].reason,
    }


def write_outputs(report: dict, frames: list, annotated: dict, out_dir: str,
                  path: str) -> str:
    """report.json, frames.csv, every violation/review frame as PNG, a sheet."""
    stem = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    d = os.path.join(out_dir, f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(os.path.join(d, "report.json"), "w") as fh:
        json.dump(clean, fh, indent=1, default=str)
    with open(os.path.join(d, "frames.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "t_s", "verdict", "integrity", "contrast", "continuity",
                     "sharpness", "contamination", "line_found", "cars",
                     "violations", "worst_margin_mm", "ms"])
        for f in frames:
            margins = [c.worst_margin_mm for c in f.cars if c.worst_margin_mm is not None]
            wr.writerow([f.index, round(f.t_s, 3), f.verdict,
                         None if f.integrity is None else round(f.integrity, 1),
                         None if f.contrast is None else round(f.contrast, 1),
                         None if f.continuity is None else round(f.continuity, 1),
                         None if f.sharpness is None else round(f.sharpness, 1),
                         None if f.contamination is None else round(f.contamination, 1),
                         f.line_found, len(f.cars),
                         sum(1 for c in f.cars if c.verdict == VIOLATION),
                         None if not margins else round(min(margins), 0), round(f.ms, 1)])
    flagged = [f for f in frames if f.verdict in (VIOLATION, REVIEW, BORDERLINE)]
    for f in flagged:
        im = annotated.get(f.index)
        if im is not None:
            cv2.imwrite(os.path.join(d, f"{f.verdict.split()[0].lower()}_frame{f.index:05d}.png"), im)
    # contact sheet: event peak frames first, else first frames
    peaks = [e["peak_frame"] for e in report["events"]] or [f.index for f in frames[:6]]
    tiles = [cv2.resize(annotated[i], (640, int(640 * frames[0].height / frames[0].width)))
             for i in peaks[:9] if i in annotated]
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(os.path.join(d, "summary.png"),
                    np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return d


def _print_report(r: dict) -> None:
    line = "=" * 78
    print(f"\n{line}\nCHRONOS ANALYSIS  ·  {r['source_name']}\n{line}")
    print(f"frames analysed      {r['frames_analysed']}   (source {r['source_fps']} fps"
          + (f", {r['duration_s']} s" if r['duration_s'] else "") + ")")
    print(f"processing           {r['wall_time_s']} s  ·  {r['throughput_fps']} fps  ·  "
          f"{r['mean_ms_per_frame']} ms/frame")
    print(f"track limit found    {r['line_found_frames']}/{r['frames_analysed']} frames "
          f"({r['line_found_pct']}%)")
    print(f"line integrity       mean {r['integrity_mean']}  ·  p05 {r['integrity_p05']}  ·  "
          f"min {r['integrity_min']}  ·  <50 on {r['integrity_below_50_frames']} frames")
    print(f"cars                 {r['cars_detected_total']} detections  ·  "
          f"{r['unique_cars']} unique")
    fv = r["frame_verdicts"]
    print(f"frame verdicts       VIOLATION {fv[VIOLATION]}  ·  REVIEW {fv[REVIEW]}  ·  "
          f"BORDERLINE {fv[BORDERLINE]}  ·  CLEAR {fv[CLEAR]}  ·  NO REF {fv[NO_REFERENCE]}")
    print(f"events               {r['violation_events']} violation  ·  "
          f"{r['review_events']} review  ·  auto-resolved {r['auto_resolved_pct']}%")
    if r["margin_mm_min"] is not None:
        print(f"margin               deepest {r['margin_mm_min']:+.0f} mm  ·  "
              f"median {r['margin_mm_median']:+.0f} mm")
    if r["violation_frames"]:
        vf = r["violation_frames"]
        print(f"violation frames     {vf[:30]}{' ...' if len(vf) > 30 else ''}")
    for e in r["events"][:12]:
        pm = "" if e["peak_margin_mm"] is None else f"{e['peak_margin_mm']:+.0f} mm"
        print(f"  car #{e['car_id']:<3} {e['verdict']:16s} frames {e['entry_frame']}-"
              f"{e['exit_frame']}  peak f{e['peak_frame']}  {pm}  "
              f"integrity {e['integrity_mean']}  trust {e['trust']:.0%}")
        print(f"      {e['reason']}")
    for f in r["per_frame"][:10] if r["frames_analysed"] <= 10 else []:
        print(f"  frame {f['index']}: {f['verdict']}  integrity "
              f"{'--' if f['integrity'] is None else round(f['integrity'])}  "
              f"line: {f['line_reason'][:70]}")
        for c in f["cars"]:
            print(f"      car #{c['car_id']} {c['verdict']}: {c['reason'][:110]}")
    if r.get("output_dir"):
        print(f"\noutputs -> {r['output_dir']}/  (report.json, frames.csv, flagged PNGs, summary.png)")


def main() -> None:
    p = argparse.ArgumentParser(description="CHRONOS -- analyse an image, folder or video")
    p.add_argument("path")
    p.add_argument("--every", type=int, default=1, help="analyse every Nth frame")
    p.add_argument("--out", default="output")
    p.add_argument("--no-yolo", action="store_true")
    a = p.parse_args()
    cfg = AnalyzeConfig(use_yolo=not a.no_yolo)
    r = analyze(a.path, cfg, a.every, a.out)
    _print_report(r)


if __name__ == "__main__":
    main()

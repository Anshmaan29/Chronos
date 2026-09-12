"""Track-limit boundary on REAL footage, using pretrained segmentation.

The colour-keyed detector in :mod:`chronos.boundary` works on the synthetic
benchmark and fails on real trackside footage, for two reasons that are
structural rather than tunable:

  * a red car defeats a red-keyed kerb detector -- on the Miami frame the kerb
    mask locked onto the Ferrari, and 79% of the returned polyline lay inside
    a car box;
  * "low saturation and mid brightness" describes asphalt, barriers,
    grandstands and overcast sky equally well, so the drivable mask grew to
    53% of the frame.

So the road is found by a model that was trained to find roads, and nothing
here is trained or fine-tuned -- pretrained checkpoints, used as published.

Pipeline::

    1. segment the drivable surface   (SAM2, or Segformer/Cityscapes 'road')
    2. subtract every car box         so a car cannot corrupt the mask
    3. take the outer contour on the track-limit side
    4. snap to the paint              local Otsu in a narrow band, outer edge
    5. fit a smooth polyline

Two things are deliberately NOT colour-keyed:

  * **which side is the track limit** is decided by where the paint actually
    is, measured, not by a hue;
  * **kerbs** are found by the PERIODICITY of their stripes -- a strong
    oscillation in luminance along the track edge -- which works on orange,
    red, blue or green kerbs alike.

The result is a :class:`chronos.boundary.BoundaryResult`, so
:mod:`chronos.integrity` consumes it unchanged: it measures the paint region
and does not care how the region was found.

Fails loudly.  An implausible road mask, or a band with no paint in it,
returns ``ok=False`` with the reason -- never a polyline.  That behaviour is
the point and is not negotiable: a system whose job is to say when its
reference cannot be established must refuse rather than guess.

STATUS, 2026-09-12
------------------
This module has NOT yet been fairly tested.  The footage it was first run on
was broadcast and press photography -- long lens, low angle, panning -- and
most of it contained no visible track-limit line at all.  Failing on frames
that do not show the reference is correct behaviour, not evidence about the
method, so that run is recorded as an INVALID CAMERA-CONDITION TEST rather
than as a result.

The open question is still whether the classical detector in
:mod:`chronos.boundary` suffices on a fixed elevated view with the white line
plainly in shot.  That is the evaluation to run first; another segmentation
model is not warranted until it has been.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import cv2
import numpy as np

from chronos.boundary import (BoundaryConfig, BoundaryResult, _circular_runs,
                              _fill_holes, _kernel, _principal_frame,
                              _resample_polyline, _smooth_polyline,
                              _turn_blocked, detect_boundary,
                              point_to_polyline_distance)

Backend = Literal["sam2", "segformer", "auto"]


@dataclass
class RealBoundaryConfig:
    """Tunables for the learned-segmentation boundary detector."""

    # --- which model ------------------------------------------------------
    backend: Backend = "auto"
    sam2_model: str = "facebook/sam2-hiera-small"
    segformer_model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    device: str = "cpu"
    seg_max_side: int = 1024          # segment at this size, then upscale

    # --- SAM2 prompting ---------------------------------------------------
    # The best prompt available is a car: a car is ON the racing surface by
    # definition, so a point just under its contact patches is guaranteed to be
    # the thing we want segmented.  "Lower centre of frame" is only a fallback,
    # and on a tight trackside crop it lands on astroturf as often as asphalt.
    prompt_under_cars: bool = True
    car_prompt_drop_px: float = 6.0    # below the box bottom, onto the surface
    prompt_points: tuple = ((0.50, 0.93), (0.38, 0.86), (0.62, 0.86),
                            (0.50, 0.78))
    sam2_multimask: bool = True

    # --- cars -------------------------------------------------------------
    car_dilate_px: int = 11           # margin around a car box before removal

    # --- plausibility of the road mask ------------------------------------
    min_road_frac: float = 0.04
    max_road_frac: float = 0.70
    min_contour_points: int = 40

    # --- contour -> chains -------------------------------------------------
    border_margin_px: int = 4
    turn_window_px: int = 9
    turn_split_deg: float = 110.0
    min_chain_points: int = 30

    # --- snapping to the paint --------------------------------------------
    paint_band_out_px: float = 22.0   # how far OUTSIDE the contour to look
    paint_band_in_px: float = 10.0    # ...and a little inside, since a road
                                      # model may stop short of the paint
    paint_step_px: float = 0.5
    paint_min_width_px: float = 1.0
    paint_max_width_px: float = 26.0
    paint_min_contrast: float = 0.10  # Michelson, paint against the road
    min_paint_stations: float = 0.35  # fraction of stations that must find paint

    # --- kerb by periodicity ----------------------------------------------
    kerb_probe_out_px: tuple = (6.0, 14.0, 24.0, 36.0)
    kerb_min_power: float = 0.18      # share of spectral power in the peak
    kerb_min_cycles: float = 4.0      # over the sampled length
    kerb_band_px: float = 46.0        # thickness of the kerb mask we emit

    # --- shape plausibility -------------------------------------------------
    # A track limit is a long smooth curve.  A car silhouette is a short closed
    # loop that turns through a full circle, and the paint-snap will happily
    # find "paint" along bright bodywork -- 98% of stations at contrast 0.75 on
    # a real frame.  Brightness cannot arbitrate that; shape can.
    max_total_turn_deg: float = 170.0   # cumulative |turning| along the line
    min_endpoint_ratio: float = 0.40    # end-to-end distance / arc length
    min_arc_px: float = 120.0           # shorter than this decides nothing
    max_inside_car_frac: float = 0.15
    """PROVISIONAL -- tuned, not validated.

    A track limit lies on the road, not on a car: a car crossing the line hides
    a small part of it, whereas a line that IS the car's underside is mostly
    inside the box.  The principle is sound; the NUMBER is not established.

    0.15 was chosen on 2026-09-12 by watching it cut false passes from 10/20 to
    3/20 on ONE clip -- broadcast footage that turned out not to contain a
    visible track-limit line at all, so that clip was an invalid test of the
    camera condition this system is for.  Treat this as a placeholder until it
    is set on fixed elevated footage where the line is actually present, and do
    not quote it as a validated threshold.
    """
    shape_points: int = 24              # the turn test is about GROSS shape, so
                                        # it is measured on a decimated copy --
                                        # summing per-station snap jitter over
                                        # 200 points reaches thousands of
                                        # degrees and means nothing

    # --- output ------------------------------------------------------------
    n_points: int = 200
    smooth_window: int = 9
    smooth_passes: int = 3   # snapping is per-station and jitters; the paint
                             # edge itself does not
    debug_dir: str = "debug"
    debug_name: str = "boundary_real.jpg"


# --------------------------------------------------------------------------
# segmentation backends
# --------------------------------------------------------------------------


class _Sam2Road:
    """SAM2, point-prompted on the road.  Pretrained, never fine-tuned."""

    def __init__(self, cfg: RealBoundaryConfig):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.p = SAM2ImagePredictor.from_pretrained(cfg.sam2_model,
                                                    device=cfg.device)
        self.cfg = cfg

    def road(self, rgb: np.ndarray,
             car_boxes: Optional[Sequence[Sequence[float]]] = None
             ) -> tuple[np.ndarray, str]:
        cfg = self.cfg
        h, w = rgb.shape[:2]
        self.p.set_image(rgb)
        pts = None
        origin = "frame fractions"
        if cfg.prompt_under_cars and car_boxes is not None and len(car_boxes):
            under = []
            for x1, y1, x2, y2 in (b[:4] for b in car_boxes):
                cx = int(round(0.5 * (x1 + x2)))
                cy = int(round(y2 + cfg.car_prompt_drop_px))
                if 0 <= cx < w and 0 <= cy < h:
                    under.append([cx, cy])
            if under:
                pts = np.array(under)
                origin = f"{len(under)} point(s) under detected car(s)"
        if pts is None:
            pts = np.array([[int(fx * w), int(fy * h)] for fx, fy in cfg.prompt_points])
        best, best_note, best_score = None, "", -1.0
        # one prompt at a time, then all together: a single point can grab a
        # kerb or a shadow, and the union of agreeing masks is steadier
        trials = [(pts[i:i + 1], np.array([1])) for i in range(len(pts))]
        trials.append((pts, np.ones(len(pts), int)))
        for coords, labels in trials:
            masks, scores, _ = self.p.predict(point_coords=coords,
                                              point_labels=labels,
                                              multimask_output=cfg.sam2_multimask)
            for m, s in zip(np.atleast_3d(masks), np.atleast_1d(scores)):
                m = m.astype(bool)
                if m.ndim != 2:
                    continue
                frac = float(m.mean())
                if not (cfg.min_road_frac <= frac <= cfg.max_road_frac):
                    continue
                if float(s) > best_score:
                    best, best_score = m, float(s)
                    best_note = (f"SAM2 {len(coords)}-point prompt ({origin}), "
                                 f"score {s:.2f}, {frac:.0%} of frame")
        if best is None:
            return np.zeros((h, w), np.uint8), "SAM2 returned no mask of a plausible size"
        return best.astype(np.uint8) * 255, best_note


class _SegformerRoad:
    """Segformer trained on Cityscapes.  Class 0 is 'road'."""

    ROAD_CLASS = 0

    def __init__(self, cfg: RealBoundaryConfig):
        import torch
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
        self.torch = torch
        self.proc = SegformerImageProcessor.from_pretrained(cfg.segformer_model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.segformer_model).to(cfg.device).eval()
        self.cfg = cfg

    def road(self, rgb: np.ndarray, car_boxes=None) -> tuple[np.ndarray, str]:
        torch = self.torch
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.cfg.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        up = torch.nn.functional.interpolate(
            logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy()
        mask = (pred == self.ROAD_CLASS).astype(np.uint8) * 255
        return mask, f"Segformer/Cityscapes road class, {mask.mean()/255:.0%} of frame"


_BACKENDS: dict = {}


def get_backend(cfg: RealBoundaryConfig) -> tuple[object, str]:
    """Load a segmentation backend, caching it.  Tries SAM2 first for 'auto'."""
    order = [cfg.backend] if cfg.backend in ("sam2", "segformer") else ["sam2", "segformer"]
    errors = []
    for name in order:
        if name in _BACKENDS:
            return _BACKENDS[name], name
        try:
            obj = _Sam2Road(cfg) if name == "sam2" else _SegformerRoad(cfg)
            _BACKENDS[name] = obj
            return obj, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no segmentation backend available -- " + " | ".join(errors))


# --------------------------------------------------------------------------
# paint snapping
# --------------------------------------------------------------------------


def _otsu_1d(values: np.ndarray) -> float:
    """Otsu's threshold on a 1-D profile.

    Local, not global: the brightness of paint against asphalt changes down
    the length of a corner with exposure, shadow and distance, and one fixed
    number for the whole frame is the thing that broke on real footage.
    """
    v = np.asarray(values, np.float64)
    if v.size < 4 or v.max() - v.min() < 1e-6:
        return float(v.max() + 1.0)
    hist, edges = np.histogram(v, bins=32)
    p = hist / max(hist.sum(), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / np.where(denom > 1e-12, denom, np.nan)
    k = int(np.nanargmax(sigma_b))
    return float(centres[k])


def snap_to_paint(gray: np.ndarray, points: np.ndarray, outward: np.ndarray,
                  cfg: RealBoundaryConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move each contour point onto the OUTER edge of the painted line.

    At every station a 1-D brightness profile is taken across the boundary,
    from a little inside the road mask to a little outside it.  Otsu splits
    that profile -- and only that profile -- into paint and not-paint.  The
    painted run nearest the contour is the edge line, and its outermost sample
    is the track limit.

    Returns ``(snapped_points, found_flags, diagnostics)``.
    """
    h, w = gray.shape
    depths = np.arange(-cfg.paint_band_in_px, cfg.paint_band_out_px + 1e-6,
                       cfg.paint_step_px)
    prof = np.zeros((len(points), len(depths)), np.float32)
    for j, d in enumerate(depths):
        q = points + d * outward
        xi = np.clip(q[:, 0].astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].astype(int), 0, h - 1)
        prof[:, j] = gray[yi, xi]

    snapped = points.copy()
    found = np.zeros(len(points), bool)
    widths, contrasts = [], []
    for i in range(len(points)):
        row = prof[i]
        thr = _otsu_1d(row)
        bright = row >= thr
        if not bright.any() or bright.all():
            continue
        dark_level = float(np.median(row[~bright])) if (~bright).any() else 0.0
        bright_level = float(np.median(row[bright]))
        contrast = (bright_level - dark_level) / max(bright_level + dark_level, 1e-6)
        if contrast < cfg.paint_min_contrast:
            continue

        # runs of bright samples; keep the one straddling or nearest the contour
        idx = np.flatnonzero(np.diff(np.concatenate([[0], bright.view(np.int8), [0]])))
        runs = [(idx[k], idx[k + 1] - 1) for k in range(0, len(idx), 2)]
        zero = int(np.argmin(np.abs(depths)))
        best = min(runs, key=lambda r: 0 if r[0] <= zero <= r[1]
                   else min(abs(r[0] - zero), abs(r[1] - zero)))
        width = (best[1] - best[0] + 1) * cfg.paint_step_px
        if not (cfg.paint_min_width_px <= width <= cfg.paint_max_width_px):
            continue
        snapped[i] = points[i] + depths[best[1]] * outward[i]
        found[i] = True
        widths.append(width)
        contrasts.append(contrast)

    diag = {"paint_frac": float(found.mean()) if len(found) else 0.0,
            "paint_width_px": float(np.median(widths)) if widths else 0.0,
            "paint_contrast": float(np.median(contrasts)) if contrasts else 0.0}
    return snapped, found, diag


# --------------------------------------------------------------------------
# kerb by periodicity -- no colour key at all
# --------------------------------------------------------------------------


def split_band_contour(contour: np.ndarray, mask: np.ndarray,
                       cfg: RealBoundaryConfig) -> list[np.ndarray]:
    """Cut a closed region outline into its two long edges.

    A segmentation mask is a closed region, so its contour is a closed loop and
    turns through a full circle no matter what it encloses.  Splitting on
    curvature alone does not reliably break that -- a smooth blob has no sharp
    corner to split at, and the "edge" that comes back is still a loop.

    A road is a band, so it has two ends and two long sides.  Projecting the
    outline onto the region's own principal axis finds the ends directly, and
    cutting there leaves exactly the two sides.
    """
    n = len(contour)
    centroid, axis, _ = _principal_frame(mask)
    t = (contour - centroid) @ axis
    a, b = int(np.argmin(t)), int(np.argmax(t))
    lo, hi = (a, b) if a < b else (b, a)
    chains = [np.arange(lo, hi + 1), np.concatenate([np.arange(hi, n), np.arange(0, lo + 1)])]
    return [c for c in chains if len(c) >= cfg.min_chain_points]


def shape_is_a_boundary(poly: np.ndarray, cfg: RealBoundaryConfig
                        ) -> tuple[bool, str, dict]:
    """Does this polyline have the shape of a track limit at all?

    Two geometric facts separate an edge line from a car outline, and neither
    involves colour or brightness:

      * a track limit turns gently -- tens of degrees over its whole length --
        while a silhouette turns through most of a circle;
      * a track limit goes somewhere: its endpoints are far apart relative to
        the distance travelled.  A loop comes back to where it started.
    """
    d_all = np.diff(poly, axis=0)
    arc = float(np.linalg.norm(d_all, axis=1).sum())
    diag = {"arc_px": arc, "turn_deg": 0.0, "endpoint_ratio": 0.0}
    if arc < cfg.min_arc_px or len(poly) < 8:
        return False, f"boundary is only {arc:.0f} px long", diag

    coarse = _resample_polyline(np.asarray(poly, np.float32),
                               min(cfg.shape_points, len(poly))).astype(np.float64)
    d = np.diff(coarse, axis=0)
    seg = np.linalg.norm(d, axis=1)
    t = d / np.maximum(seg[:, None], 1e-9)
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    dot = (t[:-1] * t[1:]).sum(axis=1)
    turn = float(np.degrees(np.abs(np.arctan2(cross, dot)).sum()))
    ratio = float(np.linalg.norm(poly[-1] - poly[0]) / max(arc, 1e-9))
    diag.update({"turn_deg": turn, "endpoint_ratio": ratio})

    if turn > cfg.max_total_turn_deg:
        return False, (f"the line turns through {turn:.0f}deg over its length "
                       f"(limit {cfg.max_total_turn_deg:.0f}deg) -- that is a "
                       "silhouette, not a track edge"), diag
    if ratio < cfg.min_endpoint_ratio:
        return False, (f"the line comes back on itself (end-to-end distance is "
                       f"{ratio:.0%} of its {arc:.0f} px length) -- that is a "
                       "closed outline, not a track edge"), diag
    return True, "", diag


def _periodicity_along(gray: np.ndarray, poly: np.ndarray,
                        cfg: RealBoundaryConfig) -> tuple[bool, dict]:
    """Is this polyline running along a striped kerb rather than a line?

    An edge line is uniform along its length.  A kerb is, by construction, a
    repeating pattern.  Sampling brightness straight down the candidate
    boundary separates the two without knowing either one's colour.
    """
    h, w = gray.shape
    n = len(poly)
    diag = {"power": 0.0, "cycles": 0.0}
    if n < 32:
        return False, diag
    xi = np.clip(poly[:, 0].astype(int), 0, w - 1)
    yi = np.clip(poly[:, 1].astype(int), 0, h - 1)
    sig = gray[yi, xi].astype(np.float64)
    sig = sig - cv2.GaussianBlur(sig.reshape(-1, 1), (1, 31), 0).reshape(-1)
    if np.std(sig) < 2.0:
        return False, diag
    spec = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    spec[:max(2, int(cfg.kerb_min_cycles))] = 0.0
    if spec.sum() <= 0:
        return False, diag
    k = int(np.argmax(spec))
    power = float(spec[k] / spec.sum())
    diag = {"power": power, "cycles": float(k)}
    return power >= cfg.kerb_min_power, diag


def detect_kerb_periodicity(gray: np.ndarray, points: np.ndarray,
                            outward: np.ndarray, cfg: RealBoundaryConfig
                            ) -> tuple[np.ndarray, dict]:
    """Find a kerb by the RHYTHM of its stripes rather than their colour.

    Every kerb in motorsport is striped, and the stripes repeat at a roughly
    constant pitch.  Walking along just outside the track limit and looking at
    the brightness signal, a kerb shows up as a strong single frequency; grass,
    asphalt and gravel do not.  Colour never enters, so this works on the
    orange and white of Miami as well as the red and white of Spa.

    Returns ``(kerb_mask, diagnostics)``.  The mask is empty when no periodic
    band is found -- that is a real answer, not a failure.
    """
    h, w = gray.shape
    best = {"power": 0.0, "cycles": 0.0, "offset": 0.0, "pitch_px": 0.0}
    n = len(points)
    if n < 32:
        return np.zeros((h, w), np.uint8), best

    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total_len = float(seg[-1])

    for off in cfg.kerb_probe_out_px:
        q = points + off * outward
        xi = np.clip(q[:, 0].astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].astype(int), 0, h - 1)
        sig = gray[yi, xi].astype(np.float64)
        sig = sig - cv2.GaussianBlur(sig.reshape(-1, 1), (1, 31), 0).reshape(-1)
        if np.std(sig) < 2.0:
            continue
        sig *= np.hanning(n)
        spec = np.abs(np.fft.rfft(sig)) ** 2
        spec[:max(2, int(cfg.kerb_min_cycles))] = 0.0     # ignore slow drift
        if spec.sum() <= 0:
            continue
        k = int(np.argmax(spec))
        power = float(spec[k] / spec.sum())
        if power > best["power"]:
            best = {"power": power, "cycles": float(k), "offset": float(off),
                    "pitch_px": total_len / max(k, 1)}

    mask = np.zeros((h, w), np.uint8)
    if best["power"] >= cfg.kerb_min_power and best["cycles"] >= cfg.kerb_min_cycles:
        inner = points + 2.0 * outward
        outer = points + cfg.kerb_band_px * outward
        poly = np.concatenate([inner, outer[::-1]], axis=0)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
    return mask, best


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def detect_boundary_real(frame: np.ndarray,
                         cfg: Optional[RealBoundaryConfig] = None,
                         car_boxes: Optional[Sequence[Sequence[float]]] = None,
                         save_debug: bool = True) -> BoundaryResult:
    """Detect the track limit on real footage with pretrained segmentation.

    Parameters
    ----------
    frame:      BGR uint8.
    car_boxes:  (x1, y1, x2, y2) per detected car.  Strongly recommended --
                removing them is what stops a car corrupting the road mask.

    Returns
    -------
    BoundaryResult, so ``chronos.integrity`` consumes it unchanged.
    ``ok=False`` with a reason whenever the road mask is implausible or no
    paint is found.  Never returns a polyline it cannot justify.
    """
    cfg = cfg or RealBoundaryConfig()
    if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
        raise ValueError("detect_boundary_real: expected a BGR uint8 frame")
    h, w = frame.shape[:2]

    def fail(reason: str, road=None, kerb=None) -> BoundaryResult:
        res = BoundaryResult(False, reason, drivable_mask=road, kerb_mask=kerb)
        if save_debug:
            res.debug_path = save_debug_image(frame, res, cfg)
        return res

    # ---- 1. segment the road -------------------------------------------
    scale = min(1.0, cfg.seg_max_side / max(h, w))
    small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    try:
        backend, name = get_backend(cfg)
        scaled = None
        if car_boxes is not None and len(car_boxes):
            scaled = [[v * scale for v in b[:4]] for b in car_boxes]
        road_small, note = backend.road(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scaled)
    except Exception as exc:
        return fail(f"segmentation backend unavailable: {type(exc).__name__}: {exc}")
    road = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST) \
        if road_small.shape[:2] != (h, w) else road_small
    if not road.any():
        return fail(f"{name}: {note}", road=road)

    # ---- 2. remove the cars --------------------------------------------
    if car_boxes is not None and len(car_boxes):
        cars = np.zeros((h, w), np.uint8)
        for box in car_boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
            cv2.rectangle(cars, (max(x1, 0), max(y1, 0)),
                          (min(x2, w), min(y2, h)), 255, -1)
        cars = cv2.dilate(cars, _kernel(cfg.car_dilate_px))
        road = cv2.bitwise_and(road, cv2.bitwise_not(cars))

    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, _kernel(5))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(road, 8)
    if n_lab <= 1:
        return fail(f"{name}: road mask empty after removing cars", road=road)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    road = _fill_holes((labels == biggest).astype(np.uint8) * 255)

    frac = float((road > 0).mean())
    if not (cfg.min_road_frac <= frac <= cfg.max_road_frac):
        return fail(f"{name}: road mask covers {frac:.0%} of the frame, outside the "
                    f"plausible {cfg.min_road_frac:.0%}-{cfg.max_road_frac:.0%}",
                    road=road)

    # ---- 3. outer contour, split into edges -----------------------------
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return fail("road region has no contour", road=road)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < cfg.min_contour_points:
        return fail(f"road contour has only {len(contour)} points", road=road)

    m = cfg.border_margin_px
    on_border = ((contour[:, 0] <= m) | (contour[:, 0] >= w - 1 - m)
                 | (contour[:, 1] <= m) | (contour[:, 1] >= h - 1 - m))
    # split the closed outline into its two long sides, then drop whatever
    # runs along the image border -- that is a crop, not a track limit
    chains = []
    for chain in split_band_contour(contour, road, cfg):
        keep = chain[~on_border[chain]]
        if len(keep) >= cfg.min_chain_points:
            # keep the longest unbroken run, so a chain cut by the border in
            # the middle does not get stitched across the gap
            breaks = np.flatnonzero(np.diff(keep) != 1)
            for part in np.split(keep, breaks + 1):
                if len(part) >= cfg.min_chain_points:
                    chains.append(part)
    if not chains:
        return fail("road outline never breaks into separate edges "
                    "(it may be clipped by the frame on every side)", road=road)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    centroid, _, _ = _principal_frame(road)

    # ---- 4. the track limit is the edge that actually has paint on it ----
    best = None
    tried = []
    for ci, ch in enumerate(chains):
        pts = _resample_polyline(contour[ch].astype(np.float32),
                                 min(cfg.n_points, len(ch))).astype(np.float64)
        if len(pts) < 16:
            continue
        t = np.gradient(pts, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        outward = np.stack([-t[:, 1], t[:, 0]], axis=1)
        # orient away from the road
        probe = pts + 6.0 * outward
        xi = np.clip(probe[:, 0].astype(int), 0, w - 1)
        yi = np.clip(probe[:, 1].astype(int), 0, h - 1)
        if (road[yi, xi] > 0).mean() > 0.5:
            outward = -outward

        snapped, found, diag = snap_to_paint(gray, pts, outward, cfg)
        tried.append((ci, diag["paint_frac"], diag["paint_contrast"]))
        score = diag["paint_frac"] * max(diag["paint_contrast"], 0.0)
        if best is None or score > best[0]:
            best = (score, pts, outward, snapped, found, diag)

    if best is None:
        return fail("no road edge long enough to test for paint", road=road)
    score, pts, outward, snapped, found, diag = best

    if diag["paint_frac"] < cfg.min_paint_stations:
        detail = ", ".join(f"edge {c}: paint at {f:.0%}, contrast {k:.2f}"
                           for c, f, k in tried)
        return fail(f"no painted line found in the search band -- {detail}. "
                    f"Need paint at {cfg.min_paint_stations:.0%} of stations.",
                    road=road)

    # ---- 5. smooth polyline through the snapped points -------------------
    keep = snapped[found]
    if len(keep) < 16:
        return fail(f"only {len(keep)} stations snapped to paint", road=road)
    poly = _resample_polyline(keep.astype(np.float32), min(cfg.n_points, len(keep)))
    for _ in range(cfg.smooth_passes):
        poly = _smooth_polyline(poly, cfg.smooth_window)

    if car_boxes is not None and len(car_boxes):
        inside = np.zeros(len(poly), bool)
        for x1, y1, x2, y2 in (b[:4] for b in car_boxes):
            inside |= ((poly[:, 0] >= x1) & (poly[:, 0] <= x2)
                       & (poly[:, 1] >= y1) & (poly[:, 1] <= y2))
        if inside.mean() > cfg.max_inside_car_frac:
            return fail(f"{inside.mean():.0%} of the line lies inside a car box -- "
                        "this is the car's own edge or its shadow, not the track "
                        "limit", road=road)

    shape_ok, shape_why, shape_diag = shape_is_a_boundary(poly, cfg)
    if not shape_ok:
        return fail(f"{shape_why}. Paint was found at {diag['paint_frac']:.0%} of "
                    "stations, so brightness alone would have accepted this.",
                    road=road)

    # A kerb's white stripes are paint too, and they are brighter and wider
    # than an edge line.  If the band we snapped into is itself periodic, we
    # have locked onto the kerb, not the track limit -- fail rather than
    # report a confident wrong line.
    on_kerb, on_kerb_diag = _periodicity_along(gray, poly, cfg)
    if on_kerb:
        return fail(
            f"the line we snapped to is periodic along its length "
            f"(power {on_kerb_diag['power']:.2f}, {on_kerb_diag['cycles']:.0f} cycles) "
            "-- that is a striped kerb, not an edge line", road=road)

    # The paint is part of the drivable surface -- the limit is its OUTER edge
    # -- so the road mask has to reach the snapped line.  Without this the mask
    # stops short, and everything downstream that measures inward from the
    # boundary (track geometry, the integrity track frame) has nothing to
    # measure into.
    ribbon = np.concatenate([pts, snapped[::-1]], axis=0)
    cv2.fillPoly(road, [np.round(ribbon).astype(np.int32)], 255)
    road = _fill_holes(road)

    kerb, kerb_diag = detect_kerb_periodicity(gray, pts, outward, cfg)

    reason = (f"{name}: {note}; paint at {diag['paint_frac']:.0%} of stations, "
              f"width {diag['paint_width_px']:.1f} px, contrast {diag['paint_contrast']:.2f}; "
              f"shape ok (turn {shape_diag['turn_deg']:.0f}deg, "
              f"end-to-end {shape_diag['endpoint_ratio']:.0%}); "
              + (f"kerb by periodicity (power {kerb_diag['power']:.2f}, "
                 f"pitch {kerb_diag['pitch_px']:.0f} px)" if kerb.any()
                 else "no periodic kerb found"))
    res = BoundaryResult(True, reason, polyline=poly, drivable_mask=road,
                         kerb_mask=kerb)
    res.agreement_px = None
    if save_debug:
        res.debug_path = save_debug_image(frame, res, cfg, extra=diag)
    return res


# --------------------------------------------------------------------------


def save_debug_image(frame: np.ndarray, res: BoundaryResult,
                     cfg: RealBoundaryConfig, extra: Optional[dict] = None) -> str:
    """2x2 panel: overlay, road mask, kerb by periodicity, extracted boundary."""
    h, w = frame.shape[:2]

    def label(img, text, color=(255, 255, 255)):
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, text[:110], (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        return img

    overlay = frame.copy()
    if res.polyline is not None and len(res.polyline) > 1:
        cv2.polylines(overlay, [np.round(res.polyline).astype(np.int32)], False,
                      (0, 255, 0), 2, cv2.LINE_AA)
    label(overlay, "GREEN = track limit (learned road + paint snap)"
          if res.ok else f"FAILED: {res.reason[:80]}",
          (255, 255, 255) if res.ok else (0, 0, 255))

    def mask_view(mask, color, text):
        view = (frame * 0.4).astype(np.uint8)
        if mask is not None and mask.any():
            view[mask > 0] = (0.45 * view[mask > 0] + 0.55 * np.array(color)).astype(np.uint8)
        return label(view, text)

    road = mask_view(res.drivable_mask, (255, 210, 60), "road, segmented (cars removed)")
    kerb = mask_view(res.kerb_mask, (60, 60, 255),
                     "kerb by stripe periodicity -- no colour key")

    edge = np.zeros_like(frame)
    if res.drivable_mask is not None:
        edge[res.drivable_mask > 0] = (55, 55, 55)
    if res.polyline is not None and len(res.polyline) > 1:
        cv2.polylines(edge, [np.round(res.polyline).astype(np.int32)], False,
                      (0, 255, 0), 2, cv2.LINE_AA)
    txt = "extracted boundary"
    if extra:
        txt += f" -- paint {extra['paint_frac']:.0%} of stations, {extra['paint_width_px']:.1f}px"
    edge = label(edge, txt)

    panel = np.vstack([np.hstack([overlay, road]), np.hstack([kerb, edge])])
    panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)
    os.makedirs(cfg.debug_dir, exist_ok=True)
    path = os.path.join(cfg.debug_dir, cfg.debug_name)
    if not cv2.imwrite(path, panel):
        raise RuntimeError(f"save_debug_image: could not write {path!r}")
    return path


def detect(frame: np.ndarray, detector: str = "auto",
           car_boxes=None, cfg: Optional[RealBoundaryConfig] = None,
           bcfg: Optional[BoundaryConfig] = None,
           save_debug: bool = True) -> BoundaryResult:
    """Dispatch between the classical and the learned detector.

    ``auto`` runs the classical one first -- it is 90 ms and exact on the
    synthetic benchmark -- and falls back to segmentation when its own
    plausibility checks say the answer cannot be trusted.
    """
    if detector == "synthetic":
        return detect_boundary(frame, bcfg or BoundaryConfig(), save_debug=save_debug)
    if detector == "real":
        return detect_boundary_real(frame, cfg, car_boxes, save_debug)
    if detector == "lane":
        from chronos.lane import LaneConfig, detect_lane
        return detect_lane(frame, LaneConfig(), save_debug=save_debug)
    if detector != "auto":
        raise ValueError(f"detect: unknown detector {detector!r}; expected "
                         "'real', 'lane', 'synthetic' or 'auto'")

    classical = detect_boundary(frame, bcfg or BoundaryConfig(), save_debug=False)
    if classical.ok and classical_is_plausible(classical, frame, car_boxes):
        # Plausible SHAPE is not enough. Ask the one question that decides
        # whether a polyline is a track limit: is there paint under it?
        from chronos.integrity import IntegrityConfig, boundary_paint_contrast
        icfg = IntegrityConfig()
        c, pl, al = boundary_paint_contrast(frame, classical, icfg)
        if c >= icfg.min_baseline_contrast:
            return classical
        classical.reason += (f" | rejected: no paint under it (paint {pl:.0f}, "
                             f"asphalt {al:.0f}, Michelson {c:+.3f})")

    # The ridge detector sits between the two on purpose. It is the one that
    # actually works on real trackside footage -- 71% of frames on the Miami
    # clip, median paint/asphalt contrast +0.21 -- and it costs milliseconds,
    # where the learned backend costs a model download and hundreds of them.
    from chronos.lane import LaneConfig, detect_lane
    lane = detect_lane(frame, LaneConfig(), save_debug=False)
    if lane.ok:
        return lane

    learned = detect_boundary_real(frame, cfg, car_boxes, save_debug)
    if not learned.ok:
        extra = []
        if classical.ok:
            extra.append("classical rejected: " + classical.reason[:50])
        extra.append("ridge rejected: " + lane.reason[:70])
        learned.reason += " | " + " | ".join(extra)
    return learned


def classical_is_plausible(res: BoundaryResult, frame: np.ndarray,
                           car_boxes=None) -> bool:
    """The same checks ``tools/real_run.py`` applies, reused for dispatch."""
    if not res.ok or res.polyline is None:
        return False
    if res.agreement_px is not None and res.agreement_px > 40.0:
        return False
    if res.drivable_mask is not None:
        road = float((res.drivable_mask > 0).mean())
        if not (0.03 <= road <= 0.45):
            return False
    if car_boxes is not None and len(car_boxes):
        pts = res.polyline
        inside = np.zeros(len(pts), bool)
        for x1, y1, x2, y2 in (b[:4] for b in car_boxes):
            inside |= ((pts[:, 0] >= x1) & (pts[:, 0] <= x2)
                       & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
        if inside.mean() > 0.30:
            return False
    return True


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS boundary on real footage")
    p.add_argument("--image", required=True)
    p.add_argument("--detector", default="real", choices=["real", "synthetic", "auto"])
    p.add_argument("--backend", default="auto", choices=["auto", "sam2", "segformer"])
    p.add_argument("--debug-name", default="boundary_real.jpg")
    args = p.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image!r}")

    boxes = None
    try:
        from chronos.car import CarConfig, YoloDetector
        boxes, _ = YoloDetector(CarConfig()).detect(frame)
        print(f"cars     : {len(boxes)} boxes removed from the road mask")
    except Exception as exc:
        print(f"cars     : none ({type(exc).__name__}) -- mask not protected")

    cfg = RealBoundaryConfig(backend=args.backend, debug_name=args.debug_name)
    res = detect(frame, args.detector, boxes, cfg)
    print(f"ok       : {res.ok}")
    print(f"reason   : {res.reason}")
    print(f"debug    : {res.debug_path}")
    if res.ok:
        print(f"polyline : {len(res.polyline)} points")


if __name__ == "__main__":
    _cli()

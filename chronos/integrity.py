"""CHRONOS Module 2 -- the Boundary Integrity Engine.  This IS the project.

Every track-limits system measures the car against the line.  This measures the
line.  It returns one number, 0-100, plus the four components it is made of:

    contrast       line luminance against the asphalt beside it
    continuity     how much of the line is still findable
    sharpness      gradient magnitude across the paint edge
    contamination  share of line pixels no longer the colour of paint

Low integrity must be able to block an automatic decision even when the car is
seen perfectly.  That is the contribution; the score only earns it by being
defensible number by number.

--------------------------------------------------------------------------
Why continuity is measured against a session baseline, not against 100%
--------------------------------------------------------------------------
Module 1 recovers 67-98% of the true boundary on a CLEAN frame, and how much
depends on the scene, not on its condition: the far field is one or two pixels
wide, and a pale run-off costs more of it than grass does.  Reporting that
shortfall as degradation would mean calling a perfectly clean corner degraded,
and the first judge to ask "is 67% dirty or just far away?" would be right.

So a baseline is captured once per session on a clean frame, and every
component is reported RELATIVE to what that scene achieved when clean.
Degradation is loss against what was achievable here, not against a
theoretical ideal.  A clean frame therefore scores ~100 by construction --
which is exactly what "we measured this corner when it was clean" should mean.

The absolute anchors in :class:`IntegrityConfig` keep that honest in the other
direction: a corner whose line was already poor at baseline cannot score 100
just because it has not got worse.  Each component takes the WORSE of its
absolute and its relative reading.

Measurements are taken along the BASELINE geometry, not along the current
detection.  If contamination stops the detector finding the line at all, we
still measure the paint where the line is known to be -- otherwise the
measurement would quietly disappear exactly when it matters most.

Run standalone::

    python -m chronos.integrity --image data/frame.jpg
    python -m chronos.integrity --image data/frame.jpg --gt data/frame_gt.json \\
        --degrade rubber --levels 0,0.3,0.6,0.9
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from chronos.boundary import (BoundaryConfig, BoundaryResult, detect_boundary,
                              point_to_polyline_distance, _resample_polyline)
from chronos.degrade import TrackFrame, build_track_frame

# --------------------------------------------------------------------------
# CONFIG -- tune these live.  No magic numbers below this block.
# --------------------------------------------------------------------------


@dataclass
class IntegrityConfig:
    """Tunables for the integrity score."""

    # --- where the paint is ----------------------------------------------
    n_samples: int = 220              # stations along the boundary
    probe_max_px: int = 34            # how far inward to look for the paint edge
    paint_min_px: float = 1.2         # thinnest paint band we will trust
    paint_max_px: float = 14.0
    asphalt_gap: tuple[float, float] = (2.2, 5.0)   # multiples of paint width
    asphalt_min_px: tuple[float, float] = (9.0, 22.0)
    min_valid_samples: int = 25       # fewer than this and the score is refused
    min_baseline_contrast: float = 0.12
    """Michelson contrast the baseline frame must show between the paint band
    and the asphalt band beside it.

    This is a check that the boundary is on a LINE at all, not a check on the
    line's quality.  A track-limit line is bright paint against dark asphalt
    -- that is what makes it a reference -- so if the two bands come back at
    the same luminance, the polyline is not lying along paint and every
    reading taken from it is meaningless.

    Measured: the synthetic clean frames sit at 0.49. On real trackside
    frames where the detector went wrong the readings are 0.002 (paint 153 /
    asphalt 153), 0.023 (203 / 196) and -0.281 (paint 134 / asphalt 207 --
    the "asphalt" band brighter than the "paint"). Without this gate those
    three produced a confident-looking integrity of 36, which is worse than
    refusing: it is the false-confident failure this engine exists to
    prevent, committed by the engine itself."""

    # --- absolute anchors: what a good line looks like regardless of scene -
    # The absolute anchors are a guard, not a quality grade: they exist so a
    # corner whose line was ALREADY bad at baseline cannot score 100 merely by
    # not getting worse.  They sit at the bottom of the clean envelope measured
    # across the 17 benchmark scenes (contrast 0.43-0.55, sharpness 154-366),
    # because lens softness and sun angle are properties of the camera and the
    # day, not of the line -- charging them as degradation is the same mistake
    # as charging a scene for its far field.  Clean frames land 95-100.
    contrast_target: float = 0.45     # Michelson, fresh paint on dark asphalt
    sharpness_target: float = 160.0   # mean Sobel magnitude across the edge
    absolute_anchors: bool = True
    """Whether the absolute anchors above are allowed to cap the score.

    They are calibrated on the synthetic corner cam, whose clean frames read
    paint 185 against asphalt 61 -- Michelson 0.49. No real camera reaches
    that: a correctly detected line on real trackside footage measures
    around 0.13, which the anchor turns into a contrast sub-score near zero
    and a total near 30 on a line that is in perfectly good condition.

    Reporting that as degradation would be the same error this engine exists
    to prevent, so on footage the anchors were not calibrated for they are
    switched off and the score is reported as RELATIVE TO THE SESSION
    BASELINE only -- which is what it was always principally measuring. The
    console labels such a reading so the two can never be confused, because
    a relative 90 and an absolute 90 are different claims."""
    # NOTE there is deliberately no absolute anchor on CONTINUITY.  Achievable
    # coverage is set by the scene's geometry -- far field, line width in
    # pixels, run-off colour -- and ranges 67-98% on clean frames.  Anchoring it
    # would report a clean grey-run-off corner as permanently degraded, which is
    # the one confound this engine must not have.  Continuity is purely relative
    # to the session baseline.

    # --- what still counts as paint --------------------------------------
    white_sat_max: int = 70
    white_val_frac: float = 0.62      # of the baseline paint luminance

    # --- response shaping (calibration lives here) ------------------------
    # Fitted against the calibration targets over all 17 scenes at once
    # (clean 90+, light rubber 0.3 -> 70-80, Miami-level 0.6 -> ~41,
    # heavy 0.9 -> under 25), with monotonicity required for all six kinds.
    contrast_gamma: float = 3.4
    continuity_gamma: float = 1.5
    sharpness_gamma: float = 3.0
    contamination_gamma: float = 1.45
    contamination_full: float = 0.60  # contaminated fraction that reads as zero

    # --- how the four combine ---------------------------------------------
    weights: tuple[float, float, float, float] = (0.26, 0.08, 0.30, 0.36)
                                      # contrast, continuity, sharpness,
                                      # contamination.  Continuity carries least
                                      # weight because Module 1 keeps finding the
                                      # line well past the point where the line
                                      # is worth trusting -- robust detection is
                                      # exactly why detection alone cannot be the
                                      # confidence signal.
    combine: str = "geometric"        # geometric = one collapsed component
                                      # drags the total down, which is the
                                      # behaviour the gating claim needs
    floor: float = 0.02               # keeps the geometric mean finite

    # --- verdict thresholds (Module 5 will consume these) -----------------
    trust_threshold: float = 60.0     # below this, do not issue a verdict
    alert_threshold: float = 75.0

    debug_dir: str = "debug"
    debug_name: str = "integrity_debug.jpg"


@dataclass
class BoundaryBaseline:
    """What this corner looked like when it was clean.

    Captured once at the start of a session.  Everything afterwards is
    measured against it, so "degraded" means "worse than this corner was",
    not "worse than a textbook".

    Attributes
    ----------
    reference:   (N, 2) the reference boundary geometry.
    track:       TrackFrame along the reference, reused for every later frame.
    paint_px:    (N,) measured paint width in pixels at each station.
    valid:       (N,) stations where the paint was actually resolvable.
    coverage:    fraction of the reference Module 1 recovered when clean --
                 the achievable coverage in THIS scene.
    contrast, sharpness, contamination: the clean readings.
    white_val:   luminance cut separating paint from not-paint, from the clean
                 frame.
    """

    reference: np.ndarray
    track: TrackFrame
    paint_px: np.ndarray
    valid: np.ndarray
    coverage: float
    contrast: float
    sharpness: float
    contamination: float
    white_val: float
    paint_luma: float
    asphalt_luma: float


@dataclass
class IntegrityScore:
    """One integrity reading.

    ``total`` and the four components are all 0-100, higher is better.
    ``detail`` carries the raw physical measurements behind them, so the score
    can be defended rather than just quoted.
    """

    total: float
    contrast: float
    continuity: float
    sharpness: float
    contamination: float
    detail: dict = field(default_factory=dict)

    def explain(self) -> str:
        """The answer to 'what does Boundary Integrity 42 actually mean?'"""
        d = self.detail
        return (f"contrast {d['contrast_rel']:.0%} of baseline, "
                f"{1 - d['coverage_rel']:.0%} of the line length no longer "
                f"detectable, {d['contaminated_frac']:.0%} of line area "
                f"contaminated, edge sharpness {d['sharpness_rel']:.0%} of baseline")


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of a single-channel image at float coordinates."""
    return cv2.remap(img, pts[:, 0].astype(np.float32).reshape(-1, 1),
                     pts[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1)


def _measure_paint_width(gray: np.ndarray, tf: TrackFrame,
                         cfg: IntegrityConfig) -> tuple[np.ndarray, np.ndarray]:
    """Find how wide the painted band is at each station, in pixels.

    Walks inward from the track limit and takes the paint to end where the
    luminance has fallen half way from its peak to the asphalt behind it.
    Returns ``(paint_px, valid)``.
    """
    depths = np.arange(0.0, cfg.probe_max_px, 0.5)
    prof = np.stack([_sample(gray, tf.points + d * tf.normals) for d in depths], axis=1)

    peak = prof[:, :8].max(axis=1)
    far = prof[:, -6:].mean(axis=1)
    half = 0.5 * (peak + far)

    paint_px = np.full(len(tf.points), cfg.paint_min_px)
    for i in range(len(prof)):
        below = np.nonzero(prof[i] < half[i])[0]
        if len(below):
            paint_px[i] = depths[below[0]]
    paint_px = np.clip(paint_px, cfg.paint_min_px, cfg.paint_max_px)

    # a station is only usable if the paint actually stood out from the road
    valid = (peak - far) > 18.0
    return paint_px, valid


def _bands(tf: TrackFrame, paint_px: np.ndarray, cfg: IntegrityConfig):
    """Sampling depths for the paint band and the asphalt beside it."""
    paint_d = [0.3, 0.55, 0.8]
    g0 = np.maximum(paint_px * cfg.asphalt_gap[0], cfg.asphalt_min_px[0])
    g1 = np.maximum(paint_px * cfg.asphalt_gap[1], cfg.asphalt_min_px[1])
    return paint_d, g0, g1


def _luma_pair(gray: np.ndarray, tf: TrackFrame, paint_px: np.ndarray,
               cfg: IntegrityConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-station (paint luminance, adjacent asphalt luminance)."""
    paint_d, g0, g1 = _bands(tf, paint_px, cfg)
    paint = np.mean([_sample(gray, tf.points + (f * paint_px)[:, None] * tf.normals)
                     for f in paint_d], axis=0)
    asphalt = np.mean([_sample(gray, tf.points + d[:, None] * tf.normals)
                       for d in (g0, 0.5 * (g0 + g1), g1)], axis=0)
    return paint, asphalt


def _edge_sharpness(gray: np.ndarray, tf: TrackFrame) -> np.ndarray:
    """Peak gradient magnitude across the paint edge, per station."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    probes = [_sample(mag, tf.points + d * tf.normals)
              for d in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    return np.max(probes, axis=0)


def _contaminated_fraction(frame: np.ndarray, tf: TrackFrame, paint_px: np.ndarray,
                           white_val: float, cfg: IntegrityConfig) -> np.ndarray:
    """Per-station share of line pixels that are no longer paint-coloured."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hits = []
    for f in (0.2, 0.4, 0.6, 0.8):
        p = tf.points + (f * paint_px)[:, None] * tf.normals
        s, v = _sample(sat, p), _sample(val, p)
        hits.append(((s <= cfg.white_sat_max) & (v >= white_val)).astype(np.float32))
    return 1.0 - np.mean(hits, axis=0)


def _coverage(detection: Optional[np.ndarray], reference: np.ndarray,
              tol_px: float = 5.0) -> float:
    """Fraction of the reference line's LENGTH that a detection recovered."""
    if detection is None or len(detection) < 2:
        return 0.0
    ref = _resample_polyline(np.asarray(reference, np.float32), 400)
    return float((point_to_polyline_distance(ref, detection) <= tol_px).mean())


# --------------------------------------------------------------------------
# baseline capture
# --------------------------------------------------------------------------


def boundary_paint_contrast(frame: np.ndarray, result: BoundaryResult,
                            cfg: Optional[IntegrityConfig] = None
                            ) -> tuple[float, float, float]:
    """Michelson contrast of the paint band against the asphalt beside it.

    Pulled out of :func:`capture_baseline` so the detector chain can ask the
    same question BEFORE committing to a boundary. Without it the chain
    accepted whichever detector answered first, and a confident wrong answer
    beats a correct later one every time -- on the Red Bull Ring frame the
    classical detector returned a plausible-looking polyline along the
    asphalt edge and the ridge detector, which had the line right, was never
    consulted.

    Returns (contrast, paint luminance, asphalt luminance); (0, 0, 0) when
    the measurement cannot be made at all.
    """
    cfg = cfg or IntegrityConfig()
    try:
        if not result.ok or result.polyline is None:
            return 0.0, 0.0, 0.0
        tf = build_track_frame(result.polyline, result.drivable_mask,
                               result.kerb_mask, n=cfg.n_samples)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        paint_px, valid = _measure_paint_width(gray, tf, cfg)
        if int(valid.sum()) < cfg.min_valid_samples:
            return 0.0, 0.0, 0.0
        paint, asphalt = _luma_pair(gray, tf, paint_px, cfg)
        c = float(np.mean((paint[valid] - asphalt[valid])
                          / np.maximum(paint[valid] + asphalt[valid], 1e-6)))
        return c, float(np.mean(paint[valid])), float(np.mean(asphalt[valid]))
    except Exception:
        return 0.0, 0.0, 0.0


def capture_baseline(clean_frame: np.ndarray,
                     reference: Optional[np.ndarray] = None,
                     cfg: Optional[IntegrityConfig] = None,
                     boundary_cfg: Optional[BoundaryConfig] = None,
                     result: Optional[BoundaryResult] = None) -> BoundaryBaseline:
    """Measure this corner while it is clean.  Everything later is relative.

    Parameters
    ----------
    clean_frame:
        A BGR frame of the corner in known-good condition.
    reference:
        The reference geometry.  Pass ``scene.gt_boundary`` in the benchmark,
        where the truth exists; leave as None in the field and Module 1's own
        clean detection becomes the reference.
    result:
        A precomputed detection for ``clean_frame``, to avoid re-running it.

    Raises
    ------
    ValueError
        If the clean frame has no findable boundary, or the paint cannot be
        resolved at enough stations.  A baseline that quietly comes out wrong
        would poison every later reading, so this refuses instead.
    """
    cfg = cfg or IntegrityConfig()
    res = result or detect_boundary(clean_frame, boundary_cfg or BoundaryConfig(),
                                    save_debug=False)
    if not res.ok:
        raise ValueError(f"capture_baseline: no boundary in the clean frame ({res.reason})")

    ref = np.asarray(reference if reference is not None else res.polyline, np.float32)
    if len(ref) < 4:
        raise ValueError("capture_baseline: reference geometry has fewer than 4 points")

    tf = build_track_frame(ref, res.drivable_mask, res.kerb_mask, n=cfg.n_samples)
    gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    paint_px, valid = _measure_paint_width(gray, tf, cfg)
    if int(valid.sum()) < cfg.min_valid_samples:
        raise ValueError(
            f"capture_baseline: paint resolvable at only {int(valid.sum())} of "
            f"{len(valid)} stations (need {cfg.min_valid_samples}) -- the clean "
            "frame may not be clean, or the line is too far away to measure")

    paint, asphalt = _luma_pair(gray, tf, paint_px, cfg)
    contrast = float(np.mean((paint[valid] - asphalt[valid])
                             / np.maximum(paint[valid] + asphalt[valid], 1e-6)))
    if contrast < cfg.min_baseline_contrast:
        pl, al = float(np.mean(paint[valid])), float(np.mean(asphalt[valid]))
        raise ValueError(
            f"capture_baseline: the detected boundary is not on a painted edge "
            f"-- paint band reads {pl:.0f}, asphalt band {al:.0f}, Michelson "
            f"contrast {contrast:+.3f} (need {cfg.min_baseline_contrast:.2f}). "
            f"A track-limit line is bright paint against dark asphalt; these "
            f"two bands are the same material, so there is no line here to "
            f"measure against")

    sharpness = float(np.mean(_edge_sharpness(gray, tf)[valid]))
    white_val = float(np.mean(paint[valid]) * cfg.white_val_frac)
    contamination = float(np.mean(
        _contaminated_fraction(clean_frame, tf, paint_px, white_val, cfg)[valid]))
    coverage = _coverage(res.polyline, ref)

    return BoundaryBaseline(reference=ref, track=tf, paint_px=paint_px, valid=valid,
                            coverage=coverage, contrast=contrast, sharpness=sharpness,
                            contamination=contamination, white_val=white_val,
                            paint_luma=float(np.mean(paint[valid])),
                            asphalt_luma=float(np.mean(asphalt[valid])))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def _shape(relative: float, absolute: float, gamma: float) -> float:
    """Turn a ratio into a 0-100 sub-score, taking the worse of two readings."""
    return 100.0 * float(np.clip(min(relative, absolute), 0.0, 1.0) ** gamma)


def _occluded_stations(baseline: BoundaryBaseline, boxes) -> np.ndarray:
    """Stations hidden behind something -- in practice, a car on the line.

    A car covering the paint does not mean the paint has degraded; it means we
    cannot see it there.  Charging that to contrast and contamination reports a
    clean line as dirty every time a car crosses it, which is both wrong and
    exactly backwards: the moments that matter most are the ones with a car in
    them.
    """
    pts = baseline.track.points
    occ = np.zeros(len(pts), bool)
    if boxes is None:
        return occ
    for box in boxes:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        occ |= ((pts[:, 0] >= x1) & (pts[:, 0] <= x2)
                & (pts[:, 1] >= y1) & (pts[:, 1] <= y2))
    return occ


def measure_boundary(frame: np.ndarray, baseline: BoundaryBaseline,
                     cfg: Optional[IntegrityConfig] = None,
                     result: Optional[BoundaryResult] = None,
                     boundary_cfg: Optional[BoundaryConfig] = None,
                     occlusion_boxes=None) -> dict:
    """The physical readings, before any scoring opinion is applied.

    Kept separate from :func:`score_from_measurement` so the calibration of the
    0-100 mapping can be re-run over a whole sweep without re-measuring, and so
    the raw numbers can be quoted on their own when someone asks what the score
    is made of.
    """
    cfg = cfg or IntegrityConfig()
    if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
        raise ValueError("measure_boundary: expected a BGR uint8 frame")

    tf, paint_px = baseline.track, baseline.paint_px
    occluded = _occluded_stations(baseline, occlusion_boxes)
    valid = baseline.valid & ~occluded
    occluded_frac = float(occluded[baseline.valid].mean()) if baseline.valid.any() else 0.0
    if int(valid.sum()) < max(8, cfg.min_valid_samples // 3):
        # too much of the line is hidden to say anything about its condition
        valid = baseline.valid
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    paint, asphalt = _luma_pair(gray, tf, paint_px, cfg)
    contrast = float(np.mean((paint[valid] - asphalt[valid])
                             / np.maximum(paint[valid] + asphalt[valid], 1e-6)))
    sharpness = float(np.mean(_edge_sharpness(gray, tf)[valid]))
    contaminated = float(np.mean(
        _contaminated_fraction(frame, tf, paint_px, baseline.white_val, cfg)[valid]))

    res = result or detect_boundary(frame, boundary_cfg or BoundaryConfig(),
                                    save_debug=False)
    reference = baseline.reference
    if occluded.any() and not occluded.all():
        visible = baseline.track.points[~occluded]
        if len(visible) >= 4:
            reference = visible
    coverage = _coverage(res.polyline if res.ok else None, reference)

    return {"contrast_michelson": contrast, "sharpness_sobel": sharpness,
            "contaminated_frac": contaminated, "coverage": coverage,
            "paint_luma": float(np.mean(paint[valid])),
            "asphalt_luma": float(np.mean(asphalt[valid])),
            "detector_ok": bool(res.ok), "stations": int(valid.sum()),
            "stations_occluded_frac": occluded_frac}


def score_from_measurement(raw: dict, baseline: BoundaryBaseline,
                           cfg: Optional[IntegrityConfig] = None) -> IntegrityScore:
    """Map physical readings onto 0-100 sub-scores and a total."""
    cfg = cfg or IntegrityConfig()
    contrast = raw["contrast_michelson"]
    sharpness = raw["sharpness_sobel"]
    coverage = raw["coverage"]
    contaminated = raw["contaminated_frac"]

    contrast_rel = contrast / max(baseline.contrast, 1e-6)
    sharpness_rel = sharpness / max(baseline.sharpness, 1e-6)
    coverage_rel = coverage / max(baseline.coverage, 1e-6)

    if cfg.absolute_anchors:
        abs_contrast = contrast / cfg.contrast_target
        abs_sharp = sharpness / cfg.sharpness_target
    else:
        abs_contrast = abs_sharp = float("inf")   # relative-only
    s_contrast = _shape(contrast_rel, abs_contrast, cfg.contrast_gamma)
    s_sharp = _shape(sharpness_rel, abs_sharp, cfg.sharpness_gamma)
    # relative only -- see the note on continuity in IntegrityConfig
    s_cont = 100.0 * float(np.clip(coverage_rel, 0.0, 1.0) ** cfg.continuity_gamma)

    # contamination is already absolute -- it is a fraction of the line area --
    # but the clean frame's own reading is subtracted so JPEG speckle at
    # baseline is not charged as dirt
    excess = max(contaminated - baseline.contamination, 0.0)
    headroom = max(cfg.contamination_full - baseline.contamination, 1e-6)
    s_contam = 100.0 * float(np.clip(1.0 - excess / headroom, 0.0, 1.0) ** cfg.contamination_gamma)

    parts = np.array([s_contrast, s_cont, s_sharp, s_contam], float)
    w = np.asarray(cfg.weights, float)
    w = w / w.sum()
    if cfg.combine == "geometric":
        total = float(np.exp(np.sum(w * np.log(np.maximum(parts / 100.0, cfg.floor)))) * 100.0)
    else:
        total = float(np.sum(w * parts))

    detail = dict(raw)
    detail.update({"contrast_rel": contrast_rel, "sharpness_rel": sharpness_rel,
                   "coverage_rel": coverage_rel,
                   "coverage_baseline": baseline.coverage})
    return IntegrityScore(total=round(total, 1), contrast=round(s_contrast, 1),
                          continuity=round(s_cont, 1), sharpness=round(s_sharp, 1),
                          contamination=round(s_contam, 1), detail=detail)


def score_integrity(frame: np.ndarray, baseline: BoundaryBaseline,
                    cfg: Optional[IntegrityConfig] = None,
                    result: Optional[BoundaryResult] = None,
                    boundary_cfg: Optional[BoundaryConfig] = None,
                    occlusion_boxes=None) -> IntegrityScore:
    """Score the condition of the boundary in one frame, 0-100.

    Parameters
    ----------
    frame:       the BGR frame to assess.
    baseline:    from :func:`capture_baseline` on a clean frame of this corner.
    result:      a precomputed Module 1 detection for ``frame``, if available.

    Returns
    -------
    IntegrityScore -- total plus the four components, with the raw physical
    measurements in ``detail``.
    """
    cfg = cfg or IntegrityConfig()
    raw = measure_boundary(frame, baseline, cfg, result, boundary_cfg,
                           occlusion_boxes)
    return score_from_measurement(raw, baseline, cfg)


# --------------------------------------------------------------------------


def save_debug_image(frame: np.ndarray, baseline: BoundaryBaseline,
                     score: IntegrityScore, cfg: Optional[IntegrityConfig] = None,
                     path: Optional[str] = None, title: str = "") -> str:
    """Frame with the measurement bands drawn on, plus the four sub-scores."""
    cfg = cfg or IntegrityConfig()
    tf, valid, paint_px = baseline.track, baseline.valid, baseline.paint_px
    vis = frame.copy()

    _, g0, g1 = _bands(tf, paint_px, cfg)
    for i in range(0, len(tf.points), 3):
        if not valid[i]:
            continue
        p, n = tf.points[i], tf.normals[i]
        a = tuple(np.round(p + 0.15 * paint_px[i] * n).astype(int))
        b = tuple(np.round(p + 0.85 * paint_px[i] * n).astype(int))
        cv2.line(vis, a, b, (0, 255, 255), 1, cv2.LINE_AA)          # paint band
        c = tuple(np.round(p + g0[i] * n).astype(int))
        d = tuple(np.round(p + g1[i] * n).astype(int))
        cv2.line(vis, c, d, (255, 140, 0), 1, cv2.LINE_AA)          # asphalt band
    cv2.polylines(vis, [np.round(baseline.reference).astype(np.int32)], False,
                  (0, 255, 0), 1, cv2.LINE_AA)

    panel = np.zeros((frame.shape[0], 330, 3), np.uint8)
    panel[:] = (24, 22, 20)
    colour = ((0, 0, 255) if score.total < cfg.trust_threshold
              else (0, 200, 255) if score.total < cfg.alert_threshold else (80, 230, 80))
    y = 40
    if title:
        cv2.putText(panel, title, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (210, 210, 210), 1, cv2.LINE_AA)
        y += 34
    cv2.putText(panel, "BOUNDARY INTEGRITY", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (170, 170, 170), 1, cv2.LINE_AA)
    y += 58
    cv2.putText(panel, f"{score.total:.0f}", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.9, colour, 3, cv2.LINE_AA)
    y += 28
    verdict = ("REVIEW REQUIRED" if score.total < cfg.trust_threshold
               else "DEGRADED" if score.total < cfg.alert_threshold else "TRUSTED")
    cv2.putText(panel, verdict, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    y += 40
    for name, v in (("contrast", score.contrast), ("continuity", score.continuity),
                    ("sharpness", score.sharpness), ("contamination", score.contamination)):
        cv2.putText(panel, f"{name:<14}{v:5.0f}", (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (16, y + 6), (16 + int(2.9 * v), y + 12),
                      (90, 200, 90) if v >= 60 else (70, 120, 240), -1)
        y += 34
    y += 10
    d = score.detail
    for line in (f"paint {d['paint_luma']:.0f} vs road {d['asphalt_luma']:.0f}",
                 f"contrast {d['contrast_michelson']:.3f}",
                 f"line found {d['coverage']:.0%} of {d['coverage_baseline']:.0%}",
                 f"contaminated {d['contaminated_frac']:.0%}",
                 f"stations {d['stations']}"):
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (150, 150, 150), 1, cv2.LINE_AA)
        y += 24

    out = np.hstack([vis, panel])
    path = path or os.path.join(cfg.debug_dir, cfg.debug_name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, out):
        raise RuntimeError(f"save_debug_image: could not write {path!r}")
    return path


def _cli() -> None:
    from chronos.degrade import DegradeConfig, KINDS, degrade

    p = argparse.ArgumentParser(description="CHRONOS Module 2 -- boundary integrity")
    p.add_argument("--image", required=True)
    p.add_argument("--gt", default=None,
                   help="ground-truth JSON; used as the reference geometry")
    p.add_argument("--degrade", choices=list(KINDS), default=None,
                   help="sweep a degradation and print the score at each level")
    p.add_argument("--levels", default="0.0,0.3,0.6,0.9")
    args = p.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image!r}")

    reference = None
    if args.gt:
        from benchmark.generate import load_ground_truth
        reference = load_ground_truth(args.gt)

    cfg = IntegrityConfig()
    clean = detect_boundary(frame, BoundaryConfig(), save_debug=False)
    baseline = capture_baseline(frame, reference, cfg, result=clean)
    print(f"baseline : paint {baseline.paint_luma:.0f} vs road {baseline.asphalt_luma:.0f}"
          f" | contrast {baseline.contrast:.3f} | sharpness {baseline.sharpness:.0f}"
          f" | achievable coverage {baseline.coverage:.0%}"
          f" | {int(baseline.valid.sum())} stations")

    if args.degrade is None:
        score = score_integrity(frame, baseline, cfg, result=clean)
        print(f"\nINTEGRITY {score.total:.0f}   contrast {score.contrast:.0f}  "
              f"continuity {score.continuity:.0f}  sharpness {score.sharpness:.0f}  "
              f"contamination {score.contamination:.0f}")
        print(f"  {score.explain()}")
        print(f"debug    : {save_debug_image(frame, baseline, score, cfg)}")
        return

    tf = build_track_frame(clean.polyline, clean.drivable_mask, clean.kerb_mask)
    dcfg = DegradeConfig()
    print(f"\n{args.degrade} sweep")
    print(f"{'level':>6} {'TOTAL':>7} {'contr':>7} {'contin':>7} {'sharp':>7} {'contam':>7}")
    tiles = []
    for lv in [float(x) for x in args.levels.split(",")]:
        img = degrade(frame, lv, args.degrade, cfg=dcfg, track_frame=tf)
        sc = score_integrity(img, baseline, cfg)
        print(f"{lv:6.2f} {sc.total:7.1f} {sc.contrast:7.1f} {sc.continuity:7.1f} "
              f"{sc.sharpness:7.1f} {sc.contamination:7.1f}")
        tiles.append(cv2.imread(save_debug_image(
            img, baseline, sc, cfg,
            path=os.path.join(cfg.debug_dir, f"_tile_{lv:.2f}.jpg"),
            title=f"{args.degrade} {lv:.2f}")))
    sheet = np.vstack(tiles)
    sheet = cv2.resize(sheet, (frame.shape[1], int(sheet.shape[0] * frame.shape[1]
                                                   / sheet.shape[1])),
                       interpolation=cv2.INTER_AREA)
    path = os.path.join(cfg.debug_dir, f"integrity_{args.degrade}.jpg")
    cv2.imwrite(path, sheet)
    for lv in [float(x) for x in args.levels.split(",")]:
        os.remove(os.path.join(cfg.debug_dir, f"_tile_{lv:.2f}.jpg"))
    print(f"debug    : {path}")


if __name__ == "__main__":
    _cli()

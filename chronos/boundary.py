"""CHRONOS Module 1 -- track-limit boundary detection.

Finds the OUTER EDGE OF THE WHITE LINE: the exact surface the regulation
refers to.  Under the rules a car is off track when no part of it is in
contact with the track, and the track ends at the outer edge of the white
line -- the kerb is already outside.

Two independent estimates are produced so that later modules can compare
them as a confidence signal:

  Channel A (primary)  segment the DRIVABLE SURFACE including the painted
                       line as one region, then take the outer edge of that
                       region's contour.  A thin-line detector dies under
                       motion blur, spray and glare; a big region does not.

  Channel B (fallback) the classical route from the plan: HSV white
                       threshold -> morphology -> Canny -> Hough, chained into
                       a curve.

The two channels are independent in their EVIDENCE, not in every input.
Channel A asks where the drivable region ends; channel B asks where the paint
is and where the intensity gradients along its outer edge are.  They share the
kerb segmentation and, when a corner is kerbed on both sides, the outer-side
decision -- so a disagreement means the two kinds of evidence disagree, which
is what Module 5 wants, but a shared kerb failure would fool both.  That is
stated here rather than in a footnote because the confidence signal is only
worth what its assumptions are.

Both channels first subtract the kerb.  Kerb classification is not optional:
the white stripes of a red/white kerb are the same colour as the paint, so a
naive surface mask swallows the kerb and the boundary lands in the wrong
place -- outside the track limit, which is exactly the error that matters.

Run standalone::

    python -m chronos.boundary --image data/frame.jpg
    python -m chronos.boundary --image data/frame.jpg --gt data/frame_gt.json
    python -m chronos.boundary --image data/frame.jpg --outer-side left
"""

from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

OuterSide = Literal["auto", "left", "right"]

# --------------------------------------------------------------------------
# CONFIG -- tune these live.  No magic numbers below this block.
# --------------------------------------------------------------------------


@dataclass
class BoundaryConfig:
    """Tunables for boundary detection.  All pixel units are full-frame."""

    # --- drivable-surface mask (channel A) --------------------------------
    surface_sat_max: int = 70        # asphalt & paint are near-grey: low S
    surface_val_min: int = 28        # reject deep shadow / black
    surface_val_max: int = 255
    open_px: int = 5                 # kill isolated specks BEFORE grouping
    pre_close_px: int = 0            # bridge cracks before grouping.  Keep at 0:
                                     # near the horizon the grass verge is only a
                                     # few pixels tall and any pre-close welds the
                                     # track to the grandstand behind it.
    close_px: int = 9                # close cracks WITHIN the chosen region
    min_area_frac: float = 0.02      # component smaller than this -> fail
    prefer_bottom: bool = True       # the track is the surface at the BOTTOM of
                                     # a corner-cam frame; sky and grandstands
                                     # are grey and low-sat too, and can be the
                                     # larger region.  Off = pure largest-area.
    bottom_band_frac: float = 0.10   # how much of the frame counts as "bottom"

    # --- kerb classification ----------------------------------------------
    kerb_hue_lo: int = 8             # red wraps the hue circle: [0,lo]+[hi,179]
    kerb_hue_hi: int = 168
    kerb_sat_min: int = 80
    kerb_val_min: int = 45
    kerb_bridge_px: int = 81         # length of the directional closing that
                                     # bridges red stripes ACROSS the white ones;
                                     # must exceed the near-field stripe gap
    kerb_bridge_angles: int = 12     # orientations tried; the kerb curves
    kerb_white_val_min: int = 0      # brightness of a kerb's WHITE stripes.  A
                                     # bridge between red stripes is only kept
                                     # where it passes through red or white --
                                     # otherwise, where two kerbs converge near
                                     # the horizon, the bridge jumps the track
                                     # and cuts the drivable surface in half.
                                     # 0 = pick it per frame (Otsu around the
                                     # kerb), which survives exposure changes.
    kerb_dilate_px: int = 7          # margin subtracted from the surface mask
    kerb_min_area_frac: float = 0.0008

    # --- outer-edge selection ---------------------------------------------
    outer_side: OuterSide = "auto"   # manual override if auto ever flips
    border_margin_px: int = 6        # contour on the image edge is not a boundary
    kerb_proximity_px: int = 14      # a contour point this close to the kerb is
                                     # on the OUTER side, by definition of a kerb
    turn_window_px: int = 9          # baseline for the turning-angle test that
                                     # finds where the contour rounds the end of
                                     # the track and comes back down the far side
    turn_split_deg: float = 105.0    # turn sharper than this ends an edge
    min_kerb_frac: float = 0.35      # a chain must be at least this kerb-adjacent
                                     # to count as the outer edge, when a kerb
                                     # is visible at all
    min_chain_frac: float = 0.25     # ignore chains shorter than this fraction of
                                     # the longest one when ranking
    min_corner_turn_deg: float = 12.0  # below this the corner is too straight for
                                       # the concave-side test to mean anything,
                                       # and we fall back to centroid distance
    n_bins: int = 160                # resample resolution of the edge polyline
    smooth_window: int = 7           # moving-average window over the polyline
    min_polyline_points: int = 12

    # --- white-paint channel (channel B) ----------------------------------
    white_sat_max: int = 55
    white_val_min: int = 0           # 0 = pick it per frame, by splitting bright
                                     # from dark WITHIN the drivable surface:
                                     # the paint is the bright part of the road.
                                     # A fixed value cannot survive an exposure
                                     # change, and dusk is not a failure mode we
                                     # get to opt out of.
    white_close_px: int = 5
    agreement_span_tol_px: float = 6.0   # how close the channels must be to
                                         # count as covering the same stretch
    canny_lo: int = 60
    canny_hi: int = 170
    hough_threshold: int = 22
    hough_min_line_px: int = 18
    hough_max_gap_px: int = 14
    hough_sample_px: float = 3.0     # spacing when sampling Hough segments
    fallback_line_gap_px: float = 16.0  # a painted stripe this close to the kerb
                                     # is the OUTER edge line; the inner line is
                                     # nowhere near a kerb
    fallback_min_line_frac: float = 0.0004  # ignore paint blobs smaller than this
                                     # fraction of the frame
    fallback_max_half_width_px: float = 25.0  # an edge line is THIN.  Pale gravel
                                     # run-off passes a white threshold happily;
                                     # it does not pass a thickness test.
    fallback_cell_px: float = 10.0   # cell size when thinning the Hough cloud
    fallback_max_hop_px: float = 40.0  # stop chaining across a gap this big

    # --- output ------------------------------------------------------------
    debug_dir: str = "debug"
    debug_name: str = "boundary_debug.jpg"


@dataclass
class BoundaryResult:
    """Outcome of one detection.

    Attributes
    ----------
    ok:
        True when a boundary was found.  ``bool(result)`` is the same thing.
    reason:
        Human-readable explanation.  On failure this says exactly which stage
        gave up and what to tune.  Never empty.
    polyline:
        (N, 2) float32 image points along the outer edge of the white line,
        ordered along the track.  ``None`` when ``ok`` is False.
    fallback_polyline:
        (M, 2) float32 polyline from the classical HSV+Canny+Hough channel,
        or None if that channel found nothing.
    agreement_px:
        Mean distance from the fallback channel's points to the primary
        polyline, or None if the fallback found nothing.  Small = the two
        independent estimates agree.  This is the raw confidence signal
        Module 5 consumes.
    agreement_span:
        Fraction of the primary boundary that the fallback channel also
        covered.  A tiny ``agreement_px`` over a 10% span is weak evidence.
    drivable_mask, kerb_mask:
        uint8 0/255 masks, full frame, for downstream modules and for tuning.
    debug_path:
        Where the debug image was written.
    """

    ok: bool
    reason: str
    polyline: Optional[np.ndarray] = None
    fallback_polyline: Optional[np.ndarray] = None
    agreement_px: Optional[float] = None
    agreement_span: Optional[float] = None
    drivable_mask: Optional[np.ndarray] = None
    kerb_mask: Optional[np.ndarray] = None
    debug_path: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


# --------------------------------------------------------------------------
# stage 1 -- masks
# --------------------------------------------------------------------------


def _kernel(px: int) -> np.ndarray:
    px = max(1, int(px) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def segment_kerb(frame: np.ndarray, cfg: BoundaryConfig) -> np.ndarray:
    """Classify the red/white kerb as its own region.

    The red stripes are the only strongly saturated red in a track scene.
    Closing that stripe pattern with a kernel wider than one stripe bridges
    the white stripes between them, recovering the whole kerb band -- white
    stripes included, which is the point.

    Returns a uint8 0/255 mask of the kerb band (empty mask if no kerb).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat_ok = (s >= cfg.kerb_sat_min) & (v >= cfg.kerb_val_min)
    red = ((h <= cfg.kerb_hue_lo) | (h >= cfg.kerb_hue_hi)) & sat_ok
    red = (red.astype(np.uint8)) * 255
    if red.sum() == 0:
        return np.zeros(frame.shape[:2], np.uint8)

    band = _directional_close(red, cfg.kerb_bridge_px, cfg.kerb_bridge_angles)

    white_min = cfg.kerb_white_val_min
    if white_min <= 0:
        # split bright from dark using only the pixels around the kerb itself,
        # so the threshold follows the exposure instead of fighting it
        neighbourhood = cv2.dilate(red, _kernel(cfg.kerb_bridge_px))
        vals = v[neighbourhood > 0]
        if len(vals) >= 32:
            white_min = float(cv2.threshold(vals.reshape(-1, 1), 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        else:
            white_min = 130.0
    kerb_white = ((s <= cfg.surface_sat_max) & (v >= white_min))
    band &= (red | (kerb_white.astype(np.uint8) * 255))
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, _kernel(3))
    band = _fill_holes(band)

    # drop specks: red bodywork, marshal flags, advertising
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    keep = np.zeros_like(band)
    min_area = cfg.kerb_min_area_frac * frame.shape[0] * frame.shape[1]
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def segment_drivable(frame: np.ndarray, cfg: BoundaryConfig,
                     kerb_mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Segment the drivable surface INCLUDING the painted line as one region.

    Returns
    -------
    (mask, reason): ``mask`` is uint8 0/255 of the largest surface component,
    ``reason`` is "" on success or an explanation when the mask is empty.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    surface = ((s <= cfg.surface_sat_max)
               & (v >= cfg.surface_val_min)
               & (v <= cfg.surface_val_max)).astype(np.uint8) * 255

    # the kerb is OUTSIDE the track limit -- cut it out, with a margin, so a
    # gravel run-off beyond it cannot leak in and drag the contour outward
    uncut = surface.copy()
    if kerb_mask.any():
        surface = surface.copy()
        surface[cv2.dilate(kerb_mask, _kernel(cfg.kerb_dilate_px)) > 0] = 0

    # Group FIRST, tidy afterwards.  Sky, grandstands and barriers are grey and
    # unsaturated too; they are separate regions only for as long as no
    # morphological operation welds them to the track across a thin verge.
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN, _kernel(cfg.open_px))
    if cfg.pre_close_px > 0:
        surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, _kernel(cfg.pre_close_px))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(surface, 8)
    if n <= 1:
        return surface, ("no low-saturation surface region found -- "
                         f"check surface_sat_max ({cfg.surface_sat_max}) and "
                         f"surface_val_min ({cfg.surface_val_min})")
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = 1 + np.argsort(areas)[::-1]

    best = int(order[0])
    if cfg.prefer_bottom:
        band_top = int(frame.shape[0] * (1.0 - cfg.bottom_band_frac))
        for idx in order:                      # largest first
            if (labels[band_top:] == idx).any():
                best = int(idx)
                break

    area = stats[best, cv2.CC_STAT_AREA]
    frac = area / float(frame.shape[0] * frame.shape[1])
    if frac < cfg.min_area_frac:
        return surface, (f"largest surface region covers {frac:.3%} of the frame, "
                         f"below min_area_frac ({cfg.min_area_frac:.3%}) -- "
                         "the track may be out of shot or too dark")

    region = (labels == best).astype(np.uint8) * 255
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, _kernel(cfg.close_px))

    # The dilation above was a knife, not a measurement: it shaved a few pixels
    # of real track off the kerb interface.  Grow the region back into the
    # ORIGINAL surface mask (minus the raw kerb) so the outer edge lands on the
    # true paint/kerb interface instead of a few pixels inside it.
    if kerb_mask.any():
        grown = cv2.dilate(region, _kernel(cfg.kerb_dilate_px + 4))
        allowed = cv2.bitwise_and(uncut, cv2.bitwise_not(kerb_mask))
        region = cv2.bitwise_and(grown, allowed)
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, _kernel(cfg.close_px))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(region, 8)
        if n2 > 1:
            region = (lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255

    return _fill_holes(region), ""


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (shadows, debris, a car sitting on the surface).

    Pads with a guaranteed-background border first -- flooding from a raw
    corner silently fills the entire frame when that corner is inside the
    mask, which is exactly what the sky does in a corner-cam view.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return mask | holes


def _directional_close(mask: np.ndarray, length: int, n_angles: int) -> np.ndarray:
    """Close gaps using LINE kernels at several orientations, unioned.

    An elliptical kernel big enough to bridge a near-field kerb stripe also
    bleeds sideways over the white line, which would drag the boundary
    outward.  A line kernel only fills gaps along its own direction, and the
    red stripes of a kerb are separated only ALONG the track -- so this
    reconnects the kerb band without widening it across the track.  Several
    angles are tried because the kerb curves through the frame.
    """
    length = max(3, int(length) | 1)
    out = mask.copy()
    for i in range(max(1, n_angles)):
        angle = 180.0 * i / max(1, n_angles)
        k = np.zeros((length, length), np.uint8)
        cv2.line(k, (0, length // 2), (length - 1, length // 2), 1, 1)
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        k = (cv2.warpAffine(k.astype(np.float32), M, (length, length)) > 0.3).astype(np.uint8)
        if k.sum() < 2:
            continue
        out |= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return out


# --------------------------------------------------------------------------
# stage 2 -- pick the outer edge of that region
# --------------------------------------------------------------------------


def _principal_frame(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA frame of a mask: (centroid, along-track axis, outward normal).

    The normal is oriented so that a positive offset means "to the right in
    the image", which is what the left/right config override refers to.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > 60000:                      # PCA does not need every pixel
        pts = pts[np.linspace(0, len(pts) - 1, 60000).astype(int)]
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    normal = np.array([axis[1], -axis[0]])
    if normal[0] < 0:
        normal = -normal
    return centroid, axis, normal


def _longest_circular_run(flags: np.ndarray) -> tuple[int, int]:
    """Longest contiguous True run in a CYCLIC boolean array.

    Returns (start, length).  The run may wrap past the end of the array,
    because a contour is a closed loop and the boundary we want is usually
    cut by the image border, not by the array's arbitrary start index.
    """
    n = len(flags)
    if flags.all():
        return 0, n
    if not flags.any():
        return 0, 0
    best_start, best_len = 0, 0
    i = 0
    doubled = np.concatenate([flags, flags])
    while i < n:
        if doubled[i]:
            j = i
            while j < i + n and doubled[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i % n, j - i
            i = j
        else:
            i += 1
    return best_start, best_len


def _chain_is_outer(chain_pts: np.ndarray, mask: np.ndarray,
                    probe_px: float = 7.0) -> tuple[Optional[bool], float]:
    """Decide whether a chain is the OUTER edge of a corner, geometrically.

    Both edges of a corner bend the same way, but the track lies on the
    CONCAVE side of the outer edge and on the CONVEX side of the inner edge.
    So: find which way the chain turns, find which side the drivable surface
    is on, and compare.

    Returns ``(is_outer, turn_deg)``.  ``is_outer`` is None when the chain is
    too straight for the test to mean anything -- ``turn_deg`` then says how
    straight, and the caller should fall back to another rule.
    """
    h, w = mask.shape
    p = _resample_polyline(np.asarray(chain_pts, np.float32), 60).astype(np.float64)
    t = np.diff(p, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    if (n < 1e-6).all():
        return None, 0.0
    t = t / np.maximum(n, 1e-9)

    # total signed turn: +ve = turning left (anticlockwise in image coords)
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    dot = (t[:-1] * t[1:]).sum(axis=1)
    turn = float(np.degrees(np.arctan2(cross, dot).sum()))

    # which side is the drivable surface on?  probe both normals
    mid = p[:-1]
    left = np.stack([-t[:, 1], t[:, 0]], axis=1)
    votes = 0
    for s_ in (-1.0, 1.0):
        q = mid + s_ * probe_px * left
        xi = np.clip(q[:, 0].astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].astype(int), 0, h - 1)
        votes += s_ * float((mask[yi, xi] > 0).mean())
    if abs(votes) < 1e-6:
        return None, abs(turn)
    mask_on_left = votes > 0

    # image y points down, so a +ve signed turn curves toward the RIGHT normal
    centre_on_left = turn < 0
    return (mask_on_left == centre_on_left), abs(turn)


def _turn_blocked(contour: np.ndarray, window: int, turn_deg: float) -> np.ndarray:
    """Mark contour points where the outline turns back on itself.

    The outline of a track band is two long edges joined at the band's two
    ends.  Measuring the turn over a ``window``-point baseline finds those
    joins -- the far end where the track narrows to nothing, and any similar
    cap -- so the two edges can be taken apart.  A one-pixel notch does not
    register, because the baseline is longer than the notch.
    """
    n = len(contour)
    i = np.arange(n)
    fwd = contour[(i + window) % n] - contour[i]
    bwd = contour[i] - contour[(i - window) % n]
    nf = np.linalg.norm(fwd, axis=1)
    nb = np.linalg.norm(bwd, axis=1)
    cos = (fwd * bwd).sum(axis=1) / np.maximum(nf * nb, 1e-9)
    return cos < math.cos(math.radians(turn_deg))


def _circular_runs(blocked: np.ndarray, min_len: int) -> list[np.ndarray]:
    """All contiguous runs of unblocked indices in a cyclic array."""
    n = len(blocked)
    if not blocked.any():
        return [np.arange(n)]
    runs, start = [], None
    for i in range(2 * n):
        j = i % n
        if not blocked[j] and start is None:
            start = i
        elif blocked[j] and start is not None:
            if i - start >= min_len and start < n:
                runs.append(np.arange(start, i) % n)
            start = None
        if i - (start if start is not None else i) >= n:
            break
    return runs


def _chain_point_cloud(cloud: np.ndarray, cell_px: float, max_hop_px: float,
                       score: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Order a thin, curved point cloud into a polyline.

    Thins the cloud to one representative point per ``cell_px`` grid cell,
    starts at one end of the cloud's longest diameter, then repeatedly hops to
    the nearest unused point.  With ``score`` given, each cell is represented
    by its LOWEST-scoring point instead of its mean -- used to pick the outer
    edge of the paint rather than the middle of the stripe.  Chaining stops at
    a gap wider than ``max_hop_px`` so a detached cluster cannot be stitched on.

    Returns the ordered (N, 2) polyline, or None if the cloud is too small.
    """
    if len(cloud) < 4:
        return None
    cells: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(cloud):
        cells.setdefault((int(p[0] // cell_px), int(p[1] // cell_px)), []).append(i)
    if score is None:
        pts = np.array([cloud[v].mean(axis=0) for v in cells.values()], dtype=np.float64)
    else:
        pts = np.array([cloud[v[int(np.argmin(score[v]))]] for v in cells.values()],
                       dtype=np.float64)
    if len(pts) < 4:
        return None

    # an endpoint of an arc is one end of its longest chord
    centre = pts.mean(axis=0)
    far = pts[int(np.argmax(np.linalg.norm(pts - centre, axis=1)))]
    start = int(np.argmax(np.linalg.norm(pts - far, axis=1)))

    remaining = np.ones(len(pts), bool)
    remaining[start] = False
    order = [start]
    cur = pts[start]
    while remaining.any():
        d = np.linalg.norm(pts - cur, axis=1)
        d[~remaining] = np.inf
        nxt = int(np.argmin(d))
        if d[nxt] > max_hop_px:
            break
        order.append(nxt)
        remaining[nxt] = False
        cur = pts[nxt]
    return pts[order] if len(order) >= 4 else None


def _resample_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced by arc length."""
    d = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return poly
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1).astype(np.float32)


def _outer_chain(mask: np.ndarray, cfg: BoundaryConfig,
                 kerb_mask: np.ndarray) -> tuple[Optional[np.ndarray], str, dict]:
    """Extract the outer-side edge of ``mask`` as an ordered polyline.

    The contour of the drivable region is a closed loop made of three kinds of
    piece: the outer track edge, the inner track edge, and stretches that run
    along the image border.  We classify every contour point as outer/not and
    take the longest CONTIGUOUS run of outer points.  Keeping the contour's own
    ordering is what makes this work on a corner -- any scheme that bins points
    along a straight axis cuts the apex off.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, "drivable region has no contour", {}
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < cfg.min_polyline_points:
        return None, f"drivable contour has only {len(contour)} points", {}

    centroid, axis, normal = _principal_frame(mask)
    offset = (contour - centroid) @ normal     # +ve = right of centroid

    m = cfg.border_margin_px
    on_border = ((contour[:, 0] <= m) | (contour[:, 0] >= w - 1 - m)
                 | (contour[:, 1] <= m) | (contour[:, 1] >= h - 1 - m))

    # --- take the outline apart at the ends of the band -------------------
    blocked = on_border | _turn_blocked(contour, cfg.turn_window_px, cfg.turn_split_deg)
    chains = _circular_runs(blocked, cfg.min_polyline_points)
    if not chains:
        return None, ("the drivable outline never breaks into separate edges -- "
                      "it may be clipped by the frame on every side"), {}

    # --- which chain is the OUTER edge? ----------------------------------
    # A kerb is outside the track limit by definition, so "this chain runs
    # along a kerb" is the strongest outer-side evidence there is.  When that
    # decides nothing -- no kerb, or kerbs on both sides -- the tie-break is
    # the plan's rule: of the candidate edges, the one further from the
    # drivable centroid is the outer one.  A curved band's centroid sits
    # inside the bend, so the outer edge really is the further of the two.
    kerb_frac = {}
    if kerb_mask.any():
        kerb_dist = cv2.distanceTransform(cv2.bitwise_not(kerb_mask), cv2.DIST_L2, 3)
        for ci, ch in enumerate(chains):
            xi = np.clip(contour[ch, 0].astype(int), 0, w - 1)
            yi = np.clip(contour[ch, 1].astype(int), 0, h - 1)
            kerb_frac[ci] = float((kerb_dist[yi, xi] <= cfg.kerb_proximity_px).mean())

    if cfg.outer_side in ("left", "right"):
        want = 1.0 if cfg.outer_side == "right" else -1.0
        pick = max(range(len(chains)),
                   key=lambda ci: want * (contour[chains[ci], 0].mean() - centroid[0]))
        side_note = f"manual override ({cfg.outer_side})"
    else:
        kerbed = [ci for ci, f in kerb_frac.items() if f >= cfg.min_kerb_frac]
        if len(kerbed) == 1:
            pick = kerbed[0]
            side_note = "auto (runs along the kerb)"
        else:
            pool = kerbed or list(range(len(chains)))
            longest = max(len(chains[ci]) for ci in pool)
            pool = [ci for ci in pool if len(chains[ci]) >= cfg.min_chain_frac * longest]
            where = "kerb on both sides" if kerbed else "no kerb"

            # the track lies inside the bend of the outer edge and outside the
            # bend of the inner one -- that settles it whenever the corner
            # actually bends
            geo = {ci: _chain_is_outer(contour[chains[ci]], mask) for ci in pool}
            outer = [ci for ci in pool
                     if geo[ci][0] is True and geo[ci][1] >= cfg.min_corner_turn_deg]
            if len(outer) == 1:
                pick = outer[0]
                side_note = f"auto ({where}, track is inside the bend)"
            elif outer:
                pick = max(outer, key=lambda ci: len(chains[ci]))
                side_note = f"auto ({where}, inside the bend, longest of {len(outer)})"
            else:
                pick = max(pool, key=lambda ci: float(
                    np.linalg.norm(contour[chains[ci]] - centroid, axis=1).mean()))
                side_note = (f"auto ({where}, corner too straight to tell -- "
                             "used furthest from centroid)")

    idx = chains[pick]
    poly = contour[idx].astype(np.float32)
    poly = _resample_polyline(poly, min(cfg.n_bins, len(poly)))
    poly = _smooth_polyline(poly, cfg.smooth_window)

    # the label is in plain image terms -- that is what the config override
    # means to whoever is tuning it at 3am, whatever PCA thinks
    side = "right" if float(poly[:, 0].mean()) > centroid[0] else "left"
    info = {"centroid": centroid, "axis": axis, "normal": normal,
            "sign": 1.0 if side == "right" else -1.0,
            "side": side, "side_note": side_note,
            "n_chains": len(chains), "edge_points": int(len(idx)),
            "contour_frac": float(len(idx)) / len(contour)}
    return poly, "", info


def _smooth_polyline(poly: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing that keeps the endpoints anchored."""
    window = max(1, int(window) | 1)
    if window == 1 or len(poly) < window:
        return poly
    pad = window // 2
    padded = np.vstack([np.repeat(poly[:1], pad, 0), poly, np.repeat(poly[-1:], pad, 0)])
    k = np.ones(window) / window
    return np.stack([np.convolve(padded[:, i], k, "valid") for i in range(2)], 1).astype(np.float32)


# --------------------------------------------------------------------------
# stage 3 -- classical fallback channel
# --------------------------------------------------------------------------


def detect_white_line_classical(frame: np.ndarray, cfg: BoundaryConfig,
                                kerb_mask: np.ndarray, frame_info: dict,
                                drivable_mask: Optional[np.ndarray] = None
                                ) -> tuple[Optional[np.ndarray], str]:
    """Independent estimate: HSV white threshold -> Canny -> Hough -> curve fit.

    This is a different kind of evidence from channel A.  Channel A asks "where
    does the drivable region end"; this asks "where is the paint, and where are
    the intensity gradients along its outer edge".  Canny runs on the real
    greyscale image (not on the binary mask) so it responds to actual image
    gradient, and Hough supplies the line support.

    Disagreement between the two channels is the confidence signal Module 5
    consumes -- so this must be allowed to be wrong in its own way.  See the
    module docstring for what the two channels do and do not share.

    Returns (polyline, reason).  ``polyline`` is None when this channel has
    nothing to say, and ``reason`` then explains why.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    val_min = float(cfg.white_val_min)
    if val_min <= 0:
        pool = val[drivable_mask > 0] if (drivable_mask is not None
                                          and drivable_mask.any()) else val.reshape(-1)
        val_min = (float(cv2.threshold(pool.reshape(-1, 1), 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                   if len(pool) >= 32 else 150.0)

    white = ((sat <= cfg.white_sat_max) & (val >= val_min)).astype(np.uint8) * 255
    # erase the kerb with the RAW mask, not a dilated one -- a dilated knife
    # eats a narrow edge line whole, and then there is no line left to find
    if kerb_mask.any():
        white[kerb_mask > 0] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, _kernel(cfg.white_close_px))
    if white.sum() == 0:
        return None, f"white-paint mask is empty (brightness cut was {val_min:.0f})"

    # --- isolate the OUTER edge line as a whole component ------------------
    # A per-pixel distance gate is brittle: the kerb band has gaps where its
    # stripes are unresolvable, and a gate loose enough to survive them also
    # lets the inner line through.  Deciding once per painted stripe is steady.
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    min_area = cfg.fallback_min_line_frac * h * w
    cands = [i for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not cands:
        return None, "no painted stripe large enough to be an edge line"

    # thickness test: half-width = deepest point inside the blob
    thickness = cv2.distanceTransform(white, cv2.DIST_L2, 5)
    thin = [i for i in cands
            if float(thickness[labels == i].max()) <= cfg.fallback_max_half_width_px]
    if not thin:
        return None, ("every pale region is too thick to be an edge line "
                      f"(thinnest half-width "
                      f"{min(float(thickness[labels == i].max()) for i in cands):.0f} px)")
    cands = thin

    kerb_dist = None
    note = "outermost painted stripe"
    if kerb_mask.any():
        kerb_dist = cv2.distanceTransform(cv2.bitwise_not(kerb_mask), cv2.DIST_L2, 3)
        gaps = {i: float(kerb_dist[labels == i].min()) for i in cands}
        kerbed = [i for i, g in gaps.items() if g <= cfg.fallback_line_gap_px]
        if not kerbed:
            return None, ("no painted stripe runs along the kerb -- nearest is "
                          f"{min(gaps.values()):.0f} px away, "
                          f"fallback_line_gap_px is {cfg.fallback_line_gap_px:.0f}")
        cands, note = kerbed, "stripe running along the kerb"

    # A corner kerbed on both sides leaves two candidate stripes; the inner one
    # is a track limit too, just not this one.  Break the tie with the outer
    # side channel A already settled -- the two channels share that decision and
    # the kerb segmentation, and differ in the evidence used to place the edge.
    centroid, normal = frame_info["centroid"], frame_info["normal"]
    sign = float(frame_info.get("sign", 1.0))
    def outerness(i: int) -> float:
        ys, xs = np.nonzero(labels == i)
        return float(((np.stack([xs, ys], 1) - centroid) @ normal).mean() * sign)
    best = max(cands, key=outerness)
    line_mask = (labels == best).astype(np.uint8) * 255

    # --- gradient evidence along that stripe -------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)
    edges[cv2.dilate(line_mask, _kernel(2 * cfg.white_close_px + 1)) == 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, cfg.hough_threshold,
                            minLineLength=cfg.hough_min_line_px,
                            maxLineGap=cfg.hough_max_gap_px)
    if lines is None:
        return None, f"Hough found no line segments along the {note}"

    # sample points along every supported segment
    chunks = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(np.float64):
        npts = max(2, int(np.hypot(x2 - x1, y2 - y1) / cfg.hough_sample_px))
        t = np.linspace(0.0, 1.0, npts)
        chunks.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
    cloud = np.concatenate(chunks, axis=0)
    if len(cloud) < cfg.min_polyline_points:
        return None, f"only {len(cloud)} Hough samples along the {note}"

    # Both edges of the stripe respond to Canny; we want the OUTER one, so
    # score every sample by its distance from the kerb and keep the nearest
    # sample in each cell when thinning.
    if kerb_dist is not None:
        xi = np.clip(cloud[:, 0].astype(int), 0, w - 1)
        yi = np.clip(cloud[:, 1].astype(int), 0, h - 1)
        cloud_score = kerb_dist[yi, xi]
    else:
        sign = float(frame_info.get("sign", 1.0))
        cloud_score = -((cloud - frame_info["centroid"]) @ frame_info["normal"]) * sign

    # Order the cloud ALONG the curve, not along a straight axis.  A corner
    # folds back on any single principal axis, and binning along one puts two
    # different stretches of boundary in the same bin -- which shows up as a
    # hook in the far field.  Thin to one point per cell, then chain by
    # nearest neighbour from one end.
    curve = _chain_point_cloud(cloud, cfg.fallback_cell_px, cfg.fallback_max_hop_px,
                               score=cloud_score)
    if curve is None or len(curve) < cfg.min_polyline_points:
        got = 0 if curve is None else len(curve)
        return None, f"paint samples would not chain into a curve ({got} points)"

    curve = _smooth_polyline(curve.astype(np.float32), cfg.smooth_window)
    return _resample_polyline(curve, min(cfg.n_bins, len(curve))), ""


# --------------------------------------------------------------------------


def point_to_polyline_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Perpendicular distance in pixels from each point to a polyline."""
    pts = np.asarray(pts, np.float64)
    poly = np.asarray(poly, np.float64)
    if len(poly) < 2:
        raise ValueError("polyline needs at least 2 points")
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(1), 1e-9)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(2) / denom[None], 0.0, 1.0)
    proj = a[None] + t[:, :, None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


def polyline_error(pred: np.ndarray, gt: np.ndarray, tol_px: float = 5.0) -> dict:
    """Pixel error between a predicted boundary and ground truth.

    Accuracy and coverage are reported separately on purpose -- a boundary that
    is exactly right over 80% of the line is a very different thing from one
    that is 20% wrong everywhere, and a single symmetric number hides which.

    Returns
    -------
    dict with
      ``dev_mean_px`` / ``dev_p95_px`` / ``dev_max_px``
          how far the DETECTED points are from the true line (accuracy).
      ``cov_frac``
          fraction of the ground truth's PIXEL LENGTH that has a detected
          point within ``tol_px`` (completeness).
      ``cov_p95_px``
          95th percentile distance from ground truth to the detection.
      ``n_pred`` / ``n_gt``
    """
    if pred is None or len(pred) < 2:
        raise ValueError("polyline_error: prediction is empty")
    if gt is None or len(gt) < 2:
        raise ValueError("polyline_error: ground truth is empty")
    # Resample both by arc length so every metric is weighted by PIXEL length
    # of boundary, not by however densely the two were sampled.  Ground truth
    # from the scene generator is uniform in world metres, which crams most of
    # its points into the far field where the track is a few pixels wide.
    pred = _resample_polyline(np.asarray(pred, np.float32), 400)
    gt = _resample_polyline(np.asarray(gt, np.float32), 400)
    d_pred = point_to_polyline_distance(pred, gt)
    d_gt = point_to_polyline_distance(gt, pred)
    return {"dev_mean_px": float(d_pred.mean()),
            "dev_p95_px": float(np.percentile(d_pred, 95)),
            "dev_max_px": float(d_pred.max()),
            "cov_frac": float((d_gt <= tol_px).mean()),
            "cov_p95_px": float(np.percentile(d_gt, 95)),
            "n_pred": int(len(pred)), "n_gt": int(len(gt))}


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def detect_boundary(frame: np.ndarray, cfg: Optional[BoundaryConfig] = None,
                    save_debug: bool = True,
                    ground_truth: Optional[np.ndarray] = None) -> BoundaryResult:
    """Detect the outer edge of the white track-limit line in one frame.

    Parameters
    ----------
    frame:
        BGR uint8 image, shape (H, W, 3).
    cfg:
        Tunables.  Defaults are calibrated on the synthetic corner cam from
        ``benchmark/generate.py``.
    save_debug:
        Write the debug overlay to ``cfg.debug_dir``.  Written on failure too.
    ground_truth:
        Optional (N, 2) polyline drawn in red on the debug image for
        eyeball comparison.  Never used by the detector itself.

    Returns
    -------
    BoundaryResult.  On failure ``ok`` is False, ``polyline`` is None and
    ``reason`` states which stage gave up.  This function does not raise for
    a detection failure -- only for bad input.
    """
    cfg = cfg or BoundaryConfig()
    if frame is None:
        raise ValueError("detect_boundary: frame is None")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"detect_boundary: expected a BGR image, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"detect_boundary: expected uint8, got {frame.dtype}")

    kerb_mask = segment_kerb(frame, cfg)
    drivable, reason = segment_drivable(frame, cfg, kerb_mask)
    if reason:
        res = BoundaryResult(False, reason, drivable_mask=drivable, kerb_mask=kerb_mask)
        if save_debug:
            res.debug_path = save_debug_image(frame, res, cfg, ground_truth)
        return res

    poly, reason, info = _outer_chain(drivable, cfg, kerb_mask)
    if reason:
        res = BoundaryResult(False, reason, drivable_mask=drivable, kerb_mask=kerb_mask)
        if save_debug:
            res.debug_path = save_debug_image(frame, res, cfg, ground_truth)
        return res

    fallback, fb_reason = detect_white_line_classical(frame, cfg, kerb_mask, info, drivable)
    agreement = agreement_span = None
    if fallback is not None:
        # Measured fallback -> primary: "where channel B has evidence, does it
        # match channel A?".  The other direction would mostly measure how much
        # of the boundary B simply never saw, which is reported separately.
        agreement = float(point_to_polyline_distance(fallback, poly).mean())
        d_back = point_to_polyline_distance(poly, fallback)
        agreement_span = float((d_back <= cfg.agreement_span_tol_px).mean())

    note = f"outer side chosen by {info['side_note']} -> {info['side']}"
    if fallback is None:
        note += f"; fallback channel silent: {fb_reason}"
    else:
        note += (f"; channels agree to {agreement:.1f} px over "
                 f"{agreement_span:.0%} of the boundary")

    res = BoundaryResult(True, note, polyline=poly, fallback_polyline=fallback,
                         agreement_px=agreement, agreement_span=agreement_span,
                         drivable_mask=drivable, kerb_mask=kerb_mask)
    if save_debug:
        res.debug_path = save_debug_image(frame, res, cfg, ground_truth)
    return res


# --------------------------------------------------------------------------
# debug view -- verify by eye in seconds
# --------------------------------------------------------------------------


def save_debug_image(frame: np.ndarray, res: BoundaryResult, cfg: BoundaryConfig,
                     ground_truth: Optional[np.ndarray] = None) -> str:
    """Write a 2x2 debug panel: overlay, surface mask, kerb mask, contour."""
    h, w = frame.shape[:2]

    def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        return img

    overlay = frame.copy()
    if ground_truth is not None and len(ground_truth) >= 2:
        cv2.polylines(overlay, [np.round(ground_truth).astype(np.int32)], False,
                      (0, 0, 255), 4, cv2.LINE_AA)
    if res.fallback_polyline is not None:
        cv2.polylines(overlay, [np.round(res.fallback_polyline).astype(np.int32)], False,
                      (255, 0, 255), 2, cv2.LINE_AA)
    if res.polyline is not None:
        cv2.polylines(overlay, [np.round(res.polyline).astype(np.int32)], False,
                      (0, 255, 0), 2, cv2.LINE_AA)
    legend = "GREEN=detected  MAGENTA=classical fallback"
    if ground_truth is not None:
        legend += "  RED=ground truth"
    label(overlay, legend if res.ok else f"FAILED: {res.reason[:70]}",
          (255, 255, 255) if res.ok else (0, 0, 255))

    def mask_view(mask: Optional[np.ndarray], color) -> np.ndarray:
        view = (frame * 0.4).astype(np.uint8)
        if mask is not None and mask.any():
            view[mask > 0] = (0.45 * view[mask > 0] + 0.55 * np.array(color)).astype(np.uint8)
        return view

    surface = label(mask_view(res.drivable_mask, (255, 210, 60)), "drivable surface (kerb removed)")
    kerb = label(mask_view(res.kerb_mask, (60, 60, 255)), "kerb -- classified separately")

    contour_view = np.zeros_like(frame)
    if res.drivable_mask is not None:
        contour_view[res.drivable_mask > 0] = (55, 55, 55)
    if res.kerb_mask is not None:
        contour_view[res.kerb_mask > 0] = (0, 0, 110)
    if res.polyline is not None:
        cv2.polylines(contour_view, [np.round(res.polyline).astype(np.int32)], False,
                      (0, 255, 0), 2, cv2.LINE_AA)
    txt = (f"agreement {res.agreement_px:.1f}px over {res.agreement_span:.0%}"
           if res.agreement_px is not None else "no fallback")
    contour_view = label(contour_view, f"extracted boundary -- {txt}")

    top = np.hstack([overlay, surface])
    bot = np.hstack([kerb, contour_view])
    panel = np.vstack([top, bot])
    panel = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)

    os.makedirs(cfg.debug_dir, exist_ok=True)
    path = os.path.join(cfg.debug_dir, cfg.debug_name)
    if not cv2.imwrite(path, panel):
        raise RuntimeError(f"save_debug_image: could not write {path!r}")
    return path


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS Module 1 -- boundary detection")
    p.add_argument("--image", required=True, help="input frame (jpg/png)")
    p.add_argument("--gt", default=None, help="ground-truth JSON from benchmark.generate")
    p.add_argument("--outer-side", choices=["auto", "left", "right"], default="auto")
    p.add_argument("--debug-name", default="boundary_debug.jpg")
    args = p.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image!r}")

    gt = None
    if args.gt:
        from benchmark.generate import load_ground_truth
        gt = load_ground_truth(args.gt)

    cfg = BoundaryConfig(outer_side=args.outer_side, debug_name=args.debug_name)
    res = detect_boundary(frame, cfg, save_debug=True, ground_truth=gt)

    print(f"ok       : {res.ok}")
    print(f"reason   : {res.reason}")
    print(f"debug    : {res.debug_path}")
    if not res.ok:
        raise SystemExit(1)
    print(f"polyline : {len(res.polyline)} points, "
          f"x[{res.polyline[:,0].min():.0f},{res.polyline[:,0].max():.0f}] "
          f"y[{res.polyline[:,1].min():.0f},{res.polyline[:,1].max():.0f}]")
    if res.agreement_px is not None:
        print(f"channels : agree to {res.agreement_px:.2f} px over "
              f"{res.agreement_span:.1%} of the boundary (confidence signal)")
    if gt is not None:
        def report(tag: str, poly: np.ndarray) -> None:
            e = polyline_error(poly, gt)
            print(f"{tag:9s}: deviation mean {e['dev_mean_px']:5.2f} px | "
                  f"p95 {e['dev_p95_px']:5.2f} px | max {e['dev_max_px']:6.2f} px | "
                  f"coverage {e['cov_frac']:.1%} of the true line")
        report("VS GT", res.polyline)
        if res.fallback_polyline is not None:
            report("fallback", res.fallback_polyline)


if __name__ == "__main__":
    _cli()

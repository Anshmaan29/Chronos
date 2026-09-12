"""Track-limit line detection for REAL footage, by ridge response.

Why this exists
---------------
:mod:`chronos.boundary` finds the boundary by segmenting the drivable
*region* and taking its outer edge. That works on the synthetic corner cam
and fails on real trackside footage for two structural reasons, both
measured rather than guessed:

* the drivable mask keys on "low saturation, mid brightness", which equally
  describes asphalt, concrete barriers, grandstands, debris fencing and an
  overcast sky -- on the Miami clip the mask swallows the barriers and the
  grandstand and the "outer edge" lands on a fence;
* the kerb classifier keys on RED hue. Miami's kerbs are blue and orange,
  Zandvoort's are red, Austin's are red and white. A red-keyed kerb finder
  returns nothing on two of those three, so the outer-side decision has
  nothing to stand on.

This module ignores regions entirely and looks for the thing the regulation
actually names: **a thin bright stripe of paint lying on dark asphalt.**
That is a ridge, and a ridge is what a morphological top-hat finds. It is
the same operator lane-departure systems have used on road markings for
twenty years, which is the point -- the project's stated second application
is exactly road-marking health from ordinary dashcam video, and it is the
same measurement.

Method
------
1. **Top-hat at several widths.** A white line is brighter than the asphalt
   on *both* sides and only a few pixels across. A top-hat with a kernel
   wider than the line keeps exactly that and removes everything broader --
   asphalt, sky, grandstand, car bodies. Perspective makes the line span
   ~3 px at the horizon and ~40 px at the camera, so the response is taken
   as the max over a range of widths rather than one.
2. **Threshold against the local asphalt, not a constant.** The road is
   darker in shadow and brighter in sun within one frame, so the cut is
   made per row band from that band's own statistics.
3. **Keep ridge-shaped components.** Paint is long and thin. Anything
   squat -- a kerb block, a sponsor board, a bollard -- is dropped on
   elongation, and anything horizontal is dropped because a track-limit
   line recedes away from a trackside camera rather than crossing it.
4. **Choose the track-limit line**, not any line. A pit-lane marking, a
   grid box and a sector line are all white paint. The track limit is the
   *outermost* long line on the driving surface, so candidates are ranked
   by length and by how far they sit toward the chosen side.
5. **Fit and return** a :class:`chronos.boundary.BoundaryResult`, so
   :mod:`chronos.integrity` and everything downstream consume it unchanged.

Fails loudly. If no candidate survives, the result carries ``ok=False`` and
a reason naming the stage that rejected it -- never a polyline.

Run standalone::

    python -m chronos.lane --image data/real/frame.jpg
    python -m chronos.lane --video data/real/clip.gif --frames 6
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

from chronos.boundary import (BoundaryConfig, BoundaryResult,
                              _resample_polyline, _smooth_polyline,
                              point_to_polyline_distance)

Side = Literal["auto", "left", "right"]


@dataclass
class LaneConfig:
    """Tunables for ridge-based line detection.  Pixel units are full-frame
    at :attr:`work_width`; the frame is scaled to that before anything runs,
    so one set of numbers holds for 720p and 4K alike."""

    work_width: int = 1024       # everything below is calibrated at this width

    # --- ridge response ---------------------------------------------------
    line_px: tuple[int, ...] = (5, 9, 15, 23, 33)
    """Kernel widths for the top-hat, in pixels.

    Each must EXCEED the line it is meant to find, so this spans the range a
    track-limit line covers between the horizon and the camera. The response
    is the per-pixel max, which is why one frame can carry a 3 px line at the
    top and a 30 px line at the bottom without retuning."""

    blur_px: int = 3
    row_bands: int = 12          # bands the adaptive threshold is computed in
    thresh_k: float = 2.4        # cut at mean + k*std of that band's response
    thresh_floor: float = 8.0    # absolute floor, stops pure noise qualifying

    # --- which blobs are paint -------------------------------------------
    min_area_px: int = 90
    min_elongation: float = 3.2  # major/minor axis of the fitted ellipse
    max_horizontal_deg: float = 62.0
    """Reject near-horizontal components.

    A track-limit line recedes from the camera, so in image space it runs
    steeply. A horizontal white bar is a grid box, a start line, a sponsor
    board or the top of a barrier -- never the limit the car is judged
    against."""

    min_span_frac: float = 0.10  # of frame height, end to end

    # --- assembling the line ---------------------------------------------
    link_gap_px: float = 55.0    # dashed paint and shadow gaps get bridged
    min_total_span_frac: float = 0.16
    n_bins: int = 160
    smooth_window: int = 15
    min_points: int = 12

    # --- picking the right line ------------------------------------------
    side: Side = "auto"
    outer_weight: float = 0.55   # how much "is it the outermost" counts
    length_weight: float = 0.45  # ... against "is it the longest"

    # --- road surface ------------------------------------------------------
    # A track-limit line lies ON the road. Without this constraint the
    # longest, outermost bright ridge in a trackside frame is the top of the
    # concrete barrier or a guardrail, and the detector confidently returns
    # it -- measured, on the Miami clip, before this was added.
    seed_band: tuple[float, float] = (0.82, 0.98)   # rows, as frame fractions
    seed_cols: tuple[float, float] = (0.20, 0.80)   # cols, as frame fractions
    chroma_block: float = 13.0
    """Lab chroma above which a pixel stops the flood.

    Measured on the Miami clip: asphalt 7.1, concrete barrier 3.8, white
    paint 7.0 -- all pass; blue kerb 29.4, orange kerb 31.7, sky 18.3 -- all
    block. It keys on chroma MAGNITUDE, never on hue, so a red kerb blocks
    exactly as a blue one does."""

    bright_block: float = 205.0  # specular sky / blown highlights
    ridge_seal_px: int = 15      # dilate paint before using it as a wall,
                                 # so a dashed or shadowed line still seals
    road_close_px: int = 25
    road_min_frac: float = 0.04  # smaller than this and we have not found road
    adjacency_px: int = 26       # how far to look sideways for road
    min_road_adjacent: float = 0.45
    """Fraction of a candidate's length that must have road beside it.

    The white edge line has asphalt on one side and kerb or run-off on the
    other, so it scores high. A barrier top has sky above and concrete below
    and scores ~0, which is exactly the separation needed."""

    # --- edge extraction ---------------------------------------------------
    min_row_px: int = 12         # ignore rows with almost no surface in them
    border_margin_px: int = 4    # an edge this close to the frame is a crop
    min_edge_rows_frac: float = 0.25
    snap_px: int = 26            # how far to walk outward onto the paint
    shoulder_frac: float = 0.45  # ridge response still counting as paint
    snap_min_response: float = 6.0
    """Ridge response a point must reach for the snap to move it.

    Below this there is no stripe on that ray, so the point stays on the
    surface edge rather than being dragged onto whatever is brightest."""

    # --- sanity ------------------------------------------------------------
    paint_band: tuple[float, float] = (1.0, 14.0)
    """Inward offsets, in pixels, that should land on the stripe."""
    asphalt_band: tuple[float, float] = (20.0, 40.0)
    """Inward offsets that should land on clean road beyond the stripe."""

    min_contrast: float = 0.12
    """Michelson contrast the winning line must show against the asphalt
    beside it.  Same gate as :class:`chronos.integrity.IntegrityConfig`, and
    here for the same reason: if the stripe is not brighter than what is
    next to it, it is not paint, and a polyline through it is a fiction."""

    horizon_skip_frac: float = 0.18
    """Fraction of the frame to ignore at the top.

    Above the horizon there is no road, and grandstand railings, catch
    fencing and cloud edges are all excellent ridges. Cheap, and it removes
    the largest single source of false candidates."""

    debug_dir: str = "debug"
    debug_name: str = "lane_debug.jpg"


# --------------------------------------------------------------------------
# stage 1 -- ridge response
# --------------------------------------------------------------------------


def ridge_response(gray: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """Bright-thin-structure response, max over the configured widths."""
    g = cv2.GaussianBlur(gray, (cfg.blur_px | 1, cfg.blur_px | 1), 0)
    out = np.zeros_like(g, np.float32)
    for w in cfg.line_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w | 1, w | 1))
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
        np.maximum(out, th.astype(np.float32), out=out)
    return out


def adaptive_mask(resp: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """Threshold the response per row band, against that band's own stats.

    A single global cut loses the far line (low response, small and dim) or
    floods the near field (high response everywhere). Banding by row is the
    cheapest correction that respects perspective.
    """
    h, w = resp.shape
    mask = np.zeros((h, w), np.uint8)
    top = int(h * cfg.horizon_skip_frac)
    edges = np.linspace(top, h, cfg.row_bands + 1).round().astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        band = resp[a:b]
        cut = max(float(band.mean() + cfg.thresh_k * band.std()), cfg.thresh_floor)
        mask[a:b] = (band >= cut).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def road_mask(small: np.ndarray, ridges: np.ndarray,
              cfg: LaneConfig) -> tuple[np.ndarray, str]:
    """Where the drivable surface is: a flood from the bottom that the kerb
    and the paint are allowed to stop.

    Matching the road by colour does not work, and the measurement says why.
    On the Miami clip asphalt sits at Lab L=126, a=+5.0, b=-5.0 and the
    concrete barrier behind the kerb sits at L=143, a=+3.8, b=+0.2 -- the
    same neutral grey, four units apart in chroma. Any tolerance loose
    enough to follow asphalt through a shadow is loose enough to swallow the
    barrier, the pit wall and the grandstand, and it did: the mask covered
    the whole frame and the "outer edge" came back along a guardrail.

    What *is* separable is the furniture between them. Kerbs are strongly
    chromatic wherever they are in the world -- Miami's blue reads chroma
    29, its orange 32, Zandvoort's red similar -- against 7 for asphalt and
    4 for concrete. And the paint itself is a bright ridge. So instead of
    asking what looks like road, this blocks the kerb and the paint and
    floods outward from the surface the camera is standing on. The flood
    stops exactly where the track stops, and it never reaches the barrier
    because the kerb is in the way.

    Hue-agnostic by construction: it keys on chroma magnitude, not on any
    particular colour, so a blue kerb, an orange one and a red one all block
    identically.
    """
    h, w = small.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128.0,
                      lab[:, :, 2].astype(np.float32) - 128.0)

    blocked = (chroma > cfg.chroma_block) | (L > cfg.bright_block)
    if ridges is not None:
        seal = cv2.dilate(ridges, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.ridge_seal_px | 1, cfg.ridge_seal_px | 1)))
        blocked |= seal > 0
    blocked[:int(h * cfg.horizon_skip_frac)] = True

    free = (~blocked).astype(np.uint8) * 255
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(free, 4)
    if n <= 1:
        return np.zeros((h, w), np.uint8), ("lane: nothing unblocked below the "
                                            "horizon -- no drivable surface")
    r0, r1 = (int(h * f) for f in cfg.seed_band)
    c0, c1 = (int(w * f) for f in cfg.seed_cols)
    seeds = lbl[r0:r1, c0:c1].ravel()
    seeds = seeds[seeds > 0]
    if len(seeds) == 0:
        return np.zeros((h, w), np.uint8), ("lane: the bottom of the frame is "
                                            "not drivable surface -- the camera "
                                            "may not be pointed at a track")
    keep = np.bincount(seeds).argmax()
    out = ((lbl == keep).astype(np.uint8)) * 255
    frac = float(out.mean()) / 255.0
    if frac < cfg.road_min_frac:
        return out, (f"lane: drivable surface is only {frac:.1%} of the frame "
                     f"(need {cfg.road_min_frac:.0%})")
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.road_close_px | 1, cfg.road_close_px | 1)))
    return out, ""


def road_adjacency(cand: "Candidate", road: np.ndarray,
                   cfg: LaneConfig) -> float:
    """Fraction of a candidate's length with road within reach sideways."""
    pts = cand.points
    if len(pts) < 3:
        return 0.0
    h, w = road.shape
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    hits = np.zeros(len(pts), bool)
    for sign in (1.0, -1.0):
        for off in (cfg.adjacency_px * 0.5, cfg.adjacency_px):
            q = pts + nrm * (sign * off)
            xs = np.clip(q[:, 0].round().astype(int), 0, w - 1)
            ys = np.clip(q[:, 1].round().astype(int), 0, h - 1)
            hits |= road[ys, xs] > 0
    return float(hits.mean())


# --------------------------------------------------------------------------
# stage 2 -- which blobs are paint
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """One surviving ridge component."""

    points: np.ndarray           # (N,2) skeleton-ish samples, ordered by y
    length: float
    span_y: float
    centre_x: float
    angle_deg: float
    area: float


def _component_points(mask_i: np.ndarray) -> np.ndarray:
    """One point per row of a component: its horizontal centre there.

    A line is single-valued in y over most of its length, so this is a
    faithful and very cheap spine -- and it orders the points for free.
    """
    ys, xs = np.nonzero(mask_i)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    order = np.argsort(ys)
    ys, xs = ys[order], xs[order]
    uy, start = np.unique(ys, return_index=True)
    means = np.add.reduceat(xs.astype(np.float64), start) / np.diff(
        np.append(start, len(xs)))
    return np.stack([means, uy], 1).astype(np.float32)


def find_candidates(mask: np.ndarray, cfg: LaneConfig) -> list[Candidate]:
    """Keep the components that look like paint and discard the rest."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out: list[Candidate] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_area_px:
            continue
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        if bh < cfg.min_span_frac * h:
            continue
        comp = (labels == i).astype(np.uint8)
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        if len(pts) < 6:
            continue
        mean = pts.mean(0)
        u, s, _ = np.linalg.svd(pts - mean, full_matrices=False)
        major, minor = float(s[0]), float(max(s[1], 1e-6))
        if major / minor < cfg.min_elongation:
            continue
        vec = np.linalg.svd(pts - mean, full_matrices=False)[2][0]
        angle = abs(np.degrees(np.arctan2(vec[1], vec[0])))
        angle = min(angle, 180.0 - angle)          # fold to 0..90
        if angle < (90.0 - cfg.max_horizontal_deg):
            continue                                # too horizontal
        spine = _component_points(comp)
        if len(spine) < 6:
            continue
        out.append(Candidate(points=spine,
                             length=float(np.hypot(*np.diff(spine, axis=0).T).sum()),
                             span_y=bh, centre_x=float(mean[0]),
                             angle_deg=angle, area=area))
    return out


# --------------------------------------------------------------------------
# stage 3 -- assemble and choose
# --------------------------------------------------------------------------


def _link(cands: list[Candidate], cfg: LaneConfig) -> list[list[Candidate]]:
    """Group candidates that continue one another.

    Real paint is broken by shadow, by a car crossing it and by the dashes
    it is sometimes painted in, so the pieces have to be put back together
    before any of them is long enough to be believed.
    """
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: cands[i].points[0, 1])
    groups: list[list[int]] = []
    used = set()
    for i in order:
        if i in used:
            continue
        chain, used_now = [i], {i}
        changed = True
        while changed:
            changed = False
            tail = cands[chain[-1]].points[-1]
            for j in order:
                if j in used or j in used_now:
                    continue
                head = cands[j].points[0]
                if (head[1] >= tail[1] - 4
                        and float(np.hypot(*(head - tail))) <= cfg.link_gap_px):
                    chain.append(j)
                    used_now.add(j)
                    tail = cands[j].points[-1]
                    changed = True
                    break
        used |= used_now
        groups.append(chain)
    return [[cands[i] for i in g] for g in groups]


def _side_of(groups, w: int, cfg: LaneConfig) -> str:
    """Which side of frame the track limit is on.

    Decided from the paint itself. The longest run of paint is taken as the
    reference stripe and the side is whichever half of the frame it sits in
    -- no hue, no kerb, so a blue kerb and a red kerb behave identically.
    """
    if cfg.side != "auto":
        return cfg.side
    best = max(groups, key=lambda g: sum(c.length for c in g))
    x = float(np.mean([c.centre_x for c in best]))
    return "right" if x >= w * 0.5 else "left"


def _score(group, w: int, side: str, cfg: LaneConfig) -> float:
    """Rank a linked run of paint as a track-limit candidate.

    Length and outerness, in that balance. Outerness matters because a pit
    box, a grid slot and a sector line are all white paint nearer the middle
    of the road; the limit is the last line before the car leaves the track.
    """
    length = sum(c.length for c in group)
    x = float(np.mean([c.centre_x for c in group]))
    outer = (x / w) if side == "right" else (1.0 - x / w)
    return cfg.length_weight * (length / w) + cfg.outer_weight * outer


def _contrast_of(gray: np.ndarray, poly: np.ndarray, side: str,
                 cfg: LaneConfig) -> tuple[float, float, float]:
    """Michelson contrast of the paint against the asphalt inside it.

    Both samples are taken INWARD, toward the road, and that matters. The
    polyline sits on the outer edge of the stripe -- the surface the
    regulation names -- so sampling symmetrically across it averages paint
    against kerb and returns ~0 for a perfectly good line, which is what an
    earlier version of this did on every frame of the Miami clip.

    Inward of the boundary lies paint, then asphalt. Comparing those two is
    the same measurement :mod:`chronos.integrity` makes, so a line that
    passes here is a line that engine can also score.

    Returns (contrast, paint luminance, asphalt luminance).
    """
    if len(poly) < 3:
        return 0.0, 0.0, 0.0
    d = np.gradient(poly, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    # point the normal at the road
    if (side == "right") == (nrm[:, 0].mean() > 0):
        nrm = -nrm
    h, w = gray.shape

    def band(a: float, b: float) -> np.ndarray:
        vals = []
        for off in np.linspace(a, b, max(2, int(b - a))):
            q = poly + nrm * off
            xs = np.clip(q[:, 0].round().astype(int), 0, w - 1)
            ys = np.clip(q[:, 1].round().astype(int), 0, h - 1)
            vals.append(gray[ys, xs].astype(np.float32))
        return np.mean(vals, 0)

    # Peak across the stripe, not a mean over a fixed band: the line's width
    # in pixels changes along its own length under perspective, so a band
    # wide enough to cover it near the camera also swallows asphalt far away
    # and drags the reading down. The brightest point on the ray is on the
    # paint wherever the paint happens to be.
    lo, hi = cfg.paint_band
    paint = np.max([band(o, o + 1.0) for o in np.arange(lo, hi, 1.0)], 0)
    asphalt = band(*cfg.asphalt_band)
    c = float(np.median((paint - asphalt) / np.maximum(paint + asphalt, 1e-6)))
    return c, float(np.mean(paint)), float(np.mean(asphalt))


def road_edge(road: np.ndarray, side: str, cfg: LaneConfig
              ) -> tuple[np.ndarray, float]:
    """The drivable surface's edge on one side, row by row.

    Once the flood has stopped in the right place, the boundary is simply
    where it stopped. Rows where the surface runs off the side of the frame
    are dropped -- there is no boundary visible there, and including them
    would pin the polyline to the image border.

    Returns the (N,2) edge and the fraction of rows that were usable.
    """
    h, w = road.shape
    xs, ys, seen = [], [], 0
    m = cfg.border_margin_px
    for y in range(h):
        row = np.nonzero(road[y])[0]
        if len(row) < cfg.min_row_px:
            continue
        seen += 1
        x = row.max() if side == "right" else row.min()
        if x >= w - 1 - m or x <= m:
            continue                      # runs off the frame: not a boundary
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return np.zeros((0, 2), np.float32), 0.0
    return (np.stack([xs, ys], 1).astype(np.float32),
            len(xs) / max(seen, 1))


def choose_side(road: np.ndarray, cfg: LaneConfig) -> tuple[str, float, float]:
    """Which side of the surface carries the track limit.

    The rule needs no hue and no kerb: the track limit is the side where the
    drivable surface **ends inside the frame**. On the other side the road
    simply runs out of picture, which is a crop, not a boundary. That makes
    it work identically at a circuit with red kerbs, blue kerbs or none.
    """
    _, left = road_edge(road, "left", cfg)
    _, right = road_edge(road, "right", cfg)
    if cfg.side != "auto":
        return cfg.side, left, right
    return ("right" if right >= left else "left"), left, right


def snap_to_paint(resp: np.ndarray, edge: np.ndarray, side: str,
                  cfg: LaneConfig) -> tuple[np.ndarray, float]:
    """Move each edge point onto the outer shoulder of the paint stripe.

    Snapping is done on the RIDGE RESPONSE, not on raw brightness, and that
    distinction is the whole of it. Walking outward in brightness does not
    work at a circuit: beyond the white line lies a bright kerb and beyond
    that pale concrete, so the profile rises into the paint and never comes
    back down -- measured on this clip as
    ``139,139,139,134,139,151,214,205,210,193,193,193...``, which sent an
    earlier version 24 px past the line and into the kerb.

    The top-hat response does come back down, because it answers a different
    question: it is large only for structures THINNER than its kernel. A
    painted line is thin and peaks; a kerb and a barrier are broad and read
    near zero however bright they are.

    Returns the moved points and the mean distance moved, which is reported
    so a snap that has clearly run away is visible rather than silent.
    """
    h, w = resp.shape
    out = edge.copy()
    step = 1 if side == "right" else -1
    moved = []
    for i, (x, y) in enumerate(edge):
        xi, yi = int(round(x)), int(round(y))
        peak_x, peak_v = None, 0.0
        for d in range(0, cfg.snap_px + 1):
            xx = xi + step * d
            if not (0 <= xx < w):
                break
            v = float(resp[yi, xx])
            if v > peak_v:
                peak_v, peak_x = v, xx
        if peak_x is None or peak_v < cfg.snap_min_response:
            moved.append(0.0)
            continue
        # outer shoulder: walk out while the ridge is still meaningfully up
        cut = peak_v * cfg.shoulder_frac
        xx = peak_x
        while 0 <= xx + step < w and float(resp[yi, xx + step]) >= cut:
            xx += step
            if abs(xx - xi) > cfg.snap_px:
                break
        out[i, 0] = float(xx)
        moved.append(abs(xx - xi))
    return out, float(np.mean(moved)) if moved else 0.0


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------


def detect_lane(frame: np.ndarray, cfg: Optional[LaneConfig] = None,
                save_debug: bool = False) -> BoundaryResult:
    """Find the track-limit line in one real frame.

    Returns a :class:`chronos.boundary.BoundaryResult` in FULL-FRAME pixel
    coordinates, so callers never have to know this ran at a working scale.
    """
    cfg = cfg or LaneConfig()
    if frame is None or frame.ndim != 3 or frame.dtype != np.uint8:
        raise ValueError("detect_lane: expected a BGR uint8 frame")

    h0, w0 = frame.shape[:2]
    scale = cfg.work_width / float(w0)
    small = (cv2.resize(frame, (cfg.work_width, max(2, int(round(h0 * scale)))),
                        interpolation=cv2.INTER_AREA) if abs(scale - 1) > 1e-3
             else frame.copy())
    h, w = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    resp = ridge_response(gray, cfg)
    mask = adaptive_mask(resp, cfg)

    road, road_reason = road_mask(small, mask, cfg)
    if road_reason:
        return BoundaryResult(False, road_reason,
                              drivable_mask=cv2.resize(road, (w0, h0),
                                                       interpolation=cv2.INTER_NEAREST))

    side, f_left, f_right = choose_side(road, cfg)
    edge, frac = road_edge(road, side, cfg)
    if len(edge) < cfg.min_points or frac < cfg.min_edge_rows_frac:
        return BoundaryResult(
            False,
            f"lane: the drivable surface does not end inside the frame on "
            f"either side (left {f_left:.0%}, right {f_right:.0%} of rows) -- "
            f"the track limit is out of shot",
            drivable_mask=cv2.resize(road, (w0, h0),
                                     interpolation=cv2.INTER_NEAREST))

    edge, snap_mean = snap_to_paint(resp, edge, side, cfg)
    poly = _smooth_polyline(
        _resample_polyline(edge, min(cfg.n_bins, len(edge))), cfg.smooth_window)

    contrast, paint_l, asph_l = _contrast_of(gray, poly, side, cfg)
    if contrast < cfg.min_contrast:
        return BoundaryResult(
            False, f"lane: the surface ends, but not at a painted line -- "
                   f"paint band reads {paint_l:.0f}, asphalt band {asph_l:.0f}, "
                   f"Michelson {contrast:+.3f} (need {cfg.min_contrast:.2f})",
            drivable_mask=cv2.resize(road, (w0, h0),
                                     interpolation=cv2.INTER_NEAREST))

    # The paint was dilated into a wall to stop the flood, so the road mask
    # stops a seal's width short of the line it stopped at. Downstream that
    # gap is fatal rather than cosmetic: build_track_frame orients its
    # normals by probing 6 px either side of the boundary for road, finds
    # none on either side, and the inward march dies at once. So the surface
    # is extended, row by row, out to the boundary it belongs to.
    filled = road.copy()
    for x, y in poly:
        yi = int(round(y))
        if not (0 <= yi < h):
            continue
        row = np.nonzero(road[yi])[0]
        if len(row) == 0:
            continue
        xi = int(np.clip(round(x), 0, w - 1))
        if side == "right":
            filled[yi, min(row.max(), xi):xi + 1] = 255
        else:
            filled[yi, xi:max(row.min(), xi) + 1] = 255

    full = poly / scale if abs(scale - 1) > 1e-3 else poly
    drivable = cv2.resize(filled, (w0, h0), interpolation=cv2.INTER_NEAREST)
    res = BoundaryResult(
        True,
        f"lane: surface edge, side {side} ({frac:.0%} of rows bounded in "
        f"frame), snapped to paint, contrast {contrast:+.2f}, spans "
        f"{np.ptp(full[:, 1]) / h0:.0%} of frame height",
        polyline=full.astype(np.float32),
        drivable_mask=drivable,
        kerb_mask=np.zeros((h0, w0), np.uint8))
    if save_debug:
        res.debug_path = save_debug_image(frame, res, resp, mask, cfg, road)
    return res


def save_debug_image(frame, res: BoundaryResult, resp, mask,
                     cfg: LaneConfig, road=None) -> str:
    """Four panels: frame+line, ridge response, mask, road surface."""
    h0, w0 = frame.shape[:2]

    def panel(img, text):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (w0, h0))
        cv2.rectangle(img, (0, 0), (w0, 26), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    over = frame.copy()
    if res.polyline is not None:
        cv2.polylines(over, [np.round(res.polyline).astype(np.int32)], False,
                      (0, 255, 0), 3, cv2.LINE_AA)
    r8 = np.clip(resp / max(resp.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    if road is None:
        road_vis = np.zeros_like(frame)
    else:
        road_vis = frame.copy()
        rr = cv2.resize(road, (w0, h0), interpolation=cv2.INTER_NEAREST)
        road_vis[rr > 0] = (0.45 * road_vis[rr > 0]
                            + 0.55 * np.array([255, 90, 0])).astype(np.uint8)
    if res.polyline is not None:
        cv2.polylines(road_vis, [np.round(res.polyline).astype(np.int32)],
                      False, (0, 255, 0), 2, cv2.LINE_AA)
    top = np.hstack([panel(over, res.reason[:96]), panel(r8, "ridge response")])
    bot = np.hstack([panel(mask, "thresholded ridges"),
                     panel(road_vis, "road surface (seeded from bottom)")])
    os.makedirs(cfg.debug_dir, exist_ok=True)
    path = os.path.join(cfg.debug_dir, cfg.debug_name)
    cv2.imwrite(path, np.vstack([top, bot]))
    return path


def _cli() -> None:
    p = argparse.ArgumentParser(description="ridge-based track-limit line")
    p.add_argument("--image")
    p.add_argument("--video")
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--out", default="debug")
    a = p.parse_args()
    cfg = LaneConfig(debug_dir=a.out)
    if a.image:
        im = cv2.imread(a.image)
        cfg.debug_name = os.path.splitext(os.path.basename(a.image))[0] + "_lane.jpg"
        r = detect_lane(im, cfg, save_debug=True)
        print(("OK   " if r.ok else "FAIL ") + r.reason)
        print("debug:", r.debug_path)
    elif a.video:
        cap = cv2.VideoCapture(a.video)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(n - 1, 0), a.frames).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if not ok:
                continue
            cfg.debug_name = f"lane_f{int(i):04d}.jpg"
            r = detect_lane(im, cfg, save_debug=True)
            print(f"f{int(i):<5} " + ("OK   " if r.ok else "FAIL ") + r.reason[:96])
        cap.release()


if __name__ == "__main__":
    _cli()

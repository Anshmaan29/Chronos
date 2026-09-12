"""CHRONOS Module 3 -- controlled degradation of a track boundary.

Takes a clean frame and a level in 0..1 and returns the same frame with a
chosen kind of contamination laid over the racing line.  This is the demo
generator and the benchmark's independent variable: no dirty footage needed,
and the amount of degradation is a number we set rather than one we guess.

``rubber`` is the case the pitch is built on -- Miami, 1 May 2026, where
support-race rubber over the white line at Turn 6 defeated the FIA's system.

Two properties matter more than realism and are tested:

  * **level 0.0 is a genuine no-op** -- the returned frame is bit-identical to
    the input, so a clean baseline is measured on the real clean frame.
  * **the pattern grows, it does not reshuffle.** Every streak is generated
    once from a fixed seed and fades in as the level rises, so dragging the
    contamination slider makes the SAME rubber build up.  A pattern that
    resampled per frame would flicker, and the slider is the whole pitch.

Run standalone::

    python -m chronos.degrade --image data/frame.jpg --kind rubber
    python -m chronos.degrade --image data/frame.jpg --kind dust --levels 0,0.5,1
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

from chronos.boundary import (BoundaryConfig, BoundaryResult, detect_boundary,
                              _resample_polyline)

DegradeKind = Literal["rubber", "dust", "wet", "fade", "glare", "shadow"]
KINDS: tuple[DegradeKind, ...] = ("rubber", "dust", "wet", "fade", "glare", "shadow")


# --------------------------------------------------------------------------
# CONFIG -- tune these live.  No magic numbers below this block.
# --------------------------------------------------------------------------


@dataclass
class DegradeConfig:
    """Tunables for every degradation kind.

    Lateral positions are given as a fraction of the LOCAL track width, not in
    pixels, so a streak sits in the same place on the road whether it is 8 px
    or 180 px wide on screen.  0.0 is the track limit itself, positive is
    inward across the track, negative is outward over the kerb.
    """

    # --- rubber: the Miami case ------------------------------------------
    rubber_streaks: int = 34
    rubber_inner_frac: float = 0.42     # how far in across the track rubber reaches
    rubber_outer_frac: float = -0.055   # dragged OVER the line, onto the kerb
    rubber_peak_frac: float = 0.015     # cars run right at the limit at a corner
                                        # exit -- this is the Miami case, and if
                                        # the deposit misses the paint the whole
                                        # premise of the demo misses with it
    rubber_spread_frac: float = 0.075   # decay length inward from the peak, for
                                        # the broad visual band across the exit
    rubber_over_line_frac: float = 0.38 # share of streaks dragged outward across
                                        # the line -- literally the Miami failure
    # A 0.18 m line on a 12 m road is 1.5% of the track width.  A deposit spread
    # over 7% of the width therefore lands almost entirely on ASPHALT and leaves
    # the paint clean -- which looks like contamination and measures like none.
    # So rubber is two populations: a broad band that sells it to the eye, and
    # narrow streaks right on the paint that are the thing actually being
    # measured.  Both are real; a corner exit has both.
    rubber_line_share: float = 0.62      # share of streaks laid on the paint
    rubber_line_span: tuple[float, float] = (-0.006, 0.014)  # lateral range
    rubber_line_half_width: tuple[float, float] = (0.003, 0.010)
    rubber_line_alpha: tuple[float, float] = (0.70, 1.0)
    rubber_line_patchiness: float = 0.30 # rubber ON the line is more continuous
                                         # than the broad band beside it
    rubber_half_width: tuple[float, float] = (0.014, 0.055)   # streak half-width
    rubber_wobble_frac: float = 0.035   # drivers do not hold one line exactly
    rubber_wobble_cycles: tuple[float, float] = (0.5, 2.2)
    rubber_segments: int = 14           # along-track pieces per streak: rubber is
                                        # laid down in patches, not painted evenly
    rubber_patchiness: float = 0.55     # 0 = even along the track, 1 = very patchy
    rubber_color: tuple[int, int, int] = (33, 31, 32)         # BGR
    rubber_max_alpha: float = 0.92
    rubber_streak_alpha: tuple[float, float] = (0.40, 1.0)
    rubber_noise_px: float = 15.0       # texture scale of the deposit
    rubber_noise_floor: float = 0.34    # 1.0 = no texture at all
    rubber_feather_px: float = 2.2
    rubber_soften_px: float = 2.2       # rubber also blurs the paint edge

    # --- dust / sand ------------------------------------------------------
    dust_color: tuple[int, int, int] = (150, 166, 184)
    dust_inner_frac: float = 0.34
    dust_outer_frac: float = -0.45      # dust collects off-line, over the kerb
    dust_max_alpha: float = 0.80
    dust_noise_px: float = 34.0
    dust_noise_floor: float = 0.25
    dust_feather_px: float = 9.0

    # --- standing water ---------------------------------------------------
    wet_inner_frac: float = 0.55
    wet_outer_frac: float = -0.20
    wet_darken: float = 0.42            # wet asphalt is darker
    wet_skylight: float = 34.0          # ...but reflects the sky, which lifts the
                                        # blacks and crushes contrast
    wet_sheen: float = 0.50             # ...and throws specular highlights
    wet_sheen_color: tuple[int, int, int] = (206, 202, 196)
    wet_noise_px: float = 40.0
    wet_blur_px: float = 3.0
    wet_feather_px: float = 7.0

    # --- worn paint -------------------------------------------------------
    fade_max: float = 0.96              # how far the paint goes toward asphalt
    fade_noise_px: float = 18.0
    fade_noise_floor: float = 0.38      # worn paint wears fairly evenly; a low
                                        # floor leaves bright patches that keep
                                        # the contrast measurement alive
    fade_sat_max: int = 60              # what counts as paint, for wearing away
    fade_val_min: int = 0               # 0 = pick per frame (Otsu), as elsewhere

    # --- sun glare --------------------------------------------------------
    glare_color: tuple[int, int, int] = (238, 242, 248)
    glare_span: float = 0.60            # fraction of the boundary it covers
    glare_radius_frac: float = 0.42     # of the frame's smaller side
    glare_max_alpha: float = 0.95
    glare_wash: float = 0.80            # local contrast crushed inside the bloom

    # --- grandstand shadow -------------------------------------------------
    shadow_span: float = 0.42
    shadow_darken: float = 0.66
    shadow_ambient: float = 26.0        # skylight still falls into a shadow.
                                        # Without this additive floor the shadow
                                        # is a pure multiply, and a pure multiply
                                        # leaves Michelson contrast EXACTLY
                                        # unchanged -- the degradation would be
                                        # invisible to the contrast sub-score,
                                        # which is a physics error, not a tuning
                                        # one.
    shadow_cool: float = 0.10           # shadows go blue, which matters to HSV
    shadow_width_frac: float = 0.34
    shadow_feather_px: float = 13.0

    seed: int = 7


# --------------------------------------------------------------------------
# track-relative geometry
# --------------------------------------------------------------------------


@dataclass
class TrackFrame:
    """A coordinate frame that follows the boundary across the road.

    Attributes
    ----------
    points:   (N, 2) resampled boundary points, the track limit itself.
    normals:  (N, 2) unit vectors pointing INWARD, across the track.
    widths:   (N,) local track width in pixels, measured from each point.
    surface:  0/255 mask of everywhere contamination may land -- road and kerb.
              Rubber that reaches the grass is not rubber, it is a bug.
    """

    points: np.ndarray
    normals: np.ndarray
    widths: np.ndarray
    surface: Optional[np.ndarray] = None

    def offset(self, frac: np.ndarray | float) -> np.ndarray:
        """Points at a lateral offset given as a fraction of local width."""
        f = np.asarray(frac, np.float64).reshape(-1, 1)
        return self.points + f * self.widths[:, None] * self.normals


def build_track_frame(boundary: np.ndarray, drivable_mask: np.ndarray,
                      kerb_mask: Optional[np.ndarray] = None,
                      n: int = 220, max_width_px: int = 900) -> TrackFrame:
    """Build an inward-pointing frame along the boundary.

    The inward direction and the local road width are both read off the
    drivable mask, so contamination lands on the road rather than in the
    run-off, at every distance from the camera.

    Raises
    ------
    ValueError
        If the boundary is too short, or the mask is empty -- both mean the
        caller has nothing to lay contamination onto.
    """
    if boundary is None or len(boundary) < 4:
        raise ValueError("build_track_frame: boundary needs at least 4 points")
    if drivable_mask is None or not drivable_mask.any():
        raise ValueError("build_track_frame: drivable mask is empty")

    h, w = drivable_mask.shape
    pts = _resample_polyline(np.asarray(boundary, np.float32), n).astype(np.float64)

    t = np.gradient(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-t[:, 1], t[:, 0]], axis=1)

    def hits(p: np.ndarray) -> np.ndarray:
        xi = np.clip(p[:, 0].astype(int), 0, w - 1)
        yi = np.clip(p[:, 1].astype(int), 0, h - 1)
        return drivable_mask[yi, xi] > 0

    # orient inward: whichever normal lands on the road more often
    probe = 6.0
    if hits(pts + probe * normals).mean() < hits(pts - probe * normals).mean():
        normals = -normals

    # march inward until the road runs out
    widths = np.zeros(len(pts))
    live = np.ones(len(pts), bool)
    step = 2.0
    d = step
    while live.any() and d < max_width_px:
        on = hits(pts + d * normals)
        widths[live & on] = d
        live &= on
        d += step

    # a point whose march died immediately is on a ragged edge; lean on its
    # neighbours rather than dropping contamination into the run-off there
    good = widths > 4
    if not good.any():
        raise ValueError("build_track_frame: road has no measurable width "
                         "inward of the boundary")
    idx = np.arange(len(pts))
    widths = np.interp(idx, idx[good], widths[good])
    widths = cv2.GaussianBlur(widths.reshape(-1, 1).astype(np.float32),
                              (1, 9), 0).reshape(-1).astype(np.float64)

    surface = drivable_mask.copy()
    if kerb_mask is not None and kerb_mask.any():
        surface = cv2.bitwise_or(surface, kerb_mask)
    surface = cv2.dilate(surface, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return TrackFrame(pts.astype(np.float32), normals, widths, surface)


def _ribbon(shape: tuple[int, int], tf: TrackFrame,
            lo: np.ndarray, hi: np.ndarray, value: float = 1.0) -> np.ndarray:
    """Rasterise the band between two lateral offset curves."""
    a = tf.offset(lo)
    b = tf.offset(hi)
    poly = np.concatenate([a, b[::-1]], axis=0)
    canvas = np.zeros(shape, np.float32)
    cv2.fillPoly(canvas, [np.round(poly).astype(np.int32)], float(value),
                 lineType=cv2.LINE_AA)
    return canvas


def _noise_field(shape: tuple[int, int], scale_px: float, floor: float,
                 rng: np.random.Generator, octaves: int = 3) -> np.ndarray:
    """Smooth multi-octave noise in [floor, 1], for texturing a deposit."""
    h, w = shape
    field = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(max(1, octaves)):
        s = max(2.0, scale_px / (2 ** o))
        small = rng.random((max(2, int(h / s)), max(2, int(w / s)))).astype(np.float32)
        field += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    field /= total
    field -= field.min()
    field /= max(float(field.max()), 1e-6)
    return floor + (1.0 - floor) * field


def _span_window(n: int, centre: float, span: float) -> np.ndarray:
    """A smooth 0..1 window over a fraction ``span`` of a polyline."""
    x = (np.arange(n) / max(n - 1, 1) - centre) / max(span / 2, 1e-6)
    return np.clip(1.0 - x ** 2, 0.0, 1.0) ** 1.5


# --------------------------------------------------------------------------
# the six kinds
# --------------------------------------------------------------------------


def _alpha_rubber(shape, tf: TrackFrame, level: float,
                  cfg: DegradeConfig) -> np.ndarray:
    """Alpha map for tyre rubber laid down over the track limit.

    Each streak is a ribbon in track coordinates, cut into segments along the
    track so the deposit is patchy the way real rubber is, and every streak is
    drawn from the SAME seed on every call, fading in at its own level.  Raise
    the slider and the existing marks thicken; nothing is re-dealt.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(tf.points)
    s_axis = np.linspace(0.0, 1.0, n)
    lo_f, hi_f = cfg.rubber_outer_frac, cfg.rubber_inner_frac
    nseg = max(1, cfg.rubber_segments)

    alpha = np.zeros(shape, np.float32)
    for k in range(cfg.rubber_streaks):
        onset = 0.86 * (k / max(cfg.rubber_streaks - 1, 1))
        weight = float(np.clip((level - onset) / 0.20, 0.0, 1.0))
        peak = cfg.rubber_peak_frac
        # sampled regardless of weight so the RNG stream stays aligned and a
        # streak keeps its shape as the level rises
        on_line = rng.random() < cfg.rubber_line_share
        if on_line:
            u = float(rng.uniform(*cfg.rubber_line_span))
            half = float(rng.uniform(*cfg.rubber_line_half_width))
        else:
            u = peak + cfg.rubber_spread_frac * float(rng.exponential(1.0))
            if rng.random() < cfg.rubber_over_line_frac:
                u -= float(rng.uniform(0.0, peak - lo_f))
            half = float(rng.uniform(*cfg.rubber_half_width))
        u = float(np.clip(u, lo_f, hi_f))
        cycles = float(rng.uniform(*cfg.rubber_wobble_cycles))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        base_a = float(rng.uniform(*(cfg.rubber_line_alpha if on_line
                                     else cfg.rubber_streak_alpha)))
        patch = rng.random(nseg + 2)
        if weight <= 0.0:
            continue

        wobble = cfg.rubber_wobble_frac * np.sin(2 * np.pi * cycles * s_axis + phase)
        centre = u + wobble
        # along-track patchiness, smoothed so segments do not read as bricks
        prof = np.interp(s_axis, np.linspace(0, 1, len(patch)), patch)
        prof = 1.0 - (cfg.rubber_line_patchiness if on_line
                      else cfg.rubber_patchiness) * prof

        edges = np.linspace(0, n, nseg + 1).astype(int)
        for j in range(nseg):
            a0, b0 = edges[j], min(edges[j + 1] + 1, n)
            if b0 - a0 < 2:
                continue
            sub = TrackFrame(tf.points[a0:b0], tf.normals[a0:b0], tf.widths[a0:b0])
            a_seg = weight * base_a * float(prof[a0:b0].mean())
            if a_seg <= 0.01:
                continue
            alpha = np.maximum(alpha, _ribbon(shape, sub, centre[a0:b0] - half,
                                              centre[a0:b0] + half, a_seg))

    if alpha.max() <= 0.0:
        return alpha
    alpha *= _noise_field(shape, cfg.rubber_noise_px, cfg.rubber_noise_floor, rng)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(cfg.rubber_feather_px, 0.6))
    alpha = np.clip(alpha * cfg.rubber_max_alpha * (0.55 + 0.45 * level), 0.0, 1.0)
    return _clip_to_surface(alpha, tf, cfg.rubber_feather_px)


def _alpha_dust(shape, tf: TrackFrame, level: float, cfg: DegradeConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 1)
    n = len(tf.points)
    lo = np.full(n, cfg.dust_outer_frac * (0.35 + 0.65 * level))
    hi = np.full(n, cfg.dust_inner_frac * (0.35 + 0.65 * level))
    alpha = _ribbon(shape, tf, lo, hi, 1.0)
    alpha *= _noise_field(shape, cfg.dust_noise_px, cfg.dust_noise_floor, rng)
    alpha = cv2.GaussianBlur(alpha, (0, 0), cfg.dust_feather_px)
    alpha = np.clip(alpha * cfg.dust_max_alpha * level, 0.0, 1.0)
    return _clip_to_surface(alpha, tf, cfg.dust_feather_px)


def _alpha_wet(shape, tf: TrackFrame, level: float, cfg: DegradeConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 2)
    n = len(tf.points)
    lo = np.full(n, cfg.wet_outer_frac)
    hi = np.full(n, cfg.wet_inner_frac)
    alpha = _ribbon(shape, tf, lo, hi, 1.0)
    alpha *= _noise_field(shape, cfg.wet_noise_px, 0.35, rng)
    alpha = cv2.GaussianBlur(alpha, (0, 0), cfg.wet_feather_px)
    alpha = np.clip(alpha * level, 0.0, 1.0)
    return _clip_to_surface(alpha, tf, cfg.wet_feather_px)


def _alpha_band(shape, tf: TrackFrame, level: float, centre: float, span: float,
                width_frac: float, feather_px: float) -> np.ndarray:
    """A band running ACROSS the track over part of the boundary's length."""
    n = len(tf.points)
    window = _span_window(n, centre, span)
    lo = -width_frac * np.ones(n)
    hi = (width_frac + 1.2) * np.ones(n)
    canvas = np.zeros(shape, np.float32)
    live = window > 0.02
    if live.sum() >= 3:
        sub = TrackFrame(tf.points[live], tf.normals[live], tf.widths[live])
        canvas = _ribbon(shape, sub, lo[live], hi[live], 1.0)
    canvas = cv2.GaussianBlur(canvas, (0, 0), feather_px)
    return np.clip(canvas * level, 0.0, 1.0)


def _clip_to_surface(alpha: np.ndarray, tf: TrackFrame, feather_px: float) -> np.ndarray:
    """Confine a deposit to the road and kerb, with a soft edge."""
    if tf.surface is None:
        return alpha
    keep = (tf.surface > 0).astype(np.float32)
    if feather_px > 0.5:
        keep = cv2.GaussianBlur(keep, (0, 0), feather_px * 0.6)
    return alpha * keep


def _tone_without(frame: np.ndarray, exclude: np.ndarray,
                  radius: int = 41) -> np.ndarray:
    """Local colour of the surface with ``exclude`` left out of the average.

    A normalised convolution: blur the frame weighted by (1 - exclude) and
    divide by the blurred weight, so the result is what the road looks like
    where the paint is not.
    """
    wgt = (1.0 - np.clip(exclude, 0.0, 1.0)).astype(np.float32)
    k = (radius | 1, radius | 1)
    num = cv2.blur(frame.astype(np.float32) * wgt[:, :, None], k)
    den = np.maximum(cv2.blur(wgt, k), 1e-3)[:, :, None]
    return num / den


def _apply_alpha(frame: np.ndarray, alpha: np.ndarray,
                 color: tuple[int, int, int]) -> np.ndarray:
    a = alpha[:, :, None]
    tgt = np.asarray(color, np.float32).reshape(1, 1, 3)
    return frame.astype(np.float32) * (1.0 - a) + tgt * a


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def degrade(frame: np.ndarray, level: float, kind: DegradeKind = "rubber",
            boundary: Optional[np.ndarray] = None,
            drivable_mask: Optional[np.ndarray] = None,
            cfg: Optional[DegradeConfig] = None,
            track_frame: Optional[TrackFrame] = None) -> np.ndarray:
    """Lay a controlled amount of contamination over the track boundary.

    Parameters
    ----------
    frame:
        Clean BGR uint8 image.
    level:
        0.0 to 1.0.  **0.0 returns an exact copy** -- a clean baseline must be
        measured on the genuinely clean frame.
    kind:
        One of ``rubber``, ``dust``, ``wet``, ``fade``, ``glare``, ``shadow``.
    boundary, drivable_mask:
        The track limit and road mask from Module 1.  Detected here if not
        supplied -- pass them in when sweeping or driving a slider, because
        re-detecting per frame costs ~90 ms and the geometry has not changed.
    track_frame:
        A prebuilt :class:`TrackFrame`; supersedes ``boundary``/``drivable_mask``.

    Returns
    -------
    A new BGR uint8 frame.  The input is never modified.

    Raises
    ------
    ValueError
        On a level outside 0..1, an unknown kind, or a frame with no findable
        boundary to contaminate -- silence here would be a rigged benchmark.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("degrade: expected a BGR image")
    if frame.dtype != np.uint8:
        raise ValueError(f"degrade: expected uint8, got {frame.dtype}")
    if not np.isfinite(level) or not (0.0 <= level <= 1.0):
        raise ValueError(f"degrade: level must be in 0..1, got {level!r}")
    if kind not in KINDS:
        raise ValueError(f"degrade: unknown kind {kind!r}, expected one of {KINDS}")

    if level == 0.0:
        return frame.copy()

    cfg = cfg or DegradeConfig()
    shape = frame.shape[:2]

    if track_frame is None:
        if boundary is None or drivable_mask is None:
            res = detect_boundary(frame, BoundaryConfig(), save_debug=False)
            if not res.ok:
                raise ValueError(
                    f"degrade: cannot place {kind} -- no boundary in this frame "
                    f"({res.reason})")
            boundary = res.polyline if boundary is None else boundary
            drivable_mask = res.drivable_mask if drivable_mask is None else drivable_mask
            track_frame = build_track_frame(boundary, drivable_mask, res.kerb_mask)
        else:
            track_frame = build_track_frame(boundary, drivable_mask)
    tf = track_frame

    out = frame.astype(np.float32)

    if kind == "rubber":
        alpha = _alpha_rubber(shape, tf, level, cfg)
        soften = alpha
        if cfg.rubber_soften_px > 0:
            blurred = cv2.GaussianBlur(frame, (0, 0), cfg.rubber_soften_px * level + 1e-3)
            s = (soften * 0.6)[:, :, None]
            out = out * (1.0 - s) + blurred.astype(np.float32) * s
        out = _apply_alpha(out.astype(np.uint8), alpha, cfg.rubber_color)

    elif kind == "dust":
        out = _apply_alpha(frame, _alpha_dust(shape, tf, level, cfg), cfg.dust_color)

    elif kind == "wet":
        rng = np.random.default_rng(cfg.seed + 12)
        alpha = _alpha_wet(shape, tf, level, cfg)
        a = alpha[:, :, None]
        out = out * (1.0 - cfg.wet_darken * a) + cfg.wet_skylight * a
        sheen = _noise_field(shape, cfg.wet_noise_px * 0.6, 0.0, rng) ** 3
        sheen = (alpha * sheen * cfg.wet_sheen * level)[:, :, None]
        out = out * (1.0 - sheen) + np.asarray(cfg.wet_sheen_color, np.float32) * sheen
        blurred = cv2.GaussianBlur(out.astype(np.float32), (0, 0),
                                   cfg.wet_blur_px * level + 1e-3)
        out = out * (1.0 - a * 0.7) + blurred * (a * 0.7)

    elif kind == "fade":
        rng = np.random.default_rng(cfg.seed + 3)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        vmin = float(cfg.fade_val_min)
        if vmin <= 0:
            road = _ribbon(shape, tf, np.full(len(tf.points), -0.05),
                           np.full(len(tf.points), 0.9)) > 0.5
            pool = val[road] if road.any() else val.reshape(-1)
            vmin = (float(cv2.threshold(pool.reshape(-1, 1), 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                    if len(pool) >= 32 else 150.0)
        paint = ((sat <= cfg.fade_sat_max) & (val >= vmin)).astype(np.float32)
        paint = cv2.GaussianBlur(paint, (0, 0), 1.2)
        wear = paint * _noise_field(shape, cfg.fade_noise_px, cfg.fade_noise_floor, rng)
        wear = np.clip(wear * cfg.fade_max * level, 0.0, 1.0)[:, :, None]
        # Worn paint tends toward the road UNDER it.  Blurring the frame would
        # smear the line into its own replacement and stop the fade short of
        # the asphalt, so the road colour is estimated with the paint left out.
        out = out * (1.0 - wear) + _tone_without(frame, paint) * wear

    elif kind == "glare":
        n = len(tf.points)
        window = _span_window(n, 0.42, cfg.glare_span)
        centre = tf.offset(np.full(n, 0.18))[int(np.argmax(window))]
        r = cfg.glare_radius_frac * min(shape)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        d = np.sqrt((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / max(r, 1.0)
        alpha = np.clip(1.0 - d ** 2, 0.0, 1.0) ** 1.4 * cfg.glare_max_alpha * level
        # a bloom both adds light and crushes the contrast underneath it
        mean = float(out.mean())
        out = mean + (out - mean) * (1.0 - cfg.glare_wash * alpha[:, :, None])
        out = _apply_alpha(out.astype(np.uint8), alpha, cfg.glare_color)

    elif kind == "shadow":
        alpha = _alpha_band(shape, tf, 1.0, 0.55, cfg.shadow_span,
                            cfg.shadow_width_frac, cfg.shadow_feather_px)
        a = (alpha * level)[:, :, None]
        cool = np.asarray([1.0 + cfg.shadow_cool, 1.0, 1.0 - cfg.shadow_cool], np.float32)
        out = out * (1.0 - cfg.shadow_darken * a) + cfg.shadow_ambient * a
        out = out * (1.0 - a + a * cool)

    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# debug view
# --------------------------------------------------------------------------


def save_debug_image(frame: np.ndarray, kind: DegradeKind, levels: list[float],
                     cfg: Optional[DegradeConfig] = None,
                     res: Optional[BoundaryResult] = None,
                     path: str = "debug/degrade_debug.jpg") -> str:
    """Contact sheet of one degradation kind across several levels."""
    cfg = cfg or DegradeConfig()
    if res is None:
        res = detect_boundary(frame, BoundaryConfig(), save_debug=False)
    if not res.ok:
        raise ValueError(f"save_debug_image: no boundary to degrade ({res.reason})")
    tf = build_track_frame(res.polyline, res.drivable_mask, res.kerb_mask)

    tiles = []
    for lv in levels:
        img = degrade(frame, lv, kind, cfg=cfg, track_frame=tf)
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f"{kind}  level {lv:.2f}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)

    cols = 2
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    if len(rows) > 1 and rows[-1].shape[1] != rows[0].shape[1]:
        pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), np.uint8)
        rows[-1] = np.hstack([rows[-1], pad])
    panel = np.vstack(rows)
    panel = cv2.resize(panel, (frame.shape[1], int(panel.shape[0] * frame.shape[1]
                                                   / panel.shape[1])),
                       interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, panel):
        raise RuntimeError(f"save_debug_image: could not write {path!r}")
    return path


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS Module 3 -- degradation generator")
    p.add_argument("--image", required=True)
    p.add_argument("--kind", choices=list(KINDS), default="rubber")
    p.add_argument("--levels", default="0.0,0.3,0.6,0.9")
    p.add_argument("--debug", default=None, help="output path for the contact sheet")
    args = p.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"could not read image: {args.image!r}")
    levels = [float(x) for x in args.levels.split(",")]
    path = args.debug or f"debug/degrade_{args.kind}.jpg"
    out = save_debug_image(frame, args.kind, levels, path=path)

    same = np.array_equal(degrade(frame, 0.0, args.kind), frame)
    print(f"kind     : {args.kind}")
    print(f"levels   : {levels}")
    print(f"level 0.0: {'identical to input (no-op)' if same else 'CHANGED -- bug'}")
    print(f"debug    : {out}")


if __name__ == "__main__":
    _cli()

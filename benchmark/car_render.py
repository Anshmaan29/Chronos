"""Render a car into a synthetic scene, with exact wheel-contact ground truth.

The temporal layer cannot be tested without a car that moves, and the whole
point of the benchmark is that the truth is known by construction rather than
annotated.  So the car is placed in world metres on the track plane, projected
through the same camera as the scene, and its four wheel contact patches are
reported both in image pixels and as a true signed margin in millimetres from
the track limit.

Margin convention, used everywhere downstream:

    margin > 0   the contact patch is INSIDE the track limit
    margin = 0   exactly on the outer edge of the white line
    margin < 0   beyond the limit

Wheel naming is from the driver's seat: FL, FR, RL, RR.  On a right-hand
corner the car's RIGHT side reaches the outer limit first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal, Optional

import cv2
import numpy as np

from benchmark.generate import (SceneConfig, _camera, _centreline, _offset_curve,
                                _project, generate_scene)

WHEELS = ("FL", "FR", "RL", "RR")
PathKind = Literal["clean", "brush", "one_wheel_on_line", "violation", "straight_wide"]


@dataclass
class CarConfig:
    """A car in metres.  Defaults are current-generation Formula 1."""

    wheelbase_m: float = 3.60
    axle_track_m: float = 2.00        # centre to centre across an axle
    tyre_width_m: float = 0.38
    tyre_radius_m: float = 0.36
    body_length_m: float = 5.10
    body_width_m: float = 1.30
    body_height_m: float = 0.72
    nose_drop: float = 0.55           # how much the nose tapers, 0..1
    rear_wing_height_m: float = 0.95
    rear_wing_width_m: float = 1.05

    color_body: tuple[int, int, int] = (44, 42, 172)      # BGR
    color_body_top: tuple[int, int, int] = (70, 68, 205)
    color_tyre: tuple[int, int, int] = (26, 26, 28)
    color_wing: tuple[int, int, int] = (32, 30, 120)
    color_cockpit: tuple[int, int, int] = (22, 22, 24)


@dataclass
class SequenceConfig:
    """One synthetic clip: a car taking a corner, optionally running wide."""

    scene: SceneConfig = field(default_factory=SceneConfig)
    car: CarConfig = field(default_factory=CarConfig)
    n_frames: int = 90
    fps: float = 50.0
    y_start_m: float = -20.0
    y_end_m: float = 34.0             # the excursion lands ~35 m from the camera,
                                      # where a tyre spans ~13 px.  A corner
                                      # camera is sited to see its corner; the
                                      # far-field case is covered by the
                                      # resolution gating in chronos.car, not by
                                      # pretending the measurement holds there.
    path: PathKind = "violation"
    lateral_peak_m: Optional[float] = None   # overrides the path preset
    peak_at: float = 0.55                    # where in the clip the car is widest
    excursion_width: float = 0.45            # fraction of the clip spent wide.
                                             # At 50 fps this puts ~320 ms of
                                             # all-four-outside in a 70-frame
                                             # clip, which is the duration a
                                             # real track-limits call turns on.
    lateral_base_m: float = 1.2              # racing line before the excursion
    jitter_m: float = 0.02                   # small steering noise
    yaw_peak_deg: float = 0.0                # car rotation at the widest point
    seed: int = 3

    def yaw_peak(self) -> float:
        """Car rotation at the widest point, in degrees.

        A car square to the track cannot put three wheels out and keep one on
        the line -- the left and right pairs share a margin exactly.  These
        presets are the rotations that make each case physically reachable.
        """
        if self.yaw_peak_deg:
            return self.yaw_peak_deg
        return {"clean": 2.0, "brush": 4.0, "one_wheel_on_line": 16.0,
                "violation": 3.0, "straight_wide": 3.0}[self.path]

    def peak(self) -> float:
        if self.lateral_peak_m is not None:
            return self.lateral_peak_m
        half = self.scene.track_half_width_m
        car_half = self.car.axle_track_m / 2 + self.car.tyre_width_m / 2
        return {
            # comfortably inside: nothing should ever fire
            "clean": half - car_half - 0.9,
            # outer wheels kiss the line, inner wheels well inside
            "brush": half - car_half + 0.55,
            # three wheels beyond, one still ON the line -- NOT a violation,
            # and the case most systems get wrong.  Reachable only with yaw;
            # see yaw_peak() and wheel_contacts_world.
            "one_wheel_on_line": 8.04,
            # All four clearly beyond the limit.  "Clearly" is quantitative:
            # the INNERMOST wheel has to clear the tyre-width threshold by more
            # than the measurement error (~115 mm abs, see chronos.track), or
            # the case is testing the noise floor rather than the rule.  The
            # marginal geometry is covered by "brush" and "one_wheel_on_line",
            # which are supposed to come out as not-a-violation.
            "violation": half + car_half + 1.15,
            "straight_wide": half + car_half + 1.15,
        }[self.path]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def _frame_at(cfg: SceneConfig, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centreline point, forward tangent and right normal at longitudinal y."""
    pts, normals = _centreline(cfg)
    ys = pts[:, 1]
    x = float(np.interp(y, ys, pts[:, 0]))
    nx = float(np.interp(y, ys, normals[:, 0]))
    ny = float(np.interp(y, ys, normals[:, 1]))
    n = np.array([nx, ny])
    n /= max(np.linalg.norm(n), 1e-9)
    f = np.array([-n[1], n[0]])          # forward is the normal turned +90 deg
    if f[1] < 0:
        f = -f
    return np.array([x, y]), f, n


def signed_margin_m(cfg: SceneConfig, pt_xy: np.ndarray) -> float:
    """True signed margin from the track limit, in metres.  Positive = inside.

    Measured as the perpendicular distance to the limit curve itself rather
    than as a difference of lateral offsets, so it stays exact through the
    corner where the two are not the same thing.
    """
    limit = _offset_curve(cfg, cfg.track_half_width_m)
    d = limit - np.asarray(pt_xy, float)[None, :]
    i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
    _, _, n = _frame_at(cfg, float(limit[i, 1]))
    return float(-np.dot(np.asarray(pt_xy, float) - limit[i], n))


def wheel_contacts_world(cfg: SceneConfig, car: CarConfig, y: float,
                         lateral_m: float, yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
    """The four contact patches in world metres, keyed FL / FR / RL / RR.

    ``yaw_deg`` rotates the car relative to the track.  It is not decoration:
    with the car square to the track the left pair and the right pair share a
    margin exactly, so the only reachable states are 4-in, 2-in-2-out and
    4-out.  The case that matters -- three wheels beyond the limit and one
    still on the line -- only exists for a car that is rotated, which is how a
    car running wide actually looks.
    """
    c, f, n = _frame_at(cfg, y)
    if yaw_deg:
        a = np.radians(yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        f, n = ca * f + sa * n, -sa * f + ca * n
    origin = c + lateral_m * n
    half_wb, half_at = car.wheelbase_m / 2, car.axle_track_m / 2
    return {"FL": origin + half_wb * f - half_at * n,
            "FR": origin + half_wb * f + half_at * n,
            "RL": origin - half_wb * f - half_at * n,
            "RR": origin - half_wb * f + half_at * n}


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _project3(pts_xyz: np.ndarray, K, R, C) -> tuple[np.ndarray, np.ndarray]:
    """Project arbitrary 3D world points (Z up) to pixels."""
    cam = (np.asarray(pts_xyz, float) - C) @ R.T
    z = cam[:, 2]
    valid = z > 0.25
    zs = np.where(valid, z, 1.0)
    return np.stack([K[0, 0] * cam[:, 0] / zs + K[0, 2],
                     K[1, 1] * cam[:, 1] / zs + K[1, 2]], axis=1), valid


def _box(origin: np.ndarray, f: np.ndarray, n: np.ndarray,
         lo: tuple[float, float, float], hi: tuple[float, float, float],
         taper: float = 0.0) -> np.ndarray:
    """8 corners of a box in car-local (forward, right, up) metres."""
    out = []
    for fx in (lo[0], hi[0]):
        for ry in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                shrink = 1.0
                if taper > 0.0 and fx > 0:
                    shrink = 1.0 - taper * (fx / max(hi[0], 1e-6))
                p = np.array([origin[0] + fx * f[0] + ry * shrink * n[0],
                              origin[1] + fx * f[1] + ry * shrink * n[1],
                              z * (1.0 if fx <= 0 else 1.0 - 0.45 * taper)])
                out.append(p)
    return np.array(out)


_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
          (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def _draw_solid(canvas: np.ndarray, corners: np.ndarray, color, K, R, C,
                shade: float = 1.0) -> None:
    uv, ok = _project3(corners, K, R, C)
    if not ok.all():
        return
    depth = np.linalg.norm(corners - C, axis=1)
    faces = sorted(_FACES, key=lambda idx: -depth[list(idx)].mean())
    for j, idx in enumerate(faces):
        lit = shade * (0.62 + 0.38 * (j + 1) / len(faces))
        col = tuple(int(np.clip(c * lit, 0, 255)) for c in color)
        cv2.fillConvexPoly(canvas, np.round(uv[list(idx)]).astype(np.int32), col,
                           lineType=cv2.LINE_AA)


def render_car(canvas: np.ndarray, cfg: SceneConfig, car: CarConfig,
               y: float, lateral_m: float, K, R, C, yaw_deg: float = 0.0) -> dict:
    """Draw the car and return its ground truth for this frame.

    Returns a dict with the four contact patches in image pixels and in world
    metres, their true signed margins in millimetres, and the car's bounding
    box in pixels.
    """
    c, f, n = _frame_at(cfg, y)
    if yaw_deg:
        a = np.radians(yaw_deg)
        ca, sa = np.cos(a), np.sin(a)
        f, n = ca * f + sa * n, -sa * f + ca * n
    origin3 = np.array([c[0] + lateral_m * n[0], c[1] + lateral_m * n[1], 0.0])
    f3, n3 = np.array([f[0], f[1], 0.0]), np.array([n[0], n[1], 0.0])
    tr, tw = car.tyre_radius_m, car.tyre_width_m
    contacts = wheel_contacts_world(cfg, car, y, lateral_m, yaw_deg)

    # wheels first: the body is drawn over them, which reads correctly from an
    # elevated camera looking down at the car
    for name in WHEELS:
        p = contacts[name]
        base = np.array([p[0], p[1], 0.0])
        _draw_solid(canvas, _box(base, f3, n3, (-tr, -tw / 2, 0.02), (tr, tw / 2, 2 * tr)),
                    car.color_tyre, K, R, C, shade=1.0)

    body_lo = (-car.body_length_m / 2, -car.body_width_m / 2, tr * 0.35)
    body_hi = (car.body_length_m / 2, car.body_width_m / 2, tr * 0.35 + car.body_height_m)
    _draw_solid(canvas, _box(origin3, f3, n3, body_lo, body_hi, taper=car.nose_drop),
                car.color_body, K, R, C)

    # cockpit, so it reads as a car and not a crate
    _draw_solid(canvas, _box(origin3, f3, n3,
                             (-0.7, -0.42, tr * 0.35 + car.body_height_m - 0.03),
                             (0.5, 0.42, tr * 0.35 + car.body_height_m + 0.26)),
                car.color_cockpit, K, R, C)

    # rear wing
    _draw_solid(canvas, _box(origin3, f3, n3,
                             (-car.body_length_m / 2 - 0.05, -car.rear_wing_width_m / 2,
                              car.rear_wing_height_m - 0.12),
                             (-car.body_length_m / 2 + 0.30, car.rear_wing_width_m / 2,
                              car.rear_wing_height_m)),
                car.color_wing, K, R, C)

    pts3 = np.array([[contacts[w][0], contacts[w][1], 0.0] for w in WHEELS])
    uv, ok = _project3(pts3, K, R, C)
    corners = _box(origin3, f3, n3, (-car.body_length_m / 2, -car.axle_track_m / 2, 0.0),
                   (car.body_length_m / 2, car.axle_track_m / 2, car.rear_wing_height_m))
    cuv, cok = _project3(corners, K, R, C)

    return {
        "contacts_px": {w: uv[i] for i, w in enumerate(WHEELS)},
        "contacts_world": {w: contacts[w] for w in WHEELS},
        "margins_mm": {w: signed_margin_m(cfg, contacts[w]) * 1000.0 for w in WHEELS},
        "visible": bool(ok.all()),
        "box_px": (float(cuv[cok][:, 0].min()), float(cuv[cok][:, 1].min()),
                   float(cuv[cok][:, 0].max()), float(cuv[cok][:, 1].max()))
        if cok.any() else None,
        "y_m": y, "lateral_m": lateral_m,
    }


# --------------------------------------------------------------------------
# sequence
# --------------------------------------------------------------------------


def _bump(cfg: SequenceConfig) -> np.ndarray:
    """0..1 excursion envelope over the clip."""
    t = np.linspace(0.0, 1.0, cfg.n_frames)
    if cfg.path == "straight_wide":
        return np.clip((t - 0.25) / 0.25, 0.0, 1.0)
    # a raised cosine: the car drifts out, holds, and comes back
    x = (t - cfg.peak_at) / max(cfg.excursion_width, 1e-3)
    return np.clip(1.0 - x ** 2, 0.0, 1.0) ** 0.75


def lateral_profile(cfg: SequenceConfig) -> np.ndarray:
    """Lateral offset in metres for every frame of the clip."""
    rng = np.random.default_rng(cfg.seed)
    d = cfg.lateral_base_m + (cfg.peak() - cfg.lateral_base_m) * _bump(cfg)
    return d + rng.normal(0.0, cfg.jitter_m, cfg.n_frames)


def generate_clip(cfg: Optional[SequenceConfig] = None) -> Iterator[dict]:
    """Yield one record per frame: the image and its exact ground truth.

    The scene is rendered once and the car composited per frame, because the
    track does not move and rendering it 90 times would be the slowest part of
    the benchmark for no gain.
    """
    cfg = cfg or SequenceConfig()
    scene = generate_scene(cfg.scene)
    K, R, C = _camera(cfg.scene, cfg.scene.width, cfg.scene.height)
    ys = np.linspace(cfg.y_start_m, cfg.y_end_m, cfg.n_frames)
    lat = lateral_profile(cfg)
    yaw = _bump(cfg) * cfg.yaw_peak()

    for i in range(cfg.n_frames):
        canvas = scene.image.copy()
        gt = render_car(canvas, cfg.scene, cfg.car, float(ys[i]), float(lat[i]),
                        K, R, C, float(yaw[i]))
        yield {"frame_index": i,
               "t_ms": 1000.0 * i / cfg.fps,
               "image": canvas,
               "scene": scene,
               **gt}


def clip_frames(cfg: Optional[SequenceConfig] = None) -> list[dict]:
    """The whole clip as a list.  Convenient; costs memory."""
    return list(generate_clip(cfg))

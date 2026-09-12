"""CHRONOS synthetic scene generator.

Renders a procedural elevated corner-camera view of a race track and returns
the *ground-truth* boundary polyline alongside the image.  Everything is built
in a flat world plane (metres) and projected through a pinhole camera, so the
ground truth is exact by construction -- it is the same world curve that was
rasterised, pushed through the same projection.

This module has two jobs:

  1. Unblock Module 1 (boundary detection) with a test frame.
  2. Be the scene source for the blind benchmark (plan section 5): the system
     sees ONLY the pixels; the ground truth is used only for scoring.

Scene layout, in signed lateral offset ``d`` from the track centreline
(positive = outer side of the corner):

    d = -half_width              outer edge of the INNER white line
    d in [-half, +half]          asphalt track surface (lines painted on it)
    d in [+half - line_w, +half] the outer white line
    d = +half_width              >>> THE TRACK LIMIT <<<  (ground truth)
    d in [+half, +half + kerb_w] red/white striped kerb  (OUTSIDE the limit)
    d beyond that                run-off (grass / gravel / asphalt)

Run standalone::

    python -m benchmark.generate --out data/frame.jpg --seed 1
    python -m benchmark.generate --out data/frame.jpg --runoff gravel --curve 0.010
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import cv2
import numpy as np

RunoffKind = Literal["grass", "gravel", "asphalt"]

# --------------------------------------------------------------------------
# CONFIG -- every tunable lives here.  Nothing magic is buried in the code.
# --------------------------------------------------------------------------


@dataclass
class SceneConfig:
    """All tunable parameters for one synthetic scene."""

    # --- output image -----------------------------------------------------
    width: int = 1280
    height: int = 720
    supersample: int = 2          # render at Nx then downscale (anti-aliasing)

    # --- track geometry, metres ------------------------------------------
    track_half_width_m: float = 6.0   # centreline -> outer edge of white line
    line_width_m: float = 0.18        # painted edge-line width
    kerb_width_m: float = 1.30        # kerb band, sits OUTSIDE the white line
    kerb_stripe_pitch_m: float = 1.00 # one red + one white = 2 * pitch
    runoff_width_m: float = 30.0
    far_cap_m: float = 90.0           # verge drawn ACROSS the far end of the
                                      # track, so the surface does not simply
                                      # stop against the backdrop -- a real
                                      # corner cam sees the track vanish behind
                                      # the verge, and an open end welds the
                                      # track to any grey barrier behind it
    inner_runoff_width_m: float = 25.0
    inner_kerb: bool = False          # kerb on the inside of the corner too

    # --- corner shape -----------------------------------------------------
    curve_k: float = 0.0060           # centreline x = curve_k * max(y,0)^2
    corner_entry_m: float = 8.0       # track is straight before this, then bends
    y_near_m: float = -34.0           # nearest point of track rendered
    y_far_m: float = 110.0            # furthest point of track rendered
    n_samples: int = 320              # polyline resolution along the track

    # --- camera pose (elevated corner cam) --------------------------------
    cam_x_m: float = -2.0
    cam_y_m: float = -26.0
    cam_z_m: float = 9.5              # height above the track plane
    cam_yaw_deg: float = 10.0         # +ve = pan right
    cam_pitch_deg: float = 13.0       # +ve = look down
    fov_deg: float = 55.0

    # --- colours, BGR -----------------------------------------------------
    color_asphalt: tuple[int, int, int] = (62, 63, 66)
    color_line: tuple[int, int, int] = (236, 238, 238)
    color_kerb_red: tuple[int, int, int] = (44, 42, 178)
    color_kerb_white: tuple[int, int, int] = (232, 234, 236)
    color_grass: tuple[int, int, int] = (58, 112, 54)
    color_gravel: tuple[int, int, int] = (124, 140, 156)
    color_background: tuple[int, int, int] = (74, 72, 78)
    color_sky: tuple[int, int, int] = (176, 168, 158)

    runoff: RunoffKind = "grass"
    inner_runoff: RunoffKind = "grass"
    far_cap: RunoffKind = "grass"     # material of the far verge.  Keep this
                                      # DIFFERENT from a grey run-off, or the
                                      # cap re-connects the track to the run-off
                                      # around the end and defeats its purpose.

    # --- realism (NOT degradation -- contamination belongs in degrade.py) --
    asphalt_grain: float = 7.0        # std-dev of per-pixel asphalt texture
    grass_grain: float = 12.0
    sensor_noise: float = 3.0         # std-dev of global sensor noise
    blur_sigma: float = 0.7           # lens softness, pixels
    vignette: float = 0.22            # 0 = off, 1 = heavy
    light_gradient: float = 0.10      # brightness ramp across the frame
    exposure: float = 1.0             # global gain
    jpeg_quality: int = 92            # re-encode to get compression artefacts

    seed: int = 0

    def __post_init__(self) -> None:
        if self.track_half_width_m <= 0:
            raise ValueError("track_half_width_m must be > 0")
        if self.kerb_width_m < 0:
            raise ValueError("kerb_width_m must be >= 0 (0 = no kerb at this corner)")
        if not (0 < self.line_width_m < self.track_half_width_m):
            raise ValueError("line_width_m must be > 0 and inside the track")
        if self.y_far_m <= self.y_near_m:
            raise ValueError("y_far_m must exceed y_near_m")
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1")


@dataclass
class SyntheticScene:
    """A rendered scene plus its exact ground truth.

    Attributes
    ----------
    image:
        BGR uint8 array, shape (height, width, 3).  This is ALL the detector
        is allowed to see.
    gt_boundary:
        (N, 2) float32 image-space polyline of the outer edge of the outer
        white line -- the regulatory track limit.  Ordered far -> near.
    gt_inner_boundary:
        (M, 2) float32 polyline of the outer edge of the inner white line.
    gt_kerb_outer:
        (K, 2) float32 polyline of the far edge of the kerb band.
    camera:
        Dict of the pose actually used, plus the 3x3 plane->image homography
        as a nested list (world (X, Y, 1) -> image (u, v, 1)).
    config:
        The SceneConfig used.
    """

    image: np.ndarray
    gt_boundary: np.ndarray
    gt_inner_boundary: np.ndarray
    gt_kerb_outer: np.ndarray
    camera: dict
    config: SceneConfig


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _centreline(cfg: SceneConfig) -> tuple[np.ndarray, np.ndarray]:
    """Sample the track centreline and its right-hand unit normal.

    Returns
    -------
    (pts, normals):
        ``pts`` is (N, 2) of (x, y) world metres, ``normals`` is (N, 2) unit
        vectors pointing to the OUTER side of the corner.
    """
    y = np.linspace(cfg.y_near_m, cfg.y_far_m, cfg.n_samples)
    # straight on approach, then a constant-k parabolic corner -- C1 continuous
    s = np.maximum(y - cfg.corner_entry_m, 0.0)
    x = cfg.curve_k * s**2
    dx = 2.0 * cfg.curve_k * s
    tangent = np.stack([dx, np.ones_like(y)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # right normal of (tx, ty) is (ty, -tx)
    normals = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    return np.stack([x, y], axis=1), normals


def _offset_curve(cfg: SceneConfig, d: float) -> np.ndarray:
    """World-space curve at signed lateral offset ``d`` metres."""
    pts, normals = _centreline(cfg)
    return pts + d * normals


def _camera(cfg: SceneConfig, width: int, height: int):
    """Build (K, R_world_to_cam, camera_centre) for the configured pose."""
    fx = fy = (width / 2.0) / math.tan(math.radians(cfg.fov_deg) / 2.0)
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    yaw = math.radians(cfg.cam_yaw_deg)
    pitch = math.radians(cfg.cam_pitch_deg)
    forward = np.array([math.sin(yaw) * math.cos(pitch),
                        math.cos(yaw) * math.cos(pitch),
                        -math.sin(pitch)])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])          # rows: world -> camera axes
    C = np.array([cfg.cam_x_m, cfg.cam_y_m, cfg.cam_z_m])
    return K, R, C


def _project(pts_xy: np.ndarray, K, R, C) -> tuple[np.ndarray, np.ndarray]:
    """Project world-plane points (Z = 0) to pixels.

    Parameters
    ----------
    pts_xy: (N, 2) world metres.

    Returns
    -------
    (uv, valid): ``uv`` is (N, 2) float64 pixels, ``valid`` is (N,) bool and is
    False for points behind the camera (which project to garbage).
    """
    P = np.concatenate([pts_xy, np.zeros((len(pts_xy), 1))], axis=1)
    cam = (P - C) @ R.T
    z = cam[:, 2]
    valid = z > 0.25
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * cam[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=1), valid


def _plane_homography(K, R, C) -> np.ndarray:
    """3x3 homography mapping world (X, Y, 1) on Z=0 to image (u, v, 1)."""
    t = -R @ C
    H = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    return H / H[2, 2]


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------


def _band_polygon(cfg: SceneConfig, d0: float, d1: float,
                  K, R, C, t_slice: Optional[slice] = None) -> Optional[np.ndarray]:
    """Image-space polygon for the strip between lateral offsets d0 and d1."""
    inner = _offset_curve(cfg, d0)
    outer = _offset_curve(cfg, d1)
    if t_slice is not None:
        inner, outer = inner[t_slice], outer[t_slice]
    uv_in, ok_in = _project(inner, K, R, C)
    uv_out, ok_out = _project(outer, K, R, C)
    keep = ok_in & ok_out
    if keep.sum() < 2:
        return None
    poly = np.concatenate([uv_in[keep], uv_out[keep][::-1]], axis=0)
    return poly


def _fill(canvas: np.ndarray, poly: Optional[np.ndarray],
          color: tuple[int, int, int], mask: Optional[np.ndarray] = None) -> None:
    if poly is None:
        return
    pts = np.round(poly).astype(np.int32)[None, :, :]
    cv2.fillPoly(canvas, pts, color, lineType=cv2.LINE_8)
    if mask is not None:
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_8)


def _runoff_color(cfg: SceneConfig, kind: RunoffKind) -> tuple[int, int, int]:
    return {"grass": cfg.color_grass,
            "gravel": cfg.color_gravel,
            "asphalt": cfg.color_asphalt}[kind]


def generate_scene(cfg: Optional[SceneConfig] = None) -> SyntheticScene:
    """Render one synthetic corner-cam frame with exact ground truth.

    Parameters
    ----------
    cfg:
        Scene parameters.  Defaults to a clean, well-lit elevated corner cam
        with grass run-off and a red/white kerb.

    Returns
    -------
    SyntheticScene -- image plus ground-truth polylines in image pixels.

    Raises
    ------
    RuntimeError
        If the configured camera pose sees none of the track, i.e. the scene
        would be empty.  Fail loudly rather than hand back a blank frame.
    """
    cfg = cfg or SceneConfig()
    rng = np.random.default_rng(cfg.seed)

    ss = cfg.supersample
    W, H = cfg.width * ss, cfg.height * ss
    K, R, C = _camera(cfg, W, H)

    half = cfg.track_half_width_m
    lw = cfg.line_width_m
    kw = cfg.kerb_width_m

    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = cfg.color_background
    mask_asphalt = np.zeros((H, W), np.uint8)
    mask_grass = np.zeros((H, W), np.uint8)

    # sky / far background above the horizon
    horizon_uv, _ = _project(np.array([[0.0, 5000.0]]), K, R, C)
    horizon_y = int(np.clip(horizon_uv[0, 1], 0, H))
    if horizon_y > 0:
        canvas[:horizon_y] = cfg.color_sky

    # --- run-off, far side (outside the kerb) ---------------------------
    _fill(canvas, _band_polygon(cfg, half + kw, half + kw + cfg.runoff_width_m, K, R, C),
          _runoff_color(cfg, cfg.runoff),
          mask_grass if cfg.runoff == "grass" else None)

    # --- run-off, inner side --------------------------------------------
    inner_kerb_span = kw if cfg.inner_kerb else 0.0
    _fill(canvas,
          _band_polygon(cfg, -half - inner_kerb_span - cfg.inner_runoff_width_m,
                        -half - inner_kerb_span, K, R, C),
          _runoff_color(cfg, cfg.inner_runoff),
          mask_grass if cfg.inner_runoff == "grass" else None)

    # --- drivable asphalt (white lines are painted on top of it) ---------
    _fill(canvas, _band_polygon(cfg, -half, half, K, R, C),
          cfg.color_asphalt, mask_asphalt)

    # --- kerbs, striped along the track ----------------------------------
    def draw_kerb(d_in: float, d_out: float) -> None:
        y = np.linspace(cfg.y_near_m, cfg.y_far_m, cfg.n_samples)
        pitch = cfg.kerb_stripe_pitch_m
        n_stripes = int((cfg.y_far_m - cfg.y_near_m) / pitch) + 1
        for i in range(n_stripes):
            y0 = cfg.y_near_m + i * pitch
            y1 = min(y0 + pitch, cfg.y_far_m)
            i0 = int(np.searchsorted(y, y0))
            i1 = int(np.searchsorted(y, y1)) + 1
            if i1 - i0 < 2:
                continue
            color = cfg.color_kerb_red if i % 2 == 0 else cfg.color_kerb_white
            _fill(canvas, _band_polygon(cfg, d_in, d_out, K, R, C, slice(i0, i1)), color)

    if kw > 0:
        draw_kerb(half, half + kw)
    if cfg.inner_kerb and kw > 0:
        draw_kerb(-half - kw, -half)

    # --- white edge lines, painted inside the track ----------------------
    _fill(canvas, _band_polygon(cfg, half - lw, half, K, R, C), cfg.color_line)
    _fill(canvas, _band_polygon(cfg, -half, -half + lw, K, R, C), cfg.color_line)

    # --- far verge: close off the end of the track ------------------------
    if cfg.far_cap_m > 0:
        cap_cfg = SceneConfig(**{**asdict(cfg),
                                 "y_near_m": cfg.y_far_m,
                                 "y_far_m": cfg.y_far_m + cfg.far_cap_m,
                                 "n_samples": 48})
        d_lo = -half - cfg.inner_runoff_width_m - inner_kerb_span
        d_hi = half + kw + cfg.runoff_width_m
        _fill(canvas, _band_polygon(cap_cfg, d_lo, d_hi, K, R, C),
              _runoff_color(cfg, cfg.far_cap),
              mask_grass if cfg.far_cap == "grass" else None)

    if mask_asphalt.sum() == 0:
        raise RuntimeError(
            "generate_scene: camera pose sees no track surface. "
            f"Check cam_x/y/z ({cfg.cam_x_m}, {cfg.cam_y_m}, {cfg.cam_z_m}), "
            f"yaw {cfg.cam_yaw_deg} deg, pitch {cfg.cam_pitch_deg} deg."
        )

    # --- surface texture --------------------------------------------------
    img = canvas.astype(np.float32)
    if cfg.asphalt_grain > 0:
        grain = rng.normal(0.0, cfg.asphalt_grain, (H, W, 1)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 1.2 * ss)[:, :, None]
        img += grain * (mask_asphalt[:, :, None] > 0)
    if cfg.grass_grain > 0:
        grain = rng.normal(0.0, cfg.grass_grain, (H, W, 1)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.9 * ss)[:, :, None]
        img += grain * (mask_grass[:, :, None] > 0)

    # --- lighting ramp + vignette ----------------------------------------
    if cfg.light_gradient:
        ramp = np.linspace(1.0 + cfg.light_gradient, 1.0 - cfg.light_gradient, W)
        img *= ramp[None, :, None].astype(np.float32)
    if cfg.vignette:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        img *= (1.0 - cfg.vignette * np.clip(r / 1.4142, 0, 1) ** 2)[:, :, None]
    img *= cfg.exposure

    frame = np.clip(img, 0, 255).astype(np.uint8)
    if ss > 1:
        frame = cv2.resize(frame, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    if cfg.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), cfg.blur_sigma)
    if cfg.sensor_noise > 0:
        frame = np.clip(frame.astype(np.float32)
                        + rng.normal(0, cfg.sensor_noise, frame.shape), 0, 255).astype(np.uint8)
    if 0 < cfg.jpeg_quality < 100:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if not ok:
            raise RuntimeError("generate_scene: JPEG re-encode failed")
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # --- ground truth, in FINAL image pixels ------------------------------
    Kf, Rf, Cf = _camera(cfg, cfg.width, cfg.height)

    def gt_polyline(d: float) -> np.ndarray:
        uv, ok = _project(_offset_curve(cfg, d), Kf, Rf, Cf)
        uv = uv[ok]
        # ground truth is what is VISIBLE -- scoring against off-frame points
        # would penalise a detector for not seeing outside the image
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < cfg.width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < cfg.height))
        return uv[inside].astype(np.float32)

    gt_boundary = gt_polyline(half)
    if len(gt_boundary) < 2:
        raise RuntimeError(
            "generate_scene: the track-limit line is not visible in frame. "
            "Widen fov_deg or move the camera back."
        )

    camera_info = {
        "cam_x_m": cfg.cam_x_m, "cam_y_m": cfg.cam_y_m, "cam_z_m": cfg.cam_z_m,
        "yaw_deg": cfg.cam_yaw_deg, "pitch_deg": cfg.cam_pitch_deg,
        "fov_deg": cfg.fov_deg,
        "H_plane_to_image": _plane_homography(Kf, Rf, Cf).tolist(),
    }

    return SyntheticScene(
        image=frame,
        gt_boundary=gt_boundary,
        gt_inner_boundary=gt_polyline(-half),
        gt_kerb_outer=gt_polyline(half + kw),
        camera=camera_info,
        config=cfg,
    )


def save_scene(scene: SyntheticScene, image_path: str,
               gt_path: Optional[str] = None) -> tuple[str, str]:
    """Write the frame and its ground-truth JSON sidecar.

    Returns the (image_path, gt_path) actually written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    if not cv2.imwrite(image_path, scene.image):
        raise RuntimeError(f"save_scene: could not write image to {image_path!r}")
    gt_path = gt_path or os.path.splitext(image_path)[0] + "_gt.json"
    payload = {
        "image": os.path.basename(image_path),
        "gt_boundary_px": scene.gt_boundary.tolist(),
        "gt_inner_boundary_px": scene.gt_inner_boundary.tolist(),
        "gt_kerb_outer_px": scene.gt_kerb_outer.tolist(),
        "world": {
            "track_half_width_m": scene.config.track_half_width_m,
            "line_width_m": scene.config.line_width_m,
            "kerb_width_m": scene.config.kerb_width_m,
            "kerb_stripe_pitch_m": scene.config.kerb_stripe_pitch_m,
        },
        "camera": scene.camera,
        "config": asdict(scene.config),
    }
    with open(gt_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    return image_path, gt_path


def load_ground_truth(gt_path: str) -> np.ndarray:
    """Load the track-limit ground-truth polyline from a sidecar JSON."""
    with open(gt_path) as fh:
        data = json.load(fh)
    if "gt_boundary_px" not in data:
        raise ValueError(f"{gt_path!r} has no 'gt_boundary_px' key")
    return np.asarray(data["gt_boundary_px"], dtype=np.float32)


def debug_image(scene: SyntheticScene, path: str = "debug/generate_debug.jpg") -> str:
    """Save the frame with its ground truth drawn on, for eyeball checking."""
    vis = scene.image.copy()
    cv2.polylines(vis, [np.round(scene.gt_kerb_outer).astype(np.int32)],
                  False, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.polylines(vis, [np.round(scene.gt_inner_boundary).astype(np.int32)],
                  False, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.polylines(vis, [np.round(scene.gt_boundary).astype(np.int32)],
                  False, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "GT track limit (outer edge of white line)", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "GT kerb outer edge", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "GT inner line", (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, vis):
        raise RuntimeError(f"debug_image: could not write {path!r}")
    return path


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS synthetic scene generator")
    p.add_argument("--out", default="data/frame.jpg", help="output image path")
    p.add_argument("--gt", default=None, help="ground-truth JSON path")
    p.add_argument("--debug", default="debug/generate_debug.jpg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runoff", choices=["grass", "gravel", "asphalt"], default="grass")
    p.add_argument("--curve", type=float, default=SceneConfig.curve_k)
    p.add_argument("--line-width", type=float, default=SceneConfig.line_width_m)
    p.add_argument("--inner-kerb", action="store_true")
    p.add_argument("--pitch", type=float, default=SceneConfig.cam_pitch_deg)
    p.add_argument("--yaw", type=float, default=SceneConfig.cam_yaw_deg)
    p.add_argument("--cam-z", type=float, default=SceneConfig.cam_z_m)
    args = p.parse_args()

    cfg = SceneConfig(seed=args.seed, runoff=args.runoff, curve_k=args.curve,
                      line_width_m=args.line_width, inner_kerb=args.inner_kerb,
                      cam_pitch_deg=args.pitch, cam_yaw_deg=args.yaw,
                      cam_z_m=args.cam_z)
    scene = generate_scene(cfg)
    img_path, gt_path = save_scene(scene, args.out, args.gt)
    dbg = debug_image(scene, args.debug)
    print(f"frame       -> {img_path}  {scene.image.shape[1]}x{scene.image.shape[0]}")
    print(f"groundtruth -> {gt_path}   {len(scene.gt_boundary)} points")
    print(f"debug       -> {dbg}")


if __name__ == "__main__":
    _cli()

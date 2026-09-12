"""Tests for Module 1 -- boundary detection, scored against known ground truth.

Run with ``pytest tests/`` or standalone with ``python tests/test_boundary.py``.

The synthetic scenes come from ``benchmark.generate``, so every assertion here
is against exact ground truth rather than against a previous run's output.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.generate import SceneConfig, generate_scene            # noqa: E402
from chronos.boundary import (                                        # noqa: E402
    BoundaryConfig, detect_boundary, polyline_error,
    point_to_polyline_distance,
)

# deviation is accuracy, coverage is completeness -- see polyline_error
MAX_DEV_PX = 3.0
MIN_COVERAGE = 0.60

VARIANTS = {
    "baseline": {},
    "gravel run-off": dict(runoff="gravel"),
    "kerb both sides": dict(inner_kerb=True),
    "no kerb": dict(kerb_width_m=0.0),
    "gentle corner": dict(curve_k=0.0020),
    "sharp corner": dict(curve_k=0.0110),
    "mirrored camera": dict(cam_x_m=9.0, cam_yaw_deg=-16.0),
    "high steep camera": dict(cam_z_m=18.0, cam_pitch_deg=26.0),
    "narrow 8cm line": dict(line_width_m=0.08),
    "overcast": dict(exposure=0.62, light_gradient=0.04),
    "soft lens and noise": dict(blur_sigma=1.8, sensor_noise=7.0, jpeg_quality=62),
}


def _check(name: str, **kw) -> None:
    scene = generate_scene(SceneConfig(**kw))
    res = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    assert res.ok, f"{name}: detection failed -- {res.reason}"
    assert res.polyline is not None and len(res.polyline) >= 12, f"{name}: polyline too short"
    err = polyline_error(res.polyline, scene.gt_boundary)
    assert err["dev_mean_px"] <= MAX_DEV_PX, (
        f"{name}: mean deviation {err['dev_mean_px']:.2f} px exceeds {MAX_DEV_PX} px")
    assert err["cov_frac"] >= MIN_COVERAGE, (
        f"{name}: covered only {err['cov_frac']:.0%} of the boundary")


def test_variants() -> None:
    for name, kw in VARIANTS.items():
        _check(name, **kw)


def test_ground_truth_is_on_the_paint() -> None:
    """The generator's own ground truth must sit on the outer edge of the line.

    Guards the benchmark itself: if this drifts, every score built on it is
    meaningless, and it would drift silently.
    """
    scene = generate_scene(SceneConfig())
    img = scene.image
    h, w = img.shape[:2]
    pts = scene.gt_boundary[::7]
    inside = pts[(pts[:, 0] > 30) & (pts[:, 0] < w - 30)
                 & (pts[:, 1] > 30) & (pts[:, 1] < h - 30)]
    assert len(inside) > 10, "ground truth is barely in frame"
    # a few pixels INSIDE the limit must be paint, i.e. bright
    gray = np.asarray(img).mean(axis=2)
    d = np.gradient(inside, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    q = inside - 4.0 * n
    paint = gray[np.clip(q[:, 1].astype(int), 0, h - 1),
                 np.clip(q[:, 0].astype(int), 0, w - 1)]
    assert paint.mean() > 120, f"ground truth is not on the paint (mean {paint.mean():.0f})"


def test_kerb_is_excluded_from_the_drivable_surface() -> None:
    """A kerb is outside the track limit; it must not be in the drivable mask."""
    scene = generate_scene(SceneConfig())
    res = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    assert res.ok, res.reason
    assert res.kerb_mask is not None and res.kerb_mask.any(), "no kerb found at a kerbed corner"
    overlap = np.logical_and(res.kerb_mask > 0, res.drivable_mask > 0).sum()
    assert overlap / max(int((res.kerb_mask > 0).sum()), 1) < 0.02, (
        "kerb leaked into the drivable surface -- the boundary will land outside the limit")


def test_boundary_sits_inside_the_kerb() -> None:
    """Every detected point must be on the track side of the kerb."""
    scene = generate_scene(SceneConfig())
    res = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    assert res.ok, res.reason
    ys, xs = np.nonzero(res.kerb_mask)
    d = point_to_polyline_distance(np.stack([xs, ys], 1)[::50], res.polyline)
    assert d.min() >= 0.0 and np.median(d) > 5.0, "boundary is sitting on the kerb"


def test_manual_side_override_changes_the_answer() -> None:
    scene = generate_scene(SceneConfig())
    right = detect_boundary(scene.image, BoundaryConfig(outer_side="right"), save_debug=False)
    left = detect_boundary(scene.image, BoundaryConfig(outer_side="left"), save_debug=False)
    assert right.ok and left.ok
    assert polyline_error(right.polyline, scene.gt_boundary)["dev_mean_px"] < 3.0
    assert polyline_error(left.polyline, scene.gt_boundary)["dev_mean_px"] > 50.0, (
        "the left override returned the same edge -- the override does nothing")


def test_failure_is_reported_not_raised() -> None:
    for name, img in {
        "black": np.zeros((480, 640, 3), np.uint8),
        "grass": np.full((480, 640, 3), (58, 112, 54), np.uint8),
    }.items():
        res = detect_boundary(img, BoundaryConfig(), save_debug=False)
        assert not res.ok, f"{name}: claimed success on a frame with no track"
        assert res.polyline is None, f"{name}: returned a polyline anyway"
        assert len(res.reason) > 20, f"{name}: reason is not explanatory: {res.reason!r}"


def test_bad_input_raises() -> None:
    for bad in (None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32)):
        try:
            detect_boundary(bad, save_debug=False)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input silently: {type(bad)}")


def test_generator_rejects_impossible_config() -> None:
    for kw in (dict(line_width_m=0.0), dict(track_half_width_m=-1.0),
               dict(y_far_m=-100.0), dict(kerb_width_m=-1.0)):
        try:
            SceneConfig(**kw)
        except ValueError:
            continue
        raise AssertionError(f"SceneConfig accepted {kw}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

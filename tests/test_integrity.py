"""Tests for Modules 2 and 3 -- integrity scoring and controlled degradation.

Run with ``pytest tests/`` or standalone with ``python tests/test_integrity.py``.

These assert the calibration contract, not a snapshot: if the score stops
meeting its targets, or stops being monotonic, or starts reporting a clean
corner as degraded, that is a failure regardless of what changed.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.generate import SceneConfig, generate_scene              # noqa: E402
from chronos.boundary import BoundaryConfig, detect_boundary            # noqa: E402
from chronos.degrade import (KINDS, DegradeConfig, build_track_frame,   # noqa: E402
                             degrade)
from chronos.integrity import (IntegrityConfig, capture_baseline,       # noqa: E402
                               score_integrity)

LEVELS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]


def _scene(**kw):
    scene = generate_scene(SceneConfig(**kw))
    clean = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    assert clean.ok, clean.reason
    baseline = capture_baseline(scene.image, scene.gt_boundary, result=clean)
    tf = build_track_frame(clean.polyline, clean.drivable_mask, clean.kerb_mask,
                           n=IntegrityConfig().n_samples)
    return scene, clean, baseline, tf


# --------------------------------------------------------------------------
# Module 3 -- degradation
# --------------------------------------------------------------------------


def test_level_zero_is_a_genuine_noop() -> None:
    scene, _, _, tf = _scene()
    for kind in KINDS:
        out = degrade(scene.image, 0.0, kind, track_frame=tf)
        assert np.array_equal(out, scene.image), f"{kind}: level 0.0 altered the frame"


def test_degrade_does_not_modify_its_input() -> None:
    scene, _, _, tf = _scene()
    before = scene.image.copy()
    degrade(scene.image, 0.8, "rubber", track_frame=tf)
    assert np.array_equal(scene.image, before), "degrade mutated the frame it was given"


def test_degrade_rejects_bad_arguments() -> None:
    scene, _, _, tf = _scene()
    for level in (-0.01, 1.01, float("nan")):
        try:
            degrade(scene.image, level, "rubber", track_frame=tf)
        except ValueError:
            continue
        raise AssertionError(f"accepted level {level}")
    try:
        degrade(scene.image, 0.5, "glitter", track_frame=tf)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("accepted an unknown degradation kind")


def test_contamination_stays_on_the_road() -> None:
    """Rubber on the grass is a bug, and it would also fake the score."""
    scene, clean, _, tf = _scene()
    dirty = degrade(scene.image, 1.0, "rubber", track_frame=tf)
    changed = np.abs(dirty.astype(int) - scene.image.astype(int)).sum(axis=2) > 12
    allowed = tf.surface > 0
    stray = int((changed & ~allowed).sum())
    assert stray / max(int(changed.sum()), 1) < 0.02, (
        f"{stray} changed pixels landed off the road surface")


def test_rubber_actually_darkens_the_paint() -> None:
    """The Miami case is rubber ON the line -- not beside it."""
    scene, _, baseline, tf = _scene()
    clean_luma = score_integrity(scene.image, baseline).detail["paint_luma"]
    dirty_luma = score_integrity(degrade(scene.image, 0.6, "rubber", track_frame=tf),
                                 baseline).detail["paint_luma"]
    assert dirty_luma < 0.8 * clean_luma, (
        f"rubber at 0.6 moved paint luminance only {clean_luma:.0f} -> {dirty_luma:.0f}; "
        "the deposit is missing the line")


# --------------------------------------------------------------------------
# Module 2 -- integrity
# --------------------------------------------------------------------------


def test_clean_scenes_score_above_90() -> None:
    """Including scenes where Module 1 cannot geometrically reach 100% coverage."""
    for name, kw in {"baseline": {}, "gravel run-off": dict(runoff="gravel"),
                     "soft lens": dict(blur_sigma=1.8, sensor_noise=7.0, jpeg_quality=62),
                     "overcast": dict(exposure=0.62)}.items():
        scene, clean, baseline, _ = _scene(**kw)
        score = score_integrity(scene.image, baseline, result=clean)
        assert score.total >= 90.0, f"{name}: clean frame scored {score.total}"


def test_continuity_is_relative_to_what_the_scene_allows() -> None:
    """THE confound test.

    Module 1 recovers ~67% of the boundary on a clean grey-run-off scene, for
    geometric reasons -- far field, thin line.  Reporting that as degradation
    would mean calling a clean corner dirty, and the demo's premise with it.
    """
    scene, clean, baseline, _ = _scene(runoff="gravel")
    assert baseline.coverage < 0.85, (
        f"this scene was expected to be geometrically hard, but clean coverage is "
        f"{baseline.coverage:.0%} -- pick a harder variant for this test")
    score = score_integrity(scene.image, baseline, result=clean)
    assert score.continuity >= 99.0, (
        f"clean frame reported continuity {score.continuity} despite being clean; "
        "the baseline-relative measurement has regressed")


def test_calibration_targets_hold_for_rubber() -> None:
    scene, clean, baseline, tf = _scene()
    got = {lv: score_integrity(degrade(scene.image, lv, "rubber", track_frame=tf),
                               baseline).total for lv in (0.0, 0.3, 0.6, 0.9)}
    assert got[0.0] >= 90, f"clean scored {got[0.0]}"
    assert 68 <= got[0.3] <= 82, f"light rubber scored {got[0.3]}, want 70-80"
    assert 34 <= got[0.6] <= 48, f"Miami-level rubber scored {got[0.6]}, want ~41"
    assert got[0.9] <= 27, f"heavy rubber scored {got[0.9]}, want under 25"


def test_every_kind_degrades_monotonically() -> None:
    scene, _, baseline, tf = _scene()
    for kind in KINDS:
        totals = [score_integrity(degrade(scene.image, lv, kind, track_frame=tf),
                                  baseline).total for lv in LEVELS]
        rises = [(LEVELS[i], totals[i], totals[i + 1]) for i in range(len(totals) - 1)
                 if totals[i + 1] > totals[i] + 0.5]
        assert not rises, f"{kind}: integrity rose with degradation at {rises}"


def test_subscores_stay_in_range() -> None:
    scene, _, baseline, tf = _scene()
    for lv in (0.0, 0.5, 1.0):
        s = score_integrity(degrade(scene.image, lv, "rubber", track_frame=tf), baseline)
        for name, v in (("total", s.total), ("contrast", s.contrast),
                        ("continuity", s.continuity), ("sharpness", s.sharpness),
                        ("contamination", s.contamination)):
            assert 0.0 <= v <= 100.0, f"{name} out of range at level {lv}: {v}"


def test_baseline_refuses_a_frame_it_cannot_measure() -> None:
    for name, img in {"black": np.zeros((480, 640, 3), np.uint8),
                      "grass": np.full((480, 640, 3), (58, 112, 54), np.uint8)}.items():
        try:
            capture_baseline(img)
        except ValueError:
            continue
        raise AssertionError(f"{name}: captured a baseline from a frame with no line")


def test_explain_reports_the_four_components() -> None:
    """'What does Boundary Integrity 42 actually mean?' must have an answer."""
    scene, _, baseline, tf = _scene()
    s = score_integrity(degrade(scene.image, 0.6, "rubber", track_frame=tf), baseline)
    text = s.explain()
    for word in ("contrast", "detectable", "contaminated", "sharpness"):
        assert word in text, f"explain() never mentions {word}: {text!r}"


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

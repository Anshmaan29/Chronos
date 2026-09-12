"""Tests for the spatiotemporal layer: cars, geometry, time series, decisions.

Run with ``pytest tests/`` or ``python tests/test_temporal.py``.

These assert behaviour that the rules require, not a snapshot of current
output: touching the line is legal, a violation needs all four wheels, a
single-frame spike must never fire, and a degraded reference must block a
verdict no matter how good the geometry is.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.car_render import (SequenceConfig, WHEELS, generate_clip)      # noqa: E402
from benchmark.detector_sim import SimDetectorConfig, SimulatedDetector       # noqa: E402
from chronos.boundary import BoundaryConfig, detect_boundary                  # noqa: E402
from chronos.car import CarConfig, CarTracker                                 # noqa: E402
from chronos.decide import DecideConfig, Outcome, Verdict, judge              # noqa: E402
from chronos.pipeline import Pipeline, PipelineConfig                         # noqa: E402
from chronos.temporal import (ExcursionEvent, TemporalConfig, WheelSample,    # noqa: E402
                              WheelState, classify)
from chronos.track import TrackGeometryConfig, build_track_geometry           # noqa: E402
from chronos.ui.timeline import render_wheel_timeline                         # noqa: E402


def _run(path: str, contamination: float = 0.0, frames: int = 70,
         occlude: int = 0, jitter: float = 1.5):
    from chronos.degrade import DegradeConfig, build_track_frame, degrade
    seq = SequenceConfig(n_frames=frames, path=path)
    clip = list(generate_clip(seq))
    scene = clip[0]["scene"]
    cfg = PipelineConfig(fps=seq.fps)
    cfg.geometry.line_width_mm = scene.config.line_width_m * 1000
    cfg.geometry.kerb_stripe_pitch_mm = scene.config.kerb_stripe_pitch_m * 1000
    cfg.temporal.tyre_width_mm = seq.car.tyre_width_m * 1000
    cfg.integrity_every = 1 if contamination > 0 else 5
    det = SimulatedDetector([r["box_px"] for r in clip],
                            SimDetectorConfig(occlusion_start=frames // 2 - 4,
                                              occlusion_frames=occlude),
                            contacts=[r["contacts_px"] for r in clip],
                            contact_jitter_px=jitter)
    pipe = Pipeline(scene.image, scene.gt_boundary, cfg, det)
    tfr = build_track_frame(pipe.boundary.polyline, pipe.boundary.drivable_mask,
                            pipe.boundary.kerb_mask)
    dcfg = DegradeConfig()
    for rec in clip:
        img = rec["image"] if contamination <= 0 else degrade(
            rec["image"], contamination, "rubber", cfg=dcfg, track_frame=tfr)
        pipe.process(img, rec["frame_index"])
    pipe.finish()
    return pipe, clip, seq


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def test_three_valued_classification() -> None:
    cfg = TemporalConfig(tyre_width_mm=380.0)
    assert classify(120.0, cfg) is WheelState.INSIDE
    assert classify(0.1, cfg) is WheelState.INSIDE
    assert classify(0.0, cfg) is WheelState.ON_LINE      # on the edge is on the line
    assert classify(-379.0, cfg) is WheelState.ON_LINE
    assert classify(-380.0, cfg) is WheelState.OUTSIDE
    assert classify(-900.0, cfg) is WheelState.OUTSIDE


def test_touching_the_line_is_not_a_violation() -> None:
    """Three wheels beyond the limit and one on the line is NOT a violation."""
    pipe, _, _ = _run("one_wheel_on_line")
    assert pipe.decisions.verdicts, "the near-miss produced no event to review at all"
    v = pipe.decisions.verdicts[0]
    assert v.outcome is Outcome.CLEAR, f"called it {v.outcome.value}: {v.reason}"
    assert "line" in v.reason.lower(), v.reason
    assert v.event.all_four_outside_frames == 0


def test_a_real_violation_is_called() -> None:
    pipe, clip, seq = _run("violation")
    assert pipe.decisions.verdicts, "no event from an unambiguous excursion"
    v = pipe.decisions.verdicts[0]
    assert v.outcome is Outcome.VIOLATION, f"{v.outcome.value}: {v.reason}"
    tw = seq.car.tyre_width_m * 1000
    truth_ms = sum(1 for r in clip
                   if all(r["margins_mm"][w] <= -tw for w in WHEELS)) * 1000 / seq.fps
    assert abs(v.event.all_four_duration_ms - truth_ms) < 0.35 * truth_ms, (
        f"reported {v.event.all_four_duration_ms:.0f} ms against a true "
        f"{truth_ms:.0f} ms")


def test_clean_laps_produce_nothing() -> None:
    for path in ("clean", "brush"):
        pipe, _, _ = _run(path)
        assert not pipe.decisions.verdicts, (
            f"{path}: invented {len(pipe.decisions.verdicts)} excursion(s)")


# --------------------------------------------------------------------------
# temporal behaviour
# --------------------------------------------------------------------------


def test_single_frame_spike_never_fires() -> None:
    """One bad frame is 20 ms. A system that reports it is accusing on noise."""
    from chronos.temporal import CarTimeline
    cfg = TemporalConfig(enter_frames=3, exit_frames=3, fps=50.0)
    tl = CarTimeline(1, cfg)
    inside = {w: np.array([100.0, 100.0]) for w in WHEELS}
    events = []
    for i in range(20):
        spike = i == 10
        ev = tl.update(i, i * 20.0, inside,
                       lambda pts, _s=spike: np.full(len(pts), -900.0 if _s else 900.0),
                       100.0, 0.95)
        if ev:
            events.append(ev)
    events += [e for e in [tl.close_open_event()] if e]
    assert not events, "a one-frame spike produced an excursion event"


def test_identity_survives_a_detection_gap() -> None:
    pipe, _, _ = _run("violation", occlude=6)
    ids = list(pipe.temporal.timelines)
    assert len(ids) == 1, f"one car became {len(ids)} identities across a 6-frame gap"


def test_losing_the_car_escalates_rather_than_clearing() -> None:
    pipe, _, _ = _run("violation", occlude=10)
    assert pipe.decisions.verdicts
    outcomes = {v.outcome for v in pipe.decisions.verdicts}
    assert Outcome.REVIEW_REQUIRED in outcomes, (
        f"a 10-frame blackout mid-excursion produced {outcomes} and no review")
    v = [x for x in pipe.decisions.verdicts if x.outcome is Outcome.REVIEW_REQUIRED][0]
    assert v.event.track_continuity < 1.0
    assert v.event.frames_missing > 0


def test_event_carries_what_a_steward_would_ask_for() -> None:
    pipe, _, _ = _run("violation")
    ev = pipe.decisions.verdicts[0].event
    assert ev.entry_frame < ev.exit_frame
    assert ev.duration_ms > 0 and ev.all_four_duration_ms > 0
    assert ev.peak_margin_mm < 0
    assert 1 <= ev.wheels_out_max <= 4
    assert set(ev.series) == set(WHEELS)
    assert all(isinstance(s, WheelSample) for s in ev.series["FL"])
    assert 0.0 <= ev.track_continuity <= 1.0
    assert 0.0 <= ev.mean_integrity <= 100.0


# --------------------------------------------------------------------------
# the gating claim
# --------------------------------------------------------------------------


def test_degraded_reference_blocks_the_same_violation() -> None:
    """The whole thesis, as a test.

    Identical car, identical geometry, identical excursion.  The only thing
    that changed is the condition of the line it is measured against.
    """
    clean, _, _ = _run("violation")
    dirty, _, _ = _run("violation", contamination=0.6)
    assert clean.decisions.verdicts and dirty.decisions.verdicts
    assert clean.decisions.verdicts[0].outcome is Outcome.VIOLATION
    assert dirty.decisions.verdicts[0].outcome is Outcome.REVIEW_REQUIRED, (
        f"contaminated reference still produced "
        f"{dirty.decisions.verdicts[0].outcome.value}")
    assert dirty.decisions.verdicts[0].event.mean_integrity < 60


def test_trust_is_the_product_of_its_three_parts() -> None:
    ev = ExcursionEvent(car_id=1, entry_frame=0, exit_frame=10, duration_ms=200.0,
                        peak_margin_mm=-900.0, wheels_out_max=4,
                        mean_integrity=80.0, geometry_confidence=0.9,
                        track_continuity=0.95, n_frames=10,
                        all_four_outside_frames=10, all_four_duration_ms=200.0)
    v = judge(ev)
    assert abs(v.trust - (0.9 * 0.95 * 0.8)) < 1e-6
    assert abs(v.components["wheel_geometry"] - 0.9) < 1e-9
    assert abs(v.components["track_continuity"] - 0.95) < 1e-9
    assert abs(v.components["boundary_integrity"] - 0.8) < 1e-9


def test_engine_never_produces_a_penalty() -> None:
    fields = set(Verdict.__dataclass_fields__)
    for banned in ("penalty", "sanction", "points", "severity", "fine"):
        assert banned not in fields, f"Verdict exposes a {banned!r} field"
    assert {o.value for o in Outcome} == {"CLEAR", "VIOLATION", "REVIEW REQUIRED"}


def test_every_verdict_has_a_readable_reason() -> None:
    for path, contam in (("violation", 0.0), ("one_wheel_on_line", 0.0),
                         ("violation", 0.6)):
        pipe, _, _ = _run(path, contamination=contam)
        for v in pipe.decisions.verdicts:
            assert len(v.reason.split()) >= 5, f"{path}: terse reason {v.reason!r}"
            assert v.reason.strip().endswith((".", "%")), v.reason


def test_session_stats_add_up() -> None:
    pipe, clip, _ = _run("violation")
    s = pipe.decisions.stats()
    assert s.frames_processed == len(clip)
    assert s.events_found == len(pipe.decisions.verdicts)
    assert s.violations + s.auto_cleared + s.escalated == s.events_found
    assert s.throughput_fps > 0 and s.race_seconds_covered > 0


# --------------------------------------------------------------------------
# geometry and wheels
# --------------------------------------------------------------------------


def test_margin_in_mm_matches_ground_truth() -> None:
    seq = SequenceConfig(n_frames=40, path="violation")
    clip = list(generate_clip(seq))
    scene = clip[0]["scene"]
    res = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    geo = build_track_geometry(scene.image, res, TrackGeometryConfig(
        line_width_mm=scene.config.line_width_m * 1000,
        kerb_stripe_pitch_mm=scene.config.kerb_stripe_pitch_m * 1000))
    errs = []
    for r in clip:
        est = geo.margin_mm_clipped(np.array([r["contacts_px"][w] for w in WHEELS]))
        for k, w in enumerate(WHEELS):
            t = r["margins_mm"][w]
            if abs(t) < 1500:
                errs.append(abs(est[k] - t))
    assert errs, "no samples inside the validity range"
    assert np.mean(errs) < 200.0, f"mean margin error {np.mean(errs):.0f} mm"


def test_margin_refuses_to_extrapolate() -> None:
    seq = SequenceConfig(n_frames=6, path="clean")
    clip = list(generate_clip(seq))
    scene = clip[0]["scene"]
    res = detect_boundary(scene.image, BoundaryConfig(), save_debug=False)
    geo = build_track_geometry(scene.image, res, TrackGeometryConfig(
        line_width_mm=scene.config.line_width_m * 1000))
    far = geo.track.points[10] + 400.0 * geo.track.normals[10]
    try:
        geo.margin_mm(far[None, :])
    except ValueError:
        return
    raise AssertionError("returned a confident margin far outside its valid range")


def test_tracker_keeps_one_id_through_low_confidence() -> None:
    """ByteTrack's second pass: a car that dims must not become a new car."""
    cfg = CarConfig()
    t = CarTracker(cfg)
    box = np.array([[100.0, 100.0, 200.0, 200.0]])
    ids = set()
    for i in range(20):
        score = 0.3 if 6 <= i <= 10 else 0.9      # a spell of low confidence
        for tr in t.update(box + i * 2.0, [score]):
            ids.add(tr.id)
    assert len(ids) == 1, f"one car produced ids {ids} across a low-confidence spell"


def test_occlusion_recovery_completes_the_rectangle() -> None:
    from chronos.car import _recover_missing
    found = {"FL": np.array([0.0, 0.0]), "FR": np.array([10.0, 0.0]),
             "RL": np.array([0.0, 20.0])}
    done, recovered = _recover_missing(found)
    assert recovered == ("RR",)
    assert np.allclose(done["RR"], [10.0, 20.0])


# --------------------------------------------------------------------------
# the picture
# --------------------------------------------------------------------------


def test_timeline_renders_and_boxes_only_real_violations() -> None:
    pipe, _, _ = _run("violation")
    cid = next(iter(pipe.temporal.timelines))
    img = render_wheel_timeline(pipe.temporal.timelines[cid].history,
                                [v.event for v in pipe.decisions.verdicts])
    assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] > 100


def test_timeline_refuses_to_draw_nothing() -> None:
    try:
        render_wheel_timeline({w: [] for w in WHEELS})
    except ValueError:
        return
    raise AssertionError("drew an empty timeline, which reads as four wheels inside")


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

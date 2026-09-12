"""CHRONOS -- the whole thing, wired together.

One clip in, verdicts out.  The order matters and is the argument:

    boundary   ->  where is the line
    geometry   ->  how many millimetres is a pixel, across the track
    integrity  ->  is that line still worth measuring against
    cars       ->  who is here, and where do their tyres touch
    temporal   ->  what did those four wheels do over time
    decide     ->  can we stand behind a verdict, and what is it

The baseline for integrity and the metric scale are both captured ONCE from a
clean frame at the start, which is what a session baseline means in practice:
the corner is measured while it is known-good, and everything afterwards is
relative to that.

Run standalone::

    python -m chronos.pipeline --path violation
    python -m chronos.pipeline --path one_wheel_on_line --contamination 0.6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from chronos.boundary import BoundaryConfig, detect_boundary
from chronos.car import CarConfig, CarTracker, detect_cars, draw_cars
from chronos.decide import DecideConfig, DecisionEngine, Verdict
from chronos.degrade import DegradeConfig, build_track_frame, degrade
from chronos.integrity import (IntegrityConfig, capture_baseline, score_integrity)
from chronos.temporal import TemporalConfig, TemporalEngine
from chronos.track import TrackGeometryConfig, build_track_geometry


@dataclass
class PipelineConfig:
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    geometry: TrackGeometryConfig = field(default_factory=TrackGeometryConfig)
    car: CarConfig = field(default_factory=CarConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    decide: DecideConfig = field(default_factory=DecideConfig)
    fps: float = 50.0
    stabilise: bool = False
    """Compensate for a camera that moves.

    Off by default, because the synthetic benchmark is a fixed corner cam and
    registration would be a no-op cost there.  Real footage -- broadcast pans,
    replay chase cams, drone orbits -- needs it on, or the baseline geometry
    describes track that has left the frame and a clean line scores 3.
    See :mod:`chronos.stabilise`.
    """
    integrity_every: int = 1
    """How often to re-score the boundary.

    A boundary's condition changes over a session, not between two frames 20 ms
    apart, so sampling it every N frames is defensible engineering rather than
    a shortcut -- and it is the lever that buys throughput when a corner has to
    keep up with a live feed.  Left at 1 for the demo, where the slider changes
    contamination faster than any real track ever would.
    """


@dataclass
class FrameResult:
    frame_index: int
    t_ms: float
    integrity: Optional[float]
    cars: list
    events: list
    verdicts: list[Verdict]
    registration: object = None
    """How this frame was aligned to the baseline, when stabilising.

    Carries ``H`` (this frame -> baseline) so a caller can map the baseline
    boundary back into this frame to draw it, and ``ok``/``reason`` so an
    unregistrable frame is reported as unmeasured rather than mismeasured.
    """

    measured: bool = True
    reason: str = ""


class Pipeline:
    """Holds the session baseline and runs frames through every stage."""

    def __init__(self, clean_frame: np.ndarray,
                 reference: Optional[np.ndarray] = None,
                 cfg: Optional[PipelineConfig] = None,
                 detector=None, boundary_fn=None):
        """``boundary_fn(frame) -> BoundaryResult`` overrides Module 1.

        Real trackside footage needs a different boundary detector from the
        synthetic corner cam -- see :mod:`chronos.lane` for why the
        region-segmentation approach cannot work there. Making it a
        parameter rather than a branch keeps the rest of the pipeline
        identical on both, which is what lets the temporal and decision
        layers be validated once and trusted on either.
        """
        self.cfg = cfg or PipelineConfig()
        self.detector = detector
        self.boundary_fn = boundary_fn or (
            lambda f: detect_boundary(f, self.cfg.boundary, save_debug=False))

        base = self.boundary_fn(clean_frame)
        if not base.ok:
            raise ValueError(f"Pipeline: no boundary in the baseline frame ({base.reason})")
        self.boundary = base
        self.baseline = capture_baseline(clean_frame, reference, self.cfg.integrity,
                                         result=base)
        self.geometry = build_track_geometry(clean_frame, base, self.cfg.geometry,
                                             self.cfg.integrity,
                                             track=self.baseline.track)
        self.track_frame = build_track_frame(base.polyline, base.drivable_mask,
                                             base.kerb_mask,
                                             n=self.cfg.integrity.n_samples)

        self.tracker = CarTracker(self.cfg.car)
        self.temporal = TemporalEngine(self.cfg.temporal)
        self.decisions = DecisionEngine(self.cfg.decide, fps=self.cfg.fps)
        self.decisions.start()
        self._last_integrity = 100.0
        self._last_score = None

        self.stabiliser = None
        if self.cfg.stabilise:
            from chronos.stabilise import Stabiliser
            base_boxes = None
            if detector is not None:
                try:
                    base_boxes, _ = detector.detect(clean_frame)
                except Exception:
                    base_boxes = None
            # give it the road mask: the plane the homography is valid for
            self.stabiliser = Stabiliser(clean_frame, baseline_boxes=base_boxes)

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        t_start = time.perf_counter()
        t_ms = 1000.0 * frame_index / max(self.cfg.fps, 1e-9)

        # Cars first: a car sitting on the line hides the paint, and hidden
        # paint must be excluded from the integrity measurement rather than
        # scored as degraded.
        cars = detect_cars(frame, frame_index, self.tracker, self.detector,
                           self.geometry, self.cfg.car,
                           self.cfg.temporal.tyre_width_mm)
        boxes = [c.box for c in cars]

        # Put the pixels back where the baseline expects them.  The corner has
        # not moved; the camera has.  Warping this frame into the baseline's
        # view keeps every station, paint width and normal the baseline
        # measured valid -- unchanged, not re-derived on a guess.
        reg = None
        measure_frame = frame
        if self.stabiliser is not None:
            reg = self.stabiliser.register(frame, boxes)
            if not reg.ok:
                self.decisions.note_frames(1, time.perf_counter() - t_start)
                return FrameResult(frame_index, t_ms, None, cars, [], [],
                                   registration=reg, measured=False,
                                   reason=reg.reason)
            measure_frame = self.stabiliser.to_baseline(frame, reg)
            boxes = [self._map_box(b, reg.H) for b in boxes]
            cars = self._cars_in_baseline(cars, reg.H)

        if frame_index % max(self.cfg.integrity_every, 1) == 0:
            # Re-detect with the SAME detector the baseline was taken with.
            # Scoring this frame's coverage against a different detector's
            # idea of the boundary would report a detector disagreement as
            # line degradation.
            try:
                live = self.boundary_fn(measure_frame)
            except Exception:
                live = None
            self._last_score = score_integrity(
                measure_frame, self.baseline, self.cfg.integrity, result=live,
                boundary_cfg=self.cfg.boundary, occlusion_boxes=boxes)
            self._last_integrity = self._last_score.total
        integrity = self._last_integrity
        events = self.temporal.update(frame_index, t_ms, cars,
                                      self.geometry.margin_mm_clipped, integrity)
        verdicts = self.decisions.ingest(events)
        self.decisions.note_frames(1, time.perf_counter() - t_start)
        return FrameResult(frame_index, t_ms, integrity, cars, events, verdicts,
                           registration=reg)

    @staticmethod
    def _map_box(box, H) -> tuple:
        from chronos.stabilise import Stabiliser
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        pts = Stabiliser.map_points(
            np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32), H)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()))

    def _cars_in_baseline(self, cars, H):
        """Re-express detections in the baseline view.

        Contact patches are FOUND in the frame the car is actually in, then
        moved into the baseline's coordinates so the one metric frame built
        from the baseline can measure them.  Detecting in a warped frame
        instead would put the tyres through an interpolation before measuring
        millimetres off them.
        """
        from copy import copy

        from chronos.stabilise import Stabiliser
        out = []
        for c in cars:
            d = copy(c)
            d.box = self._map_box(c.box, H)
            if c.wheels is not None and c.wheels.points:
                w = copy(c.wheels)
                names = list(c.wheels.points)
                pts = Stabiliser.map_points(
                    np.array([c.wheels.points[n] for n in names], np.float32), H)
                w.points = {n: pts[i] for i, n in enumerate(names)}
                d.wheels = w
            out.append(d)
        return out

    def finish(self) -> list[Verdict]:
        """Close any excursion still open and judge it."""
        return self.decisions.ingest(self.temporal.flush())

    def timeline_of(self, car_id: int):
        tl = self.temporal.timelines.get(car_id)
        return None if tl is None else tl.history


# --------------------------------------------------------------------------


def _cli() -> None:
    from benchmark.car_render import SequenceConfig, generate_clip
    from benchmark.detector_sim import SimDetectorConfig, SimulatedDetector
    from chronos.ui.timeline import render_wheel_timeline

    p = argparse.ArgumentParser(description="CHRONOS end-to-end pipeline")
    p.add_argument("--path", default="violation",
                   choices=["clean", "brush", "one_wheel_on_line", "violation",
                            "straight_wide"])
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--contamination", type=float, default=0.0)
    p.add_argument("--kind", default="rubber")
    p.add_argument("--box-contacts", action="store_true",
                   help="derive contact points from the box instead of using the "
                        "benchmark's true ones -- shows the real error budget")
    p.add_argument("--occlude", type=int, default=0,
                   help="frames of consecutive detector blackout, to stress the id")
    p.add_argument("--out", default="debug")
    args = p.parse_args()

    seq = SequenceConfig(n_frames=args.frames, path=args.path)
    clip = list(generate_clip(seq))
    scene = clip[0]["scene"]
    clean = scene.image

    cfg = PipelineConfig(fps=seq.fps)
    cfg.geometry.line_width_mm = scene.config.line_width_m * 1000
    cfg.geometry.kerb_stripe_pitch_mm = scene.config.kerb_stripe_pitch_m * 1000
    cfg.geometry.axle_track_mm = seq.car.axle_track_m * 1000
    cfg.geometry.wheelbase_mm = seq.car.wheelbase_m * 1000
    cfg.temporal.tyre_width_mm = seq.car.tyre_width_m * 1000

    det = SimulatedDetector([r["box_px"] for r in clip],
                            SimDetectorConfig(occlusion_start=args.frames // 2 - 4,
                                              occlusion_frames=args.occlude),
                            contacts=None if args.box_contacts
                            else [r["contacts_px"] for r in clip])
    pipe = Pipeline(clean, scene.gt_boundary, cfg, det)
    print(f"baseline : integrity coverage {pipe.baseline.coverage:.0%} | "
          f"scale {np.median(pipe.geometry.mm_per_px_across):.1f} mm/px across | "
          f"confidence {pipe.geometry.scale_confidence:.2f}")

    tfr = build_track_frame(pipe.boundary.polyline, pipe.boundary.drivable_mask,
                            pipe.boundary.kerb_mask)
    dcfg = DegradeConfig()
    results = []
    for rec in clip:
        img = rec["image"] if args.contamination <= 0 else degrade(
            rec["image"], args.contamination, args.kind, cfg=dcfg, track_frame=tfr)
        results.append(pipe.process(img, rec["frame_index"]))
    verdicts = pipe.finish()

    stats = pipe.decisions.stats()
    print(f"\n{stats.summary()}")
    print(f"contamination: {args.kind} {args.contamination:.2f} -> "
          f"integrity {np.mean([r.integrity for r in results]):.0f} mean")
    if not pipe.decisions.verdicts:
        print("\nno excursions found")
    for v in pipe.decisions.verdicts:
        print(f"\n  {v.as_line()}")

    os.makedirs(args.out, exist_ok=True)
    for cid, tl in pipe.temporal.timelines.items():
        evs = [v.event for v in pipe.decisions.verdicts if v.event.car_id == cid]
        img = render_wheel_timeline(
            tl.history, evs,
            title=f"car #{cid}  |  {args.path}  |  {args.kind} {args.contamination:.2f}"
                  f"  |  integrity {np.mean([r.integrity for r in results]):.0f}")
        path = os.path.join(args.out, f"timeline_{args.path}_{int(args.contamination*100)}.png")
        cv2.imwrite(path, img)
        print(f"\ntimeline : {path}")

    if pipe.decisions.verdicts:
        pkt = pipe.decisions.evidence_packet(pipe.decisions.verdicts[0])
        path = os.path.join(args.out, "evidence_packet.json")
        with open(path, "w") as fh:
            json.dump(pkt, fh, indent=1)
        print(f"evidence : {path}")

    mid = len(results) // 2
    cv2.imwrite(os.path.join(args.out, "car_debug.jpg"),
                draw_cars(clip[mid]["image"], results[mid].cars))
    print(f"cars     : {os.path.join(args.out, 'car_debug.jpg')}")


if __name__ == "__main__":
    _cli()

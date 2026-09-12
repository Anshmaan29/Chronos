"""CHRONOS console -- one window.

Left: the corner, with the boundary and the four contact patches drawn on it.
Right: what the reference is worth, and what that permits us to say.
Bottom: the contamination slider, the wheel-state timeline, the incident log.

The console reacts exactly once: when boundary integrity falls below 50 the
verdict box turns amber and a thin amber rule appears across the top.  Nothing
else moves on its own.  Movement on screen means movement in the data.

Everything the pipeline produces is treated as optional.  A stage that returns
None shows a dash and the window keeps running -- a console that dies because
one frame was unreadable is worse than useless at a race track.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QSizePolicy,
                             QSlider, QVBoxLayout, QWidget)

from chronos.degrade import KINDS, DegradeConfig, degrade
from chronos.pipeline import Pipeline, PipelineConfig
from chronos.temporal import WHEELS, WheelSample, WheelState
from chronos.ui import theme as T
from chronos.ui.timeline import render_wheel_timeline
from chronos.ui.export import export as export_run
from chronos.ui.widgets import (AnalysisReport, Bar, EvidenceStrip, HazardBar,
                                ImagePane, IncidentLog, Readout, SubScores,
                                VerdictBox, Wordmark, apply_block_shadow)

MONO_ONLY = T.MONO

STATE_BGR = {WheelState.INSIDE: (140, 214, 61),
             WheelState.ON_LINE: (0, 176, 255),
             WheelState.OUTSIDE: (77, 72, 229)}


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".gif"}


class FrameSource:
    """Frames from a video, a still image, a folder, or the synthetic clip.

    A still is served as a single repeating frame: every stage still runs, but
    nothing temporal can be learned from it and the console says so.
    """

    def __init__(self, video: Optional[str] = None, path: str = "violation",
                 frames: int = 70, detector: str = "auto"):
        self.video = video
        self.detector = detector
        self.detector_obj = None
        self.is_still = False
        self.mixed_sizes = False
        self.fps = 50.0
        # Footage is FINITE.  It used to loop -- the video rewound to frame 0
        # on EOF and a still was served forever through a modulo -- which made
        # the frame counter and "seconds of race" climb without bound and
        # meant the clip never ended, so the excursion still open at the last
        # frame was never closed and never judged.  A source now runs exactly
        # once and then says it is done.
        self.total: Optional[int] = None     # None = length unknown, read till dry
        self.kind = "video"
        self._cap = None
        self._frames: list[np.ndarray] = []
        self.reference = None            # ground-truth boundary, synthetic only
        self.label = ""

        if video and os.path.isdir(video):
            imgs = [os.path.join(video, f) for f in sorted(os.listdir(video))
                    if os.path.splitext(f)[1].lower() in IMAGE_EXT]
            if not imgs:
                raise FileNotFoundError(f"no images in folder: {video}")
            self._frames = [cv2.imread(p) for p in imgs]
            self._frames = [f for f in self._frames if f is not None]
            # Frames from one camera share one frame size. A folder whose
            # images differ in size is a collection of unrelated
            # photographs, and the session-baseline model does not apply to
            # it -- there is no single corner for a baseline to describe.
            self.mixed_sizes = len({f.shape[:2] for f in self._frames}) > 1
            self.count = self.total = len(self._frames)
            self.is_still = self.count == 1
            self.kind = "stills"
            self.label = f"{os.path.basename(video)}/  ({self.count} images)"
            self.fps = 4.0
        elif video and os.path.splitext(video)[1].lower() in IMAGE_EXT:
            img = cv2.imread(video, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"could not read image: {video}")
            self._frames = [img]
            self.count = self.total = 1
            self.is_still = True
            self.kind = "stills"
            self.fps = 4.0
            self.label = f"{os.path.basename(video)}  (still)"
        elif video:
            if not os.path.exists(video):
                raise FileNotFoundError(f"no such file: {video}")
            self._cap = cv2.VideoCapture(video)
            if not self._cap.isOpened():
                raise RuntimeError(f"--video: could not open {video!r}")
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if fps and fps > 1 else 30.0
            self.count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.total = self.count or None      # some containers do not say
            self.kind = "video"
            self.label = (f"{os.path.basename(video)}  "
                          f"({self.count} frames, {self.fps:.0f} fps)"
                          if self.count else os.path.basename(video))
        else:
            from benchmark.car_render import SequenceConfig, generate_clip
            seq = SequenceConfig(n_frames=frames, path=path)
            clip = list(generate_clip(seq))
            self._frames = [r["image"] for r in clip]
            self._contacts = [r["contacts_px"] for r in clip]
            self._boxes = [r["box_px"] for r in clip]
            self._margins = [r["margins_mm"] for r in clip]
            self.reference = clip[0]["scene"].gt_boundary
            self.scene = clip[0]["scene"]
            self.seq = seq
            self.fps = seq.fps
            self.count = self.total = len(self._frames)
            self.kind = "synthetic"
            self.label = f"synthetic / {path}"

    @property
    def has_truth(self) -> bool:
        return self.reference is not None

    def baseline(self) -> np.ndarray:
        """A clean frame to calibrate against.  The first one."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("--video: could not read the first frame")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return frame
        return self._frames[0]

    def peek(self, index: int) -> Optional[np.ndarray]:
        """Frame ``index`` without disturbing playback position.

        Used to hunt for a usable baseline. Playback is rewound afterwards
        so the run still starts at frame 0 and the frame numbering in the
        evidence strip still means what it says.
        """
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = self._cap.read()
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return frame if ok else None
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def read(self, index: int) -> Optional[np.ndarray]:
        """Frame ``index``, or None once the footage is exhausted.

        Returning None at the end rather than rewinding is what makes the
        clip finite, and a finite clip is what lets the run be finalised:
        the last excursion closed, judged, and reported.
        """
        if self.total is not None and index >= self.total:
            return None
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def truth(self, index: int):
        """True wheel margins for a frame, or None when there is no truth.

        Real footage has none, and the stats panel says so rather than showing
        an accuracy figure it cannot support.
        """
        m = getattr(self, "_margins", None)
        if not m:
            return None
        return m[index % len(m)]

    def make_detector(self):
        """A detector for this source, or None if we have nothing to offer."""
        if not hasattr(self, "_boxes"):
            from chronos.car import CarConfig, YoloDetector
            try:
                return YoloDetector(CarConfig())
            except Exception:
                return None
        from benchmark.detector_sim import SimDetectorConfig, SimulatedDetector
        return SimulatedDetector(self._boxes, SimDetectorConfig(),
                                 contacts=self._contacts)


def stamp_no_reference(vis: np.ndarray, reason: str = "") -> np.ndarray:
    """Mark a frame as unmeasured, on the frame itself.

    The stamp travels with the image into the evidence strip and out through
    the export, so a frame that was never measured cannot later be mistaken
    for one that was -- not in a screenshot, not in a slide, not in a file
    someone opens six months from now.
    """
    h, w = vis.shape[:2]
    scale = max(0.45, min(1.1, w / 1400.0))
    bar = int(30 * scale)
    cv2.rectangle(vis, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(vis, "NO REFERENCE  -  NOTHING MEASURED ON THIS FRAME",
                (int(10 * scale), int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale, (0, 184, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    if reason:
        cv2.rectangle(vis, (0, h - bar), (w, h), (0, 0, 0), -1)
        cv2.putText(vis, reason[:110], (int(10 * scale), h - int(bar * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, (148, 148, 160),
                    max(1, int(scale)), cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


class Engine(QThread):
    """Runs the pipeline off the GUI thread so the slider always feels live."""

    MIN_STABILISE_RATE = 0.6
    MAX_CAMERA_DRIFT_PX = 28.0
    """How far the boundary may move in a second and still be one corner."""

    BASELINE_SEARCH = 40
    """How many frames to look through for a clean baseline before giving up."""

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)
    done = pyqtSignal(object)      # the clip ended; here is the final word

    def __init__(self, source: FrameSource, kind: str = "rubber"):
        super().__init__()
        self.source = source
        self.kind = kind
        self.level = 0.0
        self.timeline_width = 1180
        self.timeline_height = 190
        self.paused = False
        self._stop = False
        self.pipe: Optional[Pipeline] = None
        self.stabilised = False
        self.integrity_every = 1
        self._dcfg = DegradeConfig()
        self._seen: set[int] = set()
        self.margin_errors: list[float] = []
        self._recent: dict[int, np.ndarray] = {}   # ring buffer for evidence
        # A reference that cannot be established does not stop the console.
        # The footage still plays and the frames still reach the evidence
        # strip -- with no number attached to any of them.  See
        # _process_unmeasured.
        self.reference_ok = True
        self.reference_reason = ""
        self.relative_integrity = False
        self.integrity_series: list[float] = []
        self.frames_done = 0
        self.t_start = time.perf_counter()
        self._sample_every = 0          # evidence sampling, unmeasured mode
        # Frame-analysis mode: real footage is analysed frame by frame with
        # chronos.analyze -- no session baseline, a verdict per car per frame.
        self.analysis = False
        self.analysis_report = None
        self._an = None

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------

    def _build(self) -> bool:
        if self.source.kind != "synthetic":
            return self._build_analysis()
        self.status.emit("calibrating session baseline")
        try:
            cfg = PipelineConfig(fps=self.source.fps)
            scene = getattr(self.source, "scene", None)
            if scene is not None:
                cfg.geometry.line_width_mm = scene.config.line_width_m * 1000
                cfg.geometry.kerb_stripe_pitch_mm = scene.config.kerb_stripe_pitch_m * 1000
                cfg.temporal.tyre_width_mm = self.source.seq.car.tyre_width_m * 1000
            cfg.integrity_every = self.integrity_every
            self.source.detector_obj = self.source.make_detector()
            base = self.source.baseline()

            # On real footage the boundary must be established before anything
            # downstream means anything.  If it cannot be, say so and stop --
            # that refusal is the product, not a bug to route around.
            boundary_fn = None
            real = (self.source.detector != "synthetic"
                    and not self.source.has_truth)
            if real:
                from chronos.boundary_real import RealBoundaryConfig, detect as detect_any
                rcfg = RealBoundaryConfig()
                det_name = self.source.detector

                def boundary_fn(frame, _d=det_name, _r=rcfg, _b=cfg.boundary):
                    return detect_any(frame, _d, None, _r, _b, save_debug=False)

                # Real paint is far wider in pixels than the synthetic
                # corner cam's: measured 13-20 px median on this footage
                # against the 14 px ceiling the benchmark was tuned to, so
                # the width probe found nothing and every station was
                # discarded. These are the same measurements, taken over a
                # wider window.
                cfg.integrity.paint_max_px = 30.0
                cfg.integrity.probe_max_px = 60
                # The absolute anchors were fitted on the synthetic corner
                # cam and no real camera reaches them, so on real footage
                # the score is relative to this session's baseline and is
                # labelled as such. See IntegrityConfig.absolute_anchors.
                cfg.integrity.absolute_anchors = False
                self.relative_integrity = True

            # Hunt for a baseline by trying to BUILD one, not merely by
            # detecting a boundary. A frame can yield a polyline and still
            # fail to calibrate -- the paint may be unresolvable along it,
            # or the two bands may read the same material -- and the first
            # version of this search stopped at the polyline and then died
            # one stage later. What makes a frame a baseline is that the
            # whole baseline succeeds on it.
            total = self.source.total or 1
            limit = 1 if not real else min(self.BASELINE_SEARCH, max(total, 1))
            last = ""
            self.baseline_index = 0
            for k in range(0, limit):
                cand = base if k == 0 else self.source.peek(k)
                if cand is None:
                    break
                try:
                    self.pipe = Pipeline(cand, self.source.reference, cfg,
                                         self.source.detector_obj,
                                         boundary_fn=boundary_fn)
                    self.baseline_index = k
                    break
                except Exception as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    self.pipe = None
            if self.pipe is None:
                return self._no_reference(
                    f"no usable baseline in {limit} frame(s) -- {last}")
            if self.baseline_index:
                self.status.emit(f"frame 0 unusable; session baseline taken "
                                 f"from frame {self.baseline_index}")

            if real and getattr(self.source, "mixed_sizes", False):
                sizes = sorted({f"{f.shape[1]}x{f.shape[0]}"
                                for f in self.source._frames})
                return self._no_reference(
                    f"these are {self.source.count} unrelated images, not one "
                    f"camera -- frame sizes {', '.join(sizes[:4])}"
                    f"{' ...' if len(sizes) > 4 else ''}. Boundary integrity is "
                    f"defined relative to a session baseline captured from one "
                    f"fixed camera at one corner, so there is nothing here for "
                    f"it to be relative to. Load a clip, or a folder of frames "
                    f"from a single camera")

            if real:
                # Try to hold the corner still FIRST, rather than trying to
                # prove the camera never moved.  _camera_is_fixed can only
                # compare boundaries it managed to detect, and on real footage
                # it often detects none -- which it reports as "fixed" for want
                # of evidence, and a moving camera then gets measured against a
                # static baseline.  That is how a clean line scores 3.
                fixed, moved = self._camera_is_fixed(boundary_fn)
                can_stab, rate, why = self._can_stabilise()
                if can_stab:
                    cfg.stabilise = True
                    self.pipe = Pipeline(self.source.peek(self.baseline_index),
                                         self.source.reference, cfg,
                                         self.source.detector_obj,
                                         boundary_fn=boundary_fn)
                    self.stabilised = True
                    self.status.emit(
                        f"stabilising to the corner · {rate:.0%} of probe "
                        f"frames registered"
                        + (f" · camera moves {moved:.0f} px" if not fixed else ""))
                elif fixed is False:
                    return self._no_reference(
                        f"the camera is not fixed -- the boundary moved "
                        f"{moved:.0f} px -- and the frames could not be "
                        f"registered back to the baseline either ({why}). A "
                        f"session baseline measures ONE corner from ONE "
                        f"viewpoint; with neither a fixed camera nor a "
                        f"recoverable one, there is nothing for an integrity "
                        f"score to be relative to")
                elif fixed is None:
                    return self._no_reference(
                        "cannot confirm this is one fixed viewpoint: the "
                        "boundary was not detectable on enough frames to "
                        f"compare, and registration did not hold either "
                        f"({why}). Measuring anyway would score camera motion "
                        "as a degraded line -- which is exactly the error this "
                        "engine exists to catch, so it is not going to commit "
                        "it itself")

            self.status.emit(f"baseline captured  |  "
                             f"{np.median(self.pipe.geometry.mm_per_px_across):.1f} mm/px")
            return True
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return False

    def _can_stabilise(self) -> tuple[bool, float, str]:
        """Can the camera's motion be undone well enough to measure through?

        Probes a spread of frames rather than one: a single failure proves
        nothing, and a single success proves less.  Requires most of the probe
        to register, because a clip that only aligns intermittently produces a
        reference that flickers in and out, which is worse than no reference.
        """
        try:
            from chronos.stabilise import Stabiliser
        except Exception as exc:
            return False, 0.0, f"stabiliser unavailable: {exc}"
        base = self.source.peek(self.baseline_index)
        if base is None:
            return False, 0.0, "no baseline frame"
        boxes = None
        if self.source.detector_obj is not None:
            try:
                boxes, _ = self.source.detector_obj.detect(base)
            except Exception:
                boxes = None
        # the probe must use the SAME configuration the pipeline will, or it
        # measures a stabiliser that is not the one that gets built
        try:
            st = Stabiliser(base, baseline_boxes=boxes)
        except Exception as exc:
            return False, 0.0, str(exc)

        total = self.source.total or 0
        probes = [k for k in range(self.baseline_index + 1,
                                   max(self.baseline_index + 2, total))]
        if not probes:
            return False, 0.0, "only one frame"
        step = max(1, len(probes) // 12)
        ok = tried = 0
        for k in probes[::step][:12]:
            f = self.source.peek(k)
            if f is None:
                continue
            b = None
            if self.source.detector_obj is not None:
                try:
                    b, _ = self.source.detector_obj.detect(f)
                except Exception:
                    b = None
            tried += 1
            ok += bool(st.register(f, b).ok)
        if tried == 0:
            return False, 0.0, "no probe frames readable"
        rate = ok / tried
        return rate >= self.MIN_STABILISE_RATE, rate, \
            f"only {rate:.0%} of probe frames registered"

    def _camera_is_fixed(self, boundary_fn) -> tuple[Optional[bool], float]:
        """Is this one corner seen from one viewpoint?

        The integrity score is defined relative to a session baseline: this
        corner, measured while known-good, from this camera. That definition
        quietly requires the camera to stay put. Run it on a chase cam and
        the baseline geometry describes a piece of track that is no longer
        in shot, so the engine dutifully reports the mismatch as
        degradation -- on the Miami replay clip it scored 3, which looks
        like a catastrophically dirty line and is really just a camera that
        moved. A folder of unrelated photographs is the same problem with
        the frames further apart.

        So the assumption is tested rather than assumed, at several points
        rather than one: a single probe that happens to land on a frame the
        detector fails is not evidence that the camera stayed still.
        """
        from chronos.boundary import point_to_polyline_distance
        total = self.source.total or 0
        step = max(1, int(round(self.source.fps)))
        offsets = [step, 2 * step, max(1, (total - 1) - self.baseline_index)]
        worst, tested = 0.0, 0
        for off in offsets:
            k = self.baseline_index + off
            if off <= 0 or (total and k >= total):
                continue
            try:
                later = self.source.peek(k)
                if later is None:
                    continue
                res = boundary_fn(later)
                if not res.ok or res.polyline is None:
                    continue
                d = float(np.median(point_to_polyline_distance(
                    res.polyline, self.pipe.baseline.reference)))
                worst = max(worst, d)
                tested += 1
            except Exception:
                continue
        if tested == 0:
            # Not evidence of a fixed camera -- evidence of nothing. Returning
            # "fixed" here let a moving camera be measured against a static
            # baseline and score a clean line 3. Unknown is its own answer.
            return None, 0.0
        return worst <= self.MAX_CAMERA_DRIFT_PX, worst

    def _no_reference(self, reason: str) -> bool:
        """Enter unmeasured mode.  Returns True: the thread keeps running.

        A console that goes black because the boundary failed looks exactly
        like a console that crashed, and the two must never be confusable on
        stage.  The refusal IS the product, so it is shown -- footage playing,
        every readout dashed, and the reason stated once in plain English.
        """
        self.reference_ok = False
        self.reference_reason = reason
        self._sample_every = max(1, int(round(self.source.fps * 2.0)))
        self.failed.emit("NO REFERENCE — " + reason)
        return True

    def run(self) -> None:
        if not self._build():
            return
        i = 0
        misses = 0
        period = 1.0 / max(self.source.fps, 1.0)
        while not self._stop:
            if self.paused:
                self.msleep(40)
                continue
            t0 = time.perf_counter()
            total = self.source.total
            if total is not None and i >= total:
                break
            payload = self._process(i)
            if payload is None:
                # A source of unknown length ends by running dry.  Three
                # consecutive misses is the end; one is a frame we could not
                # decode and should step over rather than abandon the clip.
                misses += 1
                if misses >= 3 and self.source.read(i) is None:
                    break
            else:
                misses = 0
                self.ready.emit(payload)
            i += 1
            slack = period - (time.perf_counter() - t0)
            if slack > 0:
                self.msleep(int(slack * 1000))
        if not self._stop:
            self._finalise(i)

    def _finalise(self, n_frames: int) -> None:
        if self.analysis:
            return self._finalise_analysis(n_frames)
        """Close the run: judge whatever was still open, then report.

        ``Pipeline.finish()`` was never called from the console before, and
        because the clip also never ended it never could be.  An excursion
        still open on the last frame was therefore never closed and never
        judged -- which is why a car could plainly run wide and the verdict
        box would sit on "waiting for the first excursion" forever.
        """
        verdicts = []
        try:
            if self.pipe is not None:
                verdicts = self.pipe.finish()
        except Exception:
            verdicts = []
        evidence = [e for e in (self._evidence(v) for v in verdicts) if e]
        stats = self.pipe.decisions.stats() if self.pipe is not None else None
        self.done.emit({
            "index": max(n_frames - 1, 0),
            "frames": n_frames,
            "verdicts": verdicts,
            "evidence": evidence,
            "stats": stats,
            "reference_ok": self.reference_ok,
            "kind": self.source.kind,
            "integrity_series": list(self.integrity_series),
        })

    # ------------------------------------------------------------------
    # frame-analysis mode (real footage)
    # ------------------------------------------------------------------

    AN_FLAGGED = ("VIOLATION", "BORDERLINE", "REVIEW REQUIRED")
    AN_WHEEL = {"far-L": "FL", "far-R": "FR", "near-L": "RL", "near-R": "RR"}

    def _build_analysis(self) -> bool:
        """Real footage: analyse every frame, no baseline, verdict per car."""
        from chronos.analyze import AnalyzeConfig, _Tracker
        self.analysis = True
        self.reference_ok = True
        self.reference_reason = ""
        cfg = AnalyzeConfig()
        self._an = SimpleNamespace(cfg=cfg, tracker=_Tracker(cfg.track_iou),
                                   frames=[], annotated={}, last_verdict={},
                                   t_start=time.perf_counter(), peak_fps=0.0,
                                   events_live=0, logged=0)
        self.status.emit(f"analysing {self.source.label} frame by frame")
        return True

    def _an_stats(self):
        a = self._an
        n = len(a.frames)
        wall = max(time.perf_counter() - a.t_start, 1e-9)
        counts = {"VIOLATION": 0, "BORDERLINE": 0, "REVIEW REQUIRED": 0, "CLEAR": 0}
        for f in a.frames:
            for c in f.cars:
                if c.verdict in counts:
                    counts[c.verdict] += 1
        events = counts["VIOLATION"] + counts["BORDERLINE"] + counts["REVIEW REQUIRED"]
        return SimpleNamespace(
            frames_processed=n,
            race_seconds_covered=n / max(self.source.fps, 1e-9),
            throughput_fps=n / wall, peak_fps=a.peak_fps,
            events_found=events, violations=counts["VIOLATION"],
            auto_cleared=counts["CLEAR"], escalated=counts["REVIEW REQUIRED"]
            + counts["BORDERLINE"])

    def _an_verdict(self, frame_index: int, car, integrity) -> SimpleNamespace:
        return SimpleNamespace(
            outcome=SimpleNamespace(value=car.verdict), reason=car.reason,
            trust=car.trust,
            event=SimpleNamespace(car_id=car.car_id, corner="--",
                                  mean_integrity=integrity,
                                  entry_frame=frame_index))

    def _an_caption(self, fr, car) -> str:
        m = "--" if car.worst_margin_mm is None else f"{car.worst_margin_mm:+.0f}mm"
        return (f"f{fr.index}  car #{car.car_id}  {m}  "
                f"{car.tyres_out}/{car.tyres_measured} tyres out")

    def _an_timeline(self, cursor: int):
        """Four tyre lanes for the car that matters most in this clip."""
        a = self._an
        if not a.frames:
            return None
        score: dict[int, int] = {}
        for f in a.frames:
            for c in f.cars:
                score[c.car_id] = score.get(c.car_id, 0) + (
                    3 if c.verdict in self.AN_FLAGGED else 1)
        if not score:
            return None
        focus = max(score, key=score.get)
        state_map = {"INSIDE": WheelState.INSIDE, "ON LINE": WheelState.ON_LINE,
                     "OUTSIDE": WheelState.OUTSIDE}
        series = {w: [] for w in WHEELS}
        run, events = [], []
        for f in a.frames:
            car = next((c for c in f.cars if c.car_id == focus), None)
            byname = {}
            if car is not None:
                for wl in car.wheels:
                    key = self.AN_WHEEL.get(wl.name)
                    if key:
                        byname[key] = wl
            t_ms = 1000.0 * f.index / max(self.source.fps, 1e-9)
            for w in WHEELS:
                wl = byname.get(w)
                ok = wl is not None and wl.state in state_map
                series[w].append(WheelSample(
                    f.index, t_ms, float(wl.margin_mm or 0.0) if wl else 0.0,
                    float(wl.margin_mm or 0.0) if wl else 0.0,
                    state_map.get(wl.state, WheelState.INSIDE) if wl else WheelState.INSIDE,
                    measured=ok))
            hit = car is not None and car.verdict == "VIOLATION"
            if hit:
                run.append((f, car))
            elif run:
                events.append(run)
                run = []
        if run:
            events.append(run)
        boxed = []
        for r in events:
            f0, f1 = r[0][0], r[-1][0]
            worst = min((c.worst_margin_mm for _, c in r if c.worst_margin_mm is not None),
                        default=0.0)
            dur = (f1.index - f0.index + 1) * 1000.0 / max(self.source.fps, 1e-9)
            boxed.append(SimpleNamespace(
                entry_frame=f0.index, exit_frame=f1.index, duration_ms=dur,
                all_four_outside_frames=len(r), all_four_duration_ms=dur,
                peak_margin_mm=worst, peak_saturated=False))
        try:
            from chronos.ui.timeline import TimelineStyle
            style = TimelineStyle(width=max(640, int(self.timeline_width)))
            lane = (int(self.timeline_height) - style.top_margin
                    - style.bottom_margin - 4 * style.lane_gap) // 4
            style.lane_height = max(12, lane)
            return render_wheel_timeline(series, boxed, style=style,
                                         title=f"car #{focus}   tyre state per frame",
                                         cursor_frame=cursor)
        except Exception:
            return None

    def _process_analysis(self, i: int) -> Optional[dict]:
        from chronos.analyze import analyze_frame, draw
        a = self._an
        try:
            raw = self.source.read(i)
            if raw is None:
                return None
            t0 = time.perf_counter()
            fr = analyze_frame(raw, i, i / max(self.source.fps, 1e-9), a.cfg,
                               annotate=False)
            order = sorted(range(len(fr.cars)), key=lambda k: -(
                (fr.cars[k].box[2] - fr.cars[k].box[0]) * (fr.cars[k].box[3] - fr.cars[k].box[1])))
            if self.source.kind == "stills":
                # separate photographs share no time base; a car in one is not
                # the "same car" as a car in the next, so ids restart per image
                from chronos.analyze import _Tracker
                a.tracker = _Tracker(a.cfg.track_iou)
            ids = a.tracker.assign([fr.cars[k].box for k in order])
            for pos, k in enumerate(order):
                fr.cars[k].car_id = ids[pos]
            annotated = draw(raw, fr, {}, 1.0, None)
            dt = time.perf_counter() - t0
            a.peak_fps = max(a.peak_fps, 1.0 / max(dt, 1e-9))
            a.frames.append(fr)
            self.frames_done = len(a.frames)
            if fr.integrity is not None:
                self.integrity_series.append(fr.integrity)
            flagged = any(c.verdict in self.AN_FLAGGED for c in fr.cars)
            if flagged or len(a.annotated) < 60 or self.source.kind == "stills":
                a.annotated[i] = annotated
            self._recent[i] = annotated
            for stale in [k for k in self._recent if k < i - 240]:
                self._recent.pop(stale, None)

            verdicts, evidence = [], []
            for car in fr.cars:
                prev = a.last_verdict.get(car.car_id)
                a.last_verdict[car.car_id] = car.verdict
                new_event = car.verdict in self.AN_FLAGGED and car.verdict != prev
                if self.source.kind == "stills" or new_event:
                    verdicts.append(self._an_verdict(i, car, fr.integrity))
                    evidence.append({
                        "image": annotated, "outcome": car.verdict,
                        "caption": self._an_caption(fr, car),
                        "color": T.verdict_color(car.verdict),
                        "frame_index": i, "has_reference": fr.line_found})
            if self.source.kind == "stills" and not fr.cars:
                evidence.append({
                    "image": annotated, "outcome": fr.verdict,
                    "caption": f"f{i}  no car found  integrity "
                               f"{'--' if fr.integrity is None else round(fr.integrity)}",
                    "color": T.verdict_color(fr.verdict), "frame_index": i,
                    "has_reference": fr.line_found})

            geom = None
            if fr.cars:
                best = max(fr.cars, key=lambda c: c.tyres_measured)
                geom = best.tyres_measured / 4.0
            cars = [SimpleNamespace(wheels=SimpleNamespace(confidence=geom))] if geom else []
            if not fr.line_found:
                self.status.emit(f"frame {i}: {fr.line_reason[:90]}")
            return {"index": i, "frame": annotated,
                    "integrity": fr.integrity,
                    "reference_ok": True,
                    "score": SimpleNamespace(contrast=fr.contrast,
                                             continuity=fr.continuity,
                                             sharpness=fr.sharpness,
                                             contamination=fr.contamination,
                                             total=fr.integrity),
                    "cars": cars, "timeline": self._an_timeline(i),
                    "verdicts": verdicts, "evidence": evidence,
                    "stats": self._an_stats(), "level": 0.0}
        except Exception:
            self.status.emit("frame skipped: " + traceback.format_exc(limit=1)
                             .strip().splitlines()[-1][:80])
            return None

    def _finalise_analysis(self, n_frames: int) -> None:
        from chronos.analyze import build_report, write_outputs
        a = self._an
        wall = time.perf_counter() - a.t_start
        report = build_report(self.source.video or self.source.label, a.frames,
                              self.source.fps, wall, a.cfg)
        try:
            report["output_dir"] = write_outputs(
                report, a.frames, a.annotated, "output",
                self.source.video or "analysis")
        except Exception as exc:
            report["output_dir"] = f"(export failed: {exc})"
        self.analysis_report = report
        evidence = []
        if self.source.kind != "stills":
            for e in report["events"]:
                img = a.annotated.get(e["peak_frame"])
                if img is None:
                    img = self._recent.get(e["peak_frame"])
                if img is None:
                    continue
                m = "--" if e["peak_margin_mm"] is None else f"{e['peak_margin_mm']:+.0f}mm"
                evidence.append({
                    "image": img, "outcome": e["verdict"],
                    "caption": f"PEAK f{e['peak_frame']}  car #{e['car_id']}  {m}  "
                               f"{e['tyres_out_max']}/4 out",
                    "color": T.verdict_color(e["verdict"]),
                    "frame_index": e["peak_frame"], "has_reference": True})
        self.done.emit({
            "index": max(n_frames - 1, 0), "frames": len(a.frames),
            "verdicts": [], "evidence": evidence, "stats": self._an_stats(),
            "reference_ok": True, "kind": "analysis",
            "integrity_series": list(self.integrity_series),
            "report": report})

    def _process(self, i: int) -> Optional[dict]:
        if self.analysis:
            return self._process_analysis(i)
        """One frame, defensively.  Never lets an exception reach the GUI."""
        if not self.reference_ok:
            return self._process_unmeasured(i)
        try:
            raw = self.source.read(i)
            if raw is None:
                return None
            # the clip loops, so a replaying detector must be seeked back with
            # it or it serves boxes for the wrong frame
            seek = getattr(self.source.detector_obj, "seek", None)
            if seek is not None and getattr(self.source, "count", 0):
                seek(i % self.source.count)
            level = float(self.level)
            frame = raw
            if level > 0.0:
                try:
                    frame = degrade(raw, level, self.kind, cfg=self._dcfg,
                                    track_frame=self.pipe.track_frame)
                except Exception:
                    frame = raw
            result = self.pipe.process(frame, i)
            if result.integrity is not None:
                self.integrity_series.append(float(result.integrity))
            self.frames_done += 1
            overlay = self._overlay(frame, result)
            self._recent[i] = overlay
            for stale in [k for k in self._recent if k < i - 240]:
                self._recent.pop(stale, None)

            new = []
            for v in self.pipe.decisions.verdicts:
                key = id(v)
                if key not in self._seen:
                    self._seen.add(key)
                    new.append(v)

            truth = self.source.truth(i)
            car_id = next(iter(self.pipe.temporal.timelines), None)
            if truth is not None and car_id is not None:
                tl = self.pipe.temporal.timelines[car_id]
                for w in WHEELS:
                    try:
                        got = tl.history[w][-1]
                        want = truth[w]
                        if got.measured and abs(want) < 1500:
                            self.margin_errors.append(abs(got.margin_mm - want))
                    except Exception:
                        pass
            timeline = None
            if car_id is not None:
                try:
                    hist = self.pipe.temporal.timelines[car_id].history
                    from chronos.ui.timeline import TimelineStyle
                    # size the render to the strip's own aspect, or the pane
                    # letterboxes it and the lanes lose half their width
                    style = TimelineStyle(width=max(640, int(self.timeline_width)))
                    lane = (int(self.timeline_height) - style.top_margin
                            - style.bottom_margin - 4 * style.lane_gap) // 4
                    style.lane_height = max(12, lane)
                    timeline = render_wheel_timeline(
                        hist, [v.event for v in self.pipe.decisions.verdicts],
                        style=style,
                        title=f"car #{car_id}   wheel state over time",
                        cursor_frame=i)
                except ValueError:
                    timeline = None

            evidence = [self._evidence(v) for v in new]
            if self.source.kind == "stills" and result.integrity is not None:
                # A still has no time base, so it can never produce an
                # excursion and would otherwise leave the strip empty. What a
                # single frame CAN carry is its reference integrity, so each
                # image is filed with its own score -- load a folder and get
                # one scored card per image.
                sc = getattr(self.pipe, "_last_score", None)
                parts = ("" if sc is None else
                         f"  c{sc.contrast:.0f} n{sc.continuity:.0f} "
                         f"s{sc.sharpness:.0f} x{sc.contamination:.0f}")
                col = (T.GREEN if result.integrity >= 75 else
                       T.AMBER if result.integrity >= T.ALERT_INTEGRITY else T.RED)
                evidence.append({
                    "image": overlay, "outcome": f"INTEGRITY {result.integrity:.0f}",
                    "caption": f"f{i}{parts}", "frame_index": i,
                    "has_reference": True, "color": col})
            return {"index": i,
                    "frame": overlay,
                    "evidence": [e for e in evidence if e is not None],
                    "integrity": result.integrity,
                    "score": getattr(self.pipe, "_last_score", None),
                    "cars": result.cars,
                    "timeline": timeline,
                    "verdicts": new,
                    "stats": self.pipe.decisions.stats(),
                    "margin_errors": self.margin_errors,
                    "level": level,
                    "reference_ok": True}
        except Exception:
            self.status.emit("frame skipped: " + traceback.format_exc(limit=1)
                             .strip().splitlines()[-1][:70])
            return None

    # ------------------------------------------------------------------

    def _process_unmeasured(self, i: int) -> Optional[dict]:
        """A frame with no established boundary behind it.

        Cars are still detected, because detecting a car does not depend on
        knowing where the line is -- and showing that the detector works while
        the reference does not is precisely the distinction the project
        exists to draw.  Nothing else runs: no geometry, no integrity, no
        wheel states, no verdict.  There is no boundary to have measured a
        margin against, and an integrity score is a statement about a
        boundary we do not have.

        The contamination slider is inert here and the console says so.
        Contamination is laid along the track frame, and there is no track
        frame -- faking one would put rubber in an arbitrary place and then
        score it, which is the exact dishonesty this mode exists to avoid.
        """
        try:
            raw = self.source.read(i)
            if raw is None:
                return None
            vis = raw.copy()
            det = self.source.detector_obj
            boxes = ()
            if det is not None:
                try:
                    boxes, _ = det.detect(raw)
                except Exception:
                    boxes = ()
            for b in () if boxes is None else boxes:
                x1, y1, x2, y2 = (int(round(float(v))) for v in b[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 172), 1,
                              cv2.LINE_AA)
                cv2.putText(vis, "car", (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 172), 1,
                            cv2.LINE_AA)
            stamp_no_reference(vis, self.reference_reason)

            self._recent[i] = vis
            for stale in [k for k in self._recent if k < i - 240]:
                self._recent.pop(stale, None)
            self.frames_done += 1

            evidence = []
            if self._sample_every and i % self._sample_every == 0:
                secs = i / max(self.source.fps, 1.0)
                evidence.append({
                    "image": vis, "outcome": "NO REFERENCE",
                    "caption": f"f{i}   {secs:5.1f}s   unmeasured",
                    "color": T.AMBER, "frame_index": i,
                    "has_reference": False})

            return {"index": i, "frame": vis, "evidence": evidence,
                    "integrity": None, "score": None, "cars": [],
                    "timeline": None, "verdicts": [], "stats": None,
                    "margin_errors": [], "level": 0.0,
                    "reference_ok": False,
                    "n_cars": 0 if boxes is None else len(boxes)}
        except Exception:
            self.status.emit("frame skipped: " + traceback.format_exc(limit=1)
                             .strip().splitlines()[-1][:70])
            return None

    def _evidence(self, verdict):
        """The frame at the deepest margin of this excursion, with its caption."""
        try:
            ev = verdict.event
            worst_i, worst_m = None, 0.0
            for w in WHEELS:
                for s in ev.series.get(w, ()):
                    if s.margin_mm < worst_m:
                        worst_m, worst_i = s.margin_mm, s.frame_index
            if worst_i is None:
                worst_i = ev.entry_frame
            img = self._recent.get(worst_i)
            if img is None and self._recent:
                near = min(self._recent, key=lambda k: abs(k - worst_i))
                img = self._recent[near]
            if img is None:
                return None
            peak = ("BEYOND RANGE" if ev.peak_saturated
                    else f"{abs(worst_m):.0f}mm")
            cap = (f"f{worst_i}  {peak}  {ev.all_four_duration_ms:.0f}ms  "
                   f"trust {verdict.trust:.0%}")
            return {"image": img, "outcome": verdict.outcome.value,
                    "caption": cap, "frame_index": int(worst_i),
                    "has_reference": True,
                    "color": T.verdict_color(verdict.outcome.value)}
        except Exception:
            return None

    def _overlay(self, frame: np.ndarray, result) -> np.ndarray:
        """Boundary, contact patches coloured by wheel state, margins in mm."""
        vis = frame.copy()
        pipe = self.pipe
        try:
            poly = pipe.baseline.reference
            if poly is not None and len(poly) > 1:
                cv2.polylines(vis, [np.round(poly).astype(np.int32)], False,
                              (227, 230, 232), 1, cv2.LINE_AA)
        except Exception:
            pass

        for car in result.cars or ():
            if car.wheels is None or not car.wheels.points:
                continue
            tl = pipe.temporal.timelines.get(car.car_id)
            for name in WHEELS:
                p = car.wheels.points.get(name)
                if p is None:
                    continue
                state, margin = None, None
                try:
                    s = tl.history[name][-1]
                    state, margin = s.state, s.margin_mm
                except Exception:
                    pass
                col = STATE_BGR.get(state, (227, 230, 232))
                q = (int(round(p[0])), int(round(p[1])))
                cv2.drawMarker(vis, q, col, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA)
                if margin is not None:
                    cv2.putText(vis, f"{name} {margin:+.0f}", (q[0] + 8, q[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        return vis


# --------------------------------------------------------------------------
# window
# --------------------------------------------------------------------------


class Console(QMainWindow):
    def __init__(self, source: FrameSource, kind: str = "rubber"):
        super().__init__()
        self._detector = source.detector
        self.setAcceptDrops(True)
        self.setWindowTitle("CHRONOS  —  reference integrity for track limits")
        self.resize(1600, 1040)
        self.setStyleSheet(T.stylesheet())

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # the one alert: a thin amber rule across the top
        self.alert_rule = HazardBar(14 if T.active().hazard else
                            (6 if T.active().name == 'f1' else 2))
        self.alert_rule.setVisible(False)
        outer.addWidget(self.alert_rule)

        body = QWidget()
        outer.addWidget(body, 1)
        grid = QVBoxLayout(body)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setSpacing(10)

        grid.addWidget(self._header(source))

        top = QHBoxLayout()
        top.setSpacing(10)
        self.video = ImagePane("no signal")
        top.addWidget(self.video, 6)
        top.addWidget(self._metrics(), 4)
        grid.addLayout(top, 5)

        strip = self._slider_strip()
        apply_block_shadow(strip)
        grid.addWidget(strip)

        # The evidence strip sits directly under the control it is produced
        # by, and above the timeline, because it is the first thing anyone
        # asks to see: not "what does it score" but "show me the frame".
        self.evidence = EvidenceStrip()
        self.evidence.jump.connect(self.on_jump)
        apply_block_shadow(self.evidence)
        grid.addWidget(self.evidence)

        self.timeline = ImagePane("wheel-state timeline")
        self.timeline.setMinimumHeight(132)
        self.timeline.setMaximumHeight(172)
        grid.addWidget(self.timeline)

        self.log = IncidentLog()
        grid.addWidget(self.log, 1)

        # full-window analysis, on R.  An overlay, never a second window: a
        # demo that has to find another window on a projector has already
        # lost the room.
        self.report = AnalysisReport()
        self.report.setParent(root)
        self.report.setVisible(False)
        self._reviewing: Optional[int] = None

        self._sweep_cache = None
        self._last_timeline = None
        self._done = False
        self.engine = Engine(source, kind)
        self.engine.ready.connect(self.on_frame)
        self.engine.failed.connect(self.on_failed)
        self.engine.done.connect(self.on_done)
        self.engine.status.connect(lambda s: self.status.setText(s))
        self.engine.start()

    # ------------------------------------------------------------------

    def _header(self, source: FrameSource) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 4)
        t = T.active()
        title = Wordmark("CHRONOS", 30 if t.name == "f1" else 24)
        sub = QLabel("REFERENCE INTEGRITY")
        sub.setObjectName("caption")
        self.status = QLabel(source.label)
        self.status.setObjectName("caption")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(title)
        h.addSpacing(10)
        h.addWidget(sub)
        h.addStretch(1)
        h.addWidget(self.status)
        return w

    def _metrics(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        apply_block_shadow(panel)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(12)

        self.integrity_cap = QLabel("BOUNDARY INTEGRITY   ·   CURRENT FRAME")
        self.integrity_cap.setObjectName("caption")
        lay.addWidget(self.integrity_cap)

        self.integrity = QLabel("--")
        self.integrity.setObjectName("huge")
        self.integrity.setStyleSheet(
            f"color: {T.DIM}; font-size: {T.active().huge_px}px; "
            f"font-weight: {900 if T.active().name == 'brutal' else 400};")
        lay.addWidget(self.integrity)

        self.integrity_bar = Bar(8 if T.active().name == "instrument" else 16)
        lay.addWidget(self.integrity_bar)

        self.subs = SubScores()
        apply_block_shadow(self.subs)
        lay.addWidget(self.subs)

        row = QHBoxLayout()
        row.setSpacing(14)
        self.car_conf = Readout("CAR GEOMETRY · LIVE")
        self.trust = Readout("TRUST · LIVE")
        row.addWidget(self.car_conf)
        row.addWidget(self.trust)
        lay.addLayout(row)

        self.verdict = VerdictBox()
        lay.addWidget(self.verdict)

        self.session = QLabel("--")
        self.session.setObjectName("caption")
        self.session.setWordWrap(True)
        lay.addWidget(self.session)
        lay.addStretch(1)
        return panel

    def _slider_strip(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        h = QHBoxLayout(panel)
        h.setContentsMargins(16, 12, 16, 12)
        h.setSpacing(16)

        cap = QLabel("CONTAMINATION")
        cap.setObjectName("caption")
        # the heavy condensed label faces are wider than mono at the same
        # size, so a single fixed width clips the word in one theme
        cap.setFixedWidth(170 if T.active().name == "brutal" else 140)
        h.addWidget(cap)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.valueChanged.connect(self.on_level)
        h.addWidget(self.slider, 1)

        self.level_label = QLabel("0.00")
        self.level_label.setStyleSheet(f"color: {T.AMBER}; font-size: 17px;")
        self.level_label.setFixedWidth(56)
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        h.addWidget(self.level_label)

        self.load = QPushButton("LOAD")
        self.load.setToolTip("open a video, image or folder  (key: O) "
                             "-- or drag one onto this window")
        self.load.clicked.connect(self.on_load)
        h.addWidget(self.load)

        self.rate = QPushButton("INTEGRITY 1:1")
        self.rate.setToolTip("how often the boundary is re-scored  (key: I)")
        self.rate.clicked.connect(self.cycle_integrity_rate)
        h.addWidget(self.rate)

        self.pause = QPushButton("PAUSE")
        self.pause.setToolTip("key: space")
        self.pause.clicked.connect(self.on_pause)
        h.addWidget(self.pause)

        self.report_btn = QPushButton("REPORT")
        self.report_btn.setToolTip("full analysis, with provenance  (key: R)")
        self.report_btn.clicked.connect(self.toggle_report)
        h.addWidget(self.report_btn)

        self.export_btn = QPushButton("EXPORT")
        self.export_btn.setToolTip("write report + evidence to ./output/  (key: E)")
        self.export_btn.clicked.connect(self.on_export)
        h.addWidget(self.export_btn)
        return panel

    # ------------------------------------------------------------------

    def on_level(self, value: int) -> None:
        level = value / 100.0
        self.engine.level = level
        self.level_label.setText(f"{level:.2f}")

    def cycle_integrity_rate(self) -> None:
        """Cycle how often the boundary is re-scored: every frame, 1:5, 1:25.

        The condition of a line changes over a session, not between two frames
        20 ms apart, so sampling it is engineering rather than a shortcut -- and
        it is the lever that decides whether a corner keeps up with a live feed.
        Wired to a key so the throughput claim can be demonstrated, not quoted.
        """
        if self.engine.pipe is None:
            return
        nxt = {1: 5, 5: 25, 25: 1}[self.engine.pipe.cfg.integrity_every]
        self.engine.pipe.cfg.integrity_every = nxt
        self.engine.integrity_every = nxt
        self.rate.setText(f"INTEGRITY 1:{nxt}")

    def _huge_css(self, colour: str) -> str:
        t = T.active()
        return (f"color: {colour}; font-family: "
                f"{t.label_font if t.name == 'brutal' else MONO_ONLY}; "
                f"font-size: {t.huge_px}px; "
                f"font-weight: {900 if t.name == 'brutal' else 400};")

    def on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open footage", os.path.abspath("data/real"),
            "Footage (*.mp4 *.mov *.avi *.mkv *.m4v *.webm *.gif "
            "*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)")
        if path:
            self.open_source(path)

    def open_source(self, path: str) -> None:
        """Swap the footage without restarting the app.

        The session baseline belongs to the footage it was captured from, so
        everything -- pipeline, tracker, timelines, incident log -- is rebuilt.
        Carrying a baseline across a cut would be measuring one corner against
        another.
        """
        # Disconnect before stopping.  Signals the old thread already
        # emitted are sitting in the event queue and will still be delivered
        # after the swap; left connected they write the previous clip's
        # timeline, verdicts and no-reference state into the new session.
        for sig in (self.engine.ready, self.engine.failed,
                    self.engine.done, self.engine.status):
            try:
                sig.disconnect()
            except TypeError:
                pass
        self.engine.stop()
        self.engine.wait(2500)
        try:
            source = FrameSource(path, detector=self._detector)
        except Exception as exc:
            self.status.setText(f"{type(exc).__name__}: {exc}")
            self.verdict.set_verdict("CANNOT OPEN", str(exc), T.AMBER)
            return
        self.log.setRowCount(0)
        self.evidence.clear()
        self.evidence.set_mode("stills" if source.kind == "stills" else "excursion")
        self._reviewing = None
        self._last_timeline = None
        self._done = False
        self.timeline.clear_frame("wheel-state timeline")
        self.video.clear_frame("no signal")
        self.alert_rule.setVisible(False)
        self.integrity.setText("--")
        self.integrity_bar.set_value(None)
        self.subs.update_scores(None)
        self.car_conf.set_value(None)
        self.trust.set_value(None)
        self.session.setText("--")
        self.slider.setEnabled(True)
        self.slider.setToolTip("")
        self.verdict.set_verdict(None, "waiting for the first excursion", T.DIM)
        self.status.setText(f"{source.label}  ·  detector {self._detector}")
        self.engine = Engine(source, self.engine.kind)
        self.engine.ready.connect(self.on_frame)
        self.engine.failed.connect(self.on_failed)
        self.engine.done.connect(self.on_done)
        self.engine.status.connect(lambda t: self.status.setText(t))
        self.engine.start()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.open_source(urls[0].toLocalFile())

    def on_pause(self) -> None:
        if getattr(self, "_done", False):
            return
        self.engine.paused = not self.engine.paused
        self.pause.setText("RESUME" if self.engine.paused else "PAUSE")

    def on_failed(self, message: str) -> None:
        """A reference that cannot be established is shown, not worked around."""
        self.status.setText(message[:120])
        self.verdict.set_verdict("NO REFERENCE", message, T.AMBER)
        self.integrity.setText("--")
        self.integrity.setStyleSheet(self._huge_css(T.DIM))
        self.integrity_bar.set_value(None)
        self.subs.update_scores(None)
        self.car_conf.set_value(None)
        self.trust.set_value(None)
        self.evidence.set_no_reference(True)
        self.slider.setEnabled(False)
        self.alert_rule.setVisible(True)

    def on_frame(self, payload: dict) -> None:
        integ = payload.get("integrity")
        ref_ok = bool(payload.get("reference_ok", True))
        # tell the worker how wide to draw the next timeline, so the strip is
        # filled rather than letterboxed at whatever size the window is now
        self.engine.timeline_width = max(640, self.timeline.width() - 4)
        self.engine.timeline_height = max(120, self.timeline.height() - 4)
        # a frame being reviewed holds the pane; live frames keep arriving
        # behind it and the strip keeps filling
        if self._reviewing is None:
            self.video.show_frame(payload.get("frame"))

        if getattr(self.engine, "relative_integrity", False):
            self.integrity_cap.setText(
                "BOUNDARY INTEGRITY   ·   RELATIVE TO SESSION BASELINE")
            self.integrity_cap.setToolTip(
                "The absolute anchors are calibrated on the synthetic corner "
                "cam and no real camera reaches them, so this reading is "
                "degradation since the baseline frame, not an absolute grade.")

        for e in payload.get("evidence") or ():
            self.evidence.add(e["image"], e["outcome"], e["caption"],
                              e["color"], e.get("frame_index", -1),
                              e.get("has_reference", True))
        if not ref_ok:
            self.evidence.set_mode("none")
            self.slider.setEnabled(False)
        elif self.engine.source.kind == "stills":
            self.evidence.set_mode("stills")
            self.slider.setToolTip(
                "inert without a reference: contamination is laid along the "
                "track frame, and there is no track frame")
        if payload.get("timeline") is not None:
            self.timeline.show_frame(payload["timeline"])
            self._last_timeline = payload["timeline"]

        if integ is None:
            self.integrity.setText("--")
            self.integrity.setStyleSheet(self._huge_css(T.DIM))
            self.integrity_bar.set_value(None)
        else:
            col = T.GREEN if integ >= 75 else T.AMBER if integ >= T.ALERT_INTEGRITY else T.RED
            self.integrity.setText(f"{integ:.0f}")
            self.integrity.setStyleSheet(self._huge_css(col))
            self.integrity_bar.set_value(integ, col)

        self.subs.update_scores(payload.get("score"))

        cars = payload.get("cars") or []
        conf = next((c.wheels.confidence for c in cars if c.wheels is not None), None)
        self.car_conf.set_value(None if conf is None else f"{conf:.0%}",
                                T.OFFWHITE if conf is None else
                                (T.GREEN if conf >= 0.7 else T.AMBER))

        # live trust is geometry x integrity; track continuity only exists once
        # an excursion has a span to measure it over, and the event trust that
        # includes it is shown inside the verdict box
        trust = None if (conf is None or integ is None) else conf * (integ / 100.0)
        if trust is None and integ is not None:
            trust = integ / 100.0        # no car in shot: the reference alone
        self.trust.set_value(None if trust is None else f"{trust:.0%}",
                             T.OFFWHITE if trust is None else
                             (T.GREEN if trust >= 0.6 else T.AMBER))

        # No reference is the loudest alert there is, and it has no integrity
        # value to trip the threshold -- so it raises the rule on its own.
        alert = (not ref_ok) or (integ is not None and integ < T.ALERT_INTEGRITY)
        self.alert_rule.setVisible(alert)

        for v in payload.get("verdicts") or ():
            colour = T.verdict_color(v.outcome.value)
            self.verdict.set_verdict(v.outcome.value, v.reason, colour,
                                     v.trust, v.event.mean_integrity)
            self.log.add(payload["index"] * 1000.0 / max(self.engine.source.fps, 1),
                         v.event.car_id, v.event.corner or "T6", v.outcome.value,
                         v.trust, v.event.mean_integrity, v.reason, colour)

        if self.report.isVisible():
            self.report.update_report(self._report_data())

        s = payload.get("stats")
        if s is not None:
            self.session.setText(
                f"{s.frames_processed} FRAMES  ·  {s.race_seconds_covered:.1f}s OF RACE  ·  "
                f"{s.throughput_fps:.0f} FPS MEAN  ·  {s.peak_fps:.0f} FPS PEAK  ·  "
                f"{s.events_found} EXCURSIONS  ·  "
                f"{s.violations} VIOLATION  ·  {s.auto_cleared} CLEARED  ·  "
                f"{s.escalated} REVIEW")

    def on_done(self, payload: dict) -> None:
        """The clip ended.  Say so, and say what it came to.

        Until now nothing ever ended: the footage looped, the counters climbed
        forever and the last excursion was never closed.  The end of a clip is
        the moment the run actually has an answer, so it is stated plainly
        rather than left to be inferred from a frame counter that stopped
        moving.
        """
        self._done = True
        self.pause.setEnabled(False)
        self.pause.setText("ENDED")
        if payload.get("kind") == "analysis":
            self._on_done_analysis(payload)
            return

        if getattr(self.engine, "relative_integrity", False):
            self.integrity_cap.setText(
                "BOUNDARY INTEGRITY   ·   RELATIVE TO SESSION BASELINE")
            self.integrity_cap.setToolTip(
                "The absolute anchors are calibrated on the synthetic corner "
                "cam and no real camera reaches them, so this reading is "
                "degradation since the baseline frame, not an absolute grade.")

        for e in payload.get("evidence") or ():
            self.evidence.add(e["image"], e["outcome"], e["caption"],
                              e["color"], e.get("frame_index", -1),
                              e.get("has_reference", True))
        for v in payload.get("verdicts") or ():
            colour = T.verdict_color(v.outcome.value)
            self.verdict.set_verdict(v.outcome.value, v.reason, colour,
                                     v.trust, v.event.mean_integrity)
            self.log.add(v.event.entry_frame * 1000.0
                         / max(self.engine.source.fps, 1),
                         v.event.car_id, v.event.corner or "T6",
                         v.outcome.value, v.trust, v.event.mean_integrity,
                         v.reason, colour)

        n = payload.get("frames", 0)
        kind = payload.get("kind", "video")
        stats = payload.get("stats")
        ref_ok = payload.get("reference_ok", True)

        if not ref_ok:
            head = f"NO REFERENCE — {n} frames read, nothing measured"
        elif kind == "stills":
            series = payload.get("integrity_series") or []
            integ = f"integrity {series[-1]:.0f}" if series else "no integrity"
            head = (f"{n} image{'s' if n != 1 else ''} scored · {integ} · "
                    f"a still has no time base, so no excursion can be judged")
        elif stats is not None and stats.events_found == 0:
            head = (f"END OF CLIP · {n} frames · no excursion found — "
                    f"no car had all four wheels outside the boundary")
        elif stats is not None:
            head = (f"END OF CLIP · {n} frames · {stats.events_found} excursion"
                    f"{'s' if stats.events_found != 1 else ''} · "
                    f"{stats.violations} violation · {stats.auto_cleared} clear · "
                    f"{stats.escalated} review")
        else:
            head = f"END OF CLIP · {n} frames"
        self.status.setText(head)

        if self.verdict.label.text() == "STANDBY":
            # An empty verdict box at the end of a clip reads as a hang.  Say
            # which of the two reasons it is: nothing happened, or we could
            # not look.
            if not ref_ok:
                self.verdict.set_verdict(
                    "NO REFERENCE", self.engine.reference_reason, T.AMBER)
            elif kind == "stills":
                self.verdict.set_verdict(
                    "NO VERDICT", "A single frame cannot show a violation. The "
                    "rule is about what four wheels did over time, so a still "
                    "is scored for reference integrity only.", T.DIM)
            else:
                self.verdict.set_verdict(
                    "NO EXCURSION", "Ran to the end of the clip without any car "
                    "putting all four wheels outside the boundary.", T.DIM)
        if self.report.isVisible():
            self.report.update_report(self._report_data())

    def _on_done_analysis(self, payload: dict) -> None:
        """End of a frame-by-frame analysis: state the answer plainly."""
        r = payload.get("report") or {}
        for e in payload.get("evidence") or ():
            self.evidence.add(e["image"], e["outcome"], e["caption"],
                              e["color"], e.get("frame_index", -1),
                              e.get("has_reference", True))
        self.evidence.set_mode("excursion")
        fv = r.get("frame_verdicts", {})
        n = r.get("frames_analysed", payload.get("frames", 0))
        head = (f"ANALYSIS DONE · {n} frame{'s' if n != 1 else ''} · "
                f"{r.get('violation_events', 0)} violation event(s) · "
                f"VIOLATION {fv.get('VIOLATION', 0)} / BORDERLINE {fv.get('BORDERLINE', 0)} / "
                f"REVIEW {fv.get('REVIEW REQUIRED', 0)} / CLEAR {fv.get('CLEAR', 0)} frames"
                f" · saved -> {r.get('output_dir', '')}")
        self.status.setText(head)
        events = r.get("events") or []
        if events:
            top = next((e for e in events if e["verdict"] == "VIOLATION"), events[0])
            m = "" if top["peak_margin_mm"] is None else f", peak {top['peak_margin_mm']:+.0f} mm"
            self.verdict.set_verdict(
                top["verdict"],
                f"car #{top['car_id']} frames {top['entry_frame']}-{top['exit_frame']}"
                f" (worst at frame {top['peak_frame']}{m}). {top['reason']}",
                T.verdict_color(top["verdict"]), top.get("trust"),
                top.get("integrity_mean"))
        elif self.verdict.label.text() in ("STANDBY", ""):
            worst = max((f for f in r.get("per_frame", [])),
                        key=lambda f: {"CLEAR": 0, "NO REFERENCE": 1, "BORDERLINE": 2,
                                       "REVIEW REQUIRED": 3, "VIOLATION": 4}.get(f["verdict"], 0),
                        default=None)
            if worst is not None and worst["cars"]:
                c = worst["cars"][0]
                self.verdict.set_verdict(c["verdict"], c["reason"],
                                         T.verdict_color(c["verdict"]), c.get("trust"),
                                         worst.get("integrity"))
            else:
                self.verdict.set_verdict(
                    "NO CAR FOUND" if r.get("line_found_frames") else "NO REFERENCE",
                    "No car could be located against the track limit in this footage."
                    if r.get("line_found_frames") else
                    "No track limit could be established in this footage.",
                    T.DIM)
        if self.report.isVisible():
            self.report.update_report(self._report_data())

    def on_jump(self, frame_index: int) -> None:
        """Review the frame a verdict was made on.

        The engine is paused and the pane is held on that frame; the pipeline
        is NOT rewound.  Re-running frames through it would re-open excursions
        and re-issue verdicts that already happened, so the incident log would
        grow every time someone clicked a thumbnail.  The timeline, the log
        and the counters stay authoritative, and what changes is only what is
        being looked at.
        """
        card = self.evidence.card_for(frame_index)
        if card is None:
            return
        self._reviewing = frame_index
        self.engine.paused = True
        self.pause.setText("RESUME")
        self.video.show_frame(card.image)
        self.evidence.select(frame_index)
        secs = frame_index / max(self.engine.source.fps, 1.0)
        self.status.setText(
            f"REVIEWING frame {frame_index}  ({secs:.2f}s)  ·  "
            f"{card.outcome}  ·  press L or space for live")

    def resume_live(self) -> None:
        self._reviewing = None
        self.evidence.select(-1)
        if getattr(self, "_done", False):
            return          # the clip has ended; there is no live to go back to
        self.engine.paused = False
        self.pause.setText("PAUSE")
        self.status.setText(self.engine.source.label)

    def toggle_report(self) -> None:
        if self.report.isVisible():
            self.report.setVisible(False)
            self.report_btn.setChecked(False)
            return
        self.report.setGeometry(self.centralWidget().rect())
        self.report.update_report(self._report_data())
        self.report.raise_()
        self.report.setVisible(True)

    def on_export(self) -> None:
        """Write the report, the evidence frames and the packets to ./output/."""
        if getattr(self.engine, "analysis", False):
            try:
                from chronos.analyze import build_report, write_outputs
                a = self.engine._an
                r = build_report(self.engine.source.video or self.engine.source.label,
                                 a.frames, self.engine.source.fps,
                                 time.perf_counter() - a.t_start, a.cfg)
                out = write_outputs(r, a.frames, a.annotated, "output",
                                    self.engine.source.video or "analysis")
                self.status.setText(f"exported {len(a.frames)} frames analysis -> {out}"
                                    f"  (report.json, frames.csv, flagged PNGs, summary.png)")
            except Exception as exc:
                self.status.setText(f"export failed: {type(exc).__name__}: {exc}")
            return
        try:
            data = self._report_data()
            pipe = self.engine.pipe
            verdicts = list(pipe.decisions.verdicts) if pipe else []
            packets = ([pipe.decisions.evidence_packet(v) for v in verdicts]
                       if pipe else [])
            out = export_run(data, self.evidence.cards(), verdicts, packets,
                             self._last_timeline)
            n = len(self.evidence.cards())
            self.status.setText(f"exported {n} frames + report -> {out}")
        except Exception as exc:
            self.status.setText(f"export failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------

    def _provenance(self) -> str:
        """The line that keeps every number under it honest.

        Named on the console and repeated at the top of every exported file,
        because a figure quoted without the conditions that produced it is
        the specific failure this project exists to argue against.
        """
        src = os.path.basename(self.engine.source.video or "") or \
            self.engine.source.label
        status = "ESTABLISHED" if self.engine.reference_ok else "NO REFERENCE"
        return (f"Source: {src}  |  Detector: {self._detector}  |  "
                f"Boundary: {status}  |  "
                f"Validated against: synthetic ground truth only")

    def _report_data(self) -> dict:
        eng = self.engine
        if getattr(eng, "analysis", False):
            return self._report_data_analysis()
        pipe = eng.pipe
        ref_ok = eng.reference_ok
        data: dict = {
            "provenance": self._provenance(),
            "reference_ok": ref_ok,
            "source_path": eng.source.video or eng.source.label,
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ref_status": "ESTABLISHED" if ref_ok else "NO REFERENCE",
            "ref_reason": (eng.reference_reason if not ref_ok
                           else (pipe.boundary.reason if pipe else None)),
            "integ_level": f"{eng.level:.2f}  ({eng.kind})" if ref_ok else None,
        }

        # frames and throughput do not depend on a boundary, so they are
        # reported in both modes -- they are facts about this run
        if pipe is not None:
            s = pipe.decisions.stats()
            resolved = (s.violations + s.auto_cleared) / max(s.events_found, 1)
            data.update({
                "frames": f"{s.frames_processed}",
                "race_s": f"{s.race_seconds_covered:.1f} s",
                "fps": f"{s.throughput_fps:.0f} mean   {s.peak_fps:.0f} peak",
                "events": f"{s.events_found}",
                "violations": f"{s.violations}",
                "cleared": f"{s.auto_cleared}",
                "escalated": f"{s.escalated}",
                "resolved": f"{resolved:.0%}" if s.events_found else None,
            })
        else:
            wall = max(time.perf_counter() - eng.t_start, 1e-9)
            data.update({
                "frames": f"{eng.frames_done}",
                "race_s": f"{eng.frames_done / max(eng.source.fps, 1e-9):.1f} s",
                "fps": f"{eng.frames_done / wall:.0f} mean",
            })

        if pipe is not None and ref_ok:
            try:
                data["ref_agreement"] = (
                    "--" if pipe.boundary.agreement_px is None else
                    f"{pipe.boundary.agreement_px:.1f} px over "
                    f"{pipe.boundary.agreement_span:.0%}")
                data["ref_coverage"] = f"{pipe.baseline.coverage:.0%}"
                data["ref_scale"] = (
                    f"{np.median(pipe.geometry.mm_per_px_across):.1f} mm/px  "
                    f"(confidence {pipe.geometry.scale_confidence:.2f})")
            except Exception:
                pass
            series = eng.integrity_series
            if series:
                below = sum(1 for v in series if v < T.ALERT_INTEGRITY)
                data.update({
                    "integ_now": f"{series[-1]:.0f}",
                    "integ_mean": f"{np.mean(series):.0f}",
                    "integ_min": f"{min(series):.0f}",
                    "integ_below": f"{below} of {len(series)}  "
                                   f"({below / len(series):.0%})",
                })

        errs = eng.margin_errors
        if errs and ref_ok:
            data["margin_mean"] = f"{np.mean(errs):.0f} mm"
            data["margin_p95"] = f"{np.percentile(errs, 95):.0f} mm"
            data["margin_n"] = f"{len(errs)}"
        data.update(self._sweep_numbers())
        return data

    def _report_data_analysis(self) -> dict:
        """Every figure from the frame-by-frame analysis, live or final."""
        from chronos.analyze import build_report
        eng = self.engine
        a = eng._an
        r = eng.analysis_report
        if r is None and a is not None and a.frames:
            r = build_report(eng.source.video or eng.source.label, a.frames,
                             eng.source.fps, time.perf_counter() - a.t_start, a.cfg)
        src = os.path.basename(eng.source.video or "") or eng.source.label
        data = {
            "provenance": (f"Source: {src}  |  Mode: frame-by-frame analysis  |  "
                           f"Limit: outer edge of white line on asphalt boundary  |  "
                           f"Integrity: absolute per frame  |  Scale: car width 2000 mm"),
            "reference_ok": True,
            "source_path": eng.source.video or eng.source.label,
            "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not r:
            data["ref_status"] = "ANALYSING"
            return data
        fv = r["frame_verdicts"]
        fmt = lambda v, suf="": "--" if v is None else f"{v}{suf}"
        ev_lines = []
        for e in r["events"][:14]:
            m = "--" if e["peak_margin_mm"] is None else f"{e['peak_margin_mm']:+.0f} mm"
            ev_lines.append(f"car #{e['car_id']} {e['verdict']}  f{e['entry_frame']}-"
                            f"{e['exit_frame']}  peak f{e['peak_frame']} {m}  "
                            f"{e['tyres_out_max']}/4 out  integ {fmt(e['integrity_mean'])}")
        data.update({
            "ref_status": f"LIMIT FOUND on {r['line_found_frames']}/{r['frames_analysed']} "
                          f"frames ({r['line_found_pct']}%)",
            "ref_reason": (r["per_frame"][-1]["line_reason"] if r["per_frame"] else None),
            "ref_scale": "car width 2000 mm (first-order, per car)",
            "integ_now": fmt(eng.integrity_series[-1] if eng.integrity_series else None),
            "integ_mean": fmt(r["integrity_mean"]),
            "integ_min": fmt(r["integrity_min"]),
            "integ_below": f"{r['integrity_below_50_frames']} of {r['frames_analysed']}",
            "frames": f"{r['frames_analysed']}",
            "race_s": fmt(r["duration_s"], " s"),
            "fps": f"{fmt(r['throughput_fps'])} mean  ·  {fmt(r['mean_ms_per_frame'])} ms/frame",
            "events": f"{len(r['events'])}",
            "violations": f"{r['violation_events']} events  ·  {fv['VIOLATION']} frames",
            "cleared": f"{fv['CLEAR']} frames",
            "escalated": f"{r['review_events']} events  ·  {fv['REVIEW REQUIRED']} frames",
            "resolved": fmt(r["auto_resolved_pct"], "%"),
            "an_cars": f"{r['cars_detected_total']} detections  ·  {r['unique_cars']} unique IDs",
            "an_verdicts": (f"VIOLATION {fv['VIOLATION']}  ·  BORDERLINE {fv['BORDERLINE']}  ·  "
                            f"REVIEW {fv['REVIEW REQUIRED']}  ·  CLEAR {fv['CLEAR']}  ·  "
                            f"NO REF {fv['NO REFERENCE']}"),
            "an_viol_frames": ", ".join(map(str, r["violation_frames"][:40])) or "none",
            "an_border_frames": ", ".join(map(str, r["borderline_frames"][:40])) or "none",
            "an_margin": ("--" if r["margin_mm_min"] is None else
                          f"deepest {r['margin_mm_min']:+.0f} mm  ·  median "
                          f"{r['margin_mm_median']:+.0f} mm"),
            "an_events": "\n".join(ev_lines) or "no violation or review events",
            "an_output": r.get("output_dir", "written at end of run"),
        })
        data.update(self._sweep_numbers())
        return data

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.report.isVisible():
            self.report.setGeometry(self.centralWidget().rect())

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            if self._reviewing is not None:
                self.resume_live()
            else:
                self.on_pause()
        elif key == Qt.Key.Key_I:
            self.cycle_integrity_rate()
        elif key == Qt.Key.Key_O:
            self.on_load()
        elif key in (Qt.Key.Key_R, Qt.Key.Key_S):
            self.toggle_report()
        elif key == Qt.Key.Key_E:
            self.on_export()
        elif key == Qt.Key.Key_L:
            self.resume_live()
        elif key == Qt.Key.Key_Escape:
            if self.report.isVisible():
                self.toggle_report()
            elif self._reviewing is not None:
                self.resume_live()
            else:
                self.close()
        else:
            super().keyPressEvent(event)

    def _sweep_numbers(self) -> dict:
        """False-confident rates from the benchmark sweep, if it has been run."""
        if getattr(self, "_sweep_cache", None) is not None:
            return self._sweep_cache
        out: dict = {}
        try:
            import json
            from benchmark.sweep import CACHE, false_confident, score_rows
            from chronos.integrity import IntegrityConfig
            with open(CACHE) as fh:
                rows = json.load(fh)
            fc = false_confident(score_rows(rows), IntegrityConfig())
            out = {"fc_baseline": f"{100 * max(fc['baseline']):.1f}% worst case",
                   "fc_chronos": f"{100 * max(fc['chronos']):.1f}% worst case",
                   "fc_abstain": f"up to {100 * max(fc['abstain']):.0f}%"}
        except Exception:
            out = {}
        self._sweep_cache = out
        return out

    def closeEvent(self, event) -> None:  # noqa: N802
        self.engine.stop()
        self.engine.wait(1500)
        super().closeEvent(event)


def run(video: Optional[str] = None, path: str = "violation",
        kind: str = "rubber", frames: int = 70, detector: str = "auto",
        theme: str = "instrument") -> int:
    T.set_theme(theme)
    app = QApplication(sys.argv[:1])
    # bundled faces must be registered before the first widget is built, or
    # Qt measures the fallback and every fixed width in the layout is wrong
    print(T.load_fonts())
    app.setStyleSheet(T.stylesheet())
    try:
        source = FrameSource(video, path, frames, detector)
    except Exception as exc:
        print(f"cannot start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    win = Console(source, kind)
    win.show()
    return app.exec()

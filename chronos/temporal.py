"""CHRONOS Module -- the temporal layer.  This is the spatiotemporal core.

Track limits are not a single-frame question.  A frame can only ever say where
the tyres appear to be right now; the RULE is about what happened over a
stretch of time, to one identified car, across four wheels at once.  Everything
here exists because of that.

--------------------------------------------------------------------------
Three-valued wheel state -- the part most systems get wrong
--------------------------------------------------------------------------
A wheel is not in or out.  It is one of three things:

    INSIDE    margin > 0                       within the limit
    ON_LINE   -tyre_width < margin <= 0        touching the line -- STILL LEGAL
    OUTSIDE   margin <= -tyre_width            fully beyond the limit

The white line is part of the track.  A car touching it has not left the
track.  So a violation requires **all four wheels OUTSIDE**; if any single
wheel is ON_LINE, there is no violation, however far the other three are out.
Collapsing this to a binary in/out is the standard mistake and it produces
confident false positives on exactly the marginal cases that get appealed.

--------------------------------------------------------------------------
Why an event object and not a frame label
--------------------------------------------------------------------------
A per-frame label cannot express "340 ms", cannot survive a two-frame
detection gap, and cannot be reviewed.  So the output is an
:class:`ExcursionEvent`: one object per excursion, carrying its duration, its
peak margin, the full per-wheel state series, the boundary integrity that held
while it happened, and whether the car's identity survived the whole thing.

Contact points are Kalman-smoothed before any of this, and entering and
leaving a violation are both hysteretic, so a single-frame spike can never
fire.  That is not a nicety: at 50 fps a one-frame glitch is 20 ms, and a
system that reports it has just accused a driver on the strength of a
compression artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import numpy as np

WHEELS = ("FL", "FR", "RL", "RR")


class WheelState(str, Enum):
    INSIDE = "INSIDE"
    ON_LINE = "ON_LINE"
    OUTSIDE = "OUTSIDE"


@dataclass
class TemporalConfig:
    """Tunables for smoothing, state classification and hysteresis."""

    tyre_width_mm: float = 380.0      # a wheel is OUTSIDE once it is this far past
    fps: float = 50.0

    excursion_min_wheels: int = 3     # wheels OUTSIDE that open an excursion.
                                      # Opening only on all four would mean the
                                      # near-miss -- three wheels out, one on
                                      # the line -- is never recorded at all, and
                                      # that case is precisely the one a steward
                                      # needs to see reported as NOT a violation.
    enter_frames: int = 3             # consecutive qualifying frames to open
    exit_frames: int = 3              # consecutive non-qualifying frames to close
    max_gap_frames: int = 15           # a longer detection gap ends the event.
                                      # Short gaps keep the event open on
                                      # purpose: losing a car mid-excursion is
                                      # what track_continuity exists to report,
                                      # not a reason to forget the excursion.
    margin_limit_mm: float = 2500.0   # must match TrackGeometry.max_margin_mm

    kalman_process: float = 4.0
    kalman_measure: float = 9.0

    history_frames: int = 600         # kept per car, for the timeline display


@dataclass
class WheelSample:
    """One wheel, one frame."""

    frame_index: int
    t_ms: float
    margin_mm: float                  # after smoothing
    raw_margin_mm: float
    state: WheelState
    measured: bool = True             # False when the car was not detected


@dataclass
class ExcursionEvent:
    """One excursion by one car: the unit the decision engine reasons about."""

    car_id: int
    entry_frame: int
    exit_frame: int
    duration_ms: float
    peak_margin_mm: float             # most negative margin reached, any wheel
    wheels_out_max: int               # most wheels simultaneously OUTSIDE
    series: dict[str, list[WheelSample]] = field(default_factory=dict)
    mean_integrity: float = 100.0
    min_integrity: float = 100.0
    geometry_confidence: float = 1.0  # mean wheel-placement confidence
    track_continuity: float = 1.0     # 1.0 = the id survived the whole event
    frames_missing: int = 0
    n_frames: int = 0
    all_four_outside_frames: int = 0
    all_four_duration_ms: float = 0.0   # how long the RULE was actually met,
                                        # which is not the length of the event
    peak_saturated: bool = False      # the peak hit the limit of what the local
                                      # scale can defend; report it as "beyond",
                                      # never as a measurement
    corner: str = "T6"

    @property
    def is_violation_geometry(self) -> bool:
        """Did the geometry alone ever satisfy the rule?

        Deliberately separate from the verdict.  This says what the wheels
        did; whether that is worth acting on is :mod:`chronos.decide`'s call.
        """
        return self.all_four_outside_frames > 0

    def wheel_never_outside(self) -> list[str]:
        """Wheels that were never fully outside -- the reason a case is not one."""
        out = []
        for w in WHEELS:
            s = self.series.get(w, [])
            if s and not any(x.state is WheelState.OUTSIDE for x in s):
                out.append(w)
        return out

    def wheel_on_line_throughout(self) -> list[str]:
        """Wheels that touched the line at every frame the rule was otherwise met."""
        return [w for w in WHEELS
                if self.series.get(w) and
                all(x.state is WheelState.ON_LINE for x in self.series[w])]


# --------------------------------------------------------------------------


def classify(margin_mm: float, cfg: TemporalConfig) -> WheelState:
    """Three-valued wheel state from a signed margin.  See the module docstring."""
    if margin_mm > 0.0:
        return WheelState.INSIDE
    if margin_mm > -cfg.tyre_width_mm:
        return WheelState.ON_LINE
    return WheelState.OUTSIDE


class _PointFilter:
    """Constant-velocity Kalman filter over one wheel's contact point."""

    def __init__(self, p: np.ndarray, cfg: TemporalConfig):
        import cv2
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * cfg.kalman_process
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * cfg.kalman_measure
        self.kf.statePost = np.array([[p[0]], [p[1]], [0], [0]], np.float32)

    def step(self, p: Optional[np.ndarray]) -> np.ndarray:
        s = self.kf.predict()
        if p is not None:
            s = self.kf.correct(np.array([[p[0]], [p[1]]], np.float32))
        return np.array([float(s[0, 0]), float(s[1, 0])])


class CarTimeline:
    """The time series of one identified car's four wheels."""

    def __init__(self, car_id: int, cfg: TemporalConfig):
        self.car_id = car_id
        self.cfg = cfg
        self.filters: dict[str, _PointFilter] = {}
        self.history: dict[str, list[WheelSample]] = {w: [] for w in WHEELS}
        self.integrity: list[float] = []
        self.geom_conf: list[float] = []
        self.present: list[bool] = []
        self.frames: list[int] = []

        self._run_in = 0          # consecutive all-four-outside frames
        self._run_out = 0         # consecutive frames not all-outside
        self._open: Optional[int] = None      # index into history where it began
        self._gap = 0
        self.last_frame = -1

    # ------------------------------------------------------------------

    def update(self, frame_index: int, t_ms: float,
               points: Optional[dict[str, np.ndarray]],
               margin_fn, integrity: float, geometry_conf: float
               ) -> Optional[ExcursionEvent]:
        """Advance one frame.  Returns an event if one closed on this frame.

        ``points`` is None when the car was not detected in this frame; the
        filters coast on their prediction and the samples are flagged
        ``measured=False``, so a gap is visible in the record instead of being
        quietly interpolated away.
        """
        measured = points is not None
        self._gap = 0 if measured else self._gap + 1

        smoothed: dict[str, np.ndarray] = {}
        for w in WHEELS:
            p = None if points is None else np.asarray(points[w], float)
            if w not in self.filters:
                if p is None:
                    return self._maybe_close(frame_index, force=True)
                self.filters[w] = _PointFilter(p, self.cfg)
                smoothed[w] = p
            else:
                smoothed[w] = self.filters[w].step(p)

        raw = margin_fn(np.array([smoothed[w] for w in WHEELS]))
        states = {}
        for i, w in enumerate(WHEELS):
            st = classify(float(raw[i]), self.cfg)
            states[w] = st
            self.history[w].append(WheelSample(frame_index, t_ms, float(raw[i]),
                                               float(raw[i]), st, measured))
            if len(self.history[w]) > self.cfg.history_frames:
                self.history[w].pop(0)

        self.integrity.append(float(integrity))
        self.geom_conf.append(float(geometry_conf))
        self.present.append(measured)
        self.frames.append(frame_index)
        for lst in (self.integrity, self.geom_conf, self.present, self.frames):
            if len(lst) > self.cfg.history_frames:
                lst.pop(0)
        self.last_frame = frame_index

        n_outside = sum(1 for w in WHEELS if states[w] is WheelState.OUTSIDE)
        engaged = n_outside >= self.cfg.excursion_min_wheels
        if engaged:
            self._run_in += 1
            self._run_out = 0
        else:
            self._run_out += 1
            self._run_in = 0

        if self._open is None:
            if self._run_in >= self.cfg.enter_frames:
                # backdate to the first frame of the qualifying run, not the
                # frame that confirmed it
                self._open = max(len(self.frames) - self._run_in, 0)
            return None

        if self._gap > self.cfg.max_gap_frames:
            return self._maybe_close(frame_index, force=True)
        if self._run_out >= self.cfg.exit_frames:
            return self._maybe_close(frame_index)
        return None

    # ------------------------------------------------------------------

    def _maybe_close(self, frame_index: int, force: bool = False) -> Optional[ExcursionEvent]:
        if self._open is None:
            return None
        start = self._open
        end = len(self.frames) - (0 if force else self.cfg.exit_frames)
        end = max(end, start + 1)
        self._open = None
        self._run_in = self._run_out = 0
        return self._build(start, end)

    def _build(self, start: int, end: int) -> ExcursionEvent:
        series = {w: self.history[w][start:end] for w in WHEELS}
        n = max(end - start, 1)
        all_out = sum(1 for i in range(start, end)
                      if all(self.history[w][i].state is WheelState.OUTSIDE for w in WHEELS))
        dt = 1000.0 / max(self.cfg.fps, 1e-6)
        margins = [s.margin_mm for w in WHEELS for s in series[w]]
        present = self.present[start:end]
        missing = int(len(present) - sum(present))
        integ = self.integrity[start:end] or [100.0]
        geom = self.geom_conf[start:end] or [1.0]
        dt_ms = 1000.0 / max(self.cfg.fps, 1e-6)

        return ExcursionEvent(
            car_id=self.car_id,
            entry_frame=self.frames[start],
            exit_frame=self.frames[min(end, len(self.frames)) - 1],
            duration_ms=n * dt_ms,
            peak_margin_mm=float(min(margins)) if margins else 0.0,
            peak_saturated=bool(margins) and min(margins) <= -(self.cfg.margin_limit_mm - 1.0),
            wheels_out_max=max(
                (sum(1 for w in WHEELS if self.history[w][i].state is WheelState.OUTSIDE)
                 for i in range(start, end)), default=0),
            series=series,
            mean_integrity=float(np.mean(integ)),
            min_integrity=float(np.min(integ)),
            geometry_confidence=float(np.mean(geom)),
            track_continuity=float(sum(present) / max(len(present), 1)),
            frames_missing=missing,
            n_frames=n,
            all_four_outside_frames=all_out,
            all_four_duration_ms=all_out * dt,
        )

    def close_open_event(self) -> Optional[ExcursionEvent]:
        """Flush an event still open at the end of a clip."""
        return self._maybe_close(self.last_frame, force=True)


class TemporalEngine:
    """Holds one :class:`CarTimeline` per tracked car id."""

    def __init__(self, cfg: Optional[TemporalConfig] = None):
        self.cfg = cfg or TemporalConfig()
        self.timelines: dict[int, CarTimeline] = {}
        self.frames_seen = 0

    def update(self, frame_index: int, t_ms: float, cars: Iterable,
               margin_fn, integrity: float) -> list[ExcursionEvent]:
        """Advance every known car by one frame and return any closed events."""
        self.frames_seen += 1
        seen = set()
        events: list[ExcursionEvent] = []

        for car in cars:
            seen.add(car.car_id)
            tl = self.timelines.setdefault(
                car.car_id, CarTimeline(car.car_id, self.cfg))
            pts = car.wheels.points if car.wheels is not None else None
            conf = car.wheels.confidence if car.wheels is not None else 0.0
            ev = tl.update(frame_index, t_ms, pts, margin_fn, integrity, conf)
            if ev is not None:
                events.append(ev)

        # cars that vanished this frame still advance, so a detection gap shows
        # up as a gap rather than as a pause in time
        for cid, tl in self.timelines.items():
            if cid in seen or not tl.filters:
                continue
            ev = tl.update(frame_index, t_ms, None, margin_fn, integrity, 0.0)
            if ev is not None:
                events.append(ev)
        return events

    def flush(self) -> list[ExcursionEvent]:
        """Close anything still open.  Call once at the end of a clip."""
        out = []
        for tl in self.timelines.values():
            ev = tl.close_open_event()
            if ev is not None:
                out.append(ev)
        return out

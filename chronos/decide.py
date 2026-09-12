"""CHRONOS Module 5 -- the reference-aware decision engine.

    trust = wheel_geometry_conf  x  track_continuity_conf  x  (integrity / 100)

Three outcomes: CLEAR, VIOLATION, REVIEW REQUIRED.

--------------------------------------------------------------------------
This engine never issues a penalty
--------------------------------------------------------------------------
There is no penalty field, no severity, no points.  It is assistive by
construction: it tells a human what the geometry did, how far it trusts its
own reference, and whether that is enough to stand behind.  The decision to
sanction a driver belongs to a steward, and a system that pretends otherwise
is claiming an authority it cannot support from a camera.

The gating is the contribution.  A car can be measured perfectly and still
produce REVIEW REQUIRED, because the line it was measured against had stopped
being readable.  Abstaining is the feature:

    Car geometry confidence:   96%
    Boundary integrity:        42
    ----------------------------------
    Verdict:  REVIEW REQUIRED
    Reason:   Reference degraded -- T6 contamination 28%

Every verdict carries a one-line reason in plain English, because a verdict
nobody can read is a verdict nobody can contest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from chronos.temporal import WHEELS, ExcursionEvent, WheelState

LONG_NAME = {"FL": "Front-left", "FR": "Front-right",
             "RL": "Rear-left", "RR": "Rear-right"}


class Outcome(str, Enum):
    CLEAR = "CLEAR"
    VIOLATION = "VIOLATION"
    REVIEW_REQUIRED = "REVIEW REQUIRED"


@dataclass
class DecideConfig:
    """Thresholds for turning measurements into a verdict."""

    trust_threshold: float = 0.60     # below this, no verdict is issued
    integrity_floor: float = 60.0     # boundary integrity below this always reviews
    min_continuity: float = 0.70      # id must survive this much of the event
    min_geometry_conf: float = 0.60
    min_duration_ms: float = 100.0    # shorter than this is not an excursion
    corner: str = "T6"


@dataclass
class Verdict:
    """One reviewable decision about one excursion."""

    outcome: Outcome
    trust: float
    reason: str
    event: ExcursionEvent
    components: dict = field(default_factory=dict)

    def as_line(self) -> str:
        return f"{self.outcome.value:<16} trust {self.trust:5.0%}  {self.reason}"


@dataclass
class SessionStats:
    """What the session did.  Volume matters: this is a per-corner workload."""

    frames_processed: int = 0
    events_found: int = 0
    auto_cleared: int = 0
    violations: int = 0
    escalated: int = 0
    wall_time_s: float = 0.0
    fps_source: float = 50.0
    peak_fps: float = 0.0
    """Fastest single frame seen, as frames per second.

    The mean is dragged down by whatever else the console is doing; the peak is
    what one frame of this pipeline actually costs.  Both are reported, because
    quoting only one of them is how a throughput claim becomes a lie.
    """

    @property
    def throughput_fps(self) -> float:
        return self.frames_processed / max(self.wall_time_s, 1e-9)

    @property
    def race_seconds_covered(self) -> float:
        return self.frames_processed / max(self.fps_source, 1e-9)

    def summary(self) -> str:
        return (f"{self.frames_processed} frames "
                f"({self.race_seconds_covered:.1f} s of race) at "
                f"{self.throughput_fps:.0f} fps mean, {self.peak_fps:.0f} peak | "
                f"{self.events_found} excursions: {self.violations} violation, "
                f"{self.auto_cleared} auto-cleared, {self.escalated} escalated to review")


# --------------------------------------------------------------------------


def _trust(event: ExcursionEvent) -> tuple[float, dict]:
    """trust = wheel geometry x track continuity x boundary integrity."""
    geom = float(max(0.0, min(1.0, event.geometry_confidence)))
    cont = float(max(0.0, min(1.0, event.track_continuity)))
    integ = float(max(0.0, min(1.0, event.mean_integrity / 100.0)))
    return geom * cont * integ, {"wheel_geometry": geom, "track_continuity": cont,
                                 "boundary_integrity": integ}


def _reason_not_a_violation(event: ExcursionEvent, cfg: DecideConfig) -> Optional[str]:
    """Why the geometry does not meet the rule, in one line -- or None."""
    if event.all_four_outside_frames == 0:
        on_line = [w for w in WHEELS if any(
            s.state is WheelState.ON_LINE for s in event.series.get(w, []))]
        never = event.wheel_never_outside()
        if on_line:
            w = on_line[0]
            return (f"{LONG_NAME[w]} on the line throughout -- not a violation. "
                    f"Touching the line is within the limit.")
        if never:
            names = ", ".join(LONG_NAME[w].lower() for w in never[:2])
            return f"{names} never fully outside -- not a violation."
        return "All four tyres were never outside together -- not a violation."
    if event.all_four_duration_ms < cfg.min_duration_ms:
        return (f"All four outside for only {event.all_four_duration_ms:.0f} ms, below "
                f"the {cfg.min_duration_ms:.0f} ms threshold -- not reported.")
    return None


def judge(event: ExcursionEvent, cfg: Optional[DecideConfig] = None) -> Verdict:
    """Turn one excursion into one reviewable verdict.  Never a penalty.

    Order matters: trust is checked BEFORE the geometry is allowed to convict.
    A system that decides first and reports its confidence afterwards has
    already made the claim.
    """
    cfg = cfg or DecideConfig()
    trust, parts = _trust(event)
    corner = event.corner or cfg.corner

    # --- can we stand behind any verdict at all? --------------------------
    if event.mean_integrity < cfg.integrity_floor:
        return Verdict(Outcome.REVIEW_REQUIRED, trust,
                       f"Boundary integrity {event.mean_integrity:.0f} during event "
                       f"at {corner}. Cannot confirm. Review.", event, parts)
    if event.track_continuity < cfg.min_continuity:
        return Verdict(Outcome.REVIEW_REQUIRED, trust,
                       f"Lost car #{event.car_id} for {event.frames_missing} of "
                       f"{event.n_frames} frames. Identity not continuous. Review.",
                       event, parts)
    if event.geometry_confidence < cfg.min_geometry_conf:
        return Verdict(Outcome.REVIEW_REQUIRED, trust,
                       f"Wheel positions inferred rather than seen "
                       f"({event.geometry_confidence:.0%} confidence). Review.",
                       event, parts)
    if trust < cfg.trust_threshold:
        return Verdict(Outcome.REVIEW_REQUIRED, trust,
                       f"Trust {trust:.0%} below threshold "
                       f"(integrity {event.mean_integrity:.0f}, continuity "
                       f"{event.track_continuity:.0%}). Cannot confirm. Review.",
                       event, parts)

    # --- trusted: now the geometry may speak ------------------------------
    not_violation = _reason_not_a_violation(event, cfg)
    if not_violation is not None:
        return Verdict(Outcome.CLEAR, trust, not_violation, event, parts)

    peak = (f"beyond the {abs(event.peak_margin_mm):.0f} mm measurement range"
            if event.peak_saturated else f"{abs(event.peak_margin_mm):.0f} mm")
    return Verdict(Outcome.VIOLATION, trust,
                   f"All four tyres outside {corner} for "
                   f"{event.all_four_duration_ms:.0f} ms, "
                   f"peak {peak}. Trust {trust:.0%}.", event, parts)


class DecisionEngine:
    """Judges events and keeps the session's books."""

    def __init__(self, cfg: Optional[DecideConfig] = None, fps: float = 50.0):
        self.cfg = cfg or DecideConfig()
        self.verdicts: list[Verdict] = []
        self._stats = SessionStats(fps_source=fps)
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def note_frames(self, n: int = 1, seconds: Optional[float] = None) -> None:
        self._stats.frames_processed += n
        if seconds and seconds > 0:
            self._stats.peak_fps = max(self._stats.peak_fps, n / seconds)

    def ingest(self, events) -> list[Verdict]:
        out = []
        for ev in events:
            v = judge(ev, self.cfg)
            self.verdicts.append(v)
            self._stats.events_found += 1
            if v.outcome is Outcome.VIOLATION:
                self._stats.violations += 1
            elif v.outcome is Outcome.CLEAR:
                self._stats.auto_cleared += 1
            else:
                self._stats.escalated += 1
            out.append(v)
        return out

    def stats(self) -> SessionStats:
        if self._t0 is not None:
            self._stats.wall_time_s = time.perf_counter() - self._t0
        return self._stats

    def evidence_packet(self, verdict: Verdict) -> dict:
        """The JSON half of an evidence packet: everything behind the verdict."""
        ev = verdict.event
        return {
            "verdict": verdict.outcome.value,
            "reason": verdict.reason,
            "trust": round(verdict.trust, 4),
            "trust_components": {k: round(v, 4) for k, v in verdict.components.items()},
            "car_id": ev.car_id,
            "corner": ev.corner or self.cfg.corner,
            "entry_frame": ev.entry_frame,
            "exit_frame": ev.exit_frame,
            "duration_ms": round(ev.duration_ms, 1),
            "all_four_outside_ms": round(ev.all_four_duration_ms, 1),
            "peak_margin_mm": round(ev.peak_margin_mm, 1),
            "wheels_out_max": ev.wheels_out_max,
            "all_four_outside_frames": ev.all_four_outside_frames,
            "frames_missing": ev.frames_missing,
            "track_continuity": round(ev.track_continuity, 4),
            "boundary_integrity_mean": round(ev.mean_integrity, 1),
            "boundary_integrity_min": round(ev.min_integrity, 1),
            "wheel_states": {w: [s.state.value for s in ev.series.get(w, [])]
                             for w in WHEELS},
            "wheel_margins_mm": {w: [round(s.margin_mm, 1) for s in ev.series.get(w, [])]
                                 for w in WHEELS},
        }

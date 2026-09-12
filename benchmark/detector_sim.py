"""A simulated car detector with controlled, repeatable noise.

Pretrained YOLO does not fire on the synthetic renders -- at low thresholds it
reports "tennis racket" and "stop sign".  The renders are flat-shaded boxes and
sit far outside its training distribution, and making them photoreal is a
different project.  So the car-detection PATH is exercised here instead, with
a detector whose failures are dialled rather than hoped for:

    box jitter     the detector is never pixel-exact
    dropouts       frames where the car is missed entirely
    low confidence frames that fall to ByteTrack's second association pass
    identity stress a burst of consecutive misses, to see whether one
                    excursion gets reported as two

This is deliberately harsher than a clean detector and tests exactly what the
temporal layer is for.  It substitutes for YOLO through the same
``detect(frame) -> (boxes, scores)`` interface, so the tracking,
contact-point, temporal and decision code under test is the same code that
runs on real footage.

What it does NOT test is whether YOLO can find a car.  That needs real
footage; see the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SimDetectorConfig:
    box_jitter_px: float = 2.5
    size_jitter: float = 0.03
    score_mean: float = 0.84
    score_jitter: float = 0.07
    dropout_prob: float = 0.05        # isolated missed frames
    low_conf_prob: float = 0.14       # drops into the second association pass
    occlusion_start: Optional[int] = None   # a run of consecutive misses
    occlusion_frames: int = 0
    seed: int = 11


class SimulatedDetector:
    """Replays known boxes with injected noise.  Advances one frame per call."""

    def __init__(self, boxes: Sequence[Optional[Sequence[float]]],
                 cfg: Optional[SimDetectorConfig] = None,
                 contacts: Optional[Sequence[dict]] = None,
                 contact_jitter_px: float = 1.5):
        self.boxes = list(boxes)
        self.cfg = cfg or SimDetectorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.i = 0
        self.dropped = 0
        self._contacts = list(contacts) if contacts is not None else None
        self._contact_jitter = contact_jitter_px
        self._last = 0

    @property
    def contact_points(self):
        """Exposed only when true contact patches are available.

        Lets the benchmark exercise the temporal and decision layers on correct
        wheel positions, so their behaviour is not confounded with the
        box-to-contact error budget measured in :mod:`chronos.car`.  Jitter is
        still injected -- a perfect input would not test the Kalman smoothing.
        """
        if self._contacts is None:
            return None

        def lookup(_box):
            i = min(self._last, len(self._contacts) - 1)
            rec = self._contacts[i]
            if rec is None:
                return {}
            return {k: np.asarray(v, float)
                    + self.rng.normal(0, self._contact_jitter, 2)
                    for k, v in rec.items()}
        return lookup

    def seek(self, i: int) -> None:
        self.i = int(i)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        i, self.i = self.i, self.i + 1
        self._last = i
        if i >= len(self.boxes) or self.boxes[i] is None:
            return np.zeros((0, 4)), np.zeros((0,))

        if cfg.occlusion_start is not None and \
                cfg.occlusion_start <= i < cfg.occlusion_start + cfg.occlusion_frames:
            self.dropped += 1
            return np.zeros((0, 4)), np.zeros((0,))
        if self.rng.random() < cfg.dropout_prob:
            self.dropped += 1
            return np.zeros((0, 4)), np.zeros((0,))

        x1, y1, x2, y2 = (float(v) for v in self.boxes[i])
        w, h = x2 - x1, y2 - y1
        jx, jy = self.rng.normal(0, cfg.box_jitter_px, 2)
        sw = w * (1 + self.rng.normal(0, cfg.size_jitter))
        sh = h * (1 + self.rng.normal(0, cfg.size_jitter))
        cx, cy = 0.5 * (x1 + x2) + jx, 0.5 * (y1 + y2) + jy
        box = np.array([[cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2]])

        score = float(np.clip(self.rng.normal(cfg.score_mean, cfg.score_jitter), 0.05, 0.99))
        if self.rng.random() < cfg.low_conf_prob:
            score = float(self.rng.uniform(0.26, 0.52))
        return box, np.array([score])

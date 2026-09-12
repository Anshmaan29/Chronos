"""Camera-motion compensation: hold the corner still while the camera moves.

The integrity score is defined relative to a session baseline -- THIS corner,
measured while known-good, from THIS viewpoint.  Taken literally that requires
a camera on a tripod, and almost no real footage is: broadcast is a pan, a
replay is a chase cam, a drone orbits.  On that footage the baseline geometry
describes a piece of track that has moved in frame, and the engine reports the
mismatch as degradation -- a clean line scoring 3 because the camera turned.

Refusing on those clips is honest but useless, and loosening the refusal would
be dishonest.  So the assumption gets removed instead of relaxed.

The corner does not move; the camera does.  Estimate the homography from the
current frame back to the baseline frame, warp the pixels into the baseline's
own view, and every station, paint width and normal the baseline measured is
valid again -- unchanged, not re-derived.  The refusal then fires only when
registration genuinely fails (a cut, a new corner, motion blur with nothing to
match), which is a real loss of reference rather than an artefact of panning.

Two details that matter:

  * **Cars are masked out of the match.** Features on a moving car describe the
    car's motion, not the camera's, and a homography fitted to them drags the
    whole reference along with the car.
  * **The transform is checked, not trusted.** A homography with too few
    inliers, or one that folds or mirrors the frame, is rejected and the frame
    is reported unmeasured rather than measured wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class StabiliseConfig:
    """Tunables for frame-to-baseline registration."""

    max_features: int = 1500
    ratio: float = 0.78            # Lowe ratio for match filtering
    min_matches: int = 20
    min_inliers: int = 18
    ransac_px: float = 4.0
    car_dilate_px: int = 25        # margin around a car before masking it out
    work_max_side: int = 960       # match at this size; scale the result back

    # sanity limits on the recovered transform
    max_scale: float = 3.0
    min_scale: float = 0.33
    min_area_ratio: float = 0.45   # baseline frame -> current, as a quad
    max_area_ratio: float = 2.2
    max_aspect_change: float = 2.0 # a folded warp keeps its area but not its
                                   # proportions; check both

    # chained tracking
    chain: bool = True             # register to the PREVIOUS frame and compose
    reanchor_inliers: int = 120    # re-anchor to the baseline when this solid
    chain_min_inliers: int = 60    # composing a weak pairwise fit compounds it
    max_chain_steps: int = 40      # composed error grows; stop trusting it

    # A homography is exact for a PLANE or a pure rotation.  A camera driving
    # down a track is doing neither -- it translates through a 3-D scene, and
    # grandstands, barriers and trackside furniture at different depths cannot
    # all be satisfied by one homography.  Fitting to them produces the folded,
    # smeared warps this flag exists to prevent.  The track surface, though, IS
    # a plane, so features are restricted to it and the homography becomes the
    # right model rather than an approximation that collapses.
    plane_only: bool = False
    """Restrict features to the track surface.

    MEASURED, 2026-09-13: correct in theory and unusable in practice on this
    footage.  Asphalt is nearly featureless -- restricting ORB to the road mask
    left 42 inliers on the baseline frame and 0-15 matches on later frames, so
    registration failed on 10 of 11 probes.  Off by default; the full frame at
    least has trackside structure to match on, and the plausibility checks
    below catch the folded warps that 3-D parallax then produces.
    """

    plane_top_frac: float = 0.45   # ignore everything above this (sky, stands)


@dataclass
class Registration:
    """One frame's transform back to the baseline view."""

    H: Optional[np.ndarray]        # current -> baseline
    inliers: int
    matches: int
    ok: bool
    reason: str = ""

    @property
    def H_inv(self) -> Optional[np.ndarray]:
        """baseline -> current, for drawing baseline geometry on the frame."""
        if self.H is None:
            return None
        try:
            return np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            return None


def _feature_mask(shape, boxes, cfg: StabiliseConfig,
                  plane: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Where features may be taken from: the road plane, minus the cars."""
    h, w = shape
    if not boxes and not cfg.plane_only and plane is None:
        return None
    m = np.full((h, w), 255, np.uint8)
    if plane is not None and plane.any():
        m = cv2.bitwise_and(m, cv2.resize(plane, (w, h),
                                          interpolation=cv2.INTER_NEAREST))
    elif cfg.plane_only:
        m[:int(h * cfg.plane_top_frac)] = 0
    if not boxes:
        return m
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        p = cfg.car_dilate_px
        cv2.rectangle(m, (max(x1 - p, 0), max(y1 - p, 0)),
                      (min(x2 + p, w), min(y2 + p, h)), 0, -1)
    return m


def _plausible(H: np.ndarray, shape, cfg: StabiliseConfig) -> tuple[bool, str]:
    """Reject transforms that are not a camera looking at the same scene.

    Tested on the frame corners rather than on matrix entries: a homography is
    only meaningful through what it does to the image, and "the four corners
    still form a sane convex quadrilateral" is the property that actually
    matters.  Thresholding H[2, :2] directly rejects ordinary perspective.
    """
    if H is None or not np.isfinite(H).all():
        return False, "transform is not finite"
    h, w = shape
    quad = cv2.perspectiveTransform(
        np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H).reshape(-1, 2)
    if not np.isfinite(quad).all():
        return False, "frame corners do not map anywhere finite"
    area = float(cv2.contourArea(quad.astype(np.float32)))
    if area <= 0:
        return False, "transform mirrors or collapses the frame"
    ratio = area / float(w * h)
    if not (cfg.min_area_ratio <= ratio <= cfg.max_area_ratio):
        return False, f"transform rescales the frame {ratio:.2f}x by area"
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False, "transform folds the frame"
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    if min(sides) <= 1e-6:
        return False, "transform collapses an edge"
    if max(sides) / min(sides) > cfg.max_aspect_change * max(w, h) / min(w, h):
        return False, "transform stretches the frame out of proportion"
    return True, ""


class Stabiliser:
    """Registers frames back to one baseline view."""

    def __init__(self, baseline: np.ndarray, cfg: Optional[StabiliseConfig] = None,
                 baseline_boxes: Optional[Sequence[Sequence[float]]] = None,
                 plane_mask: Optional[np.ndarray] = None):
        self.cfg = cfg or StabiliseConfig()
        self._plane = None
        if plane_mask is not None and plane_mask.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            self._plane = cv2.dilate(plane_mask, k)
        self.shape = baseline.shape[:2]
        self._orb = cv2.ORB_create(nfeatures=self.cfg.max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.scale = min(1.0, self.cfg.work_max_side / max(self.shape))
        self._kp_b, self._des_b = self._features(baseline, baseline_boxes)
        self._prev = None            # (kp, des) of the last registered frame
        self._prev_H = np.eye(3)     # that frame -> baseline
        self._chain_steps = 0        # how far we are from a direct anchor
        if self._des_b is None or len(self._kp_b) < self.cfg.min_matches:
            raise ValueError(
                f"Stabiliser: the baseline frame has only "
                f"{0 if self._des_b is None else len(self._kp_b)} trackable features; "
                "there is nothing to register later frames against")

    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.scale >= 1.0:
            return frame
        return cv2.resize(frame, (int(frame.shape[1] * self.scale),
                                  int(frame.shape[0] * self.scale)),
                          interpolation=cv2.INTER_AREA)

    def _features(self, frame: np.ndarray, boxes):
        small = self._resize(frame)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scaled = None
        if boxes is not None and len(boxes):
            scaled = [[v * self.scale for v in b[:4]] for b in boxes]
        mask = _feature_mask(gray.shape, scaled, self.cfg, self._plane)
        return self._orb.detectAndCompute(gray, mask)

    def register(self, frame: np.ndarray,
                 car_boxes: Optional[Sequence[Sequence[float]]] = None) -> Registration:
        """Homography taking ``frame`` back into the baseline's view.

        Tries the baseline directly first -- that is drift-free.  When the
        camera has travelled far enough that too little of the baseline is
        still in shot, it falls back to registering against the PREVIOUS frame
        and composing, which is how a pan or an orbit stays trackable at all.
        Consecutive frames always overlap; frame 1 and frame 45 need not.
        """
        kp, des = self._features(frame, car_boxes)
        if des is None or len(kp) < self.cfg.min_matches:
            return Registration(None, 0, 0, False,
                                "too few trackable features in this frame")

        direct = self._solve(kp, des, self._kp_b, self._des_b)
        if direct.ok and direct.inliers >= self.cfg.reanchor_inliers:
            self._prev, self._prev_H = (kp, des), direct.H
            self._chain_steps = 0
            return direct

        if (self.cfg.chain and self._prev is not None
                and self._chain_steps < self.cfg.max_chain_steps):
            rel = self._solve(kp, des, self._prev[0], self._prev[1])
            # a weak pairwise fit is compounded by every later composition, so
            # it is refused here rather than allowed to poison the chain
            if rel.ok and rel.inliers >= self.cfg.chain_min_inliers:
                H = self._prev_H @ rel.H
                ok, why = _plausible(H, self.shape, self.cfg)
                if ok:
                    self._prev, self._prev_H = (kp, des), H
                    self._chain_steps += 1
                    return Registration(H, rel.inliers, rel.matches, True,
                                        f"chained on {rel.inliers} inliers "
                                        f"({self._chain_steps} from anchor)")

        if direct.ok:
            self._prev, self._prev_H = (kp, des), direct.H
            self._chain_steps = 0
            return direct
        return Registration(None, direct.inliers, direct.matches, False,
                            direct.reason if not direct.ok else
                            "no transform survived the plausibility checks")

    def _solve(self, kp, des, kp_ref, des_ref) -> Registration:
        if des_ref is None or len(kp_ref) < self.cfg.min_matches:
            return Registration(None, 0, 0, False, "reference has too few features")
        pairs = self._bf.knnMatch(des, des_ref, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2)
                if m.distance < self.cfg.ratio * n.distance]
        if len(good) < self.cfg.min_matches:
            return Registration(None, 0, len(good), False,
                                f"only {len(good)} feature matches "
                                "(a cut, or a different corner)")

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC,
                                    self.cfg.ransac_px * self.scale)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < self.cfg.min_inliers:
            return Registration(None, n_in, len(good), False,
                                f"registration unstable ({n_in} inliers of "
                                f"{len(good)} matches)")

        if self.scale < 1.0:
            # lift the transform from working resolution back to full frame
            S = np.diag([self.scale, self.scale, 1.0])
            H = np.linalg.inv(S) @ H @ S

        ok, why = _plausible(H, self.shape, self.cfg)
        if not ok:
            return Registration(None, n_in, len(good), False, why)
        return Registration(H, n_in, len(good), True,
                            f"registered on {n_in} inliers")

    # ------------------------------------------------------------------

    def to_baseline(self, frame: np.ndarray, reg: Registration) -> np.ndarray:
        """Warp a frame into the baseline's view so baseline stations apply."""
        if not reg.ok or reg.H is None:
            return frame
        h, w = self.shape
        return cv2.warpPerspective(frame, reg.H, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def map_points(pts: np.ndarray, H: Optional[np.ndarray]) -> np.ndarray:
        """Apply a homography to (N, 2) points."""
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        if H is None:
            return p.reshape(-1, 2)
        return cv2.perspectiveTransform(p, H.astype(np.float64)).reshape(-1, 2)

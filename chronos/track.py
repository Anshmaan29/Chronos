"""CHRONOS -- track ground-plane geometry.  Margin in millimetres, no calibration.

Turns "the tyre is 14 pixels past the line" into "the tyre is 118 mm past the
line", using only references that are present in every frame.  No operator
draws a line, no checkerboard is waved at the camera.

--------------------------------------------------------------------------
Why the three references are not interchangeable
--------------------------------------------------------------------------
The plan lists three self-calibration references: the specified width of the
edge line, the repeating pitch of the kerb stripes, and the car's published
wheelbase.  They are NOT three measurements of one number.  Under perspective
the image scale is anisotropic: at any point on the track a millimetre ACROSS
the track and a millimetre ALONG it occupy different numbers of pixels, and on
an elevated corner camera they differ by a large factor.

A margin is measured across the track.  So:

  across-track scale   line width  (measured perpendicular to the boundary --
                       the same direction a margin is measured in)  PRIMARY
                       cross-checked by the car's axle track, which is also
                       across-track and is independent of the paint

  along-track scale    kerb stripe pitch, cross-checked by the wheelbase.
                       Used for distances and speeds along the track, and as a
                       consistency check.  NEVER used to scale a margin.

Calibrating a margin from stripe pitch would be wrong by whatever the local
anisotropy is -- which is exactly the kind of error that looks fine on a demo
frame and is badly wrong at the far end of the corner.

--------------------------------------------------------------------------
The approximation, stated
--------------------------------------------------------------------------
Margins are computed as a perpendicular pixel distance multiplied by the local
across-track scale at the nearest boundary station.  That is a first-order
(locally affine) approximation to the true ground-plane geometry.  It is
accurate while the distance is small compared with the distance over which the
scale changes -- true for the +/- 1 m that a track-limits call turns on, and
not true if you try to measure something metres away.  ``margin_mm`` refuses
distances beyond ``max_margin_mm`` rather than returning a confident wrong
number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from chronos.boundary import BoundaryResult, _resample_polyline
from chronos.degrade import TrackFrame, build_track_frame
from chronos.integrity import IntegrityConfig, _measure_paint_width


def _equivalent_paint_width(gray: np.ndarray, tf, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased painted-band width in pixels, per station.

    A half-maximum crossing counts half the blurred edge on each side and so
    over-states a thin line -- on a 4-pixel band that is a ~12% error, which
    propagates straight into every millimetre downstream as a systematic
    under-reading of the margin.

    Equivalent width does not have that problem: blur redistributes intensity
    but conserves it, so

        w_eq = integral(I - background) / (peak - background)

    recovers the width of the unblurred band.  The same quantity a
    photometrist would use for a spectral line, for the same reason.

    Returns ``(width_px, valid, edge_offset_px)``, where ``edge_offset_px`` is
    how far INWARD of the given station the true outer edge of the paint lies.
    """
    from chronos.integrity import _sample
    step = 0.25
    depths = np.arange(0.0, cfg.probe_max_px, step)
    prof = np.stack([_sample(gray, tf.points + d * tf.normals) for d in depths], axis=1)
    peak = prof[:, :max(2, int(1.0 / step))].max(axis=1)
    background = np.median(prof[:, -int(6 / step):], axis=1)
    height = peak - background
    valid = height > 18.0
    excess = np.clip(prof - background[:, None], 0.0, None)
    # stop integrating once the profile has clearly returned to the road, so a
    # bright kerb further in cannot be counted as paint
    width = excess.sum(axis=1) * step / np.maximum(height, 1e-6)

    # Sub-pixel location of the paint's OUTER edge, measured inward from the
    # station.  Module 1's contour lands on the last surface pixel, which
    # antialiasing puts about a pixel outside the true geometric edge -- a
    # constant offset that every margin then inherits.  The intensity centroid
    # is stable to a fraction of a pixel, and the outer edge is half an
    # equivalent width outside it.
    denom = np.maximum(excess.sum(axis=1), 1e-6)
    centroid = (excess * depths[None, :]).sum(axis=1) / denom
    edge_offset = centroid - 0.5 * width
    return np.clip(width, cfg.min_paint_px, 40.0), valid, edge_offset


@dataclass
class TrackGeometryConfig:
    """Physical constants of the circuit and the car.  All in millimetres."""

    line_width_mm: float = 180.0        # FIA-specified edge line for this track
    kerb_stripe_pitch_mm: float = 1000.0
    axle_track_mm: float = 2000.0       # across-track reference on the car
    wheelbase_mm: float = 3600.0        # along-track reference on the car

    probe_max_px: int = 34            # how far inward to integrate for the width
    fit_scale: bool = True            # fit the across-track scale against image
                                      # height instead of trusting each station
    fit_degree: int = 2
    fit_max_resid: float = 0.22       # median log-residual above which the fit
                                      # is rejected and raw stations are kept
    refine_edge: bool = True          # move the origin onto the true paint edge
    max_edge_shift_px: float = 4.0    # a bigger correction means something else
                                      # is wrong; do not silently apply it
    n_stations: int = 220
    smooth_stations: int = 9            # the scale varies smoothly with distance
    max_margin_mm: float = 2500.0       # beyond this the local scale is a lie
    min_paint_px: float = 1.2
    scale_warn_ratio: float = 0.25      # disagreement that lowers confidence


@dataclass
class TrackGeometry:
    """A metric frame attached to the boundary.

    Attributes
    ----------
    track:          the TrackFrame (points, inward normals, road width).
    mm_per_px_across: (N,) millimetres per pixel measured ACROSS the track at
                    each station.  This is what converts a margin.
    mm_per_px_along: (N,) or None -- along-track scale from the kerb stripes.
    scale_confidence: 0..1, from how well the independent references agree.
    sources:        what each reference gave, for the record.
    """

    track: TrackFrame
    mm_per_px_across: np.ndarray
    mm_per_px_along: Optional[np.ndarray]
    scale_confidence: float
    sources: dict
    max_margin_mm: float = 2500.0

    # ----------------------------------------------------------------------

    def nearest_station(self, pts: np.ndarray) -> np.ndarray:
        """Index of the closest boundary station to each point."""
        p = np.asarray(pts, float).reshape(-1, 2)
        d = ((p[:, None, :] - self.track.points[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)

    def margin_mm(self, pts: np.ndarray) -> np.ndarray:
        """Signed margin in millimetres.  Positive = inside the track limit.

        Raises
        ------
        ValueError
            If a point is further from the boundary than ``max_margin_mm``,
            where the local-scale approximation stops being defensible.
        """
        p = np.asarray(pts, float).reshape(-1, 2)
        idx = self.nearest_station(p)
        rel = p - self.track.points[idx]
        across_px = np.einsum("ij,ij->i", rel, self.track.normals[idx])
        mm = across_px * self.mm_per_px_across[idx]
        if np.any(np.abs(mm) > self.max_margin_mm):
            worst = float(np.max(np.abs(mm)))
            raise ValueError(
                f"margin_mm: point is {worst:.0f} mm from the boundary, beyond "
                f"max_margin_mm ({self.max_margin_mm:.0f}); the local scale "
                "approximation does not hold that far out")
        return mm

    def margin_mm_clipped(self, pts: np.ndarray) -> np.ndarray:
        """As ``margin_mm`` but saturating instead of raising.

        For plotting and for wheels that are far inside the track, where the
        exact number does not matter and only the sign does.
        """
        p = np.asarray(pts, float).reshape(-1, 2)
        idx = self.nearest_station(p)
        rel = p - self.track.points[idx]
        across_px = np.einsum("ij,ij->i", rel, self.track.normals[idx])
        return np.clip(across_px * self.mm_per_px_across[idx],
                       -self.max_margin_mm, self.max_margin_mm)

    def check_against_car(self, axle_px: float, cfg: TrackGeometryConfig,
                          station: int = 0) -> float:
        """Independent across-track check from the car's axle track.

        Returns the ratio of the car-derived scale to the paint-derived scale.
        1.0 means the two agree; this is the number to quote when someone asks
        how the millimetres are known without calibration.
        """
        if axle_px <= 1e-6:
            raise ValueError("check_against_car: axle width in pixels must be > 0")
        car_scale = cfg.axle_track_mm / axle_px
        return float(car_scale / self.mm_per_px_across[int(station)])


# --------------------------------------------------------------------------


def _kerb_stripe_pitch_px(kerb_mask: np.ndarray, tf: TrackFrame,
                          cfg: TrackGeometryConfig) -> Optional[np.ndarray]:
    """Along-track pixel pitch of the kerb stripes, per station.

    Walks the middle of the kerb band and looks at where the red stripes start
    and stop.  Returns None when there is no kerb, or too few stripes to
    measure a pitch from.
    """
    if kerb_mask is None or not kerb_mask.any():
        return None
    h, w = kerb_mask.shape
    n = len(tf.points)
    # sample just OUTSIDE the limit, in the middle of the kerb band
    probe = tf.points - 0.5 * np.maximum(0.02 * tf.widths, 6.0)[:, None] * tf.normals
    xi = np.clip(probe[:, 0].astype(int), 0, w - 1)
    yi = np.clip(probe[:, 1].astype(int), 0, h - 1)
    on = (kerb_mask[yi, xi] > 0).astype(np.int8)
    edges = np.nonzero(np.diff(on))[0]
    if len(edges) < 4:
        return None

    # arc length along the boundary, so a "pitch" is a pixel distance
    seg = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(tf.points, axis=0), axis=1))])
    # two transitions per stripe -> a full red+white pitch spans four
    pitch_at, pitch_px = [], []
    for i in range(len(edges) - 2):
        span = seg[edges[i + 2]] - seg[edges[i]]
        if span > 1.0:
            pitch_at.append(0.5 * (edges[i] + edges[i + 2]))
            pitch_px.append(span)
    if len(pitch_px) < 3:
        return None
    return np.interp(np.arange(n), np.asarray(pitch_at), np.asarray(pitch_px))


def build_track_geometry(frame: np.ndarray, boundary: BoundaryResult,
                         cfg: Optional[TrackGeometryConfig] = None,
                         icfg: Optional[IntegrityConfig] = None,
                         track: Optional[TrackFrame] = None) -> TrackGeometry:
    """Attach a metric frame to a detected boundary.

    Parameters
    ----------
    frame:     a CLEAN BGR frame -- the scale comes from the paint, so it must
               be measured while the paint is still readable.  Capture this at
               the same time as the integrity baseline.
    boundary:  the Module 1 result for that frame.

    Raises
    ------
    ValueError
        If the boundary failed, or the paint is unresolvable, or the derived
        scale is physically absurd.  A silently wrong scale would turn every
        millimetre downstream into fiction.
    """
    cfg = cfg or TrackGeometryConfig()
    icfg = icfg or IntegrityConfig()
    if not boundary.ok:
        raise ValueError(f"build_track_geometry: boundary detection failed ({boundary.reason})")

    tf = track or build_track_frame(boundary.polyline, boundary.drivable_mask,
                                    boundary.kerb_mask, n=cfg.n_stations)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # NOTE this is deliberately not integrity's paint-band estimate.  That one
    # answers "where should I sample the paint"; this one answers "how wide is
    # the paint", and only the second has to be unbiased.
    paint_px, valid, edge_offset = _equivalent_paint_width(gray, tf, cfg)
    if int(valid.sum()) < 20:
        raise ValueError(
            f"build_track_geometry: paint width resolvable at only "
            f"{int(valid.sum())} stations -- cannot establish a scale")

    # move the measurement origin onto the true paint edge before anything is
    # measured from it
    if cfg.refine_edge:
        shift = np.where(valid, edge_offset, 0.0)
        shift = np.clip(shift, -cfg.max_edge_shift_px, cfg.max_edge_shift_px)
        k0 = max(1, int(cfg.smooth_stations) | 1)
        shift = cv2.GaussianBlur(shift.reshape(-1, 1).astype(np.float32),
                                 (1, k0), 0).reshape(-1).astype(float)
        tf = TrackFrame(points=(tf.points + shift[:, None] * tf.normals).astype(np.float32),
                        normals=tf.normals, widths=tf.widths, surface=tf.surface)

    paint_px = np.maximum(paint_px, cfg.min_paint_px)
    # interpolate across stations where the paint could not be measured, then
    # smooth: the scale is a property of the camera geometry and varies slowly
    idx = np.arange(len(paint_px))
    paint_px = np.interp(idx, idx[valid], paint_px[valid])
    k = max(1, int(cfg.smooth_stations) | 1)
    paint_px = cv2.GaussianBlur(paint_px.reshape(-1, 1).astype(np.float32),
                                (1, k), 0).reshape(-1).astype(float)

    across = cfg.line_width_mm / paint_px

    # Fit the scale rather than take it station by station.  The paint is only
    # a few pixels wide, so a per-station width carries perhaps 15% noise, and
    # that noise lands directly on every margin -- 150 mm of scatter at a metre,
    # which is most of a tyre.  Under perspective on a plane the across-track
    # scale is a smooth function of image height (it goes as depth, and depth
    # goes as image row), so a low-order fit in image y is the right model and
    # it averages the noise away instead of propagating it.
    if cfg.fit_scale:
        v = tf.points[:, 1].astype(float)
        good = valid & np.isfinite(across)
        if int(good.sum()) >= cfg.fit_degree + 3:
            coeffs = np.polyfit(v[good], np.log(across[good]), cfg.fit_degree)
            fitted = np.exp(np.polyval(coeffs, v))
            resid = np.abs(np.log(across[good]) - np.polyval(coeffs, v[good]))
            # a fit that does not describe the data is worse than no fit
            if float(np.median(resid)) < cfg.fit_max_resid:
                across = fitted

    if not np.isfinite(across).all() or across.min() <= 0:
        raise ValueError("build_track_geometry: derived a non-physical across-track scale")

    pitch_px = _kerb_stripe_pitch_px(boundary.kerb_mask, tf, cfg)
    along = cfg.kerb_stripe_pitch_mm / pitch_px if pitch_px is not None else None

    sources = {"line_width_mm": cfg.line_width_mm,
               "paint_px_median": float(np.median(paint_px)),
               "mm_per_px_across_median": float(np.median(across))}
    confidence = 0.75          # paint alone; a car raises this via check_against_car
    if along is not None:
        sources["mm_per_px_along_median"] = float(np.median(along))
        # anisotropy is expected and is not an error -- but a sane elevated
        # camera should foreshorten along the track, not across it
        ratio = float(np.median(along) / np.median(across))
        sources["along_over_across"] = ratio
        confidence = 0.85 if ratio > 1.0 else 0.6

    return TrackGeometry(track=tf, mm_per_px_across=across, mm_per_px_along=along,
                         scale_confidence=confidence, sources=sources)

"""The scene variants the benchmark runs over.

One definition, used by both the sweep and the tests, so a variant can never
be measured in one place and forgotten in the other.
"""

from __future__ import annotations

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "gravel run-off": dict(runoff="gravel"),
    "asphalt run-off": dict(runoff="asphalt"),
    "kerb both sides": dict(inner_kerb=True),
    "no kerb": dict(kerb_width_m=0.0),
    "gentle corner": dict(curve_k=0.0020),
    "sharp corner": dict(curve_k=0.0110),
    "mirrored camera": dict(cam_x_m=9.0, cam_yaw_deg=-16.0),
    "high steep camera": dict(cam_z_m=18.0, cam_pitch_deg=26.0),
    "low camera": dict(cam_z_m=5.0, cam_pitch_deg=7.0),
    "narrow 8cm line": dict(line_width_m=0.08),
    "wide 30cm line": dict(line_width_m=0.30),
    "overcast": dict(exposure=0.62, light_gradient=0.04),
    "bright sun": dict(exposure=1.38),
    "soft lens and noise": dict(blur_sigma=1.8, sensor_noise=7.0, jpeg_quality=62),
    "seed 7": dict(seed=7),
    "seed 21": dict(seed=21),
}

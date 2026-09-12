# CHRONOS

**Reference Integrity for Track Limits** — TrackShift 2026.

> Every track-limits system in the world measures the car against the line.
> Not one of them measures the line. CHRONOS measures both.

Status: **working prototype.** Full pipeline — boundary, integrity,
degradation, geometry, cars, temporal, decision — plus the benchmark, the
wheel-state timeline and the PyQt6 console. 39/39 tests pass.

Runs on real footage. `chronos/lane.py` is the detector that made that
true; the degradation curve is now reproduced on a real photograph, not
only on synthetic renders. Where it still cannot measure, it says which
assumption failed — see **Real footage** below.

> Track limits is not a computer vision problem. It is a spatiotemporal one.
> A frame can only say where the tyres appear to be now. The rule is about
> what four wheels did, over time, to one identified car.

---

## Analyse any image or video (frame by frame)

Load real footage and CHRONOS analyses **every frame** — no session baseline
needed — and gives a verdict per car:

```bash
.venv/bin/python chronos.py                     # then LOAD (or drag) a video / image / folder
.venv/bin/python chronos.py --video data/real/clip.mp4
.venv/bin/python -m chronos.analyze data/real/clip.mp4     # headless, prints the full report
.venv/bin/python -m chronos.analyze data/real              # every image in a folder
.venv/bin/python tests/real_regression.py                  # 7 checks on the real test footage
```

For each frame: the track limit (outer edge of the white line, traced along the
asphalt/paint boundary), line integrity 0-100 with its four components, every car
found, each tyre classified INSIDE / ON LINE / OUTSIDE with a margin in mm, and a
verdict — **CLEAR, BORDERLINE (touching the line = legal), VIOLATION (all measured
tyres fully beyond), REVIEW REQUIRED** (line integrity < 50, or tyres on ground that
cannot be read). At the end it writes `output/<source>_<time>/` with
`report.json`, `frames.csv`, every flagged frame as PNG, and `summary.png`.

How it avoids the traps measured on real footage: asphalt is found by Lab chroma
(grey vs painted run-off, stable in shade); cars come from three physical cues
(near-black tyre mass, holes in the asphalt, YOLO of any class) because pretrained
YOLO labels F1 cars "suitcase"/"kite"; tyres are judged by distance to the visible
track surface, and a tyre in unreadable shade is UNKNOWN, never OUTSIDE.

Known limits: frames with another system's colour overlays burned in (cars tinted
orange/blue) usually report no car; millimetres are first-order (scale from the
car's 2000 mm width); validated by eye on 6 stills + one clip, not against ground truth.

## The console

```bash
.venv/bin/python chronos.py
```
```bash
.venv/bin/python chronos.py --video data/real/clip.mp4
```

One window. Left 60% is the corner with the boundary and the four contact
patches drawn on it; right is the integrity number, its four components, car
geometry, final trust and the verdict; below that the **contamination
slider**, the **evidence strip**, the wheel-state timeline, and the incident
log.

| key | does |
|---|---|
| `space` | pause / resume — or leave a reviewed frame |
| `O` | open a video, image or folder (drag-and-drop also works) |
| `I` | cycle integrity sampling 1:1 → 1:5 → 1:25 (the throughput lever) |
| `R` | the **analysis report**, full window |
| `E` | **export** report + evidence + packets to `./output/` |
| `L` | back to live from a reviewed frame |
| `esc` | close the report, leave review, or quit |

The console reacts exactly once: when boundary integrity falls below 50 the
verdict box turns amber and a rule appears across the top. Nothing else moves
on its own, so movement on screen means movement in the data.

Every stage is treated as optional — a module that returns `None` shows a dash
and the window keeps running. A console that dies because one frame was
unreadable is worse than useless at a race track.

### Footage is finite

A clip runs **once** and then ends. This is worth stating because it did not
used to: the video rewound to frame 0 on EOF and a still was served forever
through a modulo, so one photograph would report "111 frames · 27.8s of
race", and — far worse — the clip never ended, so `Pipeline.finish()` could
never be called and an excursion still open on the last frame was never
closed and never judged. A car could plainly run wide and the verdict box
would sit on "waiting for the first excursion" indefinitely.

At the end of a clip the run is **finalised**: open excursions are closed and
judged, the frame at each one's deepest margin lands in the evidence strip,
and the header states the outcome — `END OF CLIP · 60 frames · 1 excursion ·
1 violation · 0 clear · 0 review`. If nothing happened it says so
(`NO EXCURSION`) rather than leaving an empty box that reads like a hang.

**Stills are a separate mode.** A single frame has no time base, so it can
never show a violation — the rule is about what four wheels did *over time*.
A still is therefore scored for reference integrity only, and says so. Load a
folder and every image gets its own scored card in the strip, captioned with
the four components:

```
INTEGRITY 100   f0  c100 n100 s100 x100
INTEGRITY  73   f1  c70  n100 s48  x99
INTEGRITY  46   f2  c48  n100 s20  x72
```

### The evidence strip

A verdict is a claim about one moment, so the strip holds that moment: the
frame at the **deepest margin** of each excursion, with the boundary and the
four contact patches drawn on it, captioned with the margin, the duration and
the trust. Newest first.

**Click any card to review that frame.** The engine pauses and the pane holds
on it; the pipeline is deliberately *not* rewound. Re-running frames would
re-open excursions and re-issue verdicts that already happened, so the
incident log would grow every time somebody clicked a thumbnail. What changes
is what is being looked at, never what was decided.

Each card keeps its own full-resolution copy, so an incident from four minutes
ago is still reviewable after the engine's 240-frame ring buffer has rolled
past it.

### The analysis report — `R`

Everything the run measured, on one screen, headed by a provenance line:

```
Source: clip.mp4 | Detector: real | Boundary: ESTABLISHED |
Validated against: synthetic ground truth only
```

That line is not decoration. Three different kinds of number live in this
report and they are earned in completely different ways — session counters
are true of any footage; accuracy figures only mean anything where ground
truth exists, which is synthetic and nowhere else; the false-confident rate
comes from the benchmark sweep, because a rate over one excursion is not a
rate. The provenance line is what stops any of the three being quoted on
stage without the conditions that produced it.

`E` writes the whole thing to `./output/<source>_<timestamp>/` — `report.json`,
`report.txt`, one JPEG per evidence frame, an `evidence_packet` JSON per
verdict, and the timeline.

### NO REFERENCE

If the boundary cannot be established, the console **does not go black and
does not guess.** The footage keeps playing, cars are still detected — because
finding a car does not depend on knowing where the line is — and every single
readout is a dash:

- integrity, its four components, car geometry and trust all show `--`
- the verdict box reads **NO REFERENCE** with the detector's own reason
- the evidence strip still fills, with frames stamped
  `NO REFERENCE — NOTHING MEASURED ON THIS FRAME`, carrying **no margin and no
  integrity number**
- the report blanks every boundary-dependent row and turns the provenance bar
  amber
- the export **omits** those fields rather than writing them as null or zero —
  a null invites someone to read it as a missing measurement; an absent field
  cannot be

The contamination slider is disabled in this mode and says why: contamination
is laid along the track frame, and there is no track frame. Faking one would
put rubber in an arbitrary place and then score it, which is the exact
dishonesty this mode exists to avoid.

A console that went black on a failed boundary would look identical to a
console that crashed, and on stage those two must never be confusable.

### Themes and fonts

```bash
.venv/bin/python chronos.py --theme f1          # default: broadcast livery
.venv/bin/python chronos.py --theme instrument  # the original race-control look
.venv/bin/python chronos.py --theme brutal      # neo-brutalist race livery
```

`f1` is carbon black `#15151E`, brand red `#E10600`, Titillium — the face the
F1 wordmark and timing graphics are drawn from. Brand red and VIOLATION red
are deliberately the same value: red is the loudest thing the palette has, and
spending it twice, once on decoration and once on the alert, would blunt it.
So nothing else in that theme is allowed to be red, and the structural work is
done by the timing teal.

Every face is **bundled** in `assets/fonts/` under the SIL Open Font License
and registered with Qt at startup — `Impact` and `Arial Black` are not on
every machine, and a font fallback found live on a projector changes every
measured text width in the layout. See `assets/README.md`, which also explains
why `assets/logo.png` is absent and how to supply one.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy opencv-python scipy matplotlib
uv pip install --python .venv/bin/python PyQt6          # the console
uv pip install --python .venv/bin/python ultralytics    # car detection only
```

No internet is used at runtime. No training, no datasets. Every font the
console needs is bundled in `assets/fonts/`; nothing is fetched at launch.
Modules 1–3, the benchmark and the sweep need neither PyQt6 nor ultralytics.

## Run it

```bash
.venv/bin/python -m benchmark.generate --out data/frame.jpg
```
```bash
.venv/bin/python -m chronos.boundary --image data/frame.jpg --gt data/frame_gt.json
```
```bash
.venv/bin/python -m chronos.degrade --image data/frame.jpg --kind rubber
```
```bash
.venv/bin/python -m chronos.integrity --image data/frame.jpg --gt data/frame_gt.json --degrade rubber
```
```bash
.venv/bin/python -m benchmark.sweep
```
```bash
.venv/bin/python -m chronos.pipeline --path violation
```
```bash
.venv/bin/python -m chronos.pipeline --path one_wheel_on_line
```
```bash
.venv/bin/python -m chronos.pipeline --path violation --contamination 0.6
```
```bash
for t in boundary integrity temporal; do .venv/bin/python tests/test_$t.py; done
```

Every module writes a debug image to `./debug/`. Check that first, not the code.

---

## What exists

### `benchmark/generate.py` — synthetic scene generator

Renders an elevated corner-cam view and returns the **exact ground-truth
boundary polyline** beside the image. The scene is built in a flat world plane
in metres and projected through a pinhole camera, so the ground truth is the
same world curve that was rasterised, pushed through the same projection — it
is correct by construction, not by annotation.

Renders asphalt, a white edge line of configurable width, a red/white kerb with
a configurable stripe pitch, run-off (grass / gravel / asphalt), a far verge,
plus lens softness, sensor noise, vignette, a lighting ramp and JPEG artefacts.

This is the blind benchmark from plan section 5: the detector sees only pixels;
ground truth is used only for scoring.

```python
from benchmark.generate import SceneConfig, generate_scene
scene = generate_scene(SceneConfig(runoff="gravel", curve_k=0.008))
scene.image         # BGR uint8 — all the detector may see
scene.gt_boundary   # (N,2) the true track limit, in image pixels
```

### `chronos/boundary.py` — Module 1

Input a frame, output the **outer edge of the white line** as a polyline —
the exact surface the regulation names.

- **Channel A (primary).** Segments the drivable surface *including* the paint
  as one region and takes the outer edge of its contour. A thin-line detector
  dies under motion blur, spray and glare; a large region does not.
- **Channel B (fallback).** HSV white threshold → Canny → Hough, chained into a
  curve. Two estimates from different evidence; their disagreement is the
  confidence signal Module 5 will consume.
- **Kerbs are classified as their own region.** Under the rules a kerb is
  outside the track limit, and a kerb's white stripes are the same colour as
  the paint — so a mask that swallows the kerb puts the boundary in the wrong
  place. Red stripes are found by hue, then bridged across the white ones with
  oriented line kernels.

```python
from chronos.boundary import detect_boundary, BoundaryConfig
res = detect_boundary(frame, BoundaryConfig())
if not res:
    print(res.reason)        # never a silent None
else:
    res.polyline             # (N,2) the track limit
    res.agreement_px         # channel disagreement, in pixels
```

**Measured against ground truth**, over 17 scene variants (run-off material,
corner radius, camera pose, line width, exposure, lens softness, seeds):

| | |
|---|---|
| mean deviation | **0.3 – 1.4 px** |
| coverage of the true line | **67 – 98%** |
| channel agreement | 0.6 – 4.6 px, silent in 1 of 17 |

Deviation and coverage are reported separately on purpose. A boundary that is
exactly right over 80% of the line is a very different thing from one that is
20% wrong everywhere, and one symmetric number hides which.

### `chronos/degrade.py` — Module 3

Clean frame + level 0..1 + kind → contaminated frame. Six kinds: `rubber`,
`dust`, `wet`, `fade`, `glare`, `shadow`. `rubber` is the Miami case and the
one the demo is built on.

Two properties are tested rather than assumed:

- **Level 0.0 is bit-identical to the input**, so the clean baseline is
  measured on a genuinely clean frame.
- **The pattern grows, it does not reshuffle.** Every streak is generated once
  from a fixed seed and fades in at its own level, so dragging the slider
  thickens the *same* rubber. A pattern resampled per frame would flicker, and
  the slider is the whole pitch.

Lateral positions are fractions of the *local track width*, read off the
drivable mask, so a deposit sits in the same place on the road whether it is
8 px or 180 px wide on screen. Contamination is clipped to road and kerb —
rubber on the grass is a bug, and it would also fake the score.

### `chronos/integrity.py` — Module 2

The Boundary Integrity Engine. One number 0–100, plus the four components it
is made of.

**Measured against a session baseline, not against a theoretical 100%.** This
is the design decision that makes the score defensible:

> Module 1 recovers 67–98% of the true boundary on a **clean** frame, and how
> much depends on the scene, not on its condition — the far field is one or two
> pixels wide, and a pale run-off costs more of it than grass does. Reporting
> that shortfall as degradation would mean calling a perfectly clean corner
> degraded. So a baseline is captured once on a clean frame and every component
> is reported relative to what *that scene* achieved when clean.

There is deliberately **no absolute anchor on continuity** — its absolute value
is confounded by scene geometry. Contrast and sharpness keep a generous
absolute anchor, set at the bottom of the clean envelope across all 17 scenes,
so a corner whose line was *already* bad cannot score 100 by not getting worse.
Lens softness and sun angle are properties of the camera and the day, not of
the line, and are not charged as degradation.

Measurements are taken along the **baseline geometry**, not along the current
detection — otherwise the measurement would quietly disappear exactly when
contamination stops the detector finding the line at all.

**Calibration**, median over all 17 scene variants:

| rubber level | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| target | 90+ | 70–80 | ~41 | under 25 |
| **measured** | **100** | **70.6** | **43.6** | **21.4** |

Clean frames across all variants score 95.2–100. Monotonic for all six
degradation kinds, no dips. `IntegrityScore.explain()` answers the question a
judge will ask: *"contrast 58% of baseline, 0% of the line length no longer
detectable, 37% of line area contaminated, edge sharpness 22% of baseline."*

### `benchmark/sweep.py`

17 variants × 6 kinds × 8 levels = 816 measurements, ~2m15s. Raw physical
readings are cached, so re-tuning the 0–100 mapping re-plots instantly instead
of re-measuring. Writes three PNGs to `benchmark/results/`:

- `degradation_curve.png` — integrity vs level, one line per kind
- `subscores.png` — the four components separately
- `false_confident.png` — the metric that matters

**False-confident rate** — how often a system issues a confident verdict that
is wrong. Both systems run the same detector and get the same answer; the
difference is whether integrity is allowed to veto it:

| level | 0.0 | 0.3 | 0.45 | 0.6 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| baseline (ignores integrity) | 0% | 0% | 1.0% | 3.9% | 10.8% | 16.7% | 18.6% |
| **CHRONOS** (integrity-gated) | 0% | 0% | 1.0% | 0% | 1.0% | 0% | **0%** |
| price: sent to review | 0% | 0% | 14.7% | 41.2% | 81.4% | 96.1% | 100% |

A verdict counts as wrong when the boundary it would be measured against is off
by more than 8 px — about a fifth of a tyre width at these camera distances.

### The spatiotemporal layer

| module | does |
|---|---|
| `chronos/track.py` | ground-plane geometry — margin in millimetres, no calibration |
| `chronos/car.py` | pretrained YOLO, persistent ids, four contact patches, occlusion recovery |
| `chronos/temporal.py` | per-car time series of four wheel states; excursion events |
| `chronos/decide.py` | trust, three outcomes, plain-English reasons, session stats |
| `chronos/ui/timeline.py` | the four-lane wheel-state timeline |
| `chronos/pipeline.py` | all of it, end to end |

**The rule, implemented properly.** A wheel is three-valued — `INSIDE`,
`ON_LINE`, `OUTSIDE` — because the white line is part of the track and
touching it is legal. A violation needs **all four wheels OUTSIDE**; one wheel
on the line is not a violation however far the other three are out. An
excursion opens at three wheels out, so the near-miss is still *recorded and
reported as not a violation* rather than never seen.

**End-to-end behaviour**, same car, same corner:

| scenario | verdict |
|---|---|
| clean lap | no excursion |
| brush the line | no excursion |
| three wheels out, one on the line | **CLEAR** — *"Rear-left on the line throughout — not a violation."* |
| all four out, 500 ms | **VIOLATION** — *"All four tyres outside T6 for 480 ms. Trust 95%."* |
| the same excursion, rubber 0.6 | **REVIEW REQUIRED** — *"Boundary integrity 44 during event. Cannot confirm."* |
| the same excursion, 10-frame blackout | **REVIEW REQUIRED** — *"Lost car #1 for 10 of 20 frames."* |

Measured 480 ms against a true 500 ms.

**Millimetres without calibration.** The three references in the plan are not
three measurements of one number — under perspective, a millimetre *across*
the track and a millimetre *along* it occupy different numbers of pixels. A
margin is across, so it is scaled by the **line width** (cross-checked by the
car's axle track); stripe pitch and wheelbase give the *along*-track scale and
are never used to scale a margin. Two corrections that mattered:

- **Equivalent width, not half-maximum.** A half-max crossing counts the
  blurred edge and over-states a 4-pixel line by ~12%, which lands on every
  margin as a systematic under-read. Blur conserves the integral, so
  `∫(I−background)/(peak−background)` recovers the true width.
- **Fit the scale, don't sample it.** Per-station width carries ~15% noise;
  under perspective the scale is a smooth function of image height, so fitting
  it cut margin error from **189 mm to 115 mm** (p95 482 → 262 mm).

**Throughput.** Boundary integrity is re-scored every `integrity_every`
frames. A line's condition changes over a session, not between two frames
20 ms apart, so this is engineering rather than a shortcut:

| `integrity_every` | 1 | 5 | 25 |
|---|---|---|---|
| throughput | 10 fps | 49 fps | 228 fps |
| verdict | VIOLATION | VIOLATION | VIOLATION |

---

## Conventions

- Tunables live in a `@dataclass` config at the top of each module. Nothing
  magic is buried in the body.
- Every module has a `__main__` block and writes a debug image to `./debug/`.
- Failures return a result carrying a **reason string**, never a bare `None`.
  Bad *input* raises immediately.

## Real footage

```bash
.venv/bin/python -m chronos.lane --video data/real/clip.mp4 --frames 8
.venv/bin/python tools/real_test.py --frames 12
```

### Why Module 1 could not work here, and what replaced it

`chronos/boundary.py` segments the drivable *region* and takes its outer
edge. On real trackside footage that fails for two structural reasons, both
measured rather than assumed:

* **the drivable mask keys on "low saturation, mid brightness"**, which
  equally describes asphalt, concrete barriers, grandstands and overcast
  sky. Measured on the Miami clip: asphalt sits at Lab `L=126, a=+5.0,
  b=-5.0` and the concrete barrier behind the kerb at `L=143, a=+3.8,
  b=+0.2` — the same neutral grey, four units apart in chroma. No tolerance
  separates them, and the mask grew to cover the frame.
* **the kerb classifier keys on RED hue.** Miami's kerbs are blue and
  orange. A red-keyed kerb finder returns nothing there, so the outer-side
  decision had nothing to stand on.

`chronos/lane.py` ignores regions and looks for what the regulation names:
a thin bright stripe of paint on dark asphalt. That is a ridge, and a
morphological top-hat finds ridges — the same operator lane-departure
systems have used on road markings for twenty years, which is the point,
since this project's second application is road-marking health from dashcam
video.

Four things had to be right, and each was wrong first:

| stage | the failure it fixes |
|---|---|
| multi-width top-hat | one kernel cannot span a line that is 3 px at the horizon and 40 px at the camera |
| **flood blocked by chroma and by the paint** | colour cannot separate road from barrier; the *kerb* can. Chroma: asphalt 7, concrete 4, paint 7 — all pass; blue kerb 29, orange 32, sky 18 — all block. Hue-agnostic, so red kerbs behave identically |
| **the surface edge is the boundary** | the line itself gets consumed as a sealing wall, so ranking ridge candidates picked barrier tops. Where the flood stops *is* the track limit |
| **snap on ridge response, not brightness** | outward from the road the profile reads `139,139,134,139,151,214,205,210,193,193…` — it rises into the paint and never comes back down, because a bright kerb is behind it. Brightness-walking ran 24 px past the line; the top-hat response does come back down, because a kerb is broad and a line is thin |

**Measured, on the frames in `data/real/`:** 32 of 45 frames of the Miami
clip (71%), median paint-vs-asphalt contrast **+0.21**; and a correct
boundary on the Red Bull Ring photograph at **+0.22**. The frames it
refuses carry a reason naming the stage.

### The degradation curve, on a real photograph

Previously this table existed only for synthetic renders. Rubber applied to
the Red Bull Ring frame, scored against its own session baseline:

| rubber level | 0.00 | 0.15 | 0.30 | 0.45 | 0.60 | 0.75 | 0.90 |
|---|---|---|---|---|---|---|---|
| **integrity** | **100** | **87** | **76** | **62** | **53** | **48** | **42** |
| contrast | 100 | 71 | 48 | 30 | 21 | 18 | 15 |
| continuity | 100 | 87 | 59 | 30 | 17 | 13 | 14 |

Monotonic, and driven by contrast and continuity as it should be.

### Three assumptions the console now tests instead of assuming

Each of these produced a confident, wrong number before it was checked.

**1. The boundary must lie on paint.** With a wrong boundary the integrity
engine sampled its paint band and its asphalt band on the same material and
reported **36**:

| frame | paint band | asphalt band | Michelson |
|---|---|---|---|
| synthetic clean | 185 | 61 | **+0.490** |
| real, detector wrong | 153 | 153 | +0.002 |
| real, detector wrong | 134 | **207** | **−0.281** |

On the last, the "asphalt" is *brighter* than the "paint". This was never a
calibration problem — recalibrating would have buried it. `capture_baseline`
now refuses such a baseline (`min_baseline_contrast`), and the detector
chain runs the same test **before** accepting a candidate, because
previously a plausible-looking classical answer won simply by arriving
first and the correct ridge answer was never consulted.

**2. The camera must be fixed.** Integrity is defined relative to a session
baseline: this corner, from this camera. On the Miami replay — a chase cam —
the baseline describes track that is no longer in shot, and the engine
scored **3**, which looks like a catastrophically dirty line and is really a
camera that moved 427 px. The console now measures that drift and refuses.
A folder of images of differing sizes is refused for the same reason: it is
not one camera.

**3. The absolute anchors do not apply to real cameras.** They were fitted
on synthetic frames reading paint 185 / asphalt 61 — Michelson 0.49. A
*correctly detected* real line measures about 0.13, which the anchor turns
into a total near 30 on a line in good condition. On real footage the
anchors are switched off, the score is relative to the session baseline, and
the console's own heading changes to **RELATIVE TO SESSION BASELINE** so the
two claims can never be confused.

### What still does not work

*Cars.* YOLO does not recognise an F1 car — see **Known limits**.

*Moving cameras.* Refused, not supported. The session-baseline model needs a
fixed camera and there is no version of it that does not.

*Coverage.* 71% of frames on one clip is not a validated detection rate. It
is one clip, from a game replay, at one circuit. No millimetre claim is made
on any real frame, because no ground truth for one exists.

## Known limits

- The far field is where coverage is lost: the last few percent of the line is
  one or two pixels wide, and with a grey run-off it is lost sooner (~67%).
- The two channels share the kerb segmentation, so a kerb failure fools both.
  The agreement number is worth exactly what that assumption is worth.
- `outer_side` is `"auto"` by default, using the kerb and then the corner's
  bend. On a straight with no kerb, that is genuinely ambiguous — set
  `outer_side="left"` or `"right"`.
- **Validated on synthetic scenes only.** Real footage has now been *run*, and
  it does not work there yet. See **Real footage** below. Every accuracy
  number in this README is synthetic and the console says so on every screen
  that shows one.
- Continuity barely moves under rubber: Module 1 keeps finding the line well
  past the point where the line is worth trusting. That is an honest finding and
  the reason continuity carries the lowest weight — robust detection is exactly
  why detection alone cannot be the confidence signal.
- The false-confident metric tests the *boundary*, not a car verdict. `decide.py`
  now exists and issues real verdicts, but the sweep still scores the boundary,
  because a rate needs hundreds of events and the clip generator produces one
  excursion per run. Re-scoring the metric on verdicts is a known open item;
  the shape of the metric does not change.
- **YOLO does not recognise a Formula 1 car — on synthetic renders *or* on
  real footage.** On the flat-shaded renders it returns nothing at a 0.05
  confidence floor. On the six real frames in `data/real/` it returns `kite`
  0.52, `suitcase` 0.60, `book` ×27 and `train` — exactly one frame of six
  produced any COCO vehicle class at all, `truck` at 0.26. COCO has no racing
  class and an F1 car does not look like a COCO `car`; the same model finds a
  bus at 0.82 and seven people up to 0.90 in an ordinary street photo, so the
  wrapper, the class filter and the threshold are all correct. The model is
  simply the wrong model for this subject. The car-detection *path* is
  exercised by a simulated detector with injected jitter, dropouts and
  blackouts, which stresses the tracker harder than a clean detector would.
  Fixing this means a detector trained on motorsport, and that is a known
  open item, not a tuning problem.
- **Contact points from a bounding box are worth ~250–400 mm near the camera**
  and worse with distance. A tyre is 380 mm wide, so the ON_LINE/OUTSIDE call
  is *not* reliable from a box at distance. The temporal and decision layers
  are therefore validated on true contact points; the box-derived path reports
  its own measured error. Contact-patch keypoints are the first item on the
  cut list, and the published state of the art lists them as its own future
  work.
- Because of that error budget, `chronos.car` applies **resolution gating**:
  where a tyre spans under ~2 px, wheel confidence collapses and the verdict
  goes to review. The same principle as boundary integrity, applied to the
  measurement itself.
- A violation whose innermost wheel clears the threshold by less than ~150 mm
  is inside the measurement error and will not be separated reliably. The
  system abstains rather than guessing, which is the intended behaviour.
- Integrity needs a clean baseline frame per corner. There is no way to score a
  corner first seen already dirty, and the engine says so rather than guessing.

## Layout

```
chronos/chronos.py             the console entry point
chronos/chronos/ui/app.py      the console: engine thread, layout, review mode
chronos/chronos/ui/widgets.py  evidence strip, analysis report, wordmark, readouts
chronos/chronos/ui/theme.py    three themes + bundled-font registration
chronos/chronos/ui/export.py   ./output/ writer; enforces the no-reference rule
chronos/assets/fonts/          bundled OFL faces — see assets/README.md
chronos/tools/real_test.py     boundary + YOLO on real footage, reports honestly
chronos/output/                exports land here  (gitignored)
chronos/chronos/boundary.py    Module 1  — track-limit boundary (synthetic)
chronos/chronos/lane.py        Module 1b — ridge detector for REAL footage
chronos/chronos/boundary_real.py  detector dispatch: classical -> ridge -> learned
chronos/chronos/integrity.py   Module 2  — boundary integrity score
chronos/chronos/degrade.py     Module 3  — controlled degradation
chronos/benchmark/generate.py  synthetic scenes + ground truth
chronos/benchmark/scenes.py    the 17 scene variants, defined once
chronos/benchmark/sweep.py     the sweep, the curve, the plots
chronos/benchmark/results/     PNGs for the deck + the raw cache
chronos/tests/                 scored against ground truth, not snapshots
chronos/debug/                 visual output from every run
```

MIT licence.

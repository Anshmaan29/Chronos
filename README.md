# CHRONOS

**Reference Integrity for Track Limits**

> Every track-limits system in the world measures the car against the line.
> Not one of them measures the line. **CHRONOS measures both.**

Built for **TrackShift 2026** — Problem Statement: *Track Limits Detection*.
Selected in the **top 10 teams out of ~3,200 participating students.**

---

## The problem

Miami Grand Prix, May 2026. Alex Albon crossed track limits at Turn 6 in sprint qualifying.

The FIA's new AI system did not catch it — not because the car was hard to see, but because the support races that weekend had left tyre rubber across the white line. The call came too late. Liam Lawson was wrongly eliminated from SQ1 and sat in his car waiting on live TV. Albon's SQ2 times were deleted afterwards. The grid was wrong.

The same thing happened in Austria in 2022.

**The system did not fail because of the car. It failed because the line got dirty.**

Every automated officiating system models uncertainty in detecting the *subject*. None of them model uncertainty in the *reference* itself. That is the gap CHRONOS fills.

---

## What it does

CHRONOS is a **steward's assistant**, not a referee. Video in, and for every car at every corner it returns one of three verdicts:

```
CLEAR  ·  VIOLATION  ·  REVIEW REQUIRED
```

It never issues a penalty. It produces a recommendation, a trust rating, and an evidence packet a human can audit.

The difference from every other system: it also scores **how far the line itself can be trusted**, and abstains when the reference is degraded.

```
Car geometry confidence:   96%
Boundary integrity:        42%
──────────────────────────────────
Verdict:  REVIEW REQUIRED
Reason:   reference degraded — T6 contamination 28%
```

---

## Architecture

```
   VIDEO
     │
     ▼
┌─────────────────────┐
│ 1. BOUNDARY         │  outer edge of the white line
│    DETECTION        │  two independent channels
└─────────┬───────────┘
          │
          ├──────────────────────────┐
          ▼                          ▼
┌─────────────────────┐   ┌─────────────────────┐
│ 2. BOUNDARY         │   │ 3. CONTACT          │
│    INTEGRITY ENGINE │   │    GEOMETRY         │
│                     │   │                     │
│  contrast           │   │  tyre contact pts   │
│  continuity         │   │  perspective fix    │
│  edge sharpness     │   │  margin in mm       │
│  contamination      │   │  occlusion recovery │
│  → SCORE 0-100      │   │  → margin + conf    │
└─────────┬───────────┘   └─────────┬───────────┘
          │                         │
          └───────────┬─────────────┘
                      ▼
          ┌───────────────────────────┐
          │ 4. REFERENCE-AWARE        │
          │    DECISION ENGINE        │
          │                           │
          │  trust = car × integrity  │
          │  + temporal consistency   │
          └───────────┬───────────────┘
                      ▼
              EVIDENCE PACKET
```

---

## Track limits is a spatiotemporal problem, not a computer vision one

A single frame can only say where the tyres *appear* to be right now. The rule is about what **four wheels** did, **over time**, to **one identified car**.

- **Persistent car IDs** across frames — the same car must be the same car
- **Three-valued wheel states** — `INSIDE` / `ON_LINE` / `OUTSIDE`. A tyre touching the white line is **legal**, not a violation. This is the rule most systems get wrong.
- **Excursions are events with a duration**, not frames with a label
- **Kalman smoothing + hysteresis** — a single noisy frame never fires a verdict

An excursion opens at three wheels out, so that near-misses are recorded and can be reported as *"not a violation"*. All-four-out is then the violation test **inside** that event.

---

## The headline result

**False-confident rate** — how often a system issues a confident verdict that is wrong:

| Contamination level | 0.0 | 0.45 | 0.6 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| Baseline (ignores integrity) | 0% | 1.0% | 3.9% | 10.8% | 16.7% | **18.6%** |
| **CHRONOS** | 0% | 1.0% | 0% | 1.0% | 0% | **0%** |
| *price: sent to review* | 0% | 14.7% | 41.2% | 81.4% | 96.1% | 100% |

A system that ignores boundary condition is confidently wrong **18.6%** of the time at heavy contamination. CHRONOS is **1.0%** worst case.

The price is that 41% of events go to human review at Miami-level contamination — which is exactly what a steward's assistant should do.

---

## Integrity calibration

| Rubber level | 0.0 | 0.3 | 0.6 | 0.9 |
|---|---|---|---|---|
| Target | 90+ | 70–80 | ~41 | under 25 |
| **Measured (median, 17 scenes)** | **100** | **70.6** | **43.6** | **21.4** |

Monotonic across all six degradation kinds, no dips. Clean frames across every variant: **95.2–100**.

Fitted by grid-searching gammas and weights against all 17 scenes simultaneously, not tuned on a single frame.

---

## Two confounds we had to remove

**Continuity.** Boundary detection already drops to 67% coverage on a clean gravel-runoff scene for purely geometric reasons — the far field, where the line is 1–2 px wide. Reporting that as "degraded" would be wrong. Continuity is measured **relative to a per-scene clean baseline**, so degradation means loss against what was achievable in that scene. There is a regression test that fails if this breaks.

**Sharpness.** The soft-lens variant has clean Sobel sharpness of 154 against 300+ elsewhere. Lens softness is a property of the camera, not the line — charging a scene for it is the same error. Absolute anchors for contrast and sharpness sit at the bottom of the clean envelope across all 17 scenes, so they only bite when a line was already bad.

**Occlusion.** A car sitting on the measurement stations dropped integrity to 70 with contamination at zero. A car covering the line does not mean the line is degraded — it means you cannot see it there. Occluded stations are now excluded and the occluded fraction is reported. 70.0 → 92.4.

---

## What is validated, and what is not

**Validated — synthetic ground truth**

- 17 scene variants: grass / gravel / asphalt run-off, kerb on one side / both / none, gentle → sharp corner, four camera poses, 8–30 cm line widths, over/under exposure, soft lens + noise + JPEG
- Scenes are constructed in a flat world plane in metres and projected through a pinhole camera, so ground truth is the same world curve that was rasterised — **correct by construction, not by annotation**
- Boundary: **0.72 px** mean deviation, **94%** coverage
- Margin error: **128 mm** mean / **281 mm** p95
- **39/39 tests pass**

**Not validated**

- Boundary detection on real broadcast footage. Brightness and saturation cannot separate asphalt from sky, barriers and painted run-off on a hazy telephoto shot. The system **fails loudly with a named reason** rather than returning a wrong answer.
- Bounding-box contact points carry **250–400 mm** error. A tyre is 380 mm wide, so the `ON_LINE` / `OUTSIDE` call is not reliable from a box at distance. Resolution gating collapses wheel confidence where a tyre spans under ~2 px.
- A violation whose innermost wheel clears the threshold by less than ~150 mm sits inside our error band. The system abstains. That is intended behaviour.

YOLO **does** find real F1 cars (0.32 confidence, default settings, no escalation needed).

---

## The failure that became the finding

Our first real-footage run reported success. It had returned a polyline — which was tracing a car's rear wing.

`ok=True` meant "a polyline came back", not "the polyline is right".

So we built three colour-free geometric plausibility checks:

- **gross turning** on a decimated polyline — a silhouette turns 360°+, a track edge under 170°
- **endpoint distance ÷ arc length** — a loop comes back on itself
- **fraction of the line inside a car box** — took a test clip from 10 false passes down to 3

All four checks now fail on that frame:

```
[FAIL]  two channels agree       264 px apart   (synthetic: 1-3 px)
[FAIL]  road area plausible      53% of frame
[FAIL]  boundary not on a car    79% of polyline inside a car box
[FAIL]  kerb is not a car        39% of kerb mask inside a car box
```

**This is exactly the failure the project exists to prevent.** A system that does not verify its own reference will confidently produce numbers on a wrong reference. In Miami that cost a driver his qualifying. Ours refuses instead.

---

## Three measurement bugs found by measuring, not reasoning

1. **Margins read 12% short.** The paint-width estimator used a half-maximum crossing, which counts the blurred edge and over-states a 4 px line. Blur conserves the integral, so equivalent width `∫(I−bg)/(peak−bg)` is unbiased — ratio 1.010 vs 0.863.

2. **Per-station scale carried ~15% noise** straight onto every margin. Under perspective the scale is a smooth function of image height; fitting it cut error **189 → 115 mm** (p95 **482 → 262**).

3. **Stripe pitch cannot calibrate a margin.** Along-track and across-track scale differ under perspective. The line width is the across-track reference (cross-checked by axle track); kerb pitch and wheelbase are along-track only. Using pitch would have looked correct on a demo frame and been badly wrong at the far end of the corner.

---

## Install

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
    numpy opencv-python scipy matplotlib PyQt6 ultralytics
```

No internet is used at runtime.

## Run

**The console**

```bash
.venv/bin/python chronos.py                          # synthetic corner, car runs wide
.venv/bin/python chronos.py --video data/clip.mp4    # your own footage
.venv/bin/python chronos.py --theme brutal           # alternate skin
```

Or press **O** / drag a file onto the window to load a video, a still, or a folder of images.

| Flag | Values |
|---|---|
| `--path` | `clean` · `brush` · `one_wheel_on_line` · `violation` · `straight_wide` |
| `--detector` | `auto` · `real` · `synthetic` |
| `--theme` | `instrument` · `brutal` |
| `--kind` | degradation the slider applies (`rubber`, `dust`, `wet`, `fade`, `glare`, `shadow`) |

| Key | Action |
|---|---|
| `O` | open footage |
| `S` | stats panel |
| `I` | cycle integrity sampling 1:1 → 1:5 → 1:25 |
| `E` | export analysis report + violation frames to `./output/` |

**Individual modules**

```bash
.venv/bin/python -m benchmark.generate --out data/frame.jpg
.venv/bin/python -m chronos.boundary --image data/frame.jpg --gt data/frame_gt.json
.venv/bin/python -m benchmark.sweep            # degradation sweep + plots
.venv/bin/python -m tools.real_run             # real-footage plausibility report
```

**Tests**

```bash
.venv/bin/python -m pytest tests/ -v
```

Every module writes a visual debug image to `./debug/` so results can be checked by eye.

---

## Performance

| Integrity sampling | Throughput |
|---|---|
| 1:1 | 10 fps |
| 1:5 | 49 fps |
| 1:25 | **228 fps** |

Same verdict at all three. A line's condition changes over a session, not between two frames 20 ms apart.

Video in ≈10 Mbps → decisions out ≈12 kbps. Roughly **1000× data reduction**, so the footage never has to leave the circuit.

---

## Beyond motorsport

The same engine, with a different boundary.

**Lane-marking health for roads and ADAS.** Lane-keeping systems fail as road markings fade — detection quality drops directly with marking retroreflectivity. Road authorities currently survey markings with expensive specialist vans, or by eye.

CHRONOS scores marking visibility from **ordinary dashcam video**. No new hardware. A fleet of 500 buses becomes a national survey network.

```
NH-48, km 212-218, westbound.
Marking integrity 34/100.
Below ADAS reliability threshold at night. Priority repaint.
```

> In racing, a bad line costs a driver his qualifying.
> On a highway at night, a bad line means a car's lane-keeping system cannot see the road.

---

## Project layout

```
chronos/
├── chronos/
│   ├── boundary.py          classical boundary detection (1042 loc)
│   ├── boundary_real.py     learned segmentation path (834 loc)
│   ├── integrity.py         the Boundary Integrity Engine (629 loc)
│   ├── degrade.py           contamination generator (627 loc)
│   ├── car.py               detection + contact points (635 loc)
│   ├── track.py             persistent ID tracking (340 loc)
│   ├── temporal.py          wheel states + excursion events (358 loc)
│   ├── decide.py            reference-aware verdicts (250 loc)
│   ├── pipeline.py          orchestration (228 loc)
│   └── ui/                  PyQt6 console, two themes
├── benchmark/
│   ├── generate.py          synthetic scene + ground truth (539 loc)
│   ├── sweep.py             degradation sweep + plots (331 loc)
│   └── car_render.py
├── tools/
│   ├── real_run.py          real-footage plausibility report
│   └── real_test.py
└── tests/                   39 tests
```

**9,229 lines · 24 modules.** Python, OpenCV, YOLO, SAM2, Kalman filtering, PyQt6.

**No model training anywhere** — everything pretrained or classical. That was deliberate: a training run that fails at 3am costs you the project.

---

## Research direction

> *Reference Integrity in Automated Sports Officiating: measuring boundary condition as a first-class input to adjudication confidence.*

Existing systems model uncertainty in detecting the subject. None model uncertainty in the reference itself. Related work exists in road-marking retroreflectivity assessment — LiDAR survey vans, virtual-reference degradation methods — none of it applied to real-time adjudication.

The novel element is the **gating**: the system's authority to decide is conditional on the measured health of its own reference.

---

## About TrackShift

**TrackShift** is a 24-hour motorsport-inspired engineering challenge run by **Plaksha University** in partnership with the **Mphasis F1 Foundation** and the **MoneyGram Haas F1 Team**.

Teams take on problems drawn straight off the current grid. Everything must be open-source, and the challenge is explicitly aimed at translational, deployable work rather than a performative coding sprint — winning teams go on to a structured internship to push their prototypes past proof-of-concept.

The 2026 edition drew roughly **3,200 students**. CHRONOS was selected in the **top 10**.

---

## Licence

MIT.

Broadcast clips referenced during development were used for research demonstration only. All quantitative validation in this repository is against our own synthetic benchmark, which is released with the code.

"""CHRONOS degradation sweep -- the numbers and the plots for the deck.

Runs every scene variant against every degradation kind across levels 0 to 1,
scores the boundary with Module 2 at each step, and produces:

  1. integrity vs level, one line per degradation kind
  2. the four sub-scores, plotted separately
  3. the false-confident rate -- how often a system issues a confident verdict
     that is wrong, with and without integrity gating

Raw measurements are cached, because the physical readings do not change when
the 0-100 mapping is re-tuned.  Re-plotting after a calibration change is then
instant instead of a three-minute rerun.

Run::

    python -m benchmark.sweep                 # measure (or reuse cache), plot
    python -m benchmark.sweep --remeasure     # force a fresh measurement pass
    python -m benchmark.sweep --kinds rubber  # a subset, while iterating
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from benchmark.generate import SceneConfig, generate_scene
from benchmark.scenes import VARIANTS
from chronos.boundary import (BoundaryConfig, detect_boundary, polyline_error)
from chronos.degrade import KINDS, DegradeConfig, build_track_frame, degrade
from chronos.integrity import (IntegrityConfig, capture_baseline,
                               measure_boundary, score_from_measurement)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = os.path.join(RESULTS_DIR, "raw_measurements.json")
LEVELS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]

# A verdict is "wrong" when the boundary it would be measured against is off by
# more than this.  At these camera distances a Formula 1 tyre is roughly 40 px
# across in the near field, so 8 px is about a fifth of a tyre -- comfortably
# enough to flip a track-limits call.
WRONG_PX = 8.0


def measure_all(kinds: tuple[str, ...] = KINDS,
                variants: Optional[dict] = None,
                levels: Optional[list[float]] = None) -> list[dict]:
    """Measure every (variant, kind, level).  Returns one record per step."""
    variants = variants or VARIANTS
    levels = levels or LEVELS
    icfg, dcfg, bcfg = IntegrityConfig(), DegradeConfig(), BoundaryConfig()
    rows: list[dict] = []
    t0 = time.time()

    for vi, (vname, kw) in enumerate(variants.items(), 1):
        scene = generate_scene(SceneConfig(**kw))
        clean = detect_boundary(scene.image, bcfg, save_debug=False)
        if not clean.ok:
            raise RuntimeError(f"sweep: clean detection failed on {vname!r}: {clean.reason}")

        # the ground truth is the reference geometry, and the CLEAN detection
        # fixes the achievable coverage for this scene -- see integrity.py
        baseline = capture_baseline(scene.image, scene.gt_boundary, icfg, result=clean)
        tf = build_track_frame(clean.polyline, clean.drivable_mask, clean.kerb_mask,
                               n=icfg.n_samples)

        for kind in kinds:
            for lv in levels:
                img = scene.image if lv == 0.0 else degrade(
                    scene.image, lv, kind, cfg=dcfg, track_frame=tf)
                res = detect_boundary(img, bcfg, save_debug=False)
                raw = measure_boundary(img, baseline, icfg, result=res)
                err = (polyline_error(res.polyline, scene.gt_boundary)
                       if res.ok else None)
                rows.append({
                    "variant": vname, "kind": kind, "level": lv,
                    "baseline_contrast": baseline.contrast,
                    "baseline_sharpness": baseline.sharpness,
                    "baseline_coverage": baseline.coverage,
                    "baseline_contamination": baseline.contamination,
                    **raw,
                    "dev_mean_px": err["dev_mean_px"] if err else float("inf"),
                    "dev_p95_px": err["dev_p95_px"] if err else float("inf"),
                })
        print(f"  [{vi}/{len(variants)}] {vname:22s} {time.time() - t0:6.1f}s", flush=True)
    return rows


class _Baseline:
    """The few baseline fields scoring needs, rehydrated from a cached row."""

    def __init__(self, row: dict):
        self.contrast = row["baseline_contrast"]
        self.sharpness = row["baseline_sharpness"]
        self.coverage = row["baseline_coverage"]
        self.contamination = row["baseline_contamination"]


def score_rows(rows: list[dict], cfg: Optional[IntegrityConfig] = None) -> list[dict]:
    """Apply the 0-100 mapping to cached raw measurements."""
    cfg = cfg or IntegrityConfig()
    out = []
    for r in rows:
        sc = score_from_measurement(r, _Baseline(r), cfg)
        out.append({**r, "total": sc.total, "s_contrast": sc.contrast,
                    "s_continuity": sc.continuity, "s_sharpness": sc.sharpness,
                    "s_contamination": sc.contamination})
    return out


# --------------------------------------------------------------------------
# the metric that matters: false-confident rate
# --------------------------------------------------------------------------


def false_confident(scored: list[dict], cfg: IntegrityConfig,
                    wrong_px: float = WRONG_PX) -> dict:
    """How often each system is confidently wrong, by level.

    The claim is not that CHRONOS is more accurate.  Both systems run the same
    detector and get the same answer.  The claim is that CHRONOS knows when
    that answer should not be trusted:

      baseline  issues a verdict on every frame, so it is confidently wrong
                whenever the boundary is wrong.
      CHRONOS   issues a verdict only when integrity clears the threshold, so
                it is confidently wrong only when it is BOTH wrong and
                convinced the reference was healthy.

    Returns per-level rates plus the abstention rate, which is the price paid.
    """
    levels = sorted({r["level"] for r in scored})
    out = {"levels": levels, "baseline": [], "chronos": [], "abstain": [],
           "wrong": [], "n": []}
    for lv in levels:
        at = [r for r in scored if r["level"] == lv]
        wrong = np.array([r["dev_mean_px"] > wrong_px for r in at])
        trusted = np.array([r["total"] >= cfg.trust_threshold for r in at])
        out["n"].append(len(at))
        out["wrong"].append(float(wrong.mean()))
        out["baseline"].append(float(wrong.mean()))
        out["chronos"].append(float((wrong & trusted).mean()))
        out["abstain"].append(float((~trusted).mean()))
    return out


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.frameon": False,
    })
    return plt


KIND_COLOR = {"rubber": "#c0392b", "dust": "#c8922a", "wet": "#2e6fb7",
              "fade": "#7d5ba6", "glare": "#d95f9a", "shadow": "#3f7d5a"}


def _series(scored, kind, field):
    levels = sorted({r["level"] for r in scored})
    med, lo, hi = [], [], []
    for lv in levels:
        v = np.array([r[field] for r in scored if r["kind"] == kind and r["level"] == lv])
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    return np.array(levels), np.array(med), np.array(lo), np.array(hi)


def plot_curve(scored: list[dict], cfg: IntegrityConfig, path: str) -> str:
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    kinds = [k for k in KINDS if any(r["kind"] == k for r in scored)]
    for kind in kinds:
        x, m, lo, hi = _series(scored, kind, "total")
        c = KIND_COLOR.get(kind, "#444")
        ax.plot(x, m, "-o", color=c, lw=2.2, ms=4.5, label=kind, zorder=3)
        ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0, zorder=2)

    ax.axhspan(0, cfg.trust_threshold, color="#c0392b", alpha=0.06, zorder=0)
    ax.axhline(cfg.trust_threshold, color="#c0392b", lw=1.2, ls="--", zorder=1)
    ax.text(1.005, cfg.trust_threshold, f"  {cfg.trust_threshold:.0f}  no verdict issued",
            color="#c0392b", va="center", fontsize=9.5, transform=ax.get_yaxis_transform())
    ax.set_xlabel("degradation level")
    ax.set_ylabel("boundary integrity")
    ax.set_title("Boundary integrity against controlled degradation\n"
                 f"median over {len({r['variant'] for r in scored})} scene variants, "
                 "shaded band is the interquartile range", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 102)
    ax.legend(ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_subscores(scored: list[dict], path: str) -> str:
    plt = _style()
    fields = [("s_contrast", "contrast"), ("s_continuity", "continuity"),
              ("s_sharpness", "edge sharpness"), ("s_contamination", "contamination")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
    kinds = [k for k in KINDS if any(r["kind"] == k for r in scored)]
    for ax, (field, title) in zip(axes.ravel(), fields):
        for kind in kinds:
            x, m, _, _ = _series(scored, kind, field)
            ax.plot(x, m, "-o", color=KIND_COLOR.get(kind, "#444"), lw=2, ms=3.5,
                    label=kind)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 102)
        ax.set_xlim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("degradation level")
    for ax in axes[:, 0]:
        ax.set_ylabel("sub-score")
    axes[0, 0].legend(ncol=2, loc="lower left", fontsize=9)
    fig.suptitle("What the integrity score is made of", x=0.008, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_false_confident(scored: list[dict], cfg: IntegrityConfig, path: str) -> str:
    plt = _style()
    fc = false_confident(scored, cfg)
    x = np.array(fc["levels"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    ax.plot(x, 100 * np.array(fc["baseline"]), "-o", color="#c0392b", lw=2.4, ms=5,
            label="baseline (ignores integrity)")
    ax.plot(x, 100 * np.array(fc["chronos"]), "-o", color="#2e6fb7", lw=2.4, ms=5,
            label="CHRONOS (integrity-gated)")
    ax.fill_between(x, 100 * np.array(fc["chronos"]), 100 * np.array(fc["baseline"]),
                    color="#2e6fb7", alpha=0.10, lw=0)
    ax.set_xlabel("degradation level")
    ax.set_ylabel("false-confident rate  (%)")
    ax.set_title("Confident verdicts that are wrong", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    ax2.plot(x, 100 * np.array(fc["abstain"]), "-o", color="#7d5ba6", lw=2.4, ms=5)
    ax2.set_xlabel("degradation level")
    ax2.set_ylabel("frames sent to review  (%)")
    ax2.set_title("The price: how often it declines to decide", loc="left")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 102)

    fig.suptitle("CHRONOS does not detect better. It knows when not to decide.",
                 x=0.008, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(description="CHRONOS degradation sweep")
    p.add_argument("--remeasure", action="store_true", help="ignore the cache")
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--out", default=RESULTS_DIR)
    args = p.parse_args()

    kinds = tuple(k.strip() for k in args.kinds.split(","))
    for k in kinds:
        if k not in KINDS:
            raise SystemExit(f"unknown degradation kind {k!r}; expected {KINDS}")
    os.makedirs(args.out, exist_ok=True)

    if not args.remeasure and os.path.exists(CACHE):
        with open(CACHE) as fh:
            rows = json.load(fh)
        rows = [r for r in rows if r["kind"] in kinds]
        print(f"reusing {len(rows)} cached measurements ({CACHE})")
    else:
        print(f"measuring {len(VARIANTS)} variants x {len(kinds)} kinds x {len(LEVELS)} levels")
        rows = measure_all(kinds)
        with open(CACHE, "w") as fh:
            json.dump(rows, fh)
        print(f"cached -> {CACHE}")

    cfg = IntegrityConfig()
    scored = score_rows(rows, cfg)

    print("\nintegrity vs level, median over variants")
    print(f"{'kind':9s}" + "".join(f"{lv:>7.2f}" for lv in LEVELS))
    for kind in kinds:
        x, m, _, _ = _series(scored, kind, "total")
        print(f"{kind:9s}" + "".join(f"{v:7.1f}" for v in m))

    fc = false_confident(scored, cfg)
    print("\nfalse-confident rate (%)")
    print(f"{'level':9s}" + "".join(f"{lv:>7.2f}" for lv in fc["levels"]))
    print(f"{'baseline':9s}" + "".join(f"{100*v:7.1f}" for v in fc["baseline"]))
    print(f"{'chronos':9s}" + "".join(f"{100*v:7.1f}" for v in fc["chronos"]))
    print(f"{'abstain':9s}" + "".join(f"{100*v:7.1f}" for v in fc["abstain"]))

    paths = [
        plot_curve(scored, cfg, os.path.join(args.out, "degradation_curve.png")),
        plot_subscores(scored, os.path.join(args.out, "subscores.png")),
        plot_false_confident(scored, cfg, os.path.join(args.out, "false_confident.png")),
    ]
    print("\nplots:")
    for q in paths:
        print(f"  {q}")


if __name__ == "__main__":
    _cli()

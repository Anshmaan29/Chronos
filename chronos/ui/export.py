"""Write the analysis out to ``./output/`` -- key E in the console.

What leaves the console is what a steward or an engineer would need to
re-examine the call without the console in front of them: the frames, the
numbers behind each verdict, the timeline, and -- first, at the top of every
file -- the provenance line saying what produced them.

The no-reference rule is enforced *here* as well as in the UI, and
deliberately so.  An exported file outlives the window it came from and will
be read by someone who never saw the screen, so it is the one place where a
number computed against an unvalidated boundary would do the most damage.
When no boundary was established the boundary-dependent fields are not set
to null or zero -- they are **absent**, and the file says NO REFERENCE with
the reason instead.  A null invites someone to treat it as a missing
measurement; an absent field cannot be misread as one.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import Optional, Sequence

import cv2
import numpy as np

OUTPUT_DIR = "output"

#: Report keys that mean nothing without an established boundary.  Kept in
#: step with ``AnalysisReport.BOUNDARY_DEPENDENT``; the UI blanks them, this
#: omits them.
BOUNDARY_DEPENDENT = frozenset({
    "ref_agreement", "ref_coverage", "ref_scale",
    "integ_now", "integ_mean", "integ_min", "integ_below",
    "events", "violations", "cleared", "escalated", "resolved",
    "margin_mean", "margin_p95", "margin_n",
})

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, fallback: str = "run") -> str:
    out = _SAFE.sub("_", os.path.splitext(os.path.basename(text or ""))[0]).strip("_")
    return out[:48] or fallback


def export(report: dict, cards: Sequence, verdicts: Sequence = (),
           packets: Sequence[dict] = (), timeline: Optional[np.ndarray] = None,
           root: str = OUTPUT_DIR) -> str:
    """Write one run to ``<root>/<source>_<timestamp>/``.  Returns the path.

    ``cards`` are the evidence cards, newest first, each carrying the
    full-resolution overlay it was made from.  They are written oldest-first
    so the numbering reads forwards in time.
    """
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{_slug(report.get('source_path', ''), 'session')}_{stamp}"
    out = os.path.join(root, name)
    os.makedirs(out, exist_ok=True)

    ref_ok = bool(report.get("reference_ok", True))
    provenance = report.get("provenance", "Source: --")

    # --- the frames ---------------------------------------------------
    manifest = []
    for i, card in enumerate(reversed(list(cards))):
        img = getattr(card, "image", None)
        if img is None:
            continue
        outcome = _slug(getattr(card, "outcome", "frame"), "frame")
        fname = f"evidence_{i:02d}_{outcome}_f{getattr(card, 'frame_index', -1)}.jpg"
        cv2.imwrite(os.path.join(out, fname), img)
        entry = {"file": fname,
                 "frame_index": int(getattr(card, "frame_index", -1)),
                 "outcome": getattr(card, "outcome", ""),
                 "measured": bool(getattr(card, "has_reference", True))}
        # The caption carries the margin and the trust.  Under NO REFERENCE
        # there is no caption worth carrying, and none is written.
        if entry["measured"]:
            entry["caption"] = getattr(card, "caption", "")
        manifest.append(entry)

    if timeline is not None and ref_ok:
        cv2.imwrite(os.path.join(out, "timeline.png"), timeline)

    # --- the packets ----------------------------------------------------
    if ref_ok:
        for i, pkt in enumerate(packets):
            with open(os.path.join(out, f"packet_{i:02d}.json"), "w") as fh:
                json.dump(pkt, fh, indent=1)

    # --- the numbers ----------------------------------------------------
    fields = {k: v for k, v in report.items()
              if k not in {"provenance", "stamp", "source_path"} and v is not None}
    if not ref_ok:
        fields = {k: v for k, v in fields.items() if k not in BOUNDARY_DEPENDENT}

    doc = {
        "provenance": provenance,
        "exported_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "reference_established": ref_ok,
        "validated_against": "synthetic ground truth only",
        "fields": fields,
        "evidence": manifest,
    }
    if not ref_ok:
        doc["no_reference_reason"] = report.get("ref_reason", "")
        doc["note"] = ("No boundary was established, so no margin, no integrity "
                       "score and no verdict were computed. Fields that depend "
                       "on a boundary are absent rather than null.")
    if ref_ok and verdicts:
        doc["verdicts"] = [{"outcome": v.outcome.value,
                            "trust": round(float(v.trust), 4),
                            "reason": v.reason,
                            "car_id": v.event.car_id,
                            "corner": v.event.corner or "T6",
                            "entry_frame": v.event.entry_frame,
                            "exit_frame": v.event.exit_frame}
                           for v in verdicts]

    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(doc, fh, indent=2)

    # --- the same thing, readable ---------------------------------------
    lines = ["CHRONOS  ANALYSIS REPORT", "=" * 78, "", provenance, "",
             f"exported {doc['exported_at']}", ""]
    if not ref_ok:
        lines += ["!" * 78,
                  "NO REFERENCE — no boundary was established on this footage.",
                  f"reason: {report.get('ref_reason', '')}",
                  "",
                  "No margin, no integrity score and no verdict was computed.",
                  "Every boundary-dependent figure is omitted from this report",
                  "rather than shown as zero. The frames below are what the",
                  "camera saw; nothing has been measured on them.",
                  "!" * 78, ""]
    for section, rows in _TEXT_SECTIONS:
        shown = [(label, fields[key]) for label, key in rows if key in fields]
        if not shown:
            continue
        lines += [section, "-" * len(section)]
        lines += [f"  {label:<34} {value}" for label, value in shown]
        lines.append("")
    if ref_ok and verdicts:
        lines += ["VERDICTS", "-" * 8]
        for v in verdicts:
            lines.append(f"  {v.outcome.value:<16} trust {v.trust:5.0%}  {v.reason}")
        lines.append("")
    if manifest:
        lines += ["EVIDENCE", "-" * 8]
        for e in manifest:
            tail = e.get("caption", "unmeasured — no reference")
            lines.append(f"  {e['file']:<46} {tail}")
        lines.append("")
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(lines))

    return out


_TEXT_SECTIONS = (
    ("REFERENCE", (("boundary status", "ref_status"),
                   ("reason", "ref_reason"),
                   ("channel agreement", "ref_agreement"),
                   ("baseline coverage", "ref_coverage"),
                   ("scale across track", "ref_scale"))),
    ("BOUNDARY INTEGRITY", (("current", "integ_now"),
                            ("session mean", "integ_mean"),
                            ("session minimum", "integ_min"),
                            ("frames below alert (50)", "integ_below"),
                            ("contamination applied", "integ_level"))),
    ("SESSION", (("frames processed", "frames"),
                 ("race time covered", "race_s"),
                 ("throughput", "fps"),
                 ("excursions found", "events"),
                 ("violations", "violations"),
                 ("auto-cleared", "cleared"),
                 ("escalated to review", "escalated"),
                 ("auto-resolved", "resolved"))),
    ("ACCURACY  (synthetic ground truth only)",
     (("mean margin error", "margin_mean"),
      ("p95 margin error", "margin_p95"),
      ("samples", "margin_n"))),
    ("FALSE-CONFIDENT RATE  (benchmark sweep)",
     (("baseline, ignores integrity", "fc_baseline"),
      ("CHRONOS, integrity-gated", "fc_chronos"),
      ("price: sent to review", "fc_abstain"))),
)

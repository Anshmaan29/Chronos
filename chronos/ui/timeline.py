"""The wheel-state timeline: four lanes that make "spatiotemporal" visible.

One lane per wheel (FL, FR, RL, RR), one column per frame, coloured by the
three-valued wheel state.  The violation window is boxed.

This is the picture that explains the whole system without a sentence of
narration: four wheels, over time, and a box drawn only where **all four**
lanes are red at once.  An amber stripe running through a red patch is a
driver keeping a tyre on the line -- visibly not a violation, which is the
single hardest thing to explain in words and the easiest to see here.

Returns a BGR image, so it can be written to disk for the deck and dropped
straight into a Qt label in the live UI without a second implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from chronos.temporal import WHEELS, ExcursionEvent, WheelSample, WheelState

def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def state_colors() -> dict:
    """Lane colours for the active console theme.

    ``instrument`` keeps the original hand-picked values unchanged -- a reskin
    that quietly shifts the colours of the previous look is not a reskin you
    can fall back to.
    """
    try:
        from chronos.ui import theme as _T
        t = _T.active()
    except Exception:
        return dict(STATE_COLOR)
    if t.name != "brutal":
        return dict(STATE_COLOR)
    return {WheelState.INSIDE: _bgr(t.clear),
            WheelState.ON_LINE: _bgr(t.review),
            WheelState.OUTSIDE: _bgr(t.violation)}


STATE_COLOR = {
    WheelState.INSIDE: (86, 176, 92),      # green  -- within the limit
    WheelState.ON_LINE: (56, 176, 226),    # amber  -- touching the line, legal
    WheelState.OUTSIDE: (62, 62, 214),     # red    -- fully beyond
}
MISSING_COLOR = (70, 70, 70)


@dataclass
class TimelineStyle:
    width: int = 1180
    lane_height: int = 34
    lane_gap: int = 7
    left_margin: int = 74
    top_margin: int = 56      # room for the title AND a box label beneath it,
                              # which used to collide
    bottom_margin: int = 34
    background: tuple[int, int, int] = (26, 24, 22)
    text: tuple[int, int, int] = (212, 212, 212)
    dim: tuple[int, int, int] = (140, 140, 140)
    box: tuple[int, int, int] = (255, 255, 255)


def render_wheel_timeline(series: dict[str, Sequence[WheelSample]],
                          events: Optional[Sequence[ExcursionEvent]] = None,
                          style: Optional[TimelineStyle] = None,
                          title: str = "wheel state over time",
                          cursor_frame: Optional[int] = None) -> np.ndarray:
    """Draw the four-lane wheel-state timeline.

    Parameters
    ----------
    series:  per-wheel sample lists, as carried by a CarTimeline or an event.
    events:  excursions to box.  Only the all-four-outside span is boxed.
    cursor_frame: draws a playhead, for the live UI.

    Raises
    ------
    ValueError
        If no wheel has any samples -- an empty timeline would read as "all
        four wheels inside", which is a claim, not an absence of data.
    """
    style = style or TimelineStyle()
    palette = state_colors()
    separator = False
    try:
        from chronos.ui import theme as _T
        t = _T.active()
        if t.name == "brutal":
            style.background = _bgr(t.panel)
            style.text = _bgr(t.bone)
            style.dim = _bgr(t.dim)
            separator = True
    except Exception:
        pass
    lengths = [len(series.get(w, ())) for w in WHEELS]
    n = max(lengths) if lengths else 0
    if n == 0:
        raise ValueError("render_wheel_timeline: no wheel samples to draw")

    h = (style.top_margin + len(WHEELS) * (style.lane_height + style.lane_gap)
         + style.bottom_margin)
    img = np.full((h, style.width, 3), style.background, np.uint8)
    plot_w = style.width - style.left_margin - 18
    x_of = lambda i: style.left_margin + int(round(plot_w * i / max(n, 1)))

    frames = [s.frame_index for s in series[WHEELS[0]]] if series.get(WHEELS[0]) else []

    cv2.putText(img, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                style.text, 1, cv2.LINE_AA)
    legend = [("inside", STATE_COLOR[WheelState.INSIDE]),
              ("on line (legal)", STATE_COLOR[WheelState.ON_LINE]),
              ("outside", STATE_COLOR[WheelState.OUTSIDE])]
    lx = style.width - 350
    legend = [(lbl, palette.get(st, col)) for (lbl, col), st in
              zip(legend, (WheelState.INSIDE, WheelState.ON_LINE, WheelState.OUTSIDE))]
    for label, col in legend:
        cv2.rectangle(img, (lx, 10), (lx + 16, 22), col, -1)
        cv2.putText(img, label, (lx + 22, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    style.dim, 1, cv2.LINE_AA)
        lx += 26 + 8 * len(label)

    for li, w in enumerate(WHEELS):
        y0 = style.top_margin + li * (style.lane_height + style.lane_gap)
        y1 = y0 + style.lane_height
        cv2.rectangle(img, (style.left_margin, y0), (style.left_margin + plot_w, y1),
                      (44, 42, 40), -1)
        if separator:
            # hard black rules between lanes: blocks, not a heat strip
            cv2.rectangle(img, (style.left_margin, y1),
                          (style.left_margin + plot_w, y1 + style.lane_gap),
                          (0, 0, 0), -1)
        cv2.putText(img, w, (16, y1 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    style.text, 1, cv2.LINE_AA)
        for i, s in enumerate(series.get(w, ())):
            col = palette[s.state] if s.measured else MISSING_COLOR
            cv2.rectangle(img, (x_of(i), y0), (max(x_of(i + 1) - 1, x_of(i)), y1),
                          col, -1)

    # Two different windows, drawn and labelled differently, because they are
    # two different durations and showing one number for both is how a demo
    # contradicts itself on stage:
    #   thin outer box  the EXCURSION -- three or more wheels beyond the limit
    #   solid inner box ALL FOUR OUTSIDE -- the only span that is a violation
    top = style.top_margin
    bottom = (style.top_margin + len(WHEELS) * (style.lane_height + style.lane_gap)
              - style.lane_gap)
    for ev in events or ():
        if not frames:
            continue
        try:
            a = frames.index(ev.entry_frame)
            b = frames.index(ev.exit_frame)
        except ValueError:
            continue

        cv2.rectangle(img, (x_of(a) - 1, top - 4), (x_of(b + 1) + 1, bottom + 4),
                      style.dim, 1, cv2.LINE_AA)

        four = [i for i in range(a, min(b + 1, n))
                if all(series.get(w) and i < len(series[w])
                       and series[w][i].state is WheelState.OUTSIDE for w in WHEELS)]
        label = f"EXCURSION {ev.duration_ms:.0f} ms"
        if four:
            f0, f1 = four[0], four[-1]
            cv2.rectangle(img, (x_of(f0) - 1, top - 4), (x_of(f1 + 1) + 1, bottom + 4),
                          style.box, 2, cv2.LINE_AA)
            peak = (f"beyond {abs(ev.peak_margin_mm):.0f} mm range"
                    if ev.peak_saturated else f"peak {abs(ev.peak_margin_mm):.0f} mm")
            label += f"   |   ALL FOUR OUTSIDE {ev.all_four_duration_ms:.0f} ms  {peak}"
        tx = min(max(x_of(a), 14), max(style.width - 8 * len(label) - 10, 14))
        cv2.putText(img, label, (tx, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    style.box if four else style.dim, 1, cv2.LINE_AA)

    if cursor_frame is not None and frames:
        try:
            cx = x_of(frames.index(cursor_frame))
            cv2.line(img, (cx, top - 6), (cx, bottom + 6), (255, 255, 255), 1, cv2.LINE_AA)
        except ValueError:
            pass

    if frames:
        for i in range(0, n, max(1, n // 10)):
            cv2.putText(img, str(frames[i]), (x_of(i) - 8, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, style.dim, 1, cv2.LINE_AA)
        cv2.putText(img, "frame", (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    style.dim, 1, cv2.LINE_AA)
    return img

"""Instrument widgets: bars, readouts, the verdict box, the incident log.

Everything here is deliberately inert.  Nothing animates, nothing fades, and
no widget redraws itself on a timer.  The console changes only when a frame
changes it, so movement on screen always means something moved in the data.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter,
                         QPixmap, QPolygon)
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QHeaderView, QLabel,
                             QScrollArea, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from chronos.ui import theme as T


def _family(css_stack: str) -> str:
    """First real family name out of a CSS font stack.

    QFont wants one family; the stylesheets carry a stack with fallbacks.
    Taking the head is correct because :func:`theme.load_fonts` has already
    registered the bundled face, so the head is the one that is actually
    present.
    """
    head = css_stack.split(",")[0].strip()
    return head.strip('"').strip("'")


def apply_block_shadow(widget) -> None:
    """The brutalist hard offset shadow: solid, no blur, down and right.

    A blurred shadow would read as depth; this reads as a printed block, which
    is the intent.  Does nothing on a theme that does not ask for it.
    """
    t = T.active()
    if t.shadow_px <= 0:
        return
    from PyQt6.QtWidgets import QGraphicsDropShadowEffect
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(0)
    fx.setOffset(t.shadow_px, t.shadow_px)
    fx.setColor(QColor(t.accent if t.name == "brutal" else t.edge))
    widget.setGraphicsEffect(fx)


def bgr_to_pixmap(img: Optional[np.ndarray]) -> Optional[QPixmap]:
    """BGR uint8 array to QPixmap.  Returns None for anything unusable."""
    if img is None or not isinstance(img, np.ndarray) or img.ndim != 3:
        return None
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888))


class ImagePane(QLabel):
    """Scales a BGR frame to fit, keeping aspect.  Blank until given one."""

    resized = pyqtSignal(int)

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        t = T.active()
        self.setStyleSheet(f"background: {t.ink}; "
                           f"border: {t.border_px}px solid {t.edge}; color: {t.dim};")
        self._pix: Optional[QPixmap] = None
        self._placeholder = placeholder
        if placeholder:
            self.setText(placeholder)

    def clear_frame(self, placeholder: str = "") -> None:
        """Drop the held frame.

        ``setText`` alone is not enough: ``_rescale`` re-applies ``_pix`` on
        the next resize, so a cleared pane would silently bring the previous
        session's image back the moment the window moved.  That is how a
        synthetic wheel-state timeline ended up displayed underneath real
        footage it had nothing to do with.
        """
        self._pix = None
        self.setText(placeholder or self._placeholder)

    def show_frame(self, img: Optional[np.ndarray]) -> None:
        pix = bgr_to_pixmap(img)
        if pix is None:
            return
        self._pix = pix
        self._rescale()

    # A QLabel reports its pixmap's size as its size hint.  Because the pixmap
    # is scaled to the label, the label could only ever grow: going fullscreen
    # locked the layout at fullscreen size, and the window then ran past the
    # screen edge.  Hints come from the fixed minimum instead.
    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.resized.emit(self.width())
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is None:
            return
        self.setText("")
        self.setPixmap(self._pix.scaled(self.size(),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))


class HazardBar(QWidget):
    """The single alert: a full-width hazard stripe when the reference fails.

    Static by design.  A pulsing or sliding bar would be the only moving thing
    on a console whose whole claim is that movement means data changed.
    """

    def __init__(self, height: int = 14):
        super().__init__()
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        t = T.active()
        p = QPainter(self)
        if not t.hazard:
            p.fillRect(self.rect(), QColor(t.review))
            p.end()
            return
        p.fillRect(self.rect(), QColor(t.ink))
        h, w, step = self.height(), self.width(), 22
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t.review))
        for x in range(-h, w + step, step):
            p.drawPolygon(QPolygon([QPoint(x, h), QPoint(x + h, 0),
                                    QPoint(x + h + step // 2, 0),
                                    QPoint(x + step // 2, h)]))
        p.end()


class Bar(QWidget):
    """A flat 0-100 bar.  No gradient, no rounding, no animation."""

    def __init__(self, height: int = 6):
        super().__init__()
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value: Optional[float] = None
        self._color = T.OFFWHITE

    def set_value(self, value: Optional[float], color: str = T.OFFWHITE) -> None:
        self._value = None if value is None else float(max(0.0, min(100.0, value)))
        self._color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(T.DIMMER))
        if self._value is not None:
            w = int(self.width() * self._value / 100.0)
            if w > 0:
                p.fillRect(0, 0, w, self.height(), QColor(self._color))
        p.end()


class Readout(QWidget):
    """A captioned value.  ``set_value(None)`` shows a dash, never a zero."""

    def __init__(self, caption: str, with_bar: bool = False, value_size: int = 15):
        self._size = value_size
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.caption = QLabel(caption.upper())
        self.caption.setObjectName("caption")
        self.value = QLabel("--")
        self.value.setObjectName("value")
        self.value.setStyleSheet(f"font-size: {value_size}px;")
        lay.addWidget(self.caption)
        lay.addWidget(self.value)
        self.bar = Bar() if with_bar else None
        if self.bar is not None:
            lay.addWidget(self.bar)

    def set_value(self, text: Optional[str], color: str = T.OFFWHITE,
                  bar: Optional[float] = None) -> None:
        self.value.setText("--" if text is None else text)
        self.value.setStyleSheet(
            f"color: {T.DIM if text is None else color}; font-size: {self._size}px;")
        if self.bar is not None:
            self.bar.set_value(bar, color)


class SubScores(QFrame):
    """The four components the integrity score is made of."""

    NAMES = ("contrast", "continuity", "sharpness", "contamination")

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(9)
        head = QLabel("COMPONENTS")
        head.setObjectName("caption")
        lay.addWidget(head)
        self.rows = {}
        for name in self.NAMES:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            label = QLabel(name)
            label.setStyleSheet(f"color: {T.DIM}; font-size: 11px;")
            label.setFixedWidth(96)
            value = QLabel("--")
            value.setStyleSheet(f"color: {T.DIM}; font-size: 12px;")
            value.setFixedWidth(30)
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            bar = Bar(5)
            h.addWidget(label)
            h.addWidget(value)
            h.addWidget(bar, 1)
            lay.addWidget(row)
            self.rows[name] = (value, bar)

    def update_scores(self, score) -> None:
        for name in self.NAMES:
            value, bar = self.rows[name]
            v = getattr(score, name, None) if score is not None else None
            if v is None:
                value.setText("--")
                value.setStyleSheet(f"color: {T.DIM}; font-size: 12px;")
                bar.set_value(None)
                continue
            col = T.GREEN if v >= 60 else T.AMBER if v >= 35 else T.RED
            value.setText(f"{v:.0f}")
            value.setStyleSheet(f"color: {col}; font-size: 12px;")
            bar.set_value(v, col)


class VerdictBox(QFrame):
    """The final call, what it was worth, and the one line of English behind it."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.label = QLabel("STANDBY")
        self.label.setObjectName("verdict")
        self.label.setStyleSheet(
            f"color: {T.DIM}; border: {T.active().border_px}px solid {T.EDGE}; "
            f"background: {T.PANEL};")
        lay.addWidget(self.label)

        # the event's OWN trust and integrity, named so they can never be read
        # as the live panel above them
        self.trust_line = QLabel("")
        self.trust_line.setObjectName("caption")
        lay.addWidget(self.trust_line)

        self.reason = QLabel("waiting for the first excursion")
        self.reason.setObjectName("reason")
        self.reason.setWordWrap(True)
        self.reason.setMinimumHeight(40)
        self.reason.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.reason)

    def set_verdict(self, outcome: Optional[str], reason: str, color: str,
                    trust: Optional[float] = None,
                    integrity: Optional[float] = None) -> None:
        t = T.active()
        self.label.setText(outcome or "STANDBY")
        if t.invert_verdict and outcome:
            # knocked out: black type on the state colour, like a sticker
            self.label.setStyleSheet(
                f"color: {t.ink}; border: {t.border_px}px solid {color}; "
                f"background: {color};")
        else:
            self.label.setStyleSheet(
                f"color: {color}; border: {t.border_px}px solid {color}; "
                f"background: {t.panel};")
        if trust is None:
            self.trust_line.setText("")
        else:
            integ = "--" if integrity is None else f"{integrity:.0f}"
            self.trust_line.setText(
                f"AT EVENT   TRUST {trust:.0%}   ·   INTEGRITY {integ}")
        self.reason.setText(reason or "")


class F1Badge(QWidget):
    """An "F1" mark beside the wordmark in the f1 theme.

    Drop the official logo at ``assets/f1_logo.png`` and it is drawn as is
    (kept out of the repo for the trade-mark reason given on Wordmark).
    Without that file, a plain "F1" is set in heavy italic brand red --
    typed text, not a copy of the protected roundel.
    """

    def __init__(self, height: int = 30):
        super().__init__()
        self._h = height
        self.setFixedHeight(height)
        self._logo: Optional[QPixmap] = None
        path = os.path.join(T.ASSETS, "f1_logo.png")
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self._logo = pix.scaledToHeight(
                    height, Qt.TransformationMode.SmoothTransformation)
        if self._logo is not None:
            self.setFixedWidth(self._logo.width())
        else:
            self.setFixedWidth(QFontMetrics(self._font()).horizontalAdvance("F1") + 10)
        self.setToolTip("Formula 1 track limits")

    def _font(self) -> QFont:
        f = QFont(_family(T.SANS), int(self._h * 0.72))
        f.setWeight(QFont.Weight.Black)
        f.setItalic(True)
        return f

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._logo is not None:
            p.drawPixmap(0, (self.height() - self._logo.height()) // 2, self._logo)
        else:
            p.setFont(self._font())
            p.setPen(QColor(T.active().mark or T.active().accent))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter
                       | Qt.AlignmentFlag.AlignLeft, "F1")
        p.end()


class Wordmark(QWidget):
    """The console's brand mark, in the F1 visual language.

    If ``assets/logo.png`` exists it is drawn instead.  That hook is here on
    purpose and the file is deliberately absent: the Formula 1 roundel is a
    registered trade mark of Formula One Licensing BV, and shipping it inside
    an MIT-licensed public repository is a liability the project does not
    need.  Dropping a file at that path is a decision the author gets to make
    knowingly, rather than one inherited from a default.

    The drawn mark uses the same grammar as the broadcast graphics -- angled
    speed slashes in brand red, heavy condensed type -- without borrowing any
    protected asset.
    """

    def __init__(self, text: str = "CHRONOS", height: int = 30):
        super().__init__()
        self.text = text
        self._h = height
        self.setFixedHeight(height)
        self._logo: Optional[QPixmap] = None
        path = os.path.join(T.ASSETS, "logo.png")
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self._logo = pix.scaledToHeight(
                    height, Qt.TransformationMode.SmoothTransformation)
        self.setFixedWidth(self._logo.width() + 8 if self._logo
                           else self._measure())

    def _measure(self) -> int:
        t = T.active()
        f = QFont(_family(t.label_font), int(self._h * 0.62))
        f.setWeight(QFont.Weight.Black)
        f.setItalic(t.slashes)
        return QFontMetrics(f).horizontalAdvance(self.text) + self._slash_w() + 16

    def _slash_w(self) -> int:
        return int(self._h * 1.05) if T.active().slashes else 0

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._logo is not None:
            p.drawPixmap(0, 0, self._logo)
            p.end()
            return
        t = T.active()
        h = self.height()
        x = 0
        if t.slashes:
            # three slashes, tightening and brightening to the right: the
            # motif the F1 header furniture is built from
            p.setPen(Qt.PenStyle.NoPen)
            skew, w = int(h * 0.42), int(h * 0.15)
            for i, alpha in enumerate((90, 160, 255)):
                col = QColor(t.mark or t.accent)
                col.setAlpha(alpha)
                p.setBrush(col)
                ox = i * int(w * 1.9)
                p.drawPolygon(QPolygon([
                    QPoint(ox + skew, 0), QPoint(ox + skew + w, 0),
                    QPoint(ox + w, h), QPoint(ox, h)]))
            x = self._slash_w()
        f = QFont(_family(t.label_font), int(h * 0.62))
        f.setWeight(QFont.Weight.Black)
        f.setItalic(t.slashes)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        p.setFont(f)
        p.setPen(QColor(t.bone))
        p.drawText(x, 0, self.width() - x, h,
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self.text)
        p.end()


class EvidenceCard(QFrame):
    """One captured frame in the evidence strip.  Click to review it.

    The card keeps the full-resolution overlay, not just the thumbnail, so
    reviewing an incident never depends on the engine's ring buffer still
    holding that frame.  The buffer is 240 frames; an incident from four
    minutes ago is still reviewable here.
    """

    clicked = pyqtSignal(int)

    THUMB_W, THUMB_H = 208, 117

    def __init__(self, image: np.ndarray, outcome: str, caption: str,
                 color: str, frame_index: int, has_reference: bool = True):
        super().__init__()
        self.frame_index = int(frame_index)
        self.image = image
        self.outcome = outcome
        self.caption = caption
        self.has_reference = has_reference
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # fixed width, or a long caption stretches the card and letterboxes
        # the thumbnail it is captioning
        self.setFixedWidth(self.THUMB_W)
        self.setToolTip(f"review frame {frame_index}" if has_reference else
                        f"frame {frame_index} — no reference, no measurement")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.thumb = QLabel()
        pix = bgr_to_pixmap(image)
        if pix is not None:
            self.thumb.setPixmap(pix.scaled(
                self.THUMB_W - 4, self.THUMB_H, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        self.thumb.setStyleSheet(
            f"border: 2px solid {color}; background: {T.INK};")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tag = QLabel(f"{outcome}\n{caption}")
        self.tag.setWordWrap(True)
        self.tag.setStyleSheet(
            f"color: {T.INK}; background: {color}; font-family: {T.MONO}; "
            f"font-size: 9px; font-weight: 700; padding: 3px 5px; "
            f"letter-spacing: 1px;")
        v.addWidget(self.thumb)
        v.addWidget(self.tag)

    def set_selected(self, on: bool) -> None:
        t = T.active()
        self.thumb.setStyleSheet(
            f"border: {3 if on else 2}px solid "
            f"{t.accent_2 if on else T.verdict_color(self.outcome)}; "
            f"background: {T.INK};")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit(self.frame_index)


class EvidenceStrip(QFrame):
    """The frames a verdict was actually made on.

    A verdict is a claim about one moment.  Showing the moment -- the frame at
    the deepest margin of the excursion, with the boundary and the four
    contact patches drawn on it -- is the difference between a number a
    steward can contest and a number they have to take on faith.

    Newest first.  Clicking a card jumps the console to that frame.

    Under NO REFERENCE the strip still fills, because what the camera saw is
    still evidence -- but the cards carry no margin and no integrity number,
    because there is no established boundary for either to have been measured
    against.  A number on a thumbnail is still a number we published.
    """

    jump = pyqtSignal(int)
    MAX = 8

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        self.head = QLabel("EVIDENCE  ·  FRAME AT PEAK MARGIN")
        self.head.setObjectName("caption")
        lay.addWidget(self.head)
        # Cards sit in a sideways scroll area.  Laid out directly, eight
        # 208 px cards made the strip ~1.7k px wide at minimum, which on a
        # laptop pushed the right-hand metrics panel off the window at the end
        # of a long video.
        holder = QWidget()
        holder.setObjectName("evidenceRow")
        holder.setStyleSheet("QWidget#evidenceRow { background: transparent; }")
        self.row = QHBoxLayout(holder)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self.scroll.viewport().setAutoFillBackground(False)
        self.scroll.setWidget(holder)
        lay.addWidget(self.scroll)
        self.empty = QLabel("no excursion captured yet")
        self.empty.setObjectName("label")
        self.row.addWidget(self.empty)
        self.row.addStretch(1)
        self._cards: list[EvidenceCard] = []
        self._fit_height()

    def _fit_height(self) -> None:
        """Tall enough for the cards plus the sideways scrollbar, no taller."""
        # measured from the cards themselves: the holder's own hint is cached
        # until the next event-loop pass, so it still reads "empty" right
        # after a card is added
        inner = 20
        for i in range(self.row.count()):
            w = self.row.itemAt(i).widget()
            if w is not None:
                w.ensurePolished()
                inner = max(inner, w.sizeHint().height(),
                            w.minimumSizeHint().height(), w.minimumHeight())
        bar = self.scroll.horizontalScrollBar().sizeHint().height()
        self.scroll.setFixedHeight(inner + max(bar, 10) + 4)

    HEADS = {
        "excursion": "EVIDENCE  ·  FRAME AT PEAK MARGIN",
        "stills": "EVIDENCE  ·  ONE SCORE PER IMAGE",
        "none": "EVIDENCE  ·  NO REFERENCE  ·  UNMEASURED FRAMES",
    }

    def set_mode(self, mode: str) -> None:
        """What the strip is currently holding: excursions, stills, or
        frames nothing could be measured on."""
        self.head.setText(self.HEADS.get(mode, self.HEADS["excursion"]))

    def set_no_reference(self, on: bool) -> None:
        self.set_mode("none" if on else "excursion")

    def cards(self) -> list[EvidenceCard]:
        """Newest first, for the export."""
        return list(self._cards)

    def clear(self) -> None:
        for c in self._cards:
            c.setParent(None)
        self._cards.clear()
        if self.empty is None:
            self.empty = QLabel("no excursion captured yet")
            self.empty.setObjectName("label")
            self.row.insertWidget(0, self.empty)
        self._fit_height()

    def add(self, image: np.ndarray, outcome: str, caption: str, color: str,
            frame_index: int = -1, has_reference: bool = True) -> None:
        if self.empty is not None:
            self.empty.setParent(None)
            self.empty = None
        card = EvidenceCard(image, outcome, caption, color, frame_index,
                            has_reference)
        card.clicked.connect(self.jump.emit)
        self.row.insertWidget(0, card)
        self._cards.insert(0, card)
        while len(self._cards) > self.MAX:
            self._cards.pop().setParent(None)
        self._fit_height()
        self.scroll.horizontalScrollBar().setValue(0)   # newest is on the left

    def card_for(self, frame_index: int) -> Optional[EvidenceCard]:
        for c in self._cards:
            if c.frame_index == frame_index:
                return c
        return None

    def select(self, frame_index: int) -> None:
        for c in self._cards:
            c.set_selected(c.frame_index == frame_index)


class AnalysisReport(QFrame):
    """The full analysis, on a key press.  Overlays; never a second window.

    Three kinds of number live here and each is labelled as such, because
    they are earned in completely different ways:

      * **session counters** are what this run did, and are true whatever the
        footage was;
      * **accuracy** is only meaningful where ground truth exists, which means
        synthetic footage and nowhere else;
      * the **false-confident rate** comes from the benchmark sweep rather
        than from one clip, because a rate over a single excursion is not a
        rate.

    The provenance line at the top exists so that none of the three can be
    quoted on stage without also stating what produced it.  Everything below
    the reference line is blanked to "--" when no boundary was established:
    the report will say what it could not measure, and will not fill the gap.
    """

    SECTIONS = (
        ("FRAME-BY-FRAME ANALYSIS", (
            ("cars", "an_cars"),
            ("frame verdicts", "an_verdicts"),
            ("violation frames", "an_viol_frames"),
            ("borderline frames", "an_border_frames"),
            ("margin to limit", "an_margin"),
            ("events", "an_events"),
            ("saved to", "an_output"),
        )),
        ("REFERENCE", (
            ("boundary status", "ref_status"),
            ("reason", "ref_reason"),
            ("channel agreement", "ref_agreement"),
            ("baseline coverage", "ref_coverage"),
            ("scale across track", "ref_scale"),
        )),
        ("BOUNDARY INTEGRITY", (
            ("current", "integ_now"),
            ("session mean", "integ_mean"),
            ("session minimum", "integ_min"),
            ("frames below alert (50)", "integ_below"),
            ("contamination applied", "integ_level"),
        )),
        ("SESSION", (
            ("frames processed", "frames"),
            ("race time covered", "race_s"),
            ("throughput", "fps"),
            ("excursions found", "events"),
            ("violations", "violations"),
            ("auto-cleared", "cleared"),
            ("escalated to review", "escalated"),
            ("auto-resolved", "resolved"),
        )),
        ("ACCURACY  ·  SYNTHETIC GROUND TRUTH ONLY", (
            ("mean margin error", "margin_mean"),
            ("p95 margin error", "margin_p95"),
            ("samples", "margin_n"),
        )),
        ("FALSE-CONFIDENT RATE  ·  BENCHMARK SWEEP", (
            ("baseline, ignores integrity", "fc_baseline"),
            ("CHRONOS, integrity-gated", "fc_chronos"),
            ("price: sent to review", "fc_abstain"),
        )),
    )

    #: Everything that is meaningless without an established boundary.  Under
    #: NO REFERENCE these are forced to "--" no matter what is in the data.
    BOUNDARY_DEPENDENT = frozenset({
        "ref_agreement", "ref_coverage", "ref_scale",
        "integ_now", "integ_mean", "integ_min", "integ_below",
        "events", "violations", "cleared", "escalated", "resolved",
        "margin_mean", "margin_p95", "margin_n",
    })

    def __init__(self):
        super().__init__()
        self.setObjectName("report")
        t = T.active()
        self.setStyleSheet(
            f"QFrame#report {{ background: {t.bg}; "
            f"border: 1px solid {t.accent_2}; border-radius: {t.radius_px}px; }}"
            f"QFrame#report QLabel {{ border: 0px; background: transparent; }}")
        self.setAutoFillBackground(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        hb = QHBoxLayout(head)
        hb.setContentsMargins(24, 16, 24, 12)
        title = QLabel("ANALYSIS REPORT")
        title.setStyleSheet(
            f"color: {t.bone}; font-family: {t.label_font}; font-size: 18px; "
            f"font-weight: 900; letter-spacing: 4px;")
        hb.addWidget(title)
        hb.addStretch(1)
        self.stamp = QLabel("")
        self.stamp.setStyleSheet(f"color: {t.dim}; font-family: {T.MONO}; "
                                 f"font-size: 11px;")
        hb.addWidget(self.stamp)
        outer.addWidget(head)

        # provenance: the line that keeps the numbers below it honest
        self.provenance = QLabel("Source: --")
        self.provenance.setWordWrap(True)
        self.provenance.setStyleSheet(
            f"color: {t.bone}; background: {t.panel_2}; font-family: {T.MONO}; "
            f"font-size: 11px; padding: 9px 24px; "
            f"border-top: 1px solid {t.edge}; border-bottom: 1px solid {t.edge};")
        outer.addWidget(self.provenance)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 14, 24, 14)
        lay.setSpacing(5)

        self.values: dict[str, QLabel] = {}
        for section, rows in self.SECTIONS:
            sec = QLabel(section)
            sec.setStyleSheet(
                f"color: {t.accent_2}; font-family: {t.label_font}; "
                f"font-size: 10px; font-weight: 700; letter-spacing: 3px; "
                f"border-bottom: 1px solid {t.edge}; padding-bottom: 4px;")
            lay.addSpacing(12)
            lay.addWidget(sec)
            lay.addSpacing(3)
            for label, key in rows:
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                name = QLabel(label)
                name.setStyleSheet(f"color: {t.dim}; font-size: 12px;")
                value = QLabel("--")
                value.setStyleSheet(f"color: {t.dim}; font-family: {T.MONO}; "
                                    f"font-size: 12px;")
                value.setAlignment(Qt.AlignmentFlag.AlignRight)
                value.setWordWrap(True)
                h.addWidget(name, 2)
                h.addStretch(1)
                h.addWidget(value, 3)
                lay.addWidget(row)
                self.values[key] = value
        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        foot = QLabel("R closes   ·   E exports to ./output/   ·   "
                      "click any evidence card to review that frame")
        foot.setStyleSheet(
            f"color: {t.dim}; font-family: {T.MONO}; font-size: 10px; "
            f"padding: 9px 24px; border-top: 1px solid {t.edge};")
        outer.addWidget(foot)

    # ------------------------------------------------------------------

    def update_report(self, data: dict) -> None:
        t = T.active()
        self.stamp.setText(data.get("stamp", ""))
        self.provenance.setText(data.get("provenance", "Source: --"))
        no_ref = not data.get("reference_ok", True)
        self.provenance.setStyleSheet(
            f"color: {t.ink if no_ref else t.bone}; "
            f"background: {t.review if no_ref else t.panel_2}; "
            f"font-family: {T.MONO}; font-size: 11px; font-weight: "
            f"{700 if no_ref else 400}; padding: 9px 24px; "
            f"border-top: 1px solid {t.edge}; border-bottom: 1px solid {t.edge};")
        for key, widget in self.values.items():
            text = data.get(key)
            if no_ref and key in self.BOUNDARY_DEPENDENT:
                text = None
            widget.setText("--" if text is None else str(text))
            colour = t.dim if text is None else t.bone
            if key == "ref_status":
                colour = t.review if no_ref else t.clear
            widget.setStyleSheet(f"color: {colour}; font-family: {T.MONO}; "
                                 f"font-size: 12px;")


class IncidentLog(QTableWidget):
    """Every verdict, in the order it was issued.  Nothing is ever removed."""

    COLUMNS = ("T", "CAR", "CORNER", "VERDICT", "TRUST@EVT", "INTEG@EVT", "REASON")

    def __init__(self):
        super().__init__(0, len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(True)
        self.setAlternatingRowColors(bool(T.active().row_tint))
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.horizontalHeader()
        for i in range(len(self.COLUMNS) - 1):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(self.COLUMNS) - 1,
                                    QHeaderView.ResizeMode.Stretch)
        self.setMinimumHeight(110)

    def add(self, t_ms: float, car_id, corner: str, outcome: str,
            trust: Optional[float], integrity: Optional[float], reason: str,
            color: str) -> None:
        r = self.rowCount()
        self.insertRow(r)
        cells = (f"{t_ms / 1000.0:7.2f}s", f"#{car_id}", corner, outcome,
                 "--" if trust is None else f"{trust:.0%}",
                 "--" if integrity is None else f"{integrity:.0f}",
                 reason)
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(QColor(color if c == 3 else T.OFFWHITE))
            self.setItem(r, c, item)
        self.scrollToBottom()

"""Swappable console themes, and the bundled typography they depend on.

Three looks, one switch:

``f1``           the broadcast look -- F1 carbon-black, brand red, Titillium.
                 The default, because this is what the room expects a
                 timing-and-scoring screen to look like.
``instrument``   the original race-control instrument: near-black, four
                 colours, thin rules, no ornament at all.
``brutal``       neo-brutalist race livery: hazard stripes, hard offset
                 shadows, knocked-out sticker headers, acid accents.

Keeping all three means a restyle is reversible.  If ``f1`` reads badly on
the projector in the room, ``--theme instrument`` returns the previous
console exactly, with nothing else touched.

--------------------------------------------------------------------------
Colour carries meaning, and only meaning
--------------------------------------------------------------------------
A console that uses colour decoratively cannot also use colour to say "do not
trust this", and saying that is the entire product:

    CLEAR      green    within the limit
    REVIEW     amber    touching the line, or a reference we cannot vouch for
    VIOLATION  red      beyond the limit

In the ``f1`` theme the brand red and the VIOLATION red are deliberately the
same value.  Red is the loudest thing the palette has, and spending it twice
-- once on decoration, once on the alert -- would blunt it.  So it is spent
once: the wordmark and a VIOLATION.  No other furniture in that theme is
permitted to be red, which is why the slider fill, the section rules and the
focus states all use the teal accent instead.

--------------------------------------------------------------------------
Fonts are bundled, not requested
--------------------------------------------------------------------------
``Impact`` and ``Arial Black`` are not on every machine, and a font fallback
discovered live on a projector is an avoidable way to lose a demo.  Every
face the console draws with ships in ``assets/fonts/`` under the SIL Open
Font License and is registered with Qt at startup by :func:`load_fonts`.
The CSS stacks still name system fallbacks after the bundled family, so a
missing file degrades instead of crashing.

Numbers are monospace in every theme without exception.  Labels may be heavy
condensed; figures may not, because figures have to stay aligned and legible
from the back of a room.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ALERT_INTEGRITY = 50.0

# --------------------------------------------------------------------------
# bundled typography
# --------------------------------------------------------------------------

ASSETS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "assets")
FONT_DIR = os.path.join(ASSETS, "fonts")

#: Bundled faces, in the order Qt should learn them.  Missing files are
#: skipped with a warning rather than raising -- the stacks below fall back.
FONT_FILES = (
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-SemiBold.ttf",
    "TitilliumWeb-Regular.ttf",
    "TitilliumWeb-SemiBold.ttf",
    "TitilliumWeb-Bold.ttf",
    "TitilliumWeb-Black.ttf",
    "ArchivoBlack-Regular.ttf",
    "Anton-Regular.ttf",
)

# The bundled family is always first; what follows it is only insurance.
MONO = ('"IBM Plex Mono", "SF Mono", "Menlo", "Monaco", "Consolas", '
        '"DejaVu Sans Mono", monospace')
HEAVY = ('"Archivo Black", "Anton", "Impact", "Haettenschweiler", '
         '"Arial Black", "Helvetica Neue", sans-serif')
#: Titillium is the face the Formula 1 wordmark and timing graphics are drawn
#: from, and it is the closest openly licensed match to the broadcast look.
SANS = ('"Titillium Web", "Archivo Black", "Helvetica Neue", "Segoe UI", '
        "sans-serif")

_LOADED: list[str] = []
_FONT_REPORT = "fonts not loaded yet"


def load_fonts() -> str:
    """Register the bundled faces with Qt.  Call once, before any widget.

    Imports Qt lazily: this module is also imported by :mod:`chronos.ui.timeline`,
    which renders with OpenCV for the pipeline CLI and must keep working on a
    machine with no PyQt6 installed.

    Returns a one-line report, which the console prints at startup so a
    missing face is visible immediately rather than at the worst moment.
    """
    global _FONT_REPORT
    try:
        from PyQt6.QtGui import QFontDatabase
    except ImportError:
        _FONT_REPORT = "PyQt6 absent; bundled fonts not registered"
        return _FONT_REPORT

    found, missing = [], []
    for name in FONT_FILES:
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        fid = QFontDatabase.addApplicationFont(path)
        if fid < 0:
            missing.append(name + " (rejected by Qt)")
            continue
        for fam in QFontDatabase.applicationFontFamilies(fid):
            if fam not in _LOADED:
                _LOADED.append(fam)
        found.append(name)

    _FONT_REPORT = f"fonts: {len(found)} bundled faces -> {', '.join(_LOADED)}"
    if missing:
        _FONT_REPORT += f"  |  MISSING {', '.join(missing)} (falling back)"
    return _FONT_REPORT


def font_report() -> str:
    return _FONT_REPORT


def loaded_families() -> list[str]:
    return list(_LOADED)


# --------------------------------------------------------------------------
# themes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """Every colour and metric the console draws with."""

    name: str
    bg: str
    panel: str
    panel_2: str
    ink: str
    edge: str
    bone: str
    dim: str
    dimmer: str
    accent: str          # structure, captions, the system speaking normally
    accent_2: str        # secondary accent
    clear: str           # within the limit
    review: str          # touching the line / degraded reference
    violation: str       # beyond the limit
    label_font: str
    huge_px: int
    verdict_px: int
    border_px: int
    shadow_px: int       # hard offset shadow; 0 disables
    hazard: bool         # hazard-stripe furniture
    invert_verdict: bool # verdict box filled with the state colour
    row_tint: str        # alternating incident-log rows; "" disables
    radius_px: int = 0   # panel corner radius
    pill_px: int = 0     # button corner radius
    mark: str = ""       # wordmark colour; "" means use accent
    slashes: bool = False  # the angled speed-slash motif in the header
    body_font: str = MONO  # prose and table text


INSTRUMENT = Theme(
    name="instrument",
    bg="#0D0D0F", panel="#111114", panel_2="#111114", ink="#0D0D0F",
    edge="#1F1F24", bone="#E8E6E3", dim="#6A6A70", dimmer="#3A3A40",
    accent="#6A6A70", accent_2="#6A6A70",
    clear="#3DD68C", review="#FFB000", violation="#E5484D",
    label_font=MONO, huge_px=78, verdict_px=21, border_px=1, shadow_px=0,
    hazard=False, invert_verdict=False, row_tint="",
)

BRUTAL = Theme(
    name="brutal",
    bg="#0A0A0A", panel="#101010", panel_2="#161616", ink="#000000",
    edge="#2A2A2A", bone="#F2F0EA", dim="#8A8A8A", dimmer="#242424",
    accent="#E8FF00", accent_2="#00F0FF",
    clear="#00FF9D", review="#FFB800", violation="#FF1F1F",
    label_font=HEAVY, huge_px=124, verdict_px=28, border_px=3, shadow_px=4,
    hazard=True, invert_verdict=True, row_tint="#141414",
)

#: Formula 1 broadcast livery.  ``#15151E`` is the carbon black the F1 site and
#: its timing graphics sit on; ``#E10600`` is the brand red.  Teal ``#00D2BE``
#: is the timing-screen accent and does the structural work so that red can
#: stay reserved for the wordmark and a VIOLATION.
F1 = Theme(
    name="f1",
    bg="#15151E", panel="#1E1E28", panel_2="#26262F", ink="#0B0B10",
    edge="#33333F", bone="#FFFFFF", dim="#9494A0", dimmer="#2C2C38",
    accent="#E10600", accent_2="#00D2BE",
    clear="#28C76F", review="#FFB800", violation="#E10600",
    label_font=SANS, huge_px=96, verdict_px=25, border_px=1, shadow_px=0,
    hazard=False, invert_verdict=True, row_tint="#1A1A24",
    radius_px=6, pill_px=16, mark="#E10600", slashes=True, body_font=SANS,
)

THEMES = {t.name: t for t in (F1, INSTRUMENT, BRUTAL)}
_ACTIVE = F1


def set_theme(name: str) -> Theme:
    """Choose the active theme.  Call before any widget is constructed."""
    global _ACTIVE
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; expected one of {sorted(THEMES)}")
    _ACTIVE = THEMES[name]
    return _ACTIVE


def active() -> Theme:
    return _ACTIVE


def verdict_color(outcome: str, integrity: float = None) -> str:
    """Colour for a verdict.  The outcome decides it, and nothing else.

    Amber means REVIEW REQUIRED and only that.  A degraded reference already
    produces REVIEW on its own, so recolouring a VIOLATION amber would make
    one colour say two different things and make the strongest verdict read
    as a hedge.
    """
    t = active()
    return {"VIOLATION": t.violation, "CLEAR": t.clear,
            "REVIEW REQUIRED": t.review, "NO REFERENCE": t.review}.get(outcome, t.review)


def stylesheet() -> str:
    """Qt stylesheet for the active theme."""
    t = _ACTIVE
    b = t.border_px
    r = t.radius_px
    f1 = t.name == "f1"
    brut = t.name == "brutal"
    # In f1 the caption chip is a thin teal rule over plain type; in brutal it
    # is a knocked-out sticker; in instrument it is bare dim text.
    head_bg = t.accent if brut else t.ink
    head_fg = t.ink if brut else (t.accent_2 if f1 else t.accent)
    return f"""
* {{
    font-family: {t.body_font};
    border-radius: 0px;
    outline: none;
}}
QWidget {{
    background: {t.bg};
    color: {t.bone};
}}
QFrame#panel {{
    background: {t.panel};
    border: {b}px solid {t.edge};
    border-radius: {r}px;
}}
QLabel#caption {{
    color: {head_fg};
    background: {'transparent' if f1 else head_bg};
    font-family: {t.label_font};
    font-size: {11 if f1 else 10}px;
    font-weight: {700 if f1 else 900};
    letter-spacing: {2 if f1 else 3}px;
    padding: {'3px 7px' if brut else '0px'};
}}
QLabel#label {{
    color: {t.dim};
    font-size: 11px;
    letter-spacing: 1px;
}}
QLabel#value {{
    color: {t.bone};
    font-family: {MONO};
    font-size: {17 if brut else 15}px;
    font-weight: {700 if brut else 400};
}}
QLabel#huge {{
    color: {t.bone};
    font-family: {MONO};
    font-size: {t.huge_px}px;
    font-weight: {900 if brut else 400};
}}
QLabel#verdict {{
    font-family: {t.label_font};
    font-size: {t.verdict_px}px;
    font-weight: {900 if brut or f1 else 400};
    letter-spacing: {5 if brut else (3 if f1 else 2)}px;
    padding: {'14px 16px' if brut else ('12px 14px' if f1 else '10px 12px')};
    border: {b}px solid {t.edge};
    border-radius: {r}px;
    background: {t.panel};
}}
QLabel#reason {{
    color: {t.bone if brut or f1 else t.dim};
    font-size: {13 if f1 else (12 if brut else 11)}px;
}}
QSlider::groove:horizontal {{
    background: {t.dimmer};
    border: {0 if t.name == 'instrument' else b}px solid {t.edge};
    border-radius: {r // 2}px;
    height: {12 if brut else (8 if f1 else 4)}px;
}}
QSlider::sub-page:horizontal {{
    background: {t.accent if brut else (t.accent_2 if f1 else t.review)};
    border-radius: {r // 2}px;
    height: {12 if brut else (8 if f1 else 4)}px;
}}
QSlider::handle:horizontal {{
    background: {t.accent_2 if brut else (t.bone if f1 else t.bone)};
    border: {0 if t.name == 'instrument' else b}px solid {t.ink};
    border-radius: {8 if f1 else 0}px;
    width: {22 if brut else (16 if f1 else 12)}px;
    height: {34 if brut else (16 if f1 else 20)}px;
    margin: {-13 if brut else (-5 if f1 else -9)}px 0;
}}
QTableWidget {{
    background: {t.panel};
    alternate-background-color: {t.row_tint or t.panel};
    gridline-color: {t.edge};
    border: {b}px solid {t.edge};
    border-radius: {r}px;
    font-family: {MONO};
    font-size: 11px;
    color: {t.bone};
}}
QHeaderView::section {{
    background: {t.ink};
    color: {t.accent if brut else (t.accent_2 if f1 else t.dim)};
    border: 0px;
    border-bottom: {max(b, 2) if f1 else b}px solid {t.accent if brut else (t.accent_2 if f1 else t.edge)};
    padding: {6 if brut else 5}px;
    font-family: {t.label_font};
    font-size: 10px;
    font-weight: {700 if f1 else 900};
    letter-spacing: 2px;
}}
QTableWidget::item {{ padding: {4 if brut else 3}px; }}
QPushButton {{
    background: {t.ink if brut else t.panel_2};
    border: {b}px solid {t.accent if brut else t.edge};
    border-radius: {t.pill_px}px;
    color: {t.accent if brut else t.bone};
    padding: {'8px 16px' if brut else ('7px 18px' if f1 else '6px 14px')};
    font-family: {t.label_font};
    font-size: 11px;
    font-weight: {700 if f1 else 900};
    letter-spacing: {2 if brut else 1}px;
}}
QPushButton:hover {{
    border: {b}px solid {t.accent if brut else (t.accent_2 if f1 else t.dim)};
    background: {t.accent if brut else (t.accent_2 if f1 else t.panel)};
    color: {t.ink if brut or f1 else t.bone};
}}
QPushButton:checked {{
    background: {t.accent_2 if f1 else t.accent};
    color: {t.ink};
    border: {b}px solid {t.accent_2 if f1 else t.accent};
}}
QScrollBar:vertical {{ background: {t.bg}; width: {10 if brut else 8}px; }}
QScrollBar::handle:vertical {{
    background: {t.accent if brut else t.edge};
    border-radius: {4 if f1 else 0}px;
}}
QScrollBar:horizontal {{ background: {t.bg}; height: {10 if brut else 8}px; }}
QScrollBar::handle:horizontal {{
    background: {t.accent if brut else t.edge};
    border-radius: {4 if f1 else 0}px;
    min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QToolTip {{
    background: {t.ink};
    color: {t.bone};
    border: 1px solid {t.accent_2 if f1 else t.edge};
    padding: 4px 7px;
}}
"""


# --- names kept for code that reads them directly -------------------------
def __getattr__(name: str):
    """Module-level colour names resolve against the ACTIVE theme.

    Lets widget code say ``theme.AMBER`` and get the right colour after a
    switch, without every module having to re-import on a theme change.
    """
    t = _ACTIVE
    table = {
        "BG": t.bg, "PANEL": t.panel, "PANEL_2": t.panel_2, "INK": t.ink,
        "EDGE": t.edge, "BONE": t.bone, "OFFWHITE": t.bone, "DIM": t.dim,
        "DIMMER": t.dimmer, "CYAN": t.accent_2, "ACCENT": t.accent,
        "ACCENT_2": t.accent_2, "LIME": t.clear, "GREEN": t.clear,
        "AMBER": t.review, "MAGENTA": t.violation, "RED": t.violation,
        "MARK": t.mark or t.accent,
        "STYLESHEET": stylesheet(),
    }
    if name in table:
        return table[name]
    raise AttributeError(f"module 'chronos.ui.theme' has no attribute {name!r}")

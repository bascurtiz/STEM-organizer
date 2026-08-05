"""Theme tokens + QFluentWidgets integration for the PySide6 port.

Color tokens stay available for programmatic painting (logs, stems, inline
styles). The global look comes from qfluentwidgets (setTheme / setThemeColor).
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication, QWidget

from .musical_keys import CHART_KEY_COLORS  # noqa: F401 — Charts / LOG badges
# ---------------------------------------------------------------------------
# Color tokens — copied verbatim from ui_theme.COLORS / DARK
# ---------------------------------------------------------------------------

COLORS = {
    "bg":            "#1e1f26",
    "panel":         "#262833",
    "panel2":        "#2F3140",
    "fg":            "#e6e8ef",
    "fg_dim":        "#9aa0b4",
    "accent":        "#7c5cff",
    "accent_hov":    "#9077ff",
    "danger":        "#e25c5c",
    "log_bg":        "#15161c",
    "log_fg":        "#d6dae8",
    "border":        "#3a3d4d",
    "status_trough": "#343647",
    "status_pct":    "#d6dae8",  # soft white (= log_fg); not Fluent pure #ffffff
}

DARK = {
    "bg":              COLORS["bg"],
    "panel":           COLORS["panel"],
    "panel_2":         COLORS["panel2"],
    "card":            COLORS["panel2"],
    "input":           COLORS["panel"],
    "control_bg":      COLORS["panel2"],
    "border":          COLORS["border"],
    "border_soft":     COLORS["status_trough"],
    "scrollbar":       "#2a2c38",
    "scrollbar_hover": "#3a3d4d",
    "text":            COLORS["log_fg"],
    "text_dim":        COLORS["fg_dim"],
    "label":           COLORS["log_fg"],
    "text_mute":       "#7a8199",
    "entry_text":      COLORS["fg"],
    "accent":          COLORS["accent"],
    "accent_hover":    COLORS["accent_hov"],
    "accent_soft":     "#2a2540",
    "active_row":      "#44485f",
    "loading_bg":      "#2a2540",
    "changed":         COLORS["accent_hov"],
    "unchanged":       COLORS["fg_dim"],
    "list_bg":         COLORS["log_bg"],
    "list_fg":         COLORS["log_fg"],
    "waveform_bg":     COLORS["log_bg"],
    "waveform_axis":   "#343647",
    "waveform_playhead": "#ffffff",  # 1px scrubber — keep pure white for punch
    "audio":           "#10b981",
    "midi":            COLORS["accent"],
    "group":           "#a855f7",
    "danger":          COLORS["danger"],
    "badge_fg":        COLORS["log_fg"],  # soft white (#d6dae8)
    "btn":             COLORS["panel2"],
    "btn_hover":       COLORS["border"],
    "category_colors": {},
    # legacy aliases
    "fg":              COLORS["log_fg"],
    "fg_dim":          COLORS["fg_dim"],
    "panel2":          COLORS["panel2"],
    "accent_hov":      COLORS["accent_hov"],
    "log_bg":          COLORS["log_bg"],
    "log_fg":          COLORS["log_fg"],
    "status_trough":   COLORS["status_trough"],
}

# Stem colors (port of stem_player.STEM_COLORS)
STEM_COLORS = {
    "bass":         "#ef4444",
    "drums":        "#f59e0b",
    "other":        "#10b981",
    "vocals":       "#a855f7",
    "acapella":     "#a855f7",
    "vocal":        "#a855f7",
    "guitar":       "#376FAC",
    "piano":        COLORS["log_fg"],  # #d6dae8 — same fill as dry badge
    "instrumental": "#60A5FA",
    "original":     "#9aa0b4",
    "mixture":      "#9aa0b4",
}

# LOG gender/reverb chips (log_panel.LOG_GG_COLORS) — charts reuse these.
GG_CHIP_COLORS = {
    "female": "#ec4899",
    "male":   "#60A5FA",
    "dry":    COLORS["log_fg"],   # #d6dae8 — also Genre chips
    "wet":    "#262833",          # also Style chips (dark fill)
}

# Dataset overview chart fills (aliases into STEM / GG chips).
CHART_ROLE_COLORS = {
    "Instrumental": "#7C5CFF",
    # Same purple as vocal-type donut base / vocals waveform.
    "Vocal": STEM_COLORS["vocals"],
    "Samples": "#5C4EA0",
    "instrumental": "#7C5CFF",
    "vocal": STEM_COLORS["vocals"],
    "vocals": STEM_COLORS["vocals"],
    "samples": "#5C4EA0",
    "sample": "#5C4EA0",
}
CHART_GENDER_COLORS = {
    "female": GG_CHIP_COLORS["female"],
    "male": GG_CHIP_COLORS["male"],
}
CHART_REVERB_COLORS = {
    "dry": GG_CHIP_COLORS["dry"],
    "wet": GG_CHIP_COLORS["wet"],
}
# Genre uses dry chip; Style uses wet chip (same as LOG GENRE/STYLE flush).
CHART_GENRE_COLOR = GG_CHIP_COLORS["dry"]
CHART_STYLE_COLOR = GG_CHIP_COLORS["wet"]
CHART_VOCAL_TYPE_COLOR = STEM_COLORS["vocals"]  # #a855f7 — waveform / acapella


def _lighten_hex(hex_color: str, toward_white: float) -> str:
    """Blend ``hex_color`` toward white; 0 = unchanged, 1 = white."""
    c = QColor(hex_color)
    t = max(0.0, min(1.0, float(toward_white)))
    return QColor(
        int(c.red() + (255 - c.red()) * t),
        int(c.green() + (255 - c.green()) * t),
        int(c.blue() + (255 - c.blue()) * t),
    ).name()


# Vocal-type donut/badges: Roles chart Vocal purple as Singing, then +22% each step.
_VOCAL_BASE = CHART_ROLE_COLORS["Vocal"]  # same as STEM_COLORS["vocals"] / Roles
_VOCAL_STEP = 0.22
CHART_VOCAL_TYPE_COLORS = {
    "Singing": _lighten_hex(_VOCAL_BASE, 0.00),
    "Speech": _lighten_hex(_VOCAL_BASE, _VOCAL_STEP),
    "Rapping": _lighten_hex(_VOCAL_BASE, _VOCAL_STEP * 2),
    "Humming": _lighten_hex(_VOCAL_BASE, _VOCAL_STEP * 3),
    "Choir": _lighten_hex(_VOCAL_BASE, _VOCAL_STEP * 4),
}

# Dataset genre bars — same 16 custom swatches as Rename's QColorDialog.
# (track_renamer.category_palette.CATEGORY_PALETTE_COLORS[:16])
CHART_GENRE_PALETTE: tuple[str, ...] = (
    "#EF4444",
    "#F59E0B",
    "#A855F7",
    "#10B981",
    "#8D95A4",
    "#C1090B",
    "#C41D63",
    "#8351A1",
    "#5C4EA0",
    "#485FAB",
    "#398BCB",
    "#00B8D3",
    "#25BAA2",
    "#3FB655",
    "#76C043",
    "#B0D236",
)

# ---------------------------------------------------------------------------
# Layout constants — port of ui_theme numeric vocabulary
# ---------------------------------------------------------------------------

PATH_BTN_WIDTH_BROWSE = 72
PATH_BTN_WIDTH_OPEN = 64
PATH_BTN_HEIGHT = 26
COMPACT_BTN_HEIGHT = 24
ACTION_BTN_GAP = 4
# Gap between adjacent radio options (Match / Genre / Gender). CTk used 28; roomier here.
RADIO_OPTION_GAP = 40

SECTION_PADX = (10, 2)
SECTION_INNER_PAD = 6
SECTION_GAP = 6  # compact enough to avoid left-panel scrollbar

# Match LogPanel horizontal inset (log view ↔ window edge on the right).
# Use the same value as each tab's left page margin so content lines up.
PAGE_EDGE_INSET = 8
# Rename nests page + panel insets (8+8). Other tabs use this total L/R margin.
PAGE_CONTENT_INSET = PAGE_EDGE_INSET * 2

CTRL_ROW_PADY = 1
CTRL_FIELD_PAD = 2

WIN_DEFAULT_W = 1280
WIN_DEFAULT_H = 1018
WIN_MIN_W = 1040
WIN_MIN_H = 720
# Usable tab width after PAGE_CONTENT_INSET on both sides; was 560 before the inset bump.
LEFT_PANEL_WIDTH = 580

TITLE_BAR_HEIGHT = 36
WINDOW_CORNER_RADIUS = 12  # match CTk frameless SetWindowRgn clip
DIALOG_CORNER_RADIUS = 14
CARD_CORNER_RADIUS = 10
LOG_VIEW_CORNER_RADIUS = 10

# Solid fill for Fluent LineEdit / ComboBox / PushButton / SpinBox.
# Fluent QSS uses rgba(255,255,255,0.0605) which blends to ~#33353F on panel;
# override to this opaque CTk-like control surface.
CONTROL_BG = COLORS["panel2"]  # #2F3140
CONTROL_BG_HOVER = "#36384A"
CONTROL_BG_PRESSED = "#2A2C38"
# Focused LineEdit fill — Fluent's dark rgba focus reads as a gray flash; pin to window bg
INPUT_FOCUS_BG = COLORS["bg"]  # #1e1f26

# Log tag colors — match CTk stem_organizer_ui / ui_theme
LOG_OK_COLOR = "#7ee0a0"
LOG_ERR_COLOR = "#ff7a7a"
LOG_WARN_COLOR = "#ecc990"
LOG_DELETED_COLOR = "#e89292"
LOG_MARGIN_COLOR = "#9aa0b4"
LOG_PCT_COLOR = COLORS["log_fg"]  # soft white — same as body / Browse·Open labels
SDR_PASS_COLOR = "#7ee0a0"
SDR_FAIL_COLOR = "#ff7a7a"
SDR_LABEL_COLOR = COLORS["log_fg"]
GG_CONF_COLOR = "#7ee0a0"
GG_CONF_LOW_COLOR = "#e0b07a"

CHART_COMPRESSION_COLORS = {
    "lossless": LOG_OK_COLOR,
    "lossy": LOG_ERR_COLOR,
}


STATUS_FRAME_HEIGHT = 66
STATUS_PAD_TOP = 3
STATUS_ROW_GAP = 7
STATUS_PAD_BOTTOM = 26
STATUS_TOP_GAP = 12  # space above Idle / status bar (below action buttons)
ACTION_ROW_TOP_GAP = 14  # breathing room above Start/Stop/Clear/Save (below class card + log)
# Rename: air above waveform / Cancel / Rename — shortens RULES + PREVIEW stretch.
RENAME_PLAYER_TOP_GAP = 22
# Status bar is full-bleed (outside content’s 6px margin). Pad so Idle / device
# line up with tab content (content 6 + PAGE_CONTENT_INSET).
STATUS_PAD_X = 6 + PAGE_CONTENT_INSET
STATUS_PROGRESS_ROW_HEIGHT = 14
STATUS_PROGRESS_HEIGHT = 14
STATUS_IDLE_Y = STATUS_PAD_TOP + 16 + STATUS_ROW_GAP  # 26
RESOURCE_BAR_HEIGHT = 10
RESOURCE_BAR_WIDTH = 52
RESOURCE_ROW_HEIGHT = 16

STATUS_LINK_URL = "https://github.com/bascurtiz/STEM-organizer"
APP_VERSION = "1.0.8"

# ---------------------------------------------------------------------------
# Fonts — match D:\STEM-organizer\ui_theme.py (CTk sizes are pixels)
# ---------------------------------------------------------------------------

FONT_FAMILY = "Segoe UI"
FONT_FAMILY_SEMIBOLD = "Segoe UI Semibold"
FONT_FAMILY_MONO = "Consolas"

# CTk reference (ui_theme.py):
#   HEADER_DESC_FONT / ctk_ui_font  = ('Segoe UI', 12)   # body + tab desc
#   SECTION_TITLE_SIZE              = 10                 # PATHS / OPTIONS …
#   ACTION_BTN_FONT                 = ('Segoe UI Semibold', 10)
#   PATH_BTN_FONT                   = ('Segoe UI', 10)
#   STATUS_FONT                     = ('Segoe UI', 9)
#   STATUS_PCT_FONT                 = ('Segoe UI Semibold', 9)
#   LOG_FONT / PREVIEW_LOG          = Consolas 10 / 12
BODY_FONT_PX = 12
SECTION_TITLE_PX = 10
ACTION_BTN_FONT_PX = 12  # footer Start/Stop/Play — was too small at 10
PATH_BTN_FONT_PX = 12    # Browse/Open + match LineEdit/ComboBox body size
STATUS_FONT_PX = 12  # CTk STATUS_FONT=('Segoe UI', 9) is points ≈ 12px
STATUS_PCT_FONT_PX = 11
LOG_FONT_PX = 12
LOG_PCT_FONT_PX = 12
PREVIEW_LOG_FONT_PX = 12
DIALOG_FONT_PX = 13
HEADER_DESC_FONT_PX = 12
# CTk Tooltip uses Tk ('Segoe UI', 9) which is *points* ≈ 12px at 96 DPI
TOOLTIP_FONT_PX = 12

# Do not treat these as sentence ends when splitting tooltips (e.g. "e.g. Artist").
_TOOLTIP_ABBREVS = ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Mrs.", "Dr.", "St.")


def format_tooltip(text: str) -> str:
    """Put each sentence of a tooltip on its own line (Qt respects ``\\n``).

    Single-sentence tips are unchanged. Inserts a newline after ``.`` / ``?`` /
    ``!`` followed by space(s), skipping common abbreviations. Idempotent when
    tips already use ``\\n`` between sentences.
    """
    if not text or not any(m in text for m in (". ", "? ", "! ")):
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".?!" and i + 1 < n and text[i + 1] == " ":
            abbrev = False
            if ch == ".":
                for ab in _TOOLTIP_ABBREVS:
                    start = i + 1 - len(ab)
                    if start >= 0 and text[start : i + 1] == ab:
                        abbrev = True
                        break
            if not abbrev:
                out.append(ch)
                out.append("\n")
                i += 2
                while i < n and text[i] == " ":
                    i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


PATH_FIELD_HEIGHT = 28
COMPACT_SPIN_HEIGHT = 26  # dense rows; CompactSpinButton must match (Fluent default is 33)
COMPACT_SPIN_WIDTH = 64  # duration / batch / analysis spins (was 58; earlier 68, 72–88)
ACTION_BTN_HEIGHT = 30
TITLE_ICON_SIZE = 22  # match CTk TITLE_ICON_SIZE
TITLE_LABEL_FONT_PX = 13  # CTk title Label uses Segoe UI 10pt ≈ 13px

# Sub-tab bar (Match/Align, Genre/Gender, Integrity) breathing room: a few px of gap between
# the two pills and a top shift so the row isn't flush against the page top.
SUBTAB_SPACING = 6
SUBTAB_TOP_PAD = 4


def _px_font(family: str, px: int, weight: int = QFont.Normal) -> QFont:
    """Build a QFont with pixel size (CTk CTkFont.size is pixels, not points)."""
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setWeight(weight)
    return f


F_BODY = _px_font(FONT_FAMILY, BODY_FONT_PX)
F_BODY_BOLD = _px_font(FONT_FAMILY, BODY_FONT_PX, QFont.DemiBold)
F_SECTION = _px_font(FONT_FAMILY, SECTION_TITLE_PX, QFont.DemiBold)
F_SECTION.setLetterSpacing(QFont.AbsoluteSpacing, 1)
F_STATUS = _px_font(FONT_FAMILY, STATUS_FONT_PX)
F_STATUS_PCT = _px_font(FONT_FAMILY_SEMIBOLD, STATUS_PCT_FONT_PX, QFont.DemiBold)
F_LOG = _px_font(FONT_FAMILY_MONO, LOG_FONT_PX)
F_LOG_PCT = _px_font(FONT_FAMILY_MONO, LOG_PCT_FONT_PX)
F_DIALOG = _px_font(FONT_FAMILY, DIALOG_FONT_PX)
F_HEADER_DESC = _px_font(FONT_FAMILY, HEADER_DESC_FONT_PX)
F_ACTION = _px_font(FONT_FAMILY_SEMIBOLD, ACTION_BTN_FONT_PX, QFont.DemiBold)
F_PATH_BTN = _px_font(FONT_FAMILY, PATH_BTN_FONT_PX)
F_PREVIEW_LOG = _px_font(FONT_FAMILY_MONO, PREVIEW_LOG_FONT_PX)


def _qcolor(hex_or_token: str) -> QColor:
    if not hex_or_token:
        return QColor(COLORS["bg"])
    if hex_or_token.startswith("#"):
        return QColor(hex_or_token)
    return QColor(COLORS.get(hex_or_token, hex_or_token))


def qcolor(token: str) -> QColor:
    """Resolve a DARK/COLORS token to QColor."""
    if token in DARK:
        return QColor(DARK[token])
    if token in COLORS:
        return QColor(COLORS[token])
    return QColor(token)


def style_color_dialog(dlg: QWidget) -> None:
    """Dark-theme QSS for Qt's non-native QColorDialog (readable buttons + inputs)."""
    from PySide6.QtWidgets import QColorDialog

    if isinstance(dlg, QColorDialog):
        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)

    t = DARK
    c = COLORS
    dlg.setStyleSheet(
        f"""
        QColorDialog {{
            background-color: {t['panel']};
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
            font-size: {BODY_FONT_PX}px;
        }}
        QColorDialog QLabel {{
            background: transparent;
            color: {t['text_dim']};
        }}
        QColorDialog QPushButton {{
            background-color: {CONTROL_BG};
        color: {t['text']};
        border: 1px solid {t['border']};
            border-radius: 6px;
            padding: 6px 14px;
            min-height: {ACTION_BTN_HEIGHT - 4}px;
        }}
        QColorDialog QPushButton:hover {{
            background-color: {CONTROL_BG_HOVER};
            color: {t['fg']};
        }}
        QColorDialog QPushButton:pressed {{
            background-color: {CONTROL_BG_PRESSED};
        }}
        QColorDialog QPushButton:default {{
            background-color: {c['accent']};
            border: 1px solid {c['accent_hov']};
        color: {t['text']};
            font-weight: 600;
        }}
        QColorDialog QPushButton:default:hover {{
            background-color: {c['accent_hov']};
            color: {t['text']};
        }}
        QColorDialog QLineEdit,
        QColorDialog QAbstractSpinBox {{
            background-color: {CONTROL_BG};
        color: {t['text']};
        border: 1px solid {t['border']};
            border-radius: 5px;
            padding: 2px 6px;
            min-height: 22px;
            selection-background-color: {c['accent']};
            selection-color: {c['log_fg']};
        }}
        QColorDialog QLineEdit:focus,
        QColorDialog QAbstractSpinBox:focus {{
            background-color: {INPUT_FOCUS_BG};
        border: 1px solid {t['border']};
    }}
        QColorDialog QAbstractSpinBox::up-button,
        QColorDialog QAbstractSpinBox::down-button {{
            background-color: {CONTROL_BG_HOVER};
        border: none;
            width: 16px;
        }}
        QColorDialog QAbstractSpinBox::up-button:hover,
        QColorDialog QAbstractSpinBox::down-button:hover {{
            background-color: {c['accent']};
        }}
        """
    )


def style_line_edit(edit: QWidget, *, idle_bg: str | None = None) -> None:
    """Pin LineEdit colors — solid focus fill (#1e1f26); keep Fluent accent underline."""
    # Prefix fields with a category color strip own their sheet; don't clobber it.
    if edit.property("hasColorStrip") or edit.property("hasKeywordsFill"):
        return
    from qfluentwidgets import setCustomStyleSheet

    t = DARK
    bg = idle_bg if idle_bg is not None else CONTROL_BG
    hover = CONTROL_BG_HOVER if idle_bg is None else COLORS["panel2"]
    # Fluent ships LineEdit with transparent=true and a focus rule
    # `LineEdit:focus[transparent=true] { background: rgba(30,30,30,0.7) }`
    # (~#1e1e20). Force opaque fill so our idle/focus colors win.
    edit.setProperty("transparent", False)
    sheet = f"""
    LineEdit, LineEdit[transparent=false] {{
        background: {bg};
        background-color: {bg};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 5px;
        padding: 0px 8px;
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
        min-height: 24px;
        max-height: {PATH_FIELD_HEIGHT}px;
    }}
    LineEdit:hover, LineEdit[transparent=false]:hover {{
        background: {hover};
        background-color: {hover};
    }}
    LineEdit:focus, LineEdit:focus[transparent=false], LineEdit[transparent=false]:focus {{
        background: {INPUT_FOCUS_BG};
        background-color: {INPUT_FOCUS_BG};
        border: 1px solid {t['border']};
    }}
    """
    edit.setFont(F_BODY)
    edit.setStyleSheet(sheet)
    setCustomStyleSheet(edit, sheet, sheet)
    # Re-evaluate property selectors after transparent flip
    edit.style().unpolish(edit)
    edit.style().polish(edit)
    edit.update()


# ---------------------------------------------------------------------------
# App-specific QSS leftovers (custom chrome Fluent does not cover)
# ---------------------------------------------------------------------------

def build_app_overrides_qss() -> str:
    """Force dark surfaces for stock QWidgets Fluent does not restyle."""
    t = DARK
    c = COLORS
    return f"""
    /* Stock containers stay dark; Fluent widgets keep their own QSS. */
    QMainWindow, QWidget#AppRoot {{
        background-color: {c['bg']};
        color: {t['text']};
    }}
    QWidget#TitleBar {{
        background-color: {c['bg']};
        border-bottom: 1px solid {t['border']};
        color: {t['text']};
    }}
    QScrollArea {{
        background-color: {c['bg']};
        border: none;
    }}
    QScrollArea > QWidget {{
        background-color: {c['bg']};
    }}
    /* No seam between Rename tab page and PATH — the Rename body divider
       (RenameBodyDivider) is painted the page background color (#1e1f26),
       so the 4-px split between the rules and preview panels is invisible. */
    TabWidget, TabWidget > QWidget, QStackedWidget {{
        border: none;
        background-color: transparent;
    }}

    /* Native Qt scrollbars (log / leftovers) — dark thumb like CTk */
    QScrollBar:vertical {{
        background: {c['log_bg']};
        width: 8px;
        margin: 0px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {t['scrollbar']};
        min-height: 28px;
        border-radius: 4px;
        margin: 1px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {t['scrollbar_hover']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
        height: 0px;
    }}
    /* Rules panel uses the Fluent overlay scrollbar (same as the Preview
       table); its native bar is hidden, so no RulesScroll bar QSS is needed. */
    QScrollArea#RulesScroll {{
        border: none;
        background: transparent;
    }}
    QScrollBar:horizontal {{
        background: {c['log_bg']};
        height: 10px;
        margin: 0px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {t['scrollbar']};
        min-width: 28px;
        border-radius: 4px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {t['scrollbar_hover']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
        width: 0px;
    }}
    QSplitter::handle {{
        background-color: {t['border']};
    }}
    QLabel {{
        background: transparent;
        color: {t['text']};
    }}
    QLabel#Dim {{ color: {t['text_dim']}; background: transparent; }}
    QLabel#Link {{ color: {t['text_dim']}; background: transparent; }}
    QLabel#Link:hover {{ color: {t['fg']}; }}
    QLabel#HeaderDesc {{ color: {t['text_dim']}; font-size: {HEADER_DESC_FONT_PX}px; background: transparent; }}
    QLabel#ChartNote {{ color: {t['text_dim']}; font-size: {HEADER_DESC_FONT_PX}px; background: transparent; }}

    /* Native Qt tooltips — match CTk Tooltip (log_bg fill, log_fg text) */
    QToolTip {{
        background-color: {c['log_bg']};
        color: {c['log_fg']};
        border: 1px solid {c['border']};
        padding: 8px 12px;
        font-family: "{FONT_FAMILY}";
        font-size: {TOOLTIP_FONT_PX}px;
    }}

    /* Fluent ToolTip widgets (ToolTipFilter) — same CTk colors */
    ToolTip > #container {{
        background-color: {c['log_bg']};
        border: 1px solid {c['border']};
        border-radius: 4px;
    }}
    ToolTip QLabel#contentLabel {{
        color: {c['log_fg']};
        background: transparent;
        font-family: "{FONT_FAMILY}";
        font-size: {TOOLTIP_FONT_PX}px;
    }}

    HeaderCardWidget {{
        background-color: {t['panel']};
        border: 1px solid {t['border']};
        border-radius: {CARD_CORNER_RADIUS}px;
    }}
    HeaderCardWidget #headerLabel {{
        color: {t['text']};
    }}
    HeaderCardWidget > #headerView,
    HeaderCardWidget > #view {{
        background-color: transparent;
    }}

    QFrame#Section {{
        background: transparent;
        border: none;
    }}
    QFrame#Card {{
        background-color: {t['panel']};
        border: 1px solid {t['border']};
        border-radius: {CARD_CORNER_RADIUS}px;
    }}
    QLabel#SectionTitle, CaptionLabel#SectionTitle {{
        color: {t['text_dim']};
        font-family: "{FONT_FAMILY}";
        font-size: {SECTION_TITLE_PX}px;
        font-weight: 600;
        padding: 0px;
        margin: 0px;
        background: transparent;
    }}

    /* Soft CTk-like text (not pure white) + 12px body — Fluent defaults are
       white @ 14px which reads larger/harsher than the original. */
    FluentLabelBase, BodyLabel, StrongBodyLabel, SubtitleLabel {{
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
    }}
    CaptionLabel {{
        color: {t['text_dim']};
        font-family: "{FONT_FAMILY}";
        font-size: {SECTION_TITLE_PX}px;
    }}
    QCheckBox, QRadioButton, CheckBox, RadioButton {{
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
        spacing: 10px;
        padding: 0px;
        min-height: 18px;
        min-width: 0px;
    }}
    CheckBox::indicator, RadioButton::indicator,
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
    }}
    LineEdit {{
        min-height: 24px;
        max-height: {PATH_FIELD_HEIGHT}px;
        padding: 0px 8px;
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
        background-color: {CONTROL_BG};
    }}
    LineEdit:hover {{
        background-color: {CONTROL_BG_HOVER};
    }}
    LineEdit:focus {{
        background-color: {INPUT_FOCUS_BG};
        border: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']};
    }}
    ComboBox {{
        min-height: 24px;
        max-height: {PATH_FIELD_HEIGHT}px;
        padding: 0px 8px;
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
        background-color: {CONTROL_BG};
    }}
    ComboBox:hover {{
        background-color: {CONTROL_BG_HOVER};
    }}
    ComboBox:pressed {{
        background-color: {CONTROL_BG_PRESSED};
    }}
    SpinBox, DoubleSpinBox, CompactSpinBox, CompactDoubleSpinBox {{
        min-width: {COMPACT_SPIN_WIDTH}px;
        color: {t['text']};
        font-family: "{FONT_FAMILY}";
        font-size: {BODY_FONT_PX}px;
        background-color: {CONTROL_BG};
    }}
    SpinBox:hover, DoubleSpinBox:hover, CompactSpinBox:hover, CompactDoubleSpinBox:hover {{
        background-color: {CONTROL_BG_HOVER};
    }}
    PushButton, TransparentPushButton {{
        font-family: "{FONT_FAMILY}";
        font-size: {PATH_BTN_FONT_PX}px;
        color: {t['text']};
        background-color: {CONTROL_BG};
    }}
    PushButton:hover, TransparentPushButton:hover {{
        color: {t['text']};
        background-color: {CONTROL_BG_HOVER};
    }}
    PushButton:pressed, TransparentPushButton:pressed {{
        color: {t['text']};
        background-color: {CONTROL_BG_PRESSED};
    }}
    /* Disabled: keep Play/control fill — gray text alone shows state (no gray pill). */
    PushButton:disabled, TransparentPushButton:disabled,
    PrimaryPushButton:disabled, ToolButton:disabled {{
        background-color: {CONTROL_BG};
        background: {CONTROL_BG};
        border: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']};
    }}
    PushButton:disabled:hover, TransparentPushButton:disabled:hover,
    PrimaryPushButton:disabled:hover, ToolButton:disabled:hover {{
        background-color: {CONTROL_BG};
        background: {CONTROL_BG};
    }}
    PrimaryPushButton {{
        font-family: "{FONT_FAMILY}";
        font-size: {ACTION_BTN_FONT_PX}px;
        color: {t['text']};
    }}
    PrimaryPushButton:hover, PrimaryPushButton:pressed,
    PrimaryPushButton:focus {{
        color: {t['text']};
    }}
    PrimaryToolButton, ToggleButton:checked, ToggleToolButton:checked {{
        color: {t['text']};
    }}
    PrimaryToolButton:hover, PrimaryToolButton:pressed,
    ToggleButton:checked:hover, ToggleButton:checked:pressed,
    ToggleToolButton:checked:hover, ToggleToolButton:checked:pressed {{
        color: {t['text']};
    }}
    SegmentedWidget {{
        min-height: {ACTION_BTN_HEIGHT}px;
        max-height: {ACTION_BTN_HEIGHT}px;
    }}

    /* Action bar strip */
    QFrame#ActionBar, QWidget#ActionBarPage {{
        background-color: {c['bg']};
    }}

    /* Pin dark log look — no Fluent accent underline on focus */
    PlainTextEdit#LogView,
    PlainTextEdit#LogView:hover,
    PlainTextEdit#LogView:focus,
    TextEdit#LogView,
    TextEdit#LogView:hover,
    TextEdit#LogView:focus {{
        background-color: {c['log_bg']};
        color: {c['log_fg']};
        font-family: "{FONT_FAMILY_MONO}";
        font-size: {LOG_FONT_PX}px;
        border: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']};
        border-radius: {LOG_VIEW_CORNER_RADIUS}px;
        padding: 8px;
        selection-background-color: {c['accent']};
        selection-color: {c['log_fg']};
    }}

    /* Fluent MessageBox center panel */
    QFrame#centerWidget {{
        border-radius: {DIALOG_CORNER_RADIUS}px;
    }}
    """


def build_stylesheet() -> str:
    """Deprecated alias for leftover app QSS."""
    return build_app_overrides_qss()


def _apply_dark_palette(app: QApplication) -> None:
    """Match Fluent DARK tokens on the Qt palette so stock widgets aren't light."""
    from PySide6.QtGui import QPalette

    pal = QPalette()
    window = QColor(COLORS["bg"])
    panel = QColor(COLORS["panel"])
    text = QColor(DARK["text"])
    muted = QColor(DARK["text_mute"])
    accent = QColor(COLORS["accent"])
    base = QColor(COLORS["log_bg"])

    pal.setColor(QPalette.Window, window)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, panel)
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, panel)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, QColor(COLORS["danger"]))
    pal.setColor(QPalette.Highlight, accent)
    pal.setColor(QPalette.HighlightedText, QColor(COLORS["log_fg"]))
    # Tooltips — CTk Tooltip uses log_bg #15161c, border #3a3d4d, text log_fg #d6dae8.
    # Set all color groups so Windows native tips don't keep a white Inactive base.
    tip_bg = QColor(COLORS["log_bg"])
    tip_fg = QColor(COLORS["log_fg"])
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        pal.setColor(group, QPalette.ToolTipBase, tip_bg)
        pal.setColor(group, QPalette.ToolTipText, tip_fg)
    pal.setColor(QPalette.PlaceholderText, muted)
    pal.setColor(QPalette.Link, accent)
    app.setPalette(pal)


def _underline_width_for_text(font_metrics, text: str, *, minimum: float = 32.0) -> float:
    """Underline length matching label text (same rule as main tabs)."""
    return max(float(font_metrics.horizontalAdvance(text)), minimum)


# Shared status look for RMS/SI-SDR SegmentedWidget and top TabWidget items.
# Idle = dim text + soft pill; hover (unselected) = stronger pill; active = white + accent underline.
STATUS_PILL_RADIUS = 6
# Even gap between main top-tab pills; raised further so Charts meets card right edge.
MAIN_TAB_SPACING = 8


def _status_pill_color() -> QColor:
    """Idle pill fill used by SegmentedItem and top tabs."""
    from qfluentwidgets.common.style_sheet import isDarkTheme

    return QColor(255, 255, 255, 9) if isDarkTheme() else QColor(0, 0, 0, 9)


def _status_hover_pill_color() -> QColor:
    """Unselected hover pill — Fluent TabItem / SegmentedItem hover strength."""
    from qfluentwidgets.common.style_sheet import isDarkTheme

    # Dark: Fluent TabItem hover is alpha 15; SegmentedItem hover is alpha 9 over
    # a transparent idle — we keep idle at 9 and lift hover to 15 so it brightens.
    return QColor(255, 255, 255, 15) if isDarkTheme() else QColor(0, 0, 0, 10)


def _status_idle_fg() -> QColor:
    """Unselected label color (DARK text_dim / COLORS fg_dim)."""
    return QColor(DARK["text_dim"])


def _status_active_fg() -> QColor:
    """Selected label color — soft white (log_fg), not Fluent pure #ffffff."""
    from qfluentwidgets.common.style_sheet import isDarkTheme

    return QColor(COLORS["log_fg"]) if isDarkTheme() else QColor("#000000")


def _status_pill_css() -> str:
    """QSS rgba() for the idle pill (dark-theme app default)."""
    pill = _status_pill_color()
    return f"rgba({pill.red()}, {pill.green()}, {pill.blue()}, {pill.alpha()})"


def sync_compact_spin_button(spin, height: int | None = None) -> None:
    """Keep Fluent CompactSpinButton inside the field (default button is 33px tall)."""
    # Always prefer COMPACT_SPIN_HEIGHT — spin.height() is Fluent's 33 until we lock it,
    # and using that here left the field taller than equal-height filter rows (uneven gaps).
    h = COMPACT_SPIN_HEIGHT if height is None else int(height)
    if h <= 0:
        h = COMPACT_SPIN_HEIGHT
    spin.setMinimumHeight(h)
    spin.setMaximumHeight(h)
    spin.setFixedHeight(h)
    btn = getattr(spin, "compactSpinButton", None)
    if btn is not None:
        btn.setFixedSize(max(16, h - 8), h)


def _install_compact_spin_resize_sync(spin) -> None:
    """Re-sync chevron button whenever the spin is resized."""
    import types

    if getattr(spin, "_compact_spin_synced", False):
        return
    spin._compact_spin_synced = True
    _prev = spin.resizeEvent

    def _resize(self, event) -> None:
        _prev(event)
        sync_compact_spin_button(self, COMPACT_SPIN_HEIGHT)

    spin.resizeEvent = types.MethodType(_resize, spin)


def configure_segmented_widget(seg) -> None:
    """RMS / SI-SDR: half-width items, text underline on active; idle/hover pills on inactive."""
    import types

    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter
    from PySide6.QtWidgets import QSizePolicy, QWidget
    from qfluentwidgets import setCustomStyleSheet

    accent = QColor(COLORS["accent"])
    if hasattr(seg, "setIndicatorColor"):
        seg.setIndicatorColor(accent, accent)

    # Equal half-width segments across the card
    seg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    lay = getattr(seg, "hBoxLayout", None)
    if lay is not None:
        for i in range(lay.count()):
            lay.setStretch(i, 1)
            w = lay.itemAt(i).widget()
            if w is not None:
                w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    # Inactive: transparent item; parent paints idle/hover pill. Active = white + underline.
    dim = DARK["text_dim"]
    r = STATUS_PILL_RADIUS
    sheet = f"""
        SegmentedItem[isSelected=false] {{
            color: {dim};
            background-color: transparent;
            border: none;
            border-radius: {r}px;
        }}
        SegmentedItem[isSelected=false]:hover,
        SegmentedItem[isSelected=false]:pressed,
        SegmentedItem[isSelected=false]:focus {{
            color: {dim};
            background-color: transparent;
        }}
        SegmentedItem[isSelected=true],
        SegmentedItem[isSelected=true]:hover,
        SegmentedItem[isSelected=true]:pressed,
        SegmentedItem[isSelected=true]:focus {{
            color: {DARK["text"]};
            background-color: transparent;
            border: none;
        }}
    """
    setCustomStyleSheet(seg, sheet, sheet)

    def _paint_item_chrome(item, prev_paint):
        def _paint(self, e) -> None:
            selected = bool(getattr(self, "isSelected", False))
            if selected:
                # Paint recessed selected chrome ourselves so Fluent/QSS :hover
                # wash cannot flatten the pushed-in look until the mouse leaves.
                painter = QPainter(self)
                painter.setRenderHints(QPainter.Antialiasing)
                painter.setPen(Qt.NoPen)
                # Darker than idle soft pills → reads as inset / pushed-in
                painter.setBrush(QColor(COLORS["bg"]))  # #1E1F26 pushed-in
                painter.drawRoundedRect(
                    QRectF(self.rect()).adjusted(1, 3, -1, -3),
                    STATUS_PILL_RADIUS,
                    STATUS_PILL_RADIUS,
                )
                painter.setPen(_status_active_fg())
                painter.setFont(self.font())
                painter.drawText(self.rect(), Qt.AlignCenter, self.text())
                painter.setPen(Qt.NoPen)
                painter.setBrush(accent)
                uw = _underline_width_for_text(self.fontMetrics(), self.text())
                x = (self.width() - uw) / 2.0
                painter.drawRoundedRect(
                    QRectF(x, self.height() - 4.0, uw, 3.0), 1.5, 1.5
                )
                return

            prev_paint(e)

        item.paintEvent = types.MethodType(_paint, item)

    def _hook_hover_refresh(item, prev_enter, prev_leave, prev_press, prev_release):
        def _enter(self, e) -> None:
            prev_enter(e)
            seg.update()

        def _leave(self, e) -> None:
            prev_leave(e)
            seg.update()

        def _press(self, e) -> None:
            prev_press(e)
            seg.update()

        def _release(self, e) -> None:
            prev_release(e)
            # Selection applies on release — force chrome refresh immediately
            # (don't wait for mouse-leave to reveal pushed-in).
            for it in seg.items.values():
                it.update()
            seg.update()

        item.enterEvent = types.MethodType(_enter, item)
        item.leaveEvent = types.MethodType(_leave, item)
        item.mousePressEvent = types.MethodType(_press, item)
        item.mouseReleaseEvent = types.MethodType(_release, item)

    for item in getattr(seg, "items", {}).values():
        setCustomStyleSheet(item, sheet, sheet)
        item.setStyle(item.style())
        item.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        item.setFocusPolicy(Qt.NoFocus)  # click selects without stealing focus chrome
        item.setCursor(Qt.CursorShape.PointingHandCursor)
        _paint_item_chrome(item, item.paintEvent)
        _hook_hover_refresh(
            item, item.enterEvent, item.leaveEvent, item.mousePressEvent, item.mouseReleaseEvent
        )

    def _paint_seg(self, e) -> None:
        QWidget.paintEvent(self, e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        current = self.currentItem()
        for item in self.items.values():
            if item is current:
                # Selected pill also on parent (behind label) for slide/resize frames
                painter.setBrush(QColor(COLORS["bg"]))  # #1E1F26 pushed-in
                painter.drawRoundedRect(
                    item.geometry().adjusted(1, 3, -1, -3),
                    STATUS_PILL_RADIUS,
                    STATUS_PILL_RADIUS,
                )
                continue
            pill = (
                _status_hover_pill_color()
                if getattr(item, "isHover", False) or getattr(item, "isPressed", False)
                else _status_pill_color()
            )
            painter.setBrush(pill)
            painter.drawRoundedRect(
                item.geometry().adjusted(1, 3, -1, -3), STATUS_PILL_RADIUS, STATUS_PILL_RADIUS
            )

    seg.paintEvent = types.MethodType(_paint_seg, seg)

    def _refresh(*_args) -> None:
        seg.update()
        for it in seg.items.values():
            it.update()

    try:
        seg.slideAni.valueChanged.connect(_refresh)
    except Exception:
        pass
    seg.currentItemChanged.connect(_refresh)
    seg.update()


def configure_tab_widget(tabs) -> None:
    """Hide Fluent TabWidget close/+ chrome; action-button height; underline active tab.

    Status look mirrors configure_segmented_widget (RMS / SI-SDR):
    idle = dim text + soft pill; unselected hover = stronger pill; active = white + underline.
    """
    import types

    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QSizePolicy
    from qfluentwidgets import TabCloseButtonDisplayMode

    tabs.setCloseButtonDisplayMode(TabCloseButtonDisplayMode.NEVER)
    bar = getattr(tabs, "tabBar", None)
    if bar is None:
        return
    if hasattr(bar, "setAddButtonVisible"):
        bar.setAddButtonVisible(False)

    # No purple fill — transparent selected background
    transparent = QColor(0, 0, 0, 0)
    if hasattr(bar, "setTabSelectedBackgroundColor"):
        bar.setTabSelectedBackgroundColor(transparent, transparent)
    if hasattr(bar, "setTabShadowEnabled"):
        bar.setTabShadowEnabled(False)

    accent = QColor(COLORS["accent"])
    h = ACTION_BTN_HEIGHT
    bar.setFixedHeight(h)
    item_layout = getattr(bar, "itemLayout", None)
    if item_layout is not None:
        item_layout.setContentsMargins(0, 0, 0, 0)
        # Spacing set after tab widths are known (main tabs align Charts to cards).
        item_layout.setSpacing(MAIN_TAB_SPACING)
        # Fluent insertWidget(..., stretch=1) + Minimum size policy makes tabs
        # grow with the bar. Keep items content-sized and park leftover width
        # in a trailing stretch so Rename full-width matches other tabs.
        for i in range(item_layout.count()):
            item = item_layout.itemAt(i)
            w = item.widget() if item is not None else None
            if w is not None:
                item_layout.setStretch(i, 0)
                w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            elif item is not None and item.spacerItem() is not None:
                item_layout.setStretch(i, 1)
        # Ensure a single trailing stretch absorbs leftover width
        has_trailing_stretch = False
        if item_layout.count() > 0:
            last = item_layout.itemAt(item_layout.count() - 1)
            has_trailing_stretch = (
                last is not None
                and last.spacerItem() is not None
                and item_layout.stretch(item_layout.count() - 1) > 0
            )
        if not has_trailing_stretch:
            item_layout.addStretch(1)

    def _paint_underline_tab(item, _event) -> None:
        painter = QPainter(item)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if not item.isSelected:
            # Idle soft pill; hover/press brightens like SegmentedItem unselected
            hovering = getattr(item, "isHover", False) or getattr(item, "isPressed", False)
            pill = _status_hover_pill_color() if hovering else _status_pill_color()
            painter.setBrush(pill)
            painter.drawRoundedRect(
                item.rect().adjusted(1, 3, -1, -3), STATUS_PILL_RADIUS, STATUS_PILL_RADIUS
            )
            color = _status_idle_fg()
        else:
            color = _status_active_fg()
        painter.setPen(QPen(color))
        painter.setFont(item.font())
        text_rect = item.rect().adjusted(4, 0, -4, -4)
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignVCenter, item.text())
        if item.isSelected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(accent)
            uw = _underline_width_for_text(painter.fontMetrics(), item.text())
            x = (item.width() - uw) / 2.0
            painter.drawRoundedRect(QRectF(x, item.height() - 4.0, uw, 3.0), 1.5, 1.5)

    tab_items = list(getattr(bar, "items", []) or [])
    for item in tab_items:
        try:
            item.setFixedHeight(h)
            # Content-sized tabs (Fluent sizeHint uses maximumWidth → 240px)
            text_w = item.fontMetrics().horizontalAdvance(item.text()) + 20
            item.setMinimumWidth(text_w)
            item.setMaximumWidth(text_w)
            item.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            item.setSelectedBackgroundColor(transparent, transparent)
            item.setShadowEnabled(False)
            if hasattr(item, "setBorderRadius"):
                item.setBorderRadius(STATUS_PILL_RADIUS)
            item.setCursor(Qt.CursorShape.PointingHandCursor)
            item.paintEvent = types.MethodType(_paint_underline_tab, item)
            item.update()
        except Exception:
            pass

    # Main tabs: inset both sides to match cards, even gaps so Charts meets
    # the right card edge. Nested sub-tabs keep MAIN_TAB_SPACING here;
    # inset_tab_bar overwrites with SUBTAB_SPACING + its own left pad.
    if item_layout is not None and len(tab_items) >= 2:
        labels = {(it.text() or "").strip() for it in tab_items}
        if "Charts" in labels and "Classify" in labels:
            # 2px left of cards so Classify sits slightly proud; right still
            # matches card edge. Gaps even across the remaining span.
            left_inset = PAGE_CONTENT_INSET - 2
            item_layout.setContentsMargins(left_inset, 0, PAGE_CONTENT_INSET, 0)
            widths_sum = sum(it.minimumWidth() for it in tab_items)
            target = LEFT_PANEL_WIDTH - left_inset - PAGE_CONTENT_INSET
            gaps = len(tab_items) - 1
            spacing = max(MAIN_TAB_SPACING, round((target - widths_sum) / gaps))
            item_layout.setSpacing(int(spacing))


def inset_tab_bar(tabs, left: int | None = None) -> None:
    """Pad the tab bar so labels share the page content left edge.

    Sub-tabs (Match/Align, Genre/Gender, Integrity) sit above scroll content that already
    uses PAGE_CONTENT_INSET; without this the first tab sticks out left of the
    HeaderDesc line.

    Layout only — paint/idle/hover/active come from configure_tab_widget (same
    status-pill look as the top tabs). Also gives the two pills breathing room
    (SUBTAB_SPACING) and nudges the row down (SUBTAB_TOP_PAD).
    """
    if left is None:
        left = PAGE_CONTENT_INSET
    bar = getattr(tabs, "tabBar", None)
    item_layout = getattr(bar, "itemLayout", None) if bar is not None else None
    if item_layout is None:
        return
    # Grow the bar so the top padding doesn't clip the fixed-height items
    # (configure_tab_widget pins bar + items to ACTION_BTN_HEIGHT).
    bar.setFixedHeight(ACTION_BTN_HEIGHT + SUBTAB_TOP_PAD)
    item_layout.setContentsMargins(left, SUBTAB_TOP_PAD, 0, 0)
    item_layout.setSpacing(SUBTAB_SPACING)


def style_toggle_button(btn) -> None:
    """Opaque idle + accent when checked; pointing-hand cursor.

    ToggleButton subclasses PushButton, so the neutral PushButton polish sheet
    (especially ``:focus``) would override Fluent ``:checked`` accent fill —
    equal specificity, later rule wins. Call this for mode toggles (Instrument
    source, Only changed) instead of the PushButton control sheet.
    """
    from PySide6.QtCore import Qt
    from qfluentwidgets import setCustomStyleSheet
    from qfluentwidgets.common.font import setFont as fluent_set_font

    soft = DARK["text"]
    accent = COLORS["accent"]
    sheet = f"""
    ToggleButton {{
        background-color: {CONTROL_BG};
        color: {soft};
    }}
    ToggleButton:hover {{
        background-color: {CONTROL_BG_HOVER};
        color: {soft};
    }}
    ToggleButton:pressed {{
        background-color: {CONTROL_BG_PRESSED};
        color: {soft};
    }}
    ToggleButton:focus {{
        background-color: {CONTROL_BG};
        color: {soft};
    }}
    ToggleButton:checked,
    ToggleButton:checked:hover,
    ToggleButton:checked:pressed,
    ToggleButton:checked:focus {{
        background-color: {accent};
        color: {soft};
    }}
    ToggleButton:checked:hover {{
        background-color: {COLORS['accent_hov']};
    }}
    """
    fluent_set_font(btn, PATH_BTN_FONT_PX)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    setCustomStyleSheet(btn, "", sheet)


def polish_fluent_controls(root: QWidget) -> None:
    """Match CTk scale/tone: 12px Segoe UI + soft #d6dae8 text (not Fluent 14px white).

    Fluent widgets call setFont(..., 14) and hardcode white text at construction.
    Walk the tree once after UI build and re-apply CTk tokens.
    """
    from PySide6.QtCore import Qt
    from qfluentwidgets import (
        BodyLabel,
        CaptionLabel,
        CheckBox,
        ComboBox,
        CompactDoubleSpinBox,
        CompactSpinBox,
        DoubleSpinBox,
        LineEdit,
        PrimaryPushButton,
        PushButton,
        RadioButton,
        SpinBox,
        StrongBodyLabel,
        ToggleButton,
        setCustomStyleSheet,
    )
    from qfluentwidgets.common.font import setFont as fluent_set_font
    from qfluentwidgets.components.widgets.scroll_bar import ScrollBar

    soft = DARK["text"]
    accent = COLORS["accent"]

    # Fluent overlay scrollbars default to bright white thumbs — darken to CTk tones
    handle = QColor(DARK["scrollbar"])
    handle_hov = QColor(DARK["scrollbar_hover"])
    groove = QColor(COLORS["log_bg"])
    for bar in root.findChildren(ScrollBar):
        bar.setHandleColor(handle_hov, handle)
        bar.setGrooveColor(groove, groove)
        bar.setArrowColor(handle_hov, handle)

    # Opaque control surfaces — Fluent rgba white overlays blend to ~#33353F on panel
    def _control_sheet(selector: str) -> str:
        focus_bg = INPUT_FOCUS_BG if selector == "LineEdit" else CONTROL_BG
        return f"""
        {selector} {{
            background-color: {CONTROL_BG};
        }}
        {selector}:hover {{
            background-color: {CONTROL_BG_HOVER};
        }}
        {selector}:pressed {{
            background-color: {CONTROL_BG_PRESSED};
        }}
        {selector}:focus {{
            background-color: {focus_bg};
        }}
        {selector}:disabled {{
            background-color: {CONTROL_BG};
            background: {CONTROL_BG};
        }}
        {selector}:disabled:hover {{
            background-color: {CONTROL_BG};
            background: {CONTROL_BG};
        }}
        """

    def _primary_button_sheet() -> str:
        # Accent buttons: soft label on purple (Fluent defaults to black text).
        # Disabled fill matches neutral Play; gray text alone shows state.
        soft_fg = DARK["text"]
        return f"""
        PrimaryPushButton {{
            color: {soft_fg};
        }}
        PrimaryPushButton:hover {{
            color: {soft_fg};
        }}
        PrimaryPushButton:pressed {{
            color: {soft_fg};
        }}
        PrimaryPushButton:focus {{
            color: {soft_fg};
        }}
        PrimaryPushButton:disabled {{
            background-color: {CONTROL_BG};
            background: {CONTROL_BG};
            border: 1px solid {DARK['border']};
            border-bottom: 1px solid {DARK['border']};
        }}
        PrimaryPushButton:disabled:hover {{
            background-color: {CONTROL_BG};
            background: {CONTROL_BG};
        }}
        """

    hand = Qt.CursorShape.PointingHandCursor
    dim = DARK["text_dim"]
    for cb in root.findChildren(CheckBox):
        fluent_set_font(cb, BODY_FONT_PX)
        cb.setTextColor(soft, soft)
        cb.setCheckedColor(accent, accent)
        # setTextColor() only sets the enabled color — without :disabled the
        # label stays bright when setEnabled(False). Match CTk text_dim grey-out.
        soft_hex = QColor(soft).name(QColor.NameFormat.HexArgb)
        dim_hex = QColor(dim).name(QColor.NameFormat.HexArgb)
        sheet = (
            f"CheckBox{{color:{soft_hex};}}"
            f"CheckBox:disabled{{color:{dim_hex};}}"
        )
        setCustomStyleSheet(cb, sheet, sheet)
        cb.setCursor(hand)

    for rb in root.findChildren(RadioButton):
        fluent_set_font(rb, BODY_FONT_PX)
        if hasattr(rb, "setTextColor"):
            rb.setTextColor(soft, soft)
        rb.setCursor(hand)

    for lbl in root.findChildren(BodyLabel):
        if isinstance(lbl, CaptionLabel):
            continue
        # CTk HEADER_DESC_COLOR = DARK['text_dim'] (#9aa0b4)
        if lbl.objectName() in ("HeaderDesc", "ChartNote"):
            fluent_set_font(lbl, HEADER_DESC_FONT_PX)
            if hasattr(lbl, "setTextColor"):
                lbl.setTextColor(DARK["text_dim"], DARK["text_dim"])
            continue
        # How it works cards — same dim as RadioHint (AudioTester-style)
        if lbl.objectName() == "HowItWorks":
            fluent_set_font(lbl, BODY_FONT_PX)
            if hasattr(lbl, "setTextColor"):
                lbl.setTextColor(DARK["text_dim"], DARK["text_dim"])
            continue
        # "Instrument source" (Rename category table) — muted like the
        # samplepack "Found labels (N):" header so section labels match.
        if lbl.objectName() == "InstrumentSourceLbl":
            fluent_set_font(lbl, BODY_FONT_PX)
            if hasattr(lbl, "setTextColor"):
                lbl.setTextColor(DARK["text_dim"], DARK["text_dim"])
            continue
        fluent_set_font(lbl, BODY_FONT_PX)
        if hasattr(lbl, "setTextColor"):
            lbl.setTextColor(soft, soft)

    for lbl in root.findChildren(StrongBodyLabel):
        fluent_set_font(lbl, BODY_FONT_PX, QFont.DemiBold)
        if hasattr(lbl, "setTextColor"):
            lbl.setTextColor(soft, soft)

    for lbl in root.findChildren(CaptionLabel):
        fluent_set_font(lbl, SECTION_TITLE_PX, QFont.DemiBold)
        if hasattr(lbl, "setTextColor"):
            lbl.setTextColor(DARK["text_dim"], DARK["text_dim"])

    for combo in root.findChildren(ComboBox):
        fluent_set_font(combo, BODY_FONT_PX)
        setCustomStyleSheet(combo, "", _control_sheet("ComboBox"))

    # LineEdit — full sheet beats Fluent focus fill + purple underline paint.
    # Skip renamer prefix strips and category keyword fields (own sheets).
    for edit in root.findChildren(LineEdit):
        if edit.property("hasColorStrip") or edit.property("hasKeywordsFill"):
            continue
        style_line_edit(edit)

    seen_spins: set[int] = set()
    for cls, sel in (
        (CompactSpinBox, "CompactSpinBox"),
        (CompactDoubleSpinBox, "CompactDoubleSpinBox"),
        (SpinBox, "SpinBox"),
        (DoubleSpinBox, "DoubleSpinBox"),
    ):
        for spin in root.findChildren(cls):
            sid = id(spin)
            if sid in seen_spins:
                continue
            seen_spins.add(sid)
            sheet = _control_sheet(sel)
            # Fluent default is 33px + thick bottom border. Compact fields: uniform
            # border, no QSS min-height (it fights setFixedHeight → min>max → clip).
            if sel.startswith("Compact"):
                sheet += f"""
                {sel} {{
                    border: 1px solid {DARK['border']};
                    border-bottom: 1px solid {DARK['border']};
                    border-radius: 5px;
                    padding: 0px 20px 0 6px;
                }}
                """
                setCustomStyleSheet(spin, "", sheet)
                sync_compact_spin_button(spin, COMPACT_SPIN_HEIGHT)
                _install_compact_spin_resize_sync(spin)
            else:
                setCustomStyleSheet(spin, "", sheet)

    for btn in root.findChildren(PushButton):
        # Hand cursor on all Fluent action buttons (incl. custom-chrome skips).
        btn.setCursor(hand)
        # Keep custom chrome (keyword × uses setStyleSheet; danger only on hover)
        if btn.objectName() in (
            "KeywordRemove",
            "TitleClose",
            "TitleBtn",
            "RuleDelete",
            "RuleAdd",
            "RenameApply",
        ):
            continue
        # ToggleButton subclasses PushButton — must not get the neutral PushButton
        # sheet (:focus would override checked accent). Own sheet + hand cursor.
        if isinstance(btn, ToggleButton):
            style_toggle_button(btn)
            continue
        px = ACTION_BTN_FONT_PX if isinstance(btn, PrimaryPushButton) else PATH_BTN_FONT_PX
        fluent_set_font(btn, px, QFont.DemiBold if isinstance(btn, PrimaryPushButton) else QFont.Normal)
        # Accent PrimaryPushButton: white label on purple; disabled = Play fill
        if isinstance(btn, PrimaryPushButton):
            setCustomStyleSheet(btn, "", _primary_button_sheet())
            continue
        sheet = _control_sheet("PushButton")
        # Preserve action_button(danger=True) idle text color — not buttons that
        # only mention the danger hex in a :hover background rule.
        prev = (btn.styleSheet() or "").replace(" ", "").lower()
        danger_idle = f"color:{DARK['danger']}".lower()
        if danger_idle in prev and ":hover" in prev:
            # If color:danger appears only inside hover blocks, still OK to add;
            # require it outside hover by checking a simple idle rule pattern.
            idle_chunk = prev.split(":hover")[0]
            if danger_idle in idle_chunk:
                sheet += f"""
                PushButton {{ color: {DARK['danger']}; }}
                PushButton:hover {{
                    background-color: {DARK['danger']};
                    color: {DARK['text']};
                }}
                PushButton:disabled {{
                    background-color: {CONTROL_BG};
                    background: {CONTROL_BG};
                }}
                """
        elif danger_idle in prev:
            sheet += f"""
            PushButton {{ color: {DARK['danger']}; }}
            PushButton:hover {{
                background-color: {DARK['danger']};
                color: {DARK['text']};
            }}
            PushButton:disabled {{
                background-color: {CONTROL_BG};
                background: {CONTROL_BG};
            }}
            """
        else:
            # Browse / Open / Play — soft CTk text (not Fluent pure white)
            sheet += f"""
            PushButton {{ color: {DARK['text']}; }}
            PushButton:hover {{ color: {DARK['text']}; }}
            PushButton:pressed {{ color: {DARK['text']}; }}
            """
        setCustomStyleSheet(btn, "", sheet)


def _install_checker_pointing_hand(app: QApplication) -> None:
    """Pointing-hand cursor for checkers + Fluent PushButton / ToggleButton.

    Qt Style Sheets do not support ``cursor:`` (emits Unknown property cursor).
    An app event filter covers widgets created after polish (rules panel enable
    boxes, Instrument source toggles / RuleAdd / RuleDelete after re-render,
    dialogs) without per-call-site wiring. Stock QPushButton chrome (title bar)
    is left alone — only Fluent PushButton (and subclasses) are hooked.
    """
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtWidgets import QCheckBox, QRadioButton
    from qfluentwidgets import PushButton

    # Already installed (e.g. tests / re-apply)
    if getattr(app, "_stem_checker_cursor_hook", None) is not None:
        return

    hand = Qt.CursorShape.PointingHandCursor

    class _CheckerCursorHook(QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            if event.type() in (QEvent.Type.Show, QEvent.Type.Polish):
                # PushButton covers PrimaryPushButton + ToggleButton subclasses
                if isinstance(obj, (QCheckBox, QRadioButton, PushButton)):
                    if obj.cursor().shape() != hand:
                        obj.setCursor(hand)
            return False

    hook = _CheckerCursorHook(app)
    app.installEventFilter(hook)
    app._stem_checker_cursor_hook = hook  # type: ignore[attr-defined]


def apply_theme(app: QApplication) -> None:
    """Apply QFluentWidgets dark theme across the whole application."""
    try:
        from qfluentwidgets import Theme, setTheme, setThemeColor
        from qfluentwidgets.common.font import setFontFamilies
    except ImportError as exc:
        raise ImportError(
            "PySide6-Fluent-Widgets is required. Install with: pip install PySide6-Fluent-Widgets"
        ) from exc

    setFontFamilies([FONT_FAMILY, "Microsoft YaHei", "PingFang SC"])
    setTheme(Theme.DARK)
    setThemeColor(COLORS["accent"])
    _apply_dark_palette(app)
    app.setFont(F_BODY)
    # Dark surfaces for stock widgets + frameless chrome (after Fluent QSS)
    existing = app.styleSheet() or ""
    app.setStyleSheet(existing + "\n" + build_app_overrides_qss())
    _install_checker_pointing_hand(app)


def polish_widget(w: QWidget) -> None:
    """Re-apply style to a widget after dynamic property changes (used by section etc.)."""
    w.setStyle(w.style())

"""Log panel: Fluent TextEdit + colored stem chips + cross-thread drain.

Port of stem_organizer_ui LOG panel — including color-coded stem/GG badges.
Chips are painted as fixed-size images (QPlainTextEdit cannot render images).
"""
from __future__ import annotations

import queue
import re
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPixmap,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import CaptionLabel, PushButton, TextEdit

from .. import theme
from ..musical_keys import CHART_KEY_COLORS, KEY_DISPLAY_ORDER, SHORT_TO_DISPLAY

# Tag → (color_token_or_hex, bold, italic, monospace).
# Colors match CTk stem_organizer_ui log tags exactly.
TAG_STYLES: dict[str, tuple[str, bool, bool, bool]] = {
    "info":     ("text_dim", False, False, True),          # COLORS['fg_dim']
    "detail":   ("log_fg",   False, False, True),          # COLORS['log_fg']
    "ok":       (theme.LOG_OK_COLOR, True,  False, True),  # #7ee0a0
    "warn":     (theme.LOG_WARN_COLOR, False, False, True),  # #ecc990
    "err":      (theme.LOG_ERR_COLOR, True,  False, True),  # #ff7a7a
    "sdr_pass": (theme.SDR_PASS_COLOR, False, False, True),
    "sdr_fail": (theme.SDR_FAIL_COLOR, False, False, True),
    "sdr_label":(theme.SDR_LABEL_COLOR, False, False, True),
    "gg_file":  ("text_dim", False, False, True),
    "gg_key":   ("text_dim", False, False, True),
    "gg_val":   ("log_fg",   False, False, True),
    "gg_conf":  (theme.GG_CONF_COLOR, True, False, True),
    "gg_conf_low": (theme.GG_CONF_LOW_COLOR, True, False, True),
    "gg_result":("log_fg",   False, False, True),
    "log_pct":  (theme.LOG_PCT_COLOR, False, False, True),
    "log_note": (theme.LOG_WARN_COLOR, False, False, True),  # same yellow as warning/suspicious
    "log_note_err": (theme.LOG_ERR_COLOR, False, False, True),  # failed badge description (same size as log_note)
    "deleted":  (theme.LOG_DELETED_COLOR, False, False, True),
    "log_margin": (theme.LOG_MARGIN_COLOR, False, False, True),
}

LOG_CHIP_FONT_PX = 11  # CTk LOG_STEM_CHIP_FONT_SIZE was 9; slight bump for Qt
LOG_NOTE_FONT_PX = 10  # smaller description beside warning/suspicious badges
LOG_PCT_FONT_PX = 11   # was 8 — closer to body log size for readability
LOG_LINE_EXTRA_PX = 5  # extra space between log lines (QTextBlockFormat)
# Classify startup/config lines use this indent; === progress headers stay flush.
LOG_INDENT = "  "
# NBSP — Qt often skips painting chip backgrounds on regular ASCII spaces
_CHIP_PAD = "\u00A0"

# CTk LOG_STEM_COLORS / LOG_GG_COLORS — chip background fills
LOG_STEM_COLORS = {
    "bass":         "#ef4444",
    "drums":        "#f59e0b",
    "other":        "#10b981",
    "vocals":       "#a855f7",
    "instrumental": "#60A5FA",
    "acapella":     "#a855f7",
    "vocal":        "#a855f7",
    "original":     "#9aa0b4",
}
LOG_GG_COLORS = {
    "female": "#ec4899",
    "male":   "#60A5FA",
    # Dry fill stays log_fg (soft) — exception to pure-white badge text rule.
    "dry":    theme.COLORS["log_fg"],
    "wet":    "#262833",
    # Overview Compression donut — same fills as CHART_COMPRESSION_COLORS.
    "lossless": theme.LOG_OK_COLOR,   # #7ee0a0 green
    "lossy":    theme.LOG_ERR_COLOR,  # #ff7a7a red
    # flac-detective / Integrity warning — yellow badge, dark text
    "authentic": "#262833",
    "warning": theme.LOG_WARN_COLOR,
    "suspicious": theme.LOG_WARN_COLOR,
    "fake_certain": "#262833",
    # Integrity → Corruption verdicts
    "ok": theme.LOG_OK_COLOR,
    "minor": theme.LOG_WARN_COLOR,  # legacy; log maps minor→warning
    "failed": theme.LOG_ERR_COLOR,
    "suspect": theme.LOG_WARN_COLOR,  # legacy; log maps suspect→warning
    # Vocal type (Charts CHART_VOCAL_TYPE_COLORS — purple, progressively lighter)
    "singing": theme.CHART_VOCAL_TYPE_COLORS["Singing"],
    "speech": theme.CHART_VOCAL_TYPE_COLORS["Speech"],
    "rapping": theme.CHART_VOCAL_TYPE_COLORS["Rapping"],
    "humming": theme.CHART_VOCAL_TYPE_COLORS["Humming"],
    "choir": theme.CHART_VOCAL_TYPE_COLORS["Choir"],
}
# Musical keys — lowercase lookup keys; paint uses display casing.
_KEY_CHIP_DISPLAY: dict[str, str] = {}
for _disp in KEY_DISPLAY_ORDER:
    LOG_GG_COLORS[_disp.lower()] = CHART_KEY_COLORS[_disp]
    _KEY_CHIP_DISPLAY[_disp.lower()] = _disp
    _compact = _disp.replace(" / ", "/")
    if _compact != _disp:
        LOG_GG_COLORS.setdefault(_compact.lower(), CHART_KEY_COLORS[_disp])
        _KEY_CHIP_DISPLAY.setdefault(_compact.lower(), _disp)
for _short, _disp in SHORT_TO_DISPLAY.items():
    LOG_GG_COLORS.setdefault(_short.lower(), CHART_KEY_COLORS[_disp])
    _KEY_CHIP_DISPLAY.setdefault(_short.lower(), _disp)

_CHIP_FG_LIGHT = "#FFFFFF"

LOG_GG_FG = {
    "dry": "#262833",
    "wet": theme.COLORS["log_fg"],
    "lossless": "#262833",
    "lossy": "#262833",
    # AUTHENTIC / FAKE_CERTAIN: log_fg on dark badges (not pure white).
    # WARNING: dark text on yellow badge.
    "authentic": theme.COLORS["log_fg"],
    "warning": "#262833",
    "suspicious": "#262833",
    "fake_certain": theme.COLORS["log_fg"],
    "ok": "#262833",
    "minor": "#262833",
    "failed": "#262833",
    "suspect": "#262833",
    # Vocal type: dark text on all purple chips.
    "singing": "#262833",
    "speech": "#262833",
    "rapping": "#262833",
    "humming": "#262833",
    "choir": "#262833",
}
for _k in _KEY_CHIP_DISPLAY:
    LOG_GG_FG[_k] = "#262833"

LOG_SKIP_COLOR = "#636b7a"


_GENRE_CHIP_LIGHT_TEXT = frozenset({"non-music"})


def _genre_chip_fg(genre: str) -> str:
    """Genre badge text — dark by default; white on very dark fills only."""
    if (genre or "").strip().casefold() in _GENRE_CHIP_LIGHT_TEXT:
        return _CHIP_FG_LIGHT
    return "#262833"


def _genre_chip_bg(genre: str) -> str:
    from ..dataset.charts import genre_colors_for

    name = (genre or "").strip()
    if not name:
        return LOG_GG_COLORS["dry"]
    return genre_colors_for([name]).get(name, LOG_GG_COLORS["dry"])

# CTk: r'^(\s+)([a-z_]+)(?: (\d+%))?(?: \(margin [^)]+\))?(  →  .+)$'
STEM_CLASSIFY_RE = re.compile(
    r"^(\s+)([a-z_]+)(?: (\d+%))?(?: \(margin [^)]+\))?(  →  )(.+)$"
)
# Key badges include / and # (Db/C#, Abm/G#m). Longest-first for alternation.
_KEY_BADGE_ALTS = "|".join(
    sorted(
        {
            re.escape(x)
            for x in list(KEY_DISPLAY_ORDER)
            + list(SHORT_TO_DISPLAY)
            + [d.replace(" / ", "/") for d in KEY_DISPLAY_ORDER]
        },
        key=len,
        reverse=True,
    )
)
GG_BADGE_RE = re.compile(
    r"^(\s*)(female|male|dry|wet|lossless|lossy|"
    r"authentic|warning|suspicious|fake_certain|"
    r"ok|minor|failed|suspect|"
    r"singing|speech|rapping|humming|choir|"
    + _KEY_BADGE_ALTS
    + r")"
    r"(?: \(confidence [^)]+\)| (\d+%))?"
    r"(?:  (.+))?"
    r"\s*$",
    re.IGNORECASE,
)
GG_HEADER_RE = re.compile(r"^=== .+ ===\s*$")
# Classify / Genre / Gender / Vocal / Key: === [1/12] name ===
GG_PROGRESS_HEADER_RE = re.compile(
    r"^===\s+(\[\d+/\d+\])\s+(.+?)\s+===\s*$"
)
GG_PCT_ONLY_RE = re.compile(r"^(\s*)(\d+%)\s*$")
GG_RESULT_KEY_RE = re.compile(
    r"^(GENRE|STYLE|CONF|GENDER|REVERB):\s*(.*)$",
    re.IGNORECASE,
)
GG_CONF_LEGACY_RE = re.compile(
    r"^(\s*)\(confidence\s+([^)]+)\)\s*$",
    re.IGNORECASE,
)
# Distribute originals: green "matched:" · rest (folder · %) in default color
MATCHED_LINE_RE = re.compile(r"^(\s*)(matched:)\s*(.*)$", re.IGNORECASE)


class ChipRenderer:
    """Shared chip / format / line-spacing machinery used by LogPanel and the
    Rename auto-detect log.

    Chips are painted as fixed-size images (QPlainTextEdit cannot render images,
    so callers must host these on a qfluentwidgets TextEdit). State mirrors the
    CTk stem_organizer_ui layout: one Arial Bold chip font, a shared pixel width
    derived from the longest known label, and a pixmap cache keyed by label.
    """

    def __init__(self) -> None:
        # CTk: Arial bold chips, equal pixel width via space padding (stem_organizer_ui)
        self.chip_font = QFont("Arial")
        self.chip_font.setPixelSize(LOG_CHIP_FONT_PX)
        self.chip_font.setBold(True)
        if not QFontMetrics(self.chip_font).inFont(ord("A")):
            self.chip_font = QFont(theme.FONT_FAMILY_SEMIBOLD)
            self.chip_font.setPixelSize(LOG_CHIP_FONT_PX)
            self.chip_font.setBold(True)
        self._chip_cache: dict[str, str] = {}
        self._chip_pix_cache: dict[str, QPixmap] = {}
        self.chip_width_px = self._init_chip_width()
        self.chip_height_px = max(
            QFontMetrics(self.chip_font).height() + 4, LOG_CHIP_FONT_PX + 6
        )
        self._formats: dict[str, QTextCharFormat] = {}
        self._build_formats()

    # ----- chip geometry / text padding -----

    def _chip_labels(self) -> list[str]:
        return list(LOG_STEM_COLORS) + list(LOG_GG_COLORS) + ["skip"]

    def pad_chip_to_width(self, label: str, width_px: int, *, lower: bool = True) -> str:
        """CTk `_pad_chip_to_width` — pad with NBSP until measure >= width_px."""
        text = label.strip()
        if lower:
            text = text.lower()
        fm = QFontMetrics(self.chip_font)
        left = right = 1
        while fm.horizontalAdvance((_CHIP_PAD * left) + text + (_CHIP_PAD * right)) < width_px:
            if left <= right:
                left += 1
            else:
                right += 1
        return (_CHIP_PAD * left) + text + (_CHIP_PAD * right)

    def _init_chip_width(self) -> int:
        """CTk `_init_stem_chip_layout` — shared width from longest label + 16px."""
        fm = QFontMetrics(self.chip_font)
        labels = self._chip_labels()
        longest = max(labels, key=len)
        target = max(fm.horizontalAdvance(longest), fm.horizontalAdvance("n" * 9)) + 16
        provisional = {lb: self.pad_chip_to_width(lb, target) for lb in labels}
        width_px = max(fm.horizontalAdvance(s) for s in provisional.values())
        self._chip_cache = {
            lb: self.pad_chip_to_width(lb, width_px) for lb in labels
        }
        return width_px

    def pad_chip(self, label: str, *, lower: bool = True) -> str:
        """CTk `_format_stem_chip_text` — fixed-width chip so backgrounds align."""
        text = label.strip()
        if lower:
            text = text.lower()
        key = text.lower() if lower else text
        if key in self._chip_cache:
            return self._chip_cache[key]
        if self.chip_width_px > 0:
            padded = self.pad_chip_to_width(text, self.chip_width_px, lower=lower)
            self._chip_cache[key] = padded
            return padded
        return f"{_CHIP_PAD}{text}{_CHIP_PAD}"

    # ----- chip colors / pixmaps -----

    def chip_bg(self, label: str) -> str:
        key = label.strip().lower()
        if key == "skip":
            return LOG_SKIP_COLOR
        if key in LOG_GG_COLORS:
            return LOG_GG_COLORS[key]
        return LOG_STEM_COLORS.get(key, theme.COLORS["panel2"])

    def chip_fg(self, label: str) -> str:
        key = label.strip().lower()
        return LOG_GG_FG.get(key, _CHIP_FG_LIGHT)

    def chip_pixmap(self, label: str) -> QPixmap:
        """Paint a chip (min width = shared stem width; longer labels grow)."""
        key = label.strip().lower()
        bg = self.chip_bg(key)
        fg = self.chip_fg(key)
        cache_key = f"chip:v2:{key}:{bg}:{fg}"
        cached = self._chip_pix_cache.get(cache_key)
        if cached is not None:
            return cached
        fm = QFontMetrics(self.chip_font)
        # Horizontal pad so longer verdicts (fake_certain) aren't clipped.
        w = max(self.chip_width_px, fm.horizontalAdvance(key) + 12)
        h = self.chip_height_px
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(bg))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(self.chip_font)
        painter.setPen(QColor(fg))
        paint_text = _KEY_CHIP_DISPLAY.get(key, key)
        painter.drawText(img.rect(), int(Qt.AlignCenter), paint_text)
        painter.end()
        pix = QPixmap.fromImage(img)
        self._chip_pix_cache[cache_key] = pix
        return pix

    def insert_chip(self, cursor: QTextCursor, label: str) -> None:
        key = label.strip().lower()
        pix = self.chip_pixmap(key)
        # insertImage works on QTextEdit; QPlainTextEdit only shows a placeholder glyph.
        cursor.insertImage(pix.toImage(), f"stemchip_{key}")

    def category_pixmap(self, label: str, bg_hex: str) -> QPixmap:
        """Chip painted in an arbitrary category color (Rename auto-detect).

        min width = shared chip width so it lines up with stem/skip chips; longer
        labels grow. Text is white on colored category badges.
        """
        display = (label or "?").strip().lower() or "?"
        cache_key = f"cat:w:{display}:{bg_hex}"
        cached = self._chip_pix_cache.get(cache_key)
        if cached is not None:
            return cached
        padded = self.pad_chip_to_width(display, self.chip_width_px, lower=False)
        fm = QFontMetrics(self.chip_font)
        w = max(self.chip_width_px, fm.horizontalAdvance(padded))
        h = self.chip_height_px
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(bg_hex))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(self.chip_font)
        painter.setPen(QColor(_CHIP_FG_LIGHT))
        painter.drawText(img.rect(), int(Qt.AlignCenter), display)
        painter.end()
        pix = QPixmap.fromImage(img)
        self._chip_pix_cache[cache_key] = pix
        return pix

    def insert_category_chip(self, cursor: QTextCursor, label: str, bg_hex: str) -> None:
        pix = self.category_pixmap(label, bg_hex)
        safe = re.sub(r"[^\w.-]+", "_", (label or "?").strip())[:40]
        cursor.insertImage(pix.toImage(), f"catchip_{safe}_{bg_hex.lstrip('#')}")

    def gg_value_pixmap(
        self, text: str, style_key: str, *, bg_hex: Optional[str] = None
    ) -> QPixmap:
        """Genre (dry / custom fill) / Style (wet) chip — min width = stem chips."""
        display = (text or "?").strip() or "?"
        cache_key = f"ggval:v2:{style_key}:{bg_hex or ''}:{display}"
        cached = self._chip_pix_cache.get(cache_key)
        if cached is not None:
            return cached
        padded = self.pad_chip_to_width(display, self.chip_width_px, lower=False)
        fm = QFontMetrics(self.chip_font)
        w = max(self.chip_width_px, fm.horizontalAdvance(padded))
        h = self.chip_height_px
        fill = bg_hex or self.chip_bg(style_key)
        fg = _genre_chip_fg(display) if bg_hex else self.chip_fg(style_key)
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor(fill))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(self.chip_font)
        painter.setPen(QColor(fg))
        painter.drawText(img.rect(), int(Qt.AlignCenter), display)
        painter.end()
        pix = QPixmap.fromImage(img)
        self._chip_pix_cache[cache_key] = pix
        return pix

    def insert_gg_value_chip(
        self,
        cursor: QTextCursor,
        text: str,
        style_key: str,
        *,
        bg_hex: Optional[str] = None,
    ) -> None:
        pix = self.gg_value_pixmap(text, style_key, bg_hex=bg_hex)
        safe = re.sub(r"[^\w.-]+", "_", (text or "?").strip())[:40]
        cursor.insertImage(pix.toImage(), f"ggchip_{style_key}_{safe}")

    # ----- tag formats / line spacing -----

    def _build_formats(self) -> None:
        for tag, (color, bold, italic, mono) in TAG_STYLES.items():
            fmt = QTextCharFormat()
            if color in theme.DARK:
                col = QColor(theme.DARK[color])
            elif color in theme.COLORS:
                col = QColor(theme.COLORS[color])
            else:
                col = QColor(color)
            fmt.setForeground(col)
            if mono:
                font = QFont(theme.FONT_FAMILY_MONO)
                font.setPixelSize(
                    LOG_NOTE_FONT_PX
                    if tag in ("log_note", "log_note_err")
                    else LOG_PCT_FONT_PX
                    if tag == "log_pct"
                    else theme.LOG_FONT_PX
                )
            else:
                font = QFont(theme.FONT_FAMILY)
                font.setPixelSize(theme.BODY_FONT_PX)
            if bold:
                font.setWeight(QFont.Bold)
            font.setItalic(italic)
            fmt.setFont(font)
            self._formats[tag] = fmt

    def fmt(self, tag: str) -> QTextCharFormat:
        return self._formats.get(tag) or self._formats["info"]

    def insert(self, cursor: QTextCursor, text: str, tag: str = "info") -> None:
        cursor.insertText(text, self.fmt(tag))

    def apply_line_spacing(self, cursor: QTextCursor) -> None:
        block = QTextBlockFormat()
        # PySide6 wants heightType as int (LineDistanceHeight == 4)
        block.setLineHeight(
            float(LOG_LINE_EXTRA_PX),
            int(QTextBlockFormat.LineDistanceHeight.value),
        )
        cursor.setBlockFormat(block)


class LogPanel(QWidget):
    """Read-only log view + clear / save buttons."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("LogPanel")

        self.queue: queue.Queue = queue.Queue()

        layout = QVBoxLayout(self)
        # Left 0: gutter to left-panel content is only upper-layout spacing (6), so
        # window→content (6 + PAGE_CONTENT_INSET) matches content→LogView
        # (PAGE_CONTENT_INSET + 6). Right uses PAGE_CONTENT_INSET so window→LogView
        # on the right matches the left margin.
        layout.setContentsMargins(0, 7, theme.PAGE_CONTENT_INSET, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = CaptionLabel("LOG")
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)

        self.view = TextEdit()
        self.view.setObjectName("LogView")
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(20000)
        self.view.setFont(theme.F_LOG)
        # Fluent TextEdit draws an accent underline via EditLayer on focus — not wanted on LOG
        layer = getattr(self.view, "layer", None)
        if layer is not None:
            layer.hide()
            layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        # Kill Fluent LINE_EDIT focus border too (setCustomStyleSheet beats Fluent sheet)
        t = theme.DARK
        c = theme.COLORS
        log_qss = f"""
            TextEdit#LogView,
            TextEdit#LogView:hover,
            TextEdit#LogView:focus {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                font-family: "{theme.FONT_FAMILY_MONO}";
                font-size: {theme.LOG_FONT_PX}px;
                border: 1px solid {t['border']};
                border-bottom: 1px solid {t['border']};
                border-radius: {theme.LOG_VIEW_CORNER_RADIUS}px;
                padding: 8px;
                selection-background-color: {c['accent']};
                selection-color: {c['log_fg']};
            }}
            """
        self.view.setStyleSheet(log_qss)
        try:
            from qfluentwidgets import setCustomStyleSheet

            setCustomStyleSheet(self.view, log_qss, log_qss)
        except Exception:
            pass
        layout.addWidget(self.view, stretch=1)

        # Clear / Save live in the shared action row (app.py) beside Start/Stop/Play
        self.clear_btn = PushButton("Clear")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        self.clear_btn.setToolTip("Clear the log panel for a fresh run.")
        self.clear_btn.clicked.connect(self.clear)
        self.save_btn = PushButton("Save…")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        self.save_btn.setToolTip("Save the current log to a text file.")
        self.save_btn.clicked.connect(lambda: self.save_to_file(self.window()))

        # Chip / format / line-spacing state lives on ChipRenderer so other log
        # surfaces (Rename auto-detect) can reuse it without subclassing LogPanel.
        self._chips = ChipRenderer()
        # Backwards-compat aliases — existing LogPanel code referenced these directly.
        self._chip_font = self._chips.chip_font
        self._chip_pix_cache = self._chips._chip_pix_cache
        self._chip_width_px = self._chips.chip_width_px
        self._chip_height_px = self._chips.chip_height_px
        # Buffer GENRE + STYLE lines → flush as dry/wet chips (CTk _gg_pending_*)
        self._gg_pending_genre: Optional[str] = None
        self._gg_pending_style: Optional[str] = None
        # In-place Batch progress line (CTk _gg_processed mark)
        self._gg_processed_cursor: Optional[QTextCursor] = None
        # Full plain-text mirror for Save (chips + maxBlockCount drop image/text)
        self._plain_lines: list[str] = []
        self._plain_processed_idx: Optional[int] = None
        self._export_prefix: str = ""

        # Formats are owned by ChipRenderer; alias so legacy _fmt/_insert work.
        self._formats = self._chips._formats

        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(50)
        self._drain_timer.timeout.connect(self._drain)
        self._drain_timer.start()

    # ----- chip / format delegations (preserved for internal callers) -----

    def _pad_chip(self, label: str, *, lower: bool = True) -> str:
        return self._chips.pad_chip(label, lower=lower)

    def _chip_bg(self, label: str) -> str:
        return self._chips.chip_bg(label)

    def _chip_fg(self, label: str) -> str:
        return self._chips.chip_fg(label)

    def _chip_pixmap(self, label: str) -> QPixmap:
        return self._chips.chip_pixmap(label)

    def _insert_chip(self, cursor: QTextCursor, label: str) -> None:
        self._chips.insert_chip(cursor, label)

    def _gg_value_pixmap(self, text: str, style_key: str) -> QPixmap:
        return self._chips.gg_value_pixmap(text, style_key)

    def _insert_gg_value_chip(
        self,
        cursor: QTextCursor,
        text: str,
        style_key: str,
        *,
        bg_hex: Optional[str] = None,
    ) -> None:
        self._chips.insert_gg_value_chip(cursor, text, style_key, bg_hex=bg_hex)

    @staticmethod
    def _gg_norm_pct(val: str) -> str:
        conf_val = (val or "").strip()
        if not conf_val:
            return conf_val
        try:
            if not conf_val.endswith("%"):
                conf_val = f"{int(round(float(conf_val) * 100.0))}%"
        except ValueError:
            pass
        return conf_val

    def _gg_flush_genre_style_row(
        self,
        cursor: QTextCursor,
        genre: Optional[str],
        style: Optional[str],
        conf_pct: Optional[str] = None,
    ) -> None:
        """Genre chip, style chip below, optional dim pct.

        Trailing blank only when both genre and style chips are present.
        """
        genre = (genre or "").strip()
        style = (style or "").strip()
        conf = (conf_pct or "").strip()
        if not genre and not style and not conf:
            return
        if genre:
            self._apply_line_spacing(cursor)
            self._insert(cursor, "  ")
            self._insert_gg_value_chip(
                cursor, genre, "dry", bg_hex=_genre_chip_bg(genre)
            )
            if conf:
                self._insert(cursor, f"  {conf}", "log_pct")
                conf = ""
            self._insert(cursor, "\n")
        if style:
            self._apply_line_spacing(cursor)
            self._insert(cursor, "  ")
            self._insert_gg_value_chip(cursor, style, "wet")
            self._insert(cursor, "\n")
        if conf:
            self._apply_line_spacing(cursor)
            self._insert(cursor, f"  {conf}", "log_pct")
            self._insert(cursor, "\n")
        # Blank after 2+ chips (GENRE+STYLE); single chip stays tight
        n_chips = (1 if genre else 0) + (1 if style else 0)
        if n_chips >= 2:
            self._apply_line_spacing(cursor)
            self._insert(cursor, "\n")

    def _gg_flush_pending(self, cursor: QTextCursor) -> None:
        genre = self._gg_pending_genre
        style = self._gg_pending_style
        if genre or style:
            self._gg_pending_genre = None
            self._gg_pending_style = None
            self._gg_flush_genre_style_row(cursor, genre, style)

    # ----- API ----------------------------------------------------------

    def append_line(self, text: str, tag: str = "") -> None:
        """Append a line from the UI thread (no queue indirection)."""
        self._append(text, tag)

    def post(self, text: str, tag: str = "") -> None:
        """Thread-safe enqueue (workers call this)."""
        self.queue.put((text, tag))

    def clear(self) -> None:  # noqa: A003 shadowing
        self._gg_pending_genre = None
        self._gg_pending_style = None
        self._gg_processed_cursor = None
        self._plain_lines.clear()
        self._plain_processed_idx = None
        self.view.clear()

    def set_export_prefix(self, prefix: str) -> None:
        """Suggested Save filename stem, e.g. 'compression' → compression_stem_organizer.log."""
        cleaned = re.sub(r"[^\w\-]+", "_", (prefix or "").strip().lower()).strip("_")
        self._export_prefix = cleaned

    def _record_plain(self, line: str) -> None:
        self._plain_lines.append(line)

    def update_gg_processed(self, n: int, total: int) -> None:
        """Update single Batch progress line in LOG (Processed: n/total) — CTk parity."""
        self.update_live_progress(f"Processed: {int(n):,}/{int(total):,}")

    def end_live_progress(self) -> None:
        """Stop updating the current live line; next update starts a new one."""
        self._gg_processed_cursor = None
        self._plain_processed_idx = None

    def update_live_progress(self, text: str) -> None:
        """Replace a single in-place progress line (e.g. ``12/40,110 files read``)."""
        line = (text or "").rstrip("\n")
        if not line:
            return
        if self._plain_processed_idx is not None and 0 <= self._plain_processed_idx < len(
            self._plain_lines
        ):
            self._plain_lines[self._plain_processed_idx] = line
        else:
            self._plain_processed_idx = len(self._plain_lines)
            self._plain_lines.append(line)
        fmt = self._fmt("info")
        if self._gg_processed_cursor is not None and not self._gg_processed_cursor.isNull():
            c = QTextCursor(self._gg_processed_cursor)
            c.movePosition(QTextCursor.StartOfBlock)
            c.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
            c.insertText(line, fmt)
            mark = QTextCursor(c)
            mark.movePosition(QTextCursor.StartOfBlock)
            mark.setKeepPositionOnInsert(True)
            self._gg_processed_cursor = mark
        else:
            c = QTextCursor(self.view.document())
            c.movePosition(QTextCursor.End)
            self._apply_line_spacing(c)
            c.insertText(line + "\n", fmt)
            mark = QTextCursor(c)
            mark.movePosition(QTextCursor.PreviousBlock)
            mark.movePosition(QTextCursor.StartOfBlock)
            mark.setKeepPositionOnInsert(True)
            self._gg_processed_cursor = mark
        # Park caret at end with no selection — otherwise Fluent paints accent highlight
        # on the replaced block each time the count updates.
        end = QTextCursor(self.view.document())
        end.movePosition(QTextCursor.End)
        end.clearSelection()
        self.view.setTextCursor(end)
        self.view.ensureCursorVisible()

    def save_to_file(self, parent: QWidget) -> None:
        from PySide6.QtWidgets import QFileDialog

        default = (
            f"{self._export_prefix}_stem_organizer.log"
            if self._export_prefix
            else "stem_organizer.log"
        )
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save log", default, "Text files (*.txt *.log)"
        )
        if path:
            try:
                body = "\n".join(self._plain_lines)
                if self._plain_lines:
                    body += "\n"
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(body)
            except OSError:
                pass

    # ----- internal -----------------------------------------------------

    def _drain(self) -> None:
        drained = 0
        while drained < 400:
            try:
                text, tag = self.queue.get_nowait()
            except queue.Empty:
                break
            self._append(text, tag)
            drained += 1

    def _fmt(self, tag: str) -> QTextCharFormat:
        return self._formats.get(tag) or self._formats["info"]

    def _insert(self, cursor: QTextCursor, text: str, tag: str = "info") -> None:
        cursor.insertText(text, self._fmt(tag))

    def _resolve_tag(self, line: str, hinted: str) -> str:
        """Match CTk _append_log tag heuristics when no useful hint is given."""
        if hinted and hinted not in ("", "info"):
            return hinted
        s = line.strip()
        low = line.lower()
        if "[error]" in low or "[delete error]" in low:
            return "err"
        if "[deleted]" in low:
            return "deleted"
        if s == "DONE":
            return "ok"
        if "[warn]" in low or "[stopping]" in low or "cuda oom" in low:
            return "warn"
        if "[skip existing]" in low or "[skip]" in low:
            return "warn"
        if "[fail]" in low:
            return "err"
        if s.startswith("Done") or "    wrote " in line or line.lstrip().startswith("wrote "):
            return "ok"
        if line.startswith("  Successful"):
            return "ok"
        if line.startswith(("  Deleted", "  Delete failed")):
            return "deleted"
        if line.startswith(("  Skipped", "  Not processed")):
            return "warn"
        if line.startswith("  Lossless:"):
            return "ok"
        if line.startswith("  Lossy:"):
            return "err"
        # === folder titles / summaries — dim gray (CTk info = fg_dim)
        if line.startswith("===") or line.strip().startswith("==="):
            return "info"
        return hinted or "info"

    def _apply_line_spacing(self, cursor: QTextCursor) -> None:
        block = QTextBlockFormat()
        # PySide6 wants heightType as int (LineDistanceHeight == 4)
        block.setLineHeight(
            float(LOG_LINE_EXTRA_PX),
            int(QTextBlockFormat.LineDistanceHeight.value),
        )
        cursor.setBlockFormat(block)

    def _append(self, text: str, tag: str) -> None:
        if not text and not tag:
            return
        line = text.rstrip("\n")
        self._record_plain(line)
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Color-coded stem badges: "  bass 61%  →  file.wav"
        m = STEM_CLASSIFY_RE.match(line)
        if m:
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            indent, label, pct, arrow, dest = (
                m.group(1), m.group(2), m.group(3), m.group(4), m.group(5),
            )
            self._insert(cursor, indent)
            self._insert_chip(cursor, label)
            if pct:
                # Fixed field so "100%" doesn't shove the arrow right of "98%"
                self._insert(cursor, f"  {pct:>4}", "log_pct")
            self._insert(cursor, arrow, "info")  # text_dim — darker than pct / name
            self._insert(cursor, dest, "detail")
            self._insert(cursor, "\n")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        # Gender / reverb / compression fixed badges (+ optional note)
        gg = GG_BADGE_RE.match(line)
        if gg:
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            indent, label, pct, note = (
                gg.group(1), gg.group(2), gg.group(3), gg.group(4),
            )
            self._insert(cursor, indent or "  ")
            self._insert_chip(cursor, label)
            if pct:
                self._insert(cursor, f"  {pct}", "log_pct")
            if note:
                note_tag = "log_note_err" if str(label).lower() == "failed" else "log_note"
                self._insert(cursor, f"  {note.strip()}", note_tag)
            self._insert(cursor, "\n")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        # === [1/12] filename ===  (counter in log_fg; rest dim like Classify)
        prog_h = GG_PROGRESS_HEADER_RE.match(line.strip())
        if prog_h:
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            counter, rest = prog_h.group(1), prog_h.group(2)
            self._insert(cursor, "=== ", "info")
            self._insert(cursor, counter, "detail")  # log_fg
            self._insert(cursor, f" {rest} ===\n", "info")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        # === filename === headers (no counter)
        if GG_HEADER_RE.match(line.strip()):
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            self._insert(cursor, line.strip() + "\n", "info")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        # Distribute: matched: (green) + folder · % (log_fg)
        matched = MATCHED_LINE_RE.match(line)
        if matched:
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            indent, label, rest = matched.group(1), matched.group(2), matched.group(3)
            self._insert(cursor, indent or "  ")
            self._insert(cursor, label, "ok")
            if rest:
                self._insert(cursor, f" {rest}", "detail")
            self._insert(cursor, "\n")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        # Genre: buffer GENRE + STYLE, flush on CONF / bare 72%
        key_m = GG_RESULT_KEY_RE.match(line)
        if key_m:
            key = key_m.group(1).upper()
            val = (key_m.group(2) or "").strip()
            if key == "GENRE" and val:
                self._gg_pending_genre = val
                return
            if key == "STYLE":
                self._gg_pending_style = val
                return
            if key == "CONF" and val:
                genre = self._gg_pending_genre
                style = self._gg_pending_style
                self._gg_pending_genre = None
                self._gg_pending_style = None
                self._gg_flush_genre_style_row(
                    cursor, genre, style, self._gg_norm_pct(val),
                )
                self.view.setTextCursor(cursor)
                self.view.ensureCursorVisible()
                return
            self._gg_flush_pending(cursor)
            self._apply_line_spacing(cursor)
            self._insert(cursor, f"{key}: ", "gg_key")
            self._insert(cursor, val + "\n", "gg_val")
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        pct_only = GG_PCT_ONLY_RE.match(line)
        conf_legacy = GG_CONF_LEGACY_RE.match(line)
        if pct_only or conf_legacy:
            genre = self._gg_pending_genre
            style = self._gg_pending_style
            self._gg_pending_genre = None
            self._gg_pending_style = None
            if pct_only:
                conf_val = pct_only.group(2)
            else:
                conf_val = self._gg_norm_pct((conf_legacy.group(2) or "").strip())
            self._gg_flush_genre_style_row(cursor, genre, style, conf_val)
            self.view.setTextCursor(cursor)
            self.view.ensureCursorVisible()
            return

        self._gg_flush_pending(cursor)
        self._apply_line_spacing(cursor)
        resolved = self._resolve_tag(line, tag)
        self._insert(cursor, line + "\n", resolved)
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()
        self._notify_if_done(line)

    @staticmethod
    def _notify_if_done(line: str) -> None:
        """Play the Windows default beep when a job finishes with DONE."""
        if line.strip() != "DONE":
            return
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        except Exception:
            pass

    def append_sdr_line(self, filename: str, score: float, threshold: float) -> None:
        """SI-SDR result line with optional stem chip (CTk _append_sdr_log_line)."""
        from pathlib import Path

        stem_name = Path(filename).stem.lower()
        if stem_name in LOG_STEM_COLORS:
            plain = f"  {stem_name}  →  SI-SDR: {score:.1f}"
        else:
            plain = f"  {filename}  →  SI-SDR: {score:.1f}"
        self._record_plain(plain)

        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._apply_line_spacing(cursor)
        self._insert(cursor, "  ", "detail")
        if stem_name in LOG_STEM_COLORS:
            self._insert(cursor, "  ", "detail")
            self._insert_chip(cursor, stem_name)
            self._insert(cursor, "  →  ", "info")
        else:
            self._insert(cursor, f"{filename}", "detail")
            self._insert(cursor, "  →  ", "info")
        self._insert(cursor, "SI-SDR: ", "sdr_label")
        tag = "sdr_pass" if score >= threshold else "sdr_fail"
        self._insert(cursor, f"{score:.1f}", tag)
        self._insert(cursor, "\n")
        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

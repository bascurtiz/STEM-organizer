"""Fluent dialog helpers — info / yes-no / scrollable help."""
from __future__ import annotations

import re
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Union

from PySide6.QtCore import QEvent, QObject, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QFont, QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    StrongBodyLabel,
    setCustomStyleSheet,
)

from .. import theme
from .action_button import action_button

# CTk show_ctk_help_dialog — section body may be one string or bullet lines
HelpSectionBody = Union[str, Sequence[str]]

# Shared help sizing — wider than before so text wraps less; still capped vs host.
HELP_DIALOG_WIDTH = 820
HELP_SCROLL_RIGHT_GAP = 16  # space between section cards and vertical scrollbar
HELP_ABOUT_ICON_PX = 112  # match CTk ABOUT_ICON_SIZE

# Opaque OK / Yes-No card — word-wrap sizeHint is ~100px, so short prompts
# need a floor ("Browse & select a folder first." ≈ 180px; longest commons ≈ 260).
INFO_PROMPT_BODY_MIN_WIDTH = 300
INFO_PROMPT_BODY_MAX_WIDTH = 420
INFO_PROMPT_BODY_WIDE_MIN_WIDTH = 520
INFO_PROMPT_BODY_WIDE_MAX_WIDTH = 640
INFO_PROMPT_SCROLL_MAX_HEIGHT = 360

# Vertical rhythm — Track Renamer help is the reference for Match / Align /
# Genre / Gender (help_dialog default) and renamer/help_dialog.py.
# Classify About keeps its own gaps via rhythm="classify" (see HELP_ABOUT_*).
# Rename Close spacing uses HELP_RENAME_FOOTER_TOP (8); shared tab help uses
# HELP_FOOTER_TOP (40 ≈ two empty rows) for air above Close.
HELP_OUTER_MARGINS = (12, 12, 12, 12)  # dialog chrome around the panel
# Panel padding — +10 top/bottom for air above first content / below footer
# (not card padding; card vertical margins stay tight).
HELP_SHELL_MARGINS = (22, 28, 22, 26)
HELP_ICON_TO_HEADING = 6
HELP_TITLE_TO_INTRO = 8  # heading → intro (Renamer)
HELP_INTRO_TO_REPO = 4
HELP_HEADER_TO_CARDS = 22  # intro → first section card (Renamer)
HELP_CARD_GAP = 10  # between section cards
HELP_FOOTER_TOP = 40  # body → footer row (Match / Align / Genre / Gender); ~2 rows
HELP_FOOTER_SPACING = 12  # footer note ↔ Close
HELP_CARD_MARGINS = (14, 10, 14, 11)  # section card inner padding (shared)
HELP_CARD_SPACING = 4  # section header → body (tight; no extra air)

# Classify About only (logo / version / stem chips). Do not reuse for tab help.
HELP_ABOUT_TITLE_TO_VERSION = 8
HELP_ABOUT_VERSION_TO_INTRO = 4
HELP_ABOUT_HEADER_TO_CARDS = 22
HELP_ABOUT_FOOTER_TOP = 40  # ~2 empty rows above Close (was 20)
HELP_ABOUT_CARD_SPACING = 4

# Rename help keeps the original 8px body→footer gap (see renamer/help_dialog.py).
HELP_RENAME_FOOTER_TOP = 8

HelpRhythm = Literal["renamer", "classify"]


def _help_dimmer_rgba() -> str:
    """Scrim fill from theme log_bg (same dark plane as tooltips / lists)."""
    c = theme.COLORS["log_bg"].lstrip("#")
    r, g, b = (int(c[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, 160)"


# Help typography — same family as app chrome (Segoe UI / Semibold), sizes
# bumped by one shared ratio. Must be applied via stylesheet: app QSS pins
# BodyLabel/CaptionLabel to 12/10px and overrides QWidget.setFont().
HELP_FONT_SCALE = 1.15
HELP_HEADING_PX = max(1, round(18 * HELP_FONT_SCALE))
HELP_BODY_PX = max(1, round(theme.BODY_FONT_PX * HELP_FONT_SCALE))
HELP_SECTION_TITLE_PX = max(1, round(theme.SECTION_TITLE_PX * HELP_FONT_SCALE))
HELP_FOOTER_PX = max(1, round(theme.SECTION_TITLE_PX * HELP_FONT_SCALE))


def style_help_label(
    lbl: QWidget,
    px: int,
    color: str,
    *,
    bold: bool = False,
) -> None:
    """Pin help color + size to theme tokens (beats FluentLabelBase white default)."""
    family = theme.FONT_FAMILY_SEMIBOLD if bold else theme.FONT_FAMILY
    weight = 600 if bold else 400
    # FluentLabelBase.setTextColor() installs FluentLabelBase{color:#ffffff} via
    # setCustomStyleSheet — that wins over a plain QLabel sheet, so help titles/
    # bodies were painting pure white instead of DARK text / accent tokens.
    if hasattr(lbl, "setTextColor"):
        lbl.setTextColor(color, color)
    sheet = f"""
        FluentLabelBase, BodyLabel, CaptionLabel, StrongBodyLabel, SubtitleLabel, QLabel {{
            color: {color};
            font-family: "{family}";
            font-size: {px}px;
            font-weight: {weight};
            background: transparent;
        }}
    """
    lbl.setStyleSheet(sheet)
    setCustomStyleSheet(lbl, sheet, sheet)


class _DimmerHostFilter(QObject):
    """Keep the dimmer overlay sized to its host window."""

    def __init__(self, overlay: QWidget, host: QWidget) -> None:
        super().__init__(overlay)
        self._overlay = overlay
        self._host = host

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._host and event.type() == QEvent.Type.Resize:
            self._overlay.setGeometry(self._host.rect())
        return False


@contextmanager
def dim_behind(parent: Optional[QWidget]) -> Iterator[None]:
    """Dim the top-level window under a modal dialog (visual scrim)."""
    host = parent.window() if parent is not None else None
    overlay: Optional[QWidget] = None
    filt: Optional[_DimmerHostFilter] = None
    if host is not None:
        overlay = QWidget(host)
        overlay.setObjectName("ModalDimmer")
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        overlay.setStyleSheet(
            f"QWidget#ModalDimmer {{ background-color: {_help_dimmer_rgba()}; }}"
        )
        overlay.setGeometry(host.rect())
        filt = _DimmerHostFilter(overlay, host)
        host.installEventFilter(filt)
        overlay.show()
        overlay.raise_()
    try:
        yield
    finally:
        if host is not None and filt is not None:
            host.removeEventFilter(filt)
        if overlay is not None:
            overlay.hide()
            overlay.deleteLater()


def _modal_prompt(
    parent: Optional[QWidget],
    title: str,
    message: str,
    *,
    primary_text: str,
    secondary_text: Optional[str] = None,
    link_path: Optional[str] = None,
    wide: bool = False,
) -> bool:
    """Dark OK / Yes-No prompt — opaque card (no Fluent MaskDialogBase).

    Fluent ``MessageBox`` is a full-window translucent ``MaskDialogBase``. On
    Windows that layered surface often fails hit-testing: clicks pass through
    the painted OK button to the parent, which only beeps under modality and
    never closes the dialog. Use the same opaque-card pattern as ``help_dialog``
    / CTk ``show_info_dark`` instead (no system MessageBox sound either).

    When ``link_path`` is set, ``message`` is the intro line and the path is
    shown as a clickable link that opens in the system default app.

    ``wide`` uses a broader card (summary prompts with paths / multi-line stats).
    """
    host = parent.window() if parent is not None else parent
    t = theme.DARK
    r = theme.DIALOG_CORNER_RADIUS

    dlg = QDialog(host)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(12, 12, 12, 12)

    shell = QFrame()
    shell.setObjectName("InfoPromptCard")
    shell.setStyleSheet(
        f"""
        QFrame#InfoPromptCard {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(shell)
    lay.setContentsMargins(22, 18, 22, 16)
    lay.setSpacing(0)

    title_lbl = StrongBodyLabel(title, shell)
    style_help_label(title_lbl, HELP_HEADING_PX, t["text"], bold=True)
    lay.addWidget(title_lbl)
    lay.addSpacing(10)

    body_text = message
    long_body = (
        wide
        or len(body_text) > 280
        or body_text.count("\n") >= 6
        or (link_path is not None and len(link_path) > 60)
    )
    min_w = INFO_PROMPT_BODY_WIDE_MIN_WIDTH if wide else INFO_PROMPT_BODY_MIN_WIDTH
    max_w = INFO_PROMPT_BODY_WIDE_MAX_WIDTH if long_body else INFO_PROMPT_BODY_MAX_WIDTH

    body_host = QWidget(shell)
    body_lay = QVBoxLayout(body_host)
    body_lay.setContentsMargins(0, 0, 0, 0)
    body_lay.setSpacing(6)
    body = BodyLabel(body_text, body_host)
    body.setWordWrap(True)
    body.setMinimumWidth(min_w)
    body.setMaximumWidth(max_w)
    style_help_label(body, HELP_BODY_PX, t["text_dim"])
    body_lay.addWidget(body)

    if link_path:
        try:
            resolved = str(Path(link_path).expanduser().resolve(strict=False))
        except OSError:
            resolved = str(Path(link_path))
        url = QUrl.fromLocalFile(resolved).toString()
        link = QLabel(body_host)
        link.setObjectName("InfoPromptLink")
        link.setWordWrap(True)
        link.setMinimumWidth(min_w)
        link.setMaximumWidth(max_w)
        link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        link.setOpenExternalLinks(False)
        link.setCursor(Qt.PointingHandCursor)
        link.setText(
            f'<a href="{url}" style="color: {t["accent"]}; text-decoration: none;">'
            f"{resolved}</a>"
        )
        link.setStyleSheet(
            f"QLabel#InfoPromptLink {{ background: transparent; "
            f"font-family: \"{theme.FONT_FAMILY_MONO}\"; "
            f"font-size: {theme.LOG_FONT_PX}px; }}"
        )
        link.setToolTip("Open in File Explorer")

        def _open_link(_href: str = "") -> None:
            target = Path(resolved)
            if target.is_dir() or target.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
            elif target.parent.is_dir():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

        link.linkActivated.connect(_open_link)
        body_lay.addWidget(link)

    if long_body:
        scroll = QScrollArea(shell)
        scroll.setWidget(body_host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMaximumHeight(INFO_PROMPT_SCROLL_MAX_HEIGHT)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        # Wide summaries should not collapse to the default narrow min.
        scroll.setMinimumWidth(min_w)
        lay.addWidget(scroll)
    else:
        lay.addWidget(body_host)
    lay.addSpacing(18)

    footer = QHBoxLayout()
    footer.setContentsMargins(0, 0, 0, 0)
    footer.setSpacing(10)
    footer.addStretch(1)
    if secondary_text:
        no_btn = action_button(
            secondary_text,
            on_click=dlg.reject,
            accent=False,
            parent=shell,
            tip="Dismiss without continuing.",
        )
        no_btn.setMinimumWidth(72)
        footer.addWidget(no_btn)
    yes_btn = action_button(
        primary_text,
        on_click=dlg.accept,
        accent=True,
        parent=shell,
        tip="Confirm and continue.",
    )
    yes_btn.setMinimumWidth(72)
    footer.addWidget(yes_btn)
    lay.addLayout(footer)
    outer.addWidget(shell)

    dlg.adjustSize()
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(parent if parent is not None else host):
        return dlg.exec() == QDialog.DialogCode.Accepted


def show_info(
    parent: QWidget,
    title: str,
    message: str,
    *,
    link_path: Optional[str] = None,
) -> None:
    # Parent to the top-level window so the dialog centers on the full app
    # (tab pages like Rename are only a column after docking PATH/PREVIEW).
    _modal_prompt(
        parent, title, message, primary_text="OK", link_path=link_path
    )


def ask_yes_no(
    parent: QWidget,
    title: str,
    message: str,
    *,
    yes_text: str = "Yes",
    no_text: str = "No",
    link_path: Optional[str] = None,
    wide: bool = False,
) -> bool:
    return _modal_prompt(
        parent,
        title,
        message,
        primary_text=yes_text,
        secondary_text=no_text,
        link_path=link_path,
        wide=wide,
    )


def _section_body_text(body: HelpSectionBody) -> str:
    if isinstance(body, str):
        return body.strip()
    return "\n".join(line.rstrip() for line in body if str(line).strip())


def _help_code_text(line: str) -> Optional[str]:
    """Return inner text if ``line`` is a ``<pre>…</pre>`` / fenced code block."""
    s = (line or "").strip()
    if s.startswith("<pre>") and s.endswith("</pre>"):
        return s[5:-6].strip()
    if s.startswith("```") and s.endswith("```") and len(s) >= 6:
        return s[3:-3].strip()
    return None


def _help_code_block(parent: QWidget, code: str, *, text_max_width: int) -> QFrame:
    """Soft inset code chip — Consolas on panel (not near-black log_bg)."""
    t = theme.DARK
    c = theme.COLORS
    host = QFrame(parent)
    host.setObjectName("HelpCodeBlock")
    host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    # panel (#262833) sits just under the section card (#2F3140) — readable, low contrast.
    host.setStyleSheet(
        f"""
        QFrame#HelpCodeBlock {{
            background-color: {c['panel']};
            border: 1px solid {t['border']};
            border-radius: 6px;
        }}
        """
    )
    lay = QVBoxLayout(host)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(0)
    lbl = BodyLabel(code)
    lbl.setWordWrap(True)
    lbl.setMaximumWidth(max(120, text_max_width - 24))
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    font = QFont(theme.FONT_FAMILY_MONO)
    font.setPixelSize(theme.LOG_FONT_PX)
    font.setWeight(QFont.Weight.Normal)
    lbl.setFont(font)
    style_help_label(lbl, theme.LOG_FONT_PX, c["log_fg"])
    # Pin mono after style_help_label (it may reset family).
    lbl.setFont(font)
    sheet = f"""
        FluentLabelBase, BodyLabel, QLabel {{
            color: {c['log_fg']};
            font-family: "{theme.FONT_FAMILY_MONO}";
            font-size: {theme.LOG_FONT_PX}px;
            font-weight: 400;
            background: transparent;
        }}
    """
    lbl.setStyleSheet(sheet)
    setCustomStyleSheet(lbl, sheet, sheet)
    lay.addWidget(lbl)
    return host


def _fill_help_body_label(lbl: BodyLabel, text: str, *, text_max_width: int) -> None:
    """Plain or rich help body; rich text enables clickable ``<a href>`` links."""
    t = theme.DARK
    dim = t["text_dim"]
    lbl.setWordWrap(True)
    lbl.setMaximumWidth(text_max_width)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    if "<a " in text.lower():
        accent = t["accent"]
        style_help_label(lbl, HELP_BODY_PX, dim)
        font = QFont(theme.FONT_FAMILY)
        font.setPixelSize(HELP_BODY_PX)
        font.setWeight(QFont.Weight.Normal)
        lbl.setFont(font)
        sheet = f"""
            FluentLabelBase, BodyLabel, QLabel {{
                color: {dim};
                font-family: "{theme.FONT_FAMILY}";
                font-size: {HELP_BODY_PX}px;
                font-weight: 400;
                background: transparent;
            }}
            a {{
                color: {accent};
                text-decoration: none;
            }}
        """
        lbl.setTextFormat(Qt.RichText)
        lbl.setOpenExternalLinks(False)
        lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        lbl.setStyleSheet(sheet)
        setCustomStyleSheet(lbl, sheet, sheet)
        lbl.setText(
            f'<span style="font-size:{HELP_BODY_PX}px; color:{dim};">'
            f"{text.replace(chr(10), '<br>')}</span>"
        )

        def _open(href: str) -> None:
            QDesktopServices.openUrl(QUrl(href))

        lbl.linkActivated.connect(_open)
    else:
        lbl.setText(text)
        style_help_label(lbl, HELP_BODY_PX, dim)
    lbl.setFixedHeight(max(1, lbl.heightForWidth(text_max_width)))


# Stem legend lines: "bass — Kick, snare…" → chip + description (CTk About).
_STEM_LEGEND_LINE_RE = re.compile(
    r"^(?P<stem>instrumental|vocals|bass|drums|other|acapella|vocal|original|skip)"
    r"\s+[—–-]\s+(?P<desc>.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _parse_stem_legend_lines(body: HelpSectionBody) -> Optional[list[tuple[str, str]]]:
    """Return ``(stem, desc)`` rows when every non-empty line is a stem legend."""
    if isinstance(body, str):
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    else:
        lines = [str(ln).strip() for ln in body if str(ln).strip()]
    if not lines:
        return None
    parsed: list[tuple[str, str]] = []
    for line in lines:
        m = _STEM_LEGEND_LINE_RE.match(line)
        if not m:
            return None
        parsed.append((m.group("stem").lower(), m.group("desc").strip()))
    return parsed


def _help_stem_chip_font() -> QFont:
    """Same Arial bold chip face as LogPanel / CTk About legend."""
    from .log_panel import LOG_CHIP_FONT_PX

    font = QFont("Arial")
    font.setPixelSize(LOG_CHIP_FONT_PX)
    font.setBold(True)
    if not QFontMetrics(font).inFont(ord("A")):
        font = QFont(theme.FONT_FAMILY_SEMIBOLD)
        font.setPixelSize(LOG_CHIP_FONT_PX)
        font.setBold(True)
    return font


def _help_stem_chip_width() -> int:
    """Shared chip width across all About legend sections (match log chips).

    Per-section max made 4-stem / skip chips look smaller than 2-stem
    (instrumental). Use longest known stem label + 16px like LogPanel.
    """
    from .log_panel import LOG_STEM_COLORS

    fm = QFontMetrics(_help_stem_chip_font())
    labels = list(LOG_STEM_COLORS) + ["skip"]
    longest = max(labels, key=len)
    return max(fm.horizontalAdvance(longest), fm.horizontalAdvance("n" * 9)) + 16


def _help_stem_chip(parent: QWidget, label: str, *, min_width: int = 0) -> QLabel:
    """Colored stem badge — fills from ``log_panel.LOG_STEM_COLORS`` (log chips)."""
    from .log_panel import LOG_CHIP_FONT_PX, LOG_SKIP_COLOR, LOG_STEM_COLORS

    key = label.strip().lower()
    bg = LOG_SKIP_COLOR if key == "skip" else LOG_STEM_COLORS.get(
        key, theme.COLORS["panel2"]
    )
    font = _help_stem_chip_font()
    fm = QFontMetrics(font)
    pad_h = 8
    w = max(min_width, fm.horizontalAdvance(key) + pad_h * 2)
    h = max(fm.height() + 4, LOG_CHIP_FONT_PX + 6)

    chip = QLabel(key, parent)
    chip.setObjectName("HelpStemChip")
    chip.setFont(font)
    chip.setAlignment(Qt.AlignCenter)
    chip.setFixedSize(w, h)
    chip.setStyleSheet(
        f"""
        QLabel#HelpStemChip {{
            background-color: {bg};
            color: {theme.COLORS["log_fg"]};
            border: none;
            padding: 0px;
        }}
        """
    )
    return chip


class _RepoLinkLabel(BodyLabel):
    """Accent repo link — hover brightens like CTk About 'View on GitHub'."""

    def __init__(
        self,
        text: str,
        url: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        # Parent-only: Fluent's str overload can re-enter subclass __init__.
        super().__init__(parent)
        self._url = url
        self.setText(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setAlignment(Qt.AlignHCenter)
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent"])

    def enterEvent(self, event) -> None:  # noqa: N802
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent_hover"])
        font = self.font()
        font.setUnderline(True)
        self.setFont(font)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent"])
        font = self.font()
        font.setUnderline(False)
        self.setFont(font)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._url:
            webbrowser.open(self._url)
        super().mousePressEvent(event)


def _help_section_card(
    parent: QWidget,
    title: str,
    body: HelpSectionBody,
    *,
    text_max_width: int,
    rhythm: HelpRhythm = "renamer",
) -> QFrame:
    """CTk help section: bordered card, accent uppercase title, dim body."""
    t = theme.DARK
    r = theme.CARD_CORNER_RADIUS
    card_spacing = (
        HELP_ABOUT_CARD_SPACING if rhythm == "classify" else HELP_CARD_SPACING
    )
    card = QFrame(parent)
    card.setObjectName("HelpSection")
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    card.setStyleSheet(
        f"""
        QFrame#HelpSection {{
            background-color: {t['card']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(*HELP_CARD_MARGINS)
    lay.setSpacing(card_spacing)

    hdr = CaptionLabel(title.upper())
    hdr.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    style_help_label(hdr, HELP_SECTION_TITLE_PX, t["accent_hover"], bold=True)
    lay.addWidget(hdr)

    legend = _parse_stem_legend_lines(body)
    if legend is not None:
        # Shared width across 2-stem / 4-stem / skip (not per-section max).
        chip_w = _help_stem_chip_width()
        desc_max = max(120, text_max_width - chip_w - 12)
        for stem, desc in legend:
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(12)
            row.addWidget(_help_stem_chip(card, stem, min_width=chip_w), 0, Qt.AlignTop)
            desc_lbl = BodyLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setMaximumWidth(desc_max)
            style_help_label(desc_lbl, HELP_BODY_PX, t["text_dim"])
            row.addWidget(desc_lbl, 1)
            lay.addLayout(row)
    elif isinstance(body, (list, tuple)) and any(
        _help_code_text(str(ln)) for ln in body
    ):
        # Mixed prose + <pre> command chips (Convert Sources, etc.).
        prose_buf: list[str] = []

        def _flush_prose() -> None:
            nonlocal prose_buf
            if not prose_buf:
                return
            body_lbl = BodyLabel()
            _fill_help_body_label(
                body_lbl, "\n".join(prose_buf), text_max_width=text_max_width
            )
            lay.addWidget(body_lbl)
            prose_buf = []

        for raw in body:
            line = str(raw).rstrip()
            if not line.strip():
                continue
            code = _help_code_text(line)
            if code is not None:
                _flush_prose()
                lay.addWidget(
                    _help_code_block(card, code, text_max_width=text_max_width)
                )
            else:
                prose_buf.append(line)
        _flush_prose()
    else:
        body_lbl = BodyLabel()
        _fill_help_body_label(
            body_lbl, _section_body_text(body), text_max_width=text_max_width
        )
        lay.addWidget(body_lbl)
    return card


def help_dialog(
    parent: QWidget,
    *,
    title: str = "Help",
    heading: str = "",
    subtitle: str = "",
    version_line: str = "",
    intro: str = "",
    sections: Optional[Sequence[tuple[str, HelpSectionBody]]] = None,
    footer_note: str = "Hover over individual controls for more detail.",
    width: int = HELP_DIALOG_WIDTH,
    header_icon: Optional[Union[str, Path]] = None,
    repo_url: Optional[str] = None,
    repo_link_text: str = "View on GitHub",
    rhythm: HelpRhythm = "renamer",
) -> None:
    """Show a help dialog styled like CTk ``show_ctk_help_dialog``.

    ``sections`` is a list of ``(header, body)`` where body is a string or
    a sequence of lines (joined with newlines — no bullet glyphs).

    Header hierarchy (when provided): ``heading`` → ``subtitle`` →
    ``version_line`` (bold version token + regular byline) → ``intro`` (dim tagline).

    Optional ``header_icon`` / ``repo_url`` mirror CTk About (logo + GitHub link).

    ``rhythm``: ``\"renamer\"`` (default) matches Track Renamer help spacing for
    Match / Align / Genre / Gender. ``\"classify\"`` keeps About Classify gaps.
    """
    t = theme.DARK
    is_about = rhythm == "classify"
    title_to_version = HELP_ABOUT_TITLE_TO_VERSION if is_about else HELP_TITLE_TO_INTRO
    version_to_intro = HELP_ABOUT_VERSION_TO_INTRO if is_about else HELP_TITLE_TO_INTRO
    header_to_cards = (
        HELP_ABOUT_HEADER_TO_CARDS if is_about else HELP_HEADER_TO_CARDS
    )
    footer_top = HELP_ABOUT_FOOTER_TOP if is_about else HELP_FOOTER_TOP

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    host = parent.window() if parent is not None else None
    dlg_w = int(width)
    if host is not None:
        # Stay inside the parent window with a little margin (not full monitor).
        dlg_w = min(dlg_w, max(640, host.frameGeometry().width() - 80))
    else:
        screen = dlg.screen()
        if screen is not None:
            dlg_w = min(dlg_w, max(640, screen.availableGeometry().width() - 80))
    dlg.setFixedWidth(dlg_w)

    # Usable wrap width: dialog − outer − shell pads − scrollbar − card/scroll gap
    text_max = max(480, dlg_w - 24 - 44 - 10 - HELP_SCROLL_RIGHT_GAP - 28)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(*HELP_OUTER_MARGINS)

    shell = QFrame()
    shell.setObjectName("HelpCard")
    shell.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    r = theme.DIALOG_CORNER_RADIUS
    shell.setStyleSheet(
        f"""
        QFrame#HelpCard {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    layout = QVBoxLayout(shell)
    layout.setContentsMargins(*HELP_SHELL_MARGINS)
    layout.setSpacing(0)

    # Scrollable body (heading + intro + section cards); footer stays pinned
    body_host = QWidget()
    body_host.setObjectName("HelpScrollBody")
    body_host.setStyleSheet("QWidget#HelpScrollBody { background: transparent; }")
    body_lay = QVBoxLayout(body_host)
    # Right gap keeps section cards clear of the vertical scrollbar
    body_lay.setContentsMargins(0, 0, HELP_SCROLL_RIGHT_GAP, 0)
    body_lay.setSpacing(0)

    if header_icon is not None:
        icon_path = Path(header_icon)
        if icon_path.exists():
            pix = QPixmap(str(icon_path))
            if not pix.isNull():
                scaled = pix.scaled(
                    HELP_ABOUT_ICON_PX,
                    HELP_ABOUT_ICON_PX,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                icon_lbl = QLabel(body_host)
                icon_lbl.setPixmap(scaled)
                icon_lbl.setAlignment(Qt.AlignHCenter)
                icon_lbl.setStyleSheet("background: transparent; border: none;")
                # Keep pixmap alive for the dialog lifetime
                dlg._help_icon_pix = scaled  # type: ignore[attr-defined]
                body_lay.addWidget(icon_lbl)
                body_lay.addSpacing(HELP_ICON_TO_HEADING)

    center_header = header_icon is not None

    if heading:
        head = BodyLabel(heading)
        head.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        if center_header:
            head.setAlignment(Qt.AlignHCenter)
        style_help_label(head, HELP_HEADING_PX, t["text"], bold=True)
        body_lay.addWidget(head)

    if subtitle:
        sub = BodyLabel(subtitle)
        sub.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        if center_header:
            sub.setAlignment(Qt.AlignHCenter)
        style_help_label(sub, HELP_BODY_PX, t["text_dim"])
        body_lay.addWidget(sub)

    if version_line:
        # Breathing room under title before version/by line (was 0)
        if heading or subtitle:
            body_lay.addSpacing(title_to_version)
        # Mixed weight: bold "v1.0.6" + regular " — by …" when separator present
        sep = " — "
        if sep in version_line:
            ver_part, by_part = version_line.split(sep, 1)
            row = QWidget(body_host)
            row.setStyleSheet("background: transparent; border: none;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(0)
            if center_header:
                row_lay.addStretch(1)
            ver = BodyLabel(ver_part)
            by = BodyLabel(f"{sep}{by_part}")
            for lbl in (ver, by):
                lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            # Match status bar credit (theme.F_STATUS / STATUS_FONT_PX)
            style_help_label(ver, theme.STATUS_FONT_PX, t["text"], bold=True)
            style_help_label(by, theme.STATUS_FONT_PX, t["text"], bold=False)
            row_lay.addWidget(ver, 0, Qt.AlignVCenter)
            row_lay.addWidget(by, 0, Qt.AlignVCenter)
            if center_header:
                row_lay.addStretch(1)
            body_lay.addWidget(row)
        else:
            ver = BodyLabel(version_line)
            ver.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            if center_header:
                ver.setAlignment(Qt.AlignHCenter)
            style_help_label(ver, theme.STATUS_FONT_PX, t["text"], bold=True)
            body_lay.addWidget(ver)

    if intro:
        # Title block → intro: tight under version_line, else title→intro gap.
        if version_line:
            body_lay.addSpacing(version_to_intro)
        elif heading or subtitle:
            body_lay.addSpacing(HELP_TITLE_TO_INTRO)
        ilbl = BodyLabel(body_host)
        ilbl.setWordWrap(True)
        ilbl.setMaximumWidth(text_max)
        ilbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        if center_header:
            ilbl.setAlignment(Qt.AlignHCenter)
        if "<a " in intro.lower():
            # RichText uses the widget's QFont, not stylesheet font-size — pin it
            # so the intro matches section card body text (HELP_BODY_PX).
            dim = t["text_dim"]
            accent = t["accent"]
            style_help_label(ilbl, HELP_BODY_PX, dim)
            font = QFont(theme.FONT_FAMILY)
            font.setPixelSize(HELP_BODY_PX)
            font.setWeight(QFont.Weight.Normal)
            ilbl.setFont(font)
            sheet = f"""
                FluentLabelBase, BodyLabel, QLabel {{
                    color: {dim};
                    font-family: "{theme.FONT_FAMILY}";
                    font-size: {HELP_BODY_PX}px;
                    font-weight: 400;
                    background: transparent;
                }}
                a {{
                    color: {accent};
                    text-decoration: none;
                }}
            """
            ilbl.setTextFormat(Qt.RichText)
            ilbl.setOpenExternalLinks(True)
            ilbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
            ilbl.setStyleSheet(sheet)
            setCustomStyleSheet(ilbl, sheet, sheet)
            ilbl.setText(
                f'<span style="font-size:{HELP_BODY_PX}px; color:{dim};">'
                f"{intro}</span>"
            )
        else:
            ilbl.setText(intro)
            style_help_label(ilbl, HELP_BODY_PX, t["text_dim"])
        body_lay.addWidget(ilbl)

    if repo_url:
        body_lay.addSpacing(HELP_INTRO_TO_REPO)
        body_lay.addWidget(
            _RepoLinkLabel(repo_link_text, repo_url, parent=body_host),
            0,
            Qt.AlignHCenter,
        )

    # Breathing room under title + intro before the first card
    body_lay.addSpacing(header_to_cards)

    for i, (header, body) in enumerate(sections or []):
        if i:
            body_lay.addSpacing(HELP_CARD_GAP)
        body_lay.addWidget(
            _help_section_card(
                body_host,
                header,
                body,
                text_max_width=text_max,
                rhythm=rhythm,
            )
        )

    scroll = QScrollArea()
    scroll.setWidget(body_host)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    scroll.setStyleSheet(
        f"""
        QScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollBar:vertical {{
            background: {t['panel']};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {t['border']};
            border-radius: 4px;
            min-height: 24px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        """
    )
    layout.addWidget(scroll)

    footer = QHBoxLayout()
    # Match body_lay right inset so Close lines up with section card edges
    footer.setContentsMargins(0, footer_top, HELP_SCROLL_RIGHT_GAP, 0)
    footer.setSpacing(HELP_FOOTER_SPACING)
    if footer_note:
        fn = CaptionLabel(footer_note)
        style_help_label(fn, HELP_FOOTER_PX, t["text_dim"])
        footer.addWidget(fn, 1)
    else:
        footer.addStretch(1)
    close_btn = action_button(
        "Close", on_click=dlg.accept, accent=True, parent=shell, tip="Close this dialog."
    )
    close_btn.setMinimumWidth(72)
    footer.addWidget(close_btn, 0, Qt.AlignRight | Qt.AlignVCenter)
    layout.addLayout(footer)

    outer.addWidget(shell)

    # Fit scroll to content when short; cap when taller than parent/screen
    body_host.adjustSize()
    content_h = max(body_host.sizeHint().height(), body_host.minimumSizeHint().height())
    host = parent.window() if parent is not None else None
    max_body = 820
    if host is not None:
        # Leave room for shell chrome + footer + outer margins
        max_body = max(400, int(host.frameGeometry().height() * 0.92) - 120)
    else:
        screen = dlg.screen()
        if screen is not None:
            max_body = max(400, int(screen.availableGeometry().height() * 0.92) - 120)
    scroll.setFixedHeight(min(content_h + 4, max_body))
    dlg.adjustSize()

    # Center on parent window (CTk geometry)
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(parent):
        dlg.exec()

"""Renamer help dialog — QDialog port of track_renamer.gui.help_dialog."""
from __future__ import annotations

import webbrowser
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CaptionLabel

from .. import theme
from ..widgets.action_button import action_button
from ..widgets.dialogs import (
    HELP_BODY_PX,
    HELP_CARD_GAP,
    HELP_CARD_MARGINS,
    HELP_FOOTER_PX,
    HELP_FOOTER_SPACING,
    HELP_HEADER_TO_CARDS,
    HELP_HEADING_PX,
    HELP_OUTER_MARGINS,
    HELP_RENAME_FOOTER_TOP,
    HELP_SECTION_TITLE_PX,
    HELP_SHELL_MARGINS,
    HELP_TITLE_TO_INTRO,
    _help_section_card,
    dim_behind,
    style_help_label,
)

# Match previous wrap width for this non-scroll rename help card
_RENAME_HELP_TEXT_MAX = 620

DOCS_URL = "https://enableton-renamer-docs.vercel.app/"


def _key_chip(parent: QWidget, text: str) -> BodyLabel:
    """CTk-style keyboard key badge (border_soft fill, mono text)."""
    from qfluentwidgets import setCustomStyleSheet

    t = theme.DARK
    chip = BodyLabel(text, parent)
    chip.setAlignment(Qt.AlignCenter)
    if hasattr(chip, "setTextColor"):
        chip.setTextColor(t["text"], t["text"])
    sheet = f"""
        FluentLabelBase, BodyLabel, QLabel {{
            color: {t['text']};
            background-color: {t['border_soft']};
            border-radius: 4px;
            padding: 3px 7px;
            font-family: "{theme.FONT_FAMILY_MONO}";
            font-size: {HELP_FOOTER_PX}px;
            font-weight: 700;
        }}
    """
    chip.setStyleSheet(sheet)
    setCustomStyleSheet(chip, sheet, sheet)
    return chip


class _DocsLink(BodyLabel):
    """Accent docs link — theme tokens, hover brightens like CTk."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Parent-only: Fluent's str overload does self.__init__(parent), which
        # re-enters a subclass __init__ that passes text → RecursionError.
        super().__init__(parent)
        self.setText("Open full rule reference")
        self.setCursor(Qt.PointingHandCursor)
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent"])

    def enterEvent(self, event) -> None:  # noqa: N802
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent_hover"])
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        style_help_label(self, HELP_BODY_PX, theme.DARK["accent"])
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            webbrowser.open(DOCS_URL)
        super().mousePressEvent(event)


def show_rename_help_dialog(parent: Optional[QWidget] = None) -> None:
    t = theme.DARK
    dlg = QDialog(parent)
    dlg.setWindowTitle("Rename Files — Help")
    dlg.setModal(True)
    dlg.setMinimumWidth(680)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(*HELP_OUTER_MARGINS)

    card = QFrame()
    card.setObjectName("HelpCard")
    r = theme.DIALOG_CORNER_RADIUS
    card.setStyleSheet(
        f"""
        QFrame#HelpCard {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*HELP_SHELL_MARGINS)
    # setSpacing(0) + explicit addSpacing — same tokens as dialogs.help_dialog
    layout.setSpacing(0)

    title = BodyLabel("Track Renamer")
    title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    style_help_label(title, HELP_HEADING_PX, t["text"], bold=True)
    layout.addWidget(title)
    layout.addSpacing(HELP_TITLE_TO_INTRO)

    sub = BodyLabel("Build a rule stack, verify every filename, then rename safely")
    sub.setWordWrap(True)
    sub.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    style_help_label(sub, HELP_BODY_PX, t["text_dim"])
    layout.addWidget(sub)
    layout.addSpacing(HELP_HEADER_TO_CARDS)

    layout.addWidget(_help_section_card(
        card, "Workflow",
        "1. Browse & select a folder. Choose whether subfolders are included.\n"
        "2. Add, reorder, enable, or edit rules on the left.\n"
        "3. Click Apply only when rules have changed, then inspect the preview.\n"
        "4. Select the files you want and click Rename.",
        text_max_width=_RENAME_HELP_TEXT_MAX,
    ))
    layout.addSpacing(HELP_CARD_GAP)
    layout.addWidget(_help_section_card(
        card, "Preview & Audio",
        "Original names are struck through; the resulting category badge and filename "
        "appear after the arrow. Click a row to load its waveform. Space toggles playback, "
        "Left/Right seeks 3 seconds, and Up/Down changes the active file.",
        text_max_width=_RENAME_HELP_TEXT_MAX,
    ))
    layout.addSpacing(HELP_CARD_GAP)
    layout.addWidget(_help_section_card(
        card, "Safe File Operations",
        "Extensions are preserved. Naming conflicts receive _1, _2, and so on. After renaming, "
        "you can optionally choose a destination where BASS, DRUMS, VOCALS, and other prefix "
        "folders are created automatically.",
        text_max_width=_RENAME_HELP_TEXT_MAX,
    ))
    layout.addSpacing(HELP_CARD_GAP)
    layout.addWidget(_help_section_card(
        card, "Sources",
        "Auto-detect uses "
        '<a href="https://research.atspotify.com/publications/openmic-2018-an-open-dataset-for-multiple-instrument-recognition">'
        "OpenMIC-2018</a> instrument labels via "
        '<a href="https://github.com/kkoutini/passt_hear21">hear21passt</a> '
        "(PaSST), mapped to Category Macro PREFIX rows.",
        text_max_width=_RENAME_HELP_TEXT_MAX,
    ))

    keys = QFrame(card)
    keys.setObjectName("HelpSection")
    keys.setStyleSheet(
        f"""
        QFrame#HelpSection {{
            background-color: {t['card']};
            border: 1px solid {t['border']};
            border-radius: {theme.CARD_CORNER_RADIUS}px;
        }}
        """
    )
    keys_layout = QGridLayout(keys)
    keys_layout.setContentsMargins(*HELP_CARD_MARGINS)
    keys_layout.setHorizontalSpacing(12)
    keys_layout.setVerticalSpacing(8)
    h = CaptionLabel("KEYBOARD")
    style_help_label(h, HELP_SECTION_TITLE_PX, t["accent_hover"], bold=True)
    keys_layout.addWidget(h, 0, 0, 1, 4)
    for index, (k, v) in enumerate((
        ("↑ / ↓", "Select file"),
        ("Page Up / Down", "Scroll preview"),
        ("Enter", "Toggle include ✓"),
        ("Space", "Play / pause"),
        ("← / →", "Seek 3 seconds"),
        ("Shift / Ctrl-click", "Multi-select (then right-click → Change to:)"),
    )):
        column = (index % 2) * 2
        row = 1 + index // 2
        keys_layout.addWidget(_key_chip(keys, k), row, column, Qt.AlignLeft)
        desc = CaptionLabel(v)
        style_help_label(desc, HELP_FOOTER_PX, t["text_dim"])
        keys_layout.addWidget(desc, row, column + 1, Qt.AlignLeft | Qt.AlignVCenter)
    layout.addSpacing(HELP_CARD_GAP)
    layout.addWidget(keys)

    layout.addSpacing(HELP_CARD_GAP)
    layout.addWidget(_DocsLink(card))

    btn_row = QHBoxLayout()
    btn_row.setContentsMargins(0, HELP_RENAME_FOOTER_TOP, 0, 0)
    btn_row.setSpacing(HELP_FOOTER_SPACING)
    note = CaptionLabel("Hover over individual controls for more detail.")
    style_help_label(note, HELP_FOOTER_PX, t["text_dim"])
    btn_row.addWidget(note, 1)
    close_btn = action_button(
        "Close", on_click=dlg.accept, accent=True, parent=card, tip="Close this dialog."
    )
    close_btn.setMinimumWidth(72)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    outer.addWidget(card)
    with dim_behind(parent):
        dlg.exec()

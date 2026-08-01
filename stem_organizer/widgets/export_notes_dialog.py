"""Optional NOTES text before Charts report export."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, PlainTextEdit, StrongBodyLabel

from .. import theme
from .action_button import action_button
from .dialogs import dim_behind


def ask_export_notes(parent: QWidget, *, initial_text: str = "") -> Optional[str]:
    """Prompt for NOTES card text. Return text (may be empty) or None if cancelled."""
    t = theme.DARK
    host = parent.window() if parent is not None else parent
    dlg = QDialog(host)
    dlg.setWindowTitle("Report notes")
    dlg.setModal(True)
    dlg.setMinimumWidth(440)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(12, 12, 12, 12)

    card = QFrame()
    card.setObjectName("ExportNotesCard")
    r = theme.DIALOG_CORNER_RADIUS
    card.setStyleSheet(
        f"""
        QFrame#ExportNotesCard {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(20, 18, 20, 16)
    lay.setSpacing(10)

    title = StrongBodyLabel("NOTES")
    if hasattr(title, "setTextColor"):
        title.setTextColor(t["text"], t["text"])
    lay.addWidget(title)

    hint = BodyLabel(
        "Optional text for the NOTES section on PNG and PDF reports. "
        "Leave blank to keep it empty."
    )
    hint.setWordWrap(True)
    dim = theme.COLORS["fg_dim"]
    if hasattr(hint, "setTextColor"):
        hint.setTextColor(dim, dim)
    lay.addWidget(hint)

    edit = PlainTextEdit()
    edit.setPlaceholderText("Type notes for this export…")
    edit.setFixedHeight(160)
    note_px = theme.BODY_FONT_PX + 2
    bg = theme.CONTROL_BG
    edit.setStyleSheet(
        f"""
        PlainTextEdit {{
            background-color: {bg};
            color: {theme.COLORS['fg']};
            border: 1px solid {theme.COLORS['border']};
            border-radius: 6px;
            padding: 8px;
            font-family: "{theme.FONT_FAMILY}";
            font-size: {note_px}px;
        }}
        PlainTextEdit:focus {{
            background-color: {theme.INPUT_FOCUS_BG};
            border: 1px solid {theme.COLORS['accent']};
        }}
        """
    )
    lay.addWidget(edit)
    if initial_text:
        edit.setPlainText(initial_text)

    row = QHBoxLayout()
    row.addStretch(1)
    cancel_btn = action_button("Cancel", on_click=dlg.reject, parent=card)
    ok_btn = action_button("Continue…", on_click=dlg.accept, accent=True, parent=card)
    row.addWidget(cancel_btn)
    row.addWidget(ok_btn)
    lay.addLayout(row)
    outer.addWidget(card)

    edit.setFocus()
    dlg.adjustSize()
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(host):
        if dlg.exec() != QDialog.Accepted:
            return None
    return edit.toPlainText().strip()

"""Section card — compact titled panel (dense left-column layout).

Uses a lightweight QFrame instead of Fluent HeaderCardWidget, which is too
tall (48px header + 24px padding) for this app's settings columns.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel

from .. import theme


class Section(QFrame):
    """A titled card. Body content is packed into ``self.body``."""

    def __init__(self, parent: QWidget, title: str) -> None:
        super().__init__(parent)
        self.setObjectName("Section")
        # Hug content — don't absorb leftover scroll-viewport height into the card
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        # Title↔card and section↔section share SECTION_GAP for even rhythm
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.SECTION_GAP)
        if title:
            lbl = CaptionLabel(title.upper())
            lbl.setObjectName("SectionTitle")
            lbl.setFont(theme.F_SECTION)
            outer.addWidget(lbl)

        self.body = QFrame()
        self.body.setObjectName("Card")
        self.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(10, 8, 10, 8)
        body_layout.setSpacing(theme.CTRL_FIELD_PAD)
        body_layout.setAlignment(Qt.AlignTop)
        outer.addWidget(self.body)


def section(parent: QWidget, title: str) -> Section:
    """Convenience factory mirroring ui_theme.ctk_section."""
    return Section(parent, title)

"""Info icon — clickable "?" hover-bright circle.

Port of stem_organizer_ui.InfoIcon (tk.Canvas → small QWidget paintEvent).
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .. import theme


class InfoIcon(QWidget):
    """Small "?" badge that opens a help dialog on click."""

    def __init__(
        self,
        parent: QWidget,
        on_click: Optional[Callable[[], None]] = None,
        *,
        size: int = 18,
    ) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Open help for this panel.")
        self._on_click = on_click
        self._hover = False

    def set_on_click(self, cb: Callable[[], None]) -> None:
        self._on_click = cb

    def enterEvent(self, event) -> None:  # noqa: N802 Qt name
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 Qt name
        self._hover = False
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 Qt name
        if event.button() == Qt.LeftButton and self._on_click is not None:
            self._on_click()

    def paintEvent(self, event) -> None:  # noqa: N802 Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        dim = QColor(theme.DARK["text_dim"])
        bright = QColor(theme.DARK["text"])
        ring_color = bright if self._hover else dim
        text_color = bright if self._hover else dim
        w, h = self.width(), self.height()
        r = min(w, h) / 2 - 1
        rect = QRectF(w / 2 - r, h / 2 - r, 2 * r, 2 * r)
        pen = QPen(ring_color, 1.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(rect)
        p.setPen(text_color)
        font = p.font()
        font.setBold(True)
        font.setPointSize(max(7, int(r * 1.1)))
        p.setFont(font)
        p.drawText(rect, Qt.AlignCenter, "?")

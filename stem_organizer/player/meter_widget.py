"""Peak meter widget — vertical gradient bar (green→yellow→red)."""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

from .. import theme


def meter_color(level: float) -> QColor:
    """Green → yellow → red based on 0..1 level."""
    level = max(0.0, min(1.0, level))
    if level < 0.6:
        # green to yellow
        t = level / 0.6
        r = int(46 + (245 - 46) * t)
        g = int(204 + (158 - 204) * t)
        b = int(113 + (11 - 113) * t)
    else:
        # yellow to red
        t = (level - 0.6) / 0.4
        r = int(245 + (239 - 245) * t)
        g = int(158 + (68 - 158) * t)
        b = int(11 + (68 - 11) * t)
    return QColor(r, g, b)


class MeterWidget(QWidget):
    """Paints a vertical level bar that decays each tick."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(72, 14)
        self._level = 0.0

    def set_level(self, level: float) -> None:
        # Decay: keep max of new and slightly-decayed previous for a smoother fall
        target = max(0.0, min(1.0, level))
        self._level = max(target, self._level * 0.85)
        self.update()

    def reset(self) -> None:
        self._level = 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(theme.COLORS["status_trough"]))
        fill_w = int(round(w * self._level))
        if fill_w > 0:
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(46, 204, 113))
            grad.setColorAt(0.6, QColor(245, 158, 11))
            grad.setColorAt(1.0, QColor(239, 68, 68))
            p.fillRect(0, 0, fill_w, h, grad)

"""Colored vertical-bar icon for category menu items.

Renders the same "vertical dash" prefix the samplepack chips use, as a small
QIcon so it can be attached to a qfluentwidgets ``Action`` in a ``RoundMenu``.
RoundMenu aligns all items to the icon column once any item has an icon, so a
matched/colored bar on each category item lines up cleanly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

# Bar geometry, tuned for the default RoundMenu icon size (16x16).
_BAR_W = 3
_BAR_H = 14
_PIXMAP = 16


def category_bar_icon(color_hex: str) -> QIcon:
    """Return a 16×16 icon holding a vertical bar filled with ``color_hex``.

    Transparent background; the bar is centered horizontally. Falls back to a
    muted dim color when ``color_hex`` is empty/invalid.
    """
    color = QColor(color_hex)
    if not color.isValid():
        color = QColor("#9aa0b4")  # text_dim — same as unmatched chips

    pm = QPixmap(_PIXMAP, _PIXMAP)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    x = (_PIXMAP - _BAR_W) // 2
    y = (_PIXMAP - _BAR_H) // 2
    p.drawRoundedRect(x, y, _BAR_W, _BAR_H, _BAR_W / 2, _BAR_W / 2)
    p.end()
    return QIcon(pm)

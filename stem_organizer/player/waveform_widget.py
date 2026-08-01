"""Waveform widget — QPainter polygon + playhead + filename overlay.

Replaces stem_player._draw_waveform / _update_playhead / _draw_waveform_filename
(the tk.Canvas calls). The window owns the math (view_start, view_zoom, peaks);
this widget just paints whatever the window tells it to via set_peaks().

The waveform polygon is cached in a QPixmap so playhead ticks (every ~33 ms)
only blit the cache and draw a 1 px line — not rebuild thousands of path points.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme


def _max_pool_peaks(peaks: np.ndarray, bins: int) -> np.ndarray:
    """Downsample peak amplitudes with per-column max (envelope-safe)."""
    bins = max(1, int(bins))
    arr = np.asarray(peaks, dtype=np.float32).ravel()
    n = int(arr.size)
    if n == 0:
        return np.zeros(bins, dtype=np.float32)
    if n <= bins:
        return arr
    bucket = np.minimum((np.arange(n, dtype=np.int64) * bins) // n, bins - 1)
    out = np.zeros(bins, dtype=np.float32)
    np.maximum.at(out, bucket, arr)
    return out


class WaveformWidget(QWidget):
    """Paints one track's waveform + playhead + filename overlay."""

    # Fraction 0..1 across the drawable width (avoids stale cached pixel widths).
    clicked = Signal(float)
    # Emitted when drawable width changes so the parent can re-bin peaks.
    width_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self._peaks: Optional[np.ndarray] = None
        self._color: QColor = QColor(theme.COLORS["accent"])
        self._dimmed: bool = False
        self._playhead_x: Optional[float] = None
        self._filename: str = ""
        self._wave_pixmap: Optional[QPixmap] = None
        self._wave_cache_valid = False

    # ----- configuration -----

    def set_peaks(self, peaks: Optional[np.ndarray]) -> None:
        self._peaks = peaks
        self._invalidate_wave_cache()
        self.update()

    def set_color(self, hex_color: str, *, dimmed: bool = False) -> None:
        color = QColor(hex_color)
        if dimmed:
            color = self._blend(color, QColor(theme.COLORS["log_bg"]), 0.68)
        if self._color == color and self._dimmed == dimmed:
            return
        self._color = color
        self._dimmed = dimmed
        self._invalidate_wave_cache()
        self.update()

    def set_playhead(self, frac: Optional[float]) -> None:
        """Playhead as 0..1 across the drawable width, or None to hide."""
        if frac is self._playhead_x:
            return
        if (
            frac is not None
            and self._playhead_x is not None
            and abs(frac - self._playhead_x) < 1e-5
        ):
            return
        self._playhead_x = frac
        self.update()

    def set_filename(self, text: str) -> None:
        if text == self._filename:
            return
        self._filename = text
        self._invalidate_wave_cache()
        self.update()

    def _invalidate_wave_cache(self) -> None:
        self._wave_cache_valid = False

    @staticmethod
    def _blend(fg: QColor, bg: QColor, t: float) -> QColor:
        r = int(fg.red() * (1 - t) + bg.red() * t)
        g = int(fg.green() * (1 - t) + bg.green() * t)
        b = int(fg.blue() * (1 - t) + bg.blue() * t)
        return QColor(r, g, b)

    # ----- paint -----

    def mousePressEvent(self, event) -> None:  # noqa: N802 Qt name
        if event.button() == Qt.LeftButton:
            # Map click against this widget's live width (CTk uses event.widget
            # winfo_width). Emitting a fraction avoids stale _wave_w caches and
            # any DPI/margin mismatch between click X and a foreign width.
            rect = self.contentsRect()
            w = max(1.0, float(rect.width()))
            x = float(event.position().x()) - float(rect.x())
            self.clicked.emit(max(0.0, min(1.0, x / w)))

    def resizeEvent(self, event) -> None:  # noqa: N802
        old_w = event.oldSize().width()
        new_w = event.size().width()
        self._invalidate_wave_cache()
        super().resizeEvent(event)
        if old_w != new_w and new_w >= 2:
            self.width_changed.emit()

    def paintEvent(self, event) -> None:  # noqa: N802 Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        rect = self.contentsRect()
        # Full-widget fill (margins outside contentsRect)
        p.fillRect(self.rect(), QColor(theme.COLORS["log_bg"]))

        if not self._wave_cache_valid or self._wave_pixmap is None:
            self._rebuild_wave_pixmap(rect)

        pm = self._wave_pixmap
        if pm is not None and not pm.isNull():
            p.drawPixmap(rect.topLeft(), pm)

        self._draw_playhead(p, rect.x(), rect.y(), rect.width(), rect.height())

    def _rebuild_wave_pixmap(self, rect) -> None:
        w = max(1, rect.width())
        h = max(1, rect.height())
        pm = QPixmap(w, h)
        pm.fill(QColor(theme.COLORS["log_bg"]))
        qp = QPainter(pm)
        qp.setRenderHint(QPainter.Antialiasing, False)

        peaks = self._peaks
        if peaks is None or len(peaks) == 0:
            qp.setPen(QPen(QColor(theme.COLORS["border"]), 1))
            qp.drawLine(0, h // 2, w, h // 2)
        else:
            peaks = np.asarray(peaks, dtype=np.float32)
            # Parent normally sends one bin per pixel. If denser, max-pool —
            # never linspace-pick (that drops peaks → chunky triangles).
            if peaks.size > w:
                peaks = _max_pool_peaks(peaks, w)
            n = int(peaks.size)
            mid = h / 2.0
            max_amp = h / 2.0 - 6
            amps = np.minimum(peaks, 1.0) * max_amp
            bar_w = w / max(1, n)
            x_left = np.arange(n, dtype=np.float32) * bar_w
            x_right = x_left + max(1.0, bar_w - 0.5)
            y_top = mid - amps
            y_bot = mid + amps

            # Visit both left and right of each column so wide bars stay
            # rectangular instead of diagonal polygons between sparse points.
            path = QPainterPath()
            path.moveTo(float(x_left[0]), float(y_top[0]))
            for i in range(n):
                path.lineTo(float(x_left[i]), float(y_top[i]))
                path.lineTo(float(x_right[i]), float(y_top[i]))
            for i in range(n - 1, -1, -1):
                path.lineTo(float(x_right[i]), float(y_bot[i]))
                path.lineTo(float(x_left[i]), float(y_bot[i]))
            path.closeSubpath()
            qp.fillPath(path, self._color)

        self._draw_filename(qp, 0, 0, w, h)
        qp.end()
        self._wave_pixmap = pm
        self._wave_cache_valid = True

    def _draw_filename(self, p: QPainter, ox: int, oy: int, w: int, h: int) -> None:
        if not self._filename:
            return
        p.setRenderHint(QPainter.TextAntialiasing, True)
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        metrics = p.fontMetrics()
        text_w = metrics.horizontalAdvance(self._filename)
        text_h = metrics.height()
        pad_x, pad_y = 6, 3
        rect_x = ox + w - text_w - pad_x * 2 - 4
        rect_y = oy + h - text_h - pad_y * 2 - 4
        if rect_x < ox or rect_y < oy:
            return
        bg = QColor(theme.COLORS["panel"])
        bg.setAlpha(180)
        p.fillRect(QRectF(rect_x, rect_y, text_w + pad_x * 2, text_h + pad_y * 2), bg)
        p.setPen(QColor(theme.COLORS["fg"]))
        p.drawText(
            QRectF(rect_x + pad_x, rect_y + pad_y, text_w, text_h),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._filename,
        )

    def _draw_playhead(self, p: QPainter, ox: int, oy: int, w: int, h: int) -> None:
        if self._playhead_x is None:
            return
        # _playhead_x is 0..1 fraction of the drawable width
        x = ox + float(self._playhead_x) * w
        if ox <= x <= ox + w:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(int(round(x)), oy, int(round(x)), oy + h)

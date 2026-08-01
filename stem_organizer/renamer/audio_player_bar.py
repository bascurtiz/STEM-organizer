"""Audio player bar — port of track_renamer.gui.audio_player.AudioPlayerBar.

A play button + a custom waveform QWidget. Polls AudioPreviewService.events on
a 100 ms QTimer.

Text placement (match Stem Player wave filename overlay):
- filename (ready / playing): bottom-right chip — same style as WaveformWidget._draw_filename
- idle hint (no file): centered (no waveform axis stroke)
- other status (loading / unavailable): centered
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QMouseEvent, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget

from track_renamer.audio_preview import AudioPreviewService
from track_renamer.category_palette import (
    default_category_color,
    parse_category_prefix_display,
)

from .. import theme

# Idle empty-state hint — centered over empty waveform (no axis line).
_IDLE_HINT = "Select audio file to preview"

# Play/pause: paint QIcons — do not use U+23F8 (Segoe blue emoji tile) or
# U+275A ❚ (often missing / blank on Windows+Qt fonts).
# Do not use Fluent PushButton: its paintEvent places the icon at
# x = 12 + (width - minHint)/2, which clips off a fixed 32px transport.
_ICON_SIZE = 14
_ICON_CACHE: dict[str, QIcon] = {}


def _media_icon(kind: str) -> QIcon:
    """Monochrome play triangle / pause bars as a pixmap icon."""
    cached = _ICON_CACHE.get(kind)
    if cached is not None:
        return cached

    size = _ICON_SIZE
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(theme.COLORS["fg"]))  # light on dark muted button
    if kind == "pause":
        bar_w = max(2, size // 5)
        gap = max(2, size // 5)
        total = bar_w * 2 + gap
        x0 = (size - total) // 2
        y0 = max(2, size // 5)
        h = size - 2 * y0
        p.drawRect(x0, y0, bar_w, h)
        p.drawRect(x0 + bar_w + gap, y0, bar_w, h)
    else:
        m = max(2, size // 6)
        p.drawPolygon(
            QPolygon(
                [
                    QPoint(m, m),
                    QPoint(size - m, size // 2),
                    QPoint(m, size - m),
                ]
            )
        )
    p.end()

    icon = QIcon(pix)
    _ICON_CACHE[kind] = icon
    return icon


def _set_transport_icon(btn, *, playing: bool) -> None:
    btn.setIcon(_media_icon("pause" if playing else "play"))
    btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))


def _make_transport_button(parent: QWidget, on_click) -> QToolButton:
    """Plain icon button — Fluent PushButton clips icons on a 32px width."""
    btn = QToolButton(parent)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFixedSize(32, theme.PATH_BTN_HEIGHT)
    btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
    btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
    btn.setAutoRaise(False)
    btn.setFocusPolicy(Qt.NoFocus)
    # Match muted Fluent control chrome (CONTROL_BG), icon-only — no text padding.
    btn.setStyleSheet(
        f"""
        QToolButton {{
            background-color: {theme.CONTROL_BG};
            border: 1px solid {theme.DARK["border"]};
            border-radius: 5px;
            padding: 0px;
            margin: 0px;
        }}
        QToolButton:hover {{
            background-color: {theme.CONTROL_BG_HOVER};
        }}
        QToolButton:pressed {{
            background-color: {theme.CONTROL_BG_PRESSED};
        }}
        QToolButton:disabled {{
            background-color: {theme.CONTROL_BG};
            border: 1px solid {theme.DARK["border"]};
        }}
        """
    )
    btn.clicked.connect(on_click)
    return btn


def _is_black_prefix_color(hex_color: str) -> bool:
    """True for palette black (#000000) — invisible on dark waveform_bg."""
    raw = (hex_color or "").strip().lower()
    if raw in ("#000000", "#000", "black"):
        return True
    if not raw.startswith("#") or len(raw) not in (4, 7):
        return False
    try:
        c = QColor(raw)
        return c.isValid() and c.red() == 0 and c.green() == 0 and c.blue() == 0
    except Exception:
        return False


def preview_waveform_color(new_display: str, category_colors: dict) -> str:
    """Category/prefix color for the waveform; black → light (CTk #e6e8ef)."""
    parsed = parse_category_prefix_display(new_display or "")
    if parsed:
        cat = parsed[0]
        if category_colors and cat in category_colors:
            color = category_colors[cat]
        else:
            color = default_category_color(cat)
        # CTk: FX (default #000000) draws as #e6e8ef so the wave stays visible.
        # Same rule for any prefix that chose the black palette swatch.
        if _is_black_prefix_color(color):
            return "#e6e8ef"
        return color
    return theme.DARK["accent"]


class _WaveView(QWidget):
    """Waveform + playhead + status text (CTk canvas layout)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(theme.PATH_BTN_HEIGHT)
        self.setFixedHeight(theme.PATH_BTN_HEIGHT)
        self._peaks = ()
        self._duration = 0.0
        self._position = 0.0
        self._color = QColor(theme.DARK["accent"])
        self._status = _IDLE_HINT
        self._filename_state = False  # idle hint is centered
        self._subdued = True

    def set_peaks(self, peaks) -> None:
        self._peaks = peaks or ()
        self.update()

    def set_duration(self, duration: float) -> None:
        self._duration = max(0.0, float(duration))

    def set_position(self, position: float) -> None:
        self._position = max(0.0, float(position))
        self.update()

    def set_color(self, hex_color: str) -> None:
        self._color = QColor(hex_color)
        self.update()

    def set_status(self, text: str, *, filename_state: bool = False, subdued: bool = False) -> None:
        self._status = text or ""
        self._filename_state = bool(filename_state)
        self._subdued = bool(subdued)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._duration > 0:
            w = max(1, self.width())
            frac = max(0.0, min(1.0, event.position().x() / w))
            p = self.parent()
            if isinstance(p, AudioPlayerBar):
                p.seek(frac * self._duration)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        bg = QColor(theme.DARK["waveform_bg"])
        p.fillRect(0, 0, w, h, bg)

        peaks = self._peaks
        # Axis + waveform only when peaks exist — idle/loading stay clean behind centered text.
        if peaks:
            p.setPen(QPen(QColor(theme.DARK["waveform_axis"]), 1))
            p.drawLine(0, h // 2, w, h // 2)

            n = len(peaks)
            center = h / 2.0
            p.setPen(Qt.NoPen)
            p.setBrush(self._color)
            for x in range(w):
                idx = min(n - 1, int(x * n / w))
                lo, hi = peaks[idx]
                y1 = int(center - hi * (center - 2))
                y2 = int(center - lo * (center - 2))
                p.drawRect(x, y1, 1, max(1, y2 - y1))

            if self._duration > 0 and self._position >= 0:
                progress = min(1.0, self._position / self._duration)
                px = int(progress * max(w - 1, 1))
                p.setPen(QPen(QColor(theme.DARK["waveform_playhead"]), 1))
                p.drawLine(px, 1, px, h - 1)

        if not self._status:
            return

        p.setRenderHint(QPainter.TextAntialiasing, True)
        text = self._status

        if self._filename_state:
            # Match Stem Player WaveformWidget._draw_filename
            font = QFont(theme.FONT_FAMILY)
            font.setPointSize(7)
            p.setFont(font)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            pad_x, pad_y = 6, 3
            rect_x = w - tw - pad_x * 2 - 4
            rect_y = h - th - pad_y * 2 - 4
            if rect_x < 0 or rect_y < 0:
                return
            chip_bg = QColor(theme.COLORS["panel"])
            chip_bg.setAlpha(180)
            p.fillRect(QRectF(rect_x, rect_y, tw + pad_x * 2, th + pad_y * 2), chip_bg)
            p.setPen(QColor(theme.COLORS["fg"]))
            p.drawText(
                QRectF(rect_x + pad_x, rect_y + pad_y, tw, th),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )
        else:
            # Idle hint / loading / unavailable — centered, slightly larger for readability
            font = QFont(theme.FONT_FAMILY)
            font.setPointSize(9)
            p.setFont(font)
            color = QColor("#4a4e62") if self._subdued else QColor(theme.DARK["text_dim"])
            p.setPen(color)
            p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, text)


class AudioPlayerBar(QWidget):
    """Play button + waveform area. Drives AudioPreviewService."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)  # match CTk padx=(0, 10)
        self.setFixedHeight(theme.PATH_BTN_HEIGHT)

        self.service = AudioPreviewService()
        self.active_track = None
        self.active_row = None
        self.category_colors: dict = {}
        self._last_state = "stopped"
        self._status_text = _IDLE_HINT

        # Plain QToolButton — Fluent PushButton clips painted icons on 32px width.
        self.play_btn = _make_transport_button(self, self.toggle_playback)
        _set_transport_icon(self.play_btn, playing=False)
        self.play_btn.setEnabled(False)
        self.play_btn.setToolTip("Play / pause the selected preview file.")
        layout.addWidget(self.play_btn)

        self.wave = _WaveView(self)
        layout.addWidget(self.wave, stretch=1)
        self._refresh_status_paint()

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # ----- API used by app -----

    def set_active(self, track, row) -> None:
        if track is None:
            self.reset()
            return

        self.wave.set_color(
            preview_waveform_color(getattr(row, "new_display", "") or "", self.category_colors)
        )
        same_file = (
            self.active_track is not None
            and getattr(self.active_track, "file_path", None) == getattr(track, "file_path", None)
        )
        self.active_track = track
        self.active_row = row

        if same_file:
            state = self._playback_state()
            if state in ("playing", "paused"):
                self._apply_playback_state(state)
            elif self.wave._peaks:
                self._set_status(getattr(track, "display_name", "") or "")
            self.wave.update()
            return

        self.wave.set_peaks(())
        self.wave.set_duration(0.0)
        self.wave.set_position(0.0)
        self._last_state = "stopped"
        _set_transport_icon(self.play_btn, playing=False)

        path = getattr(track, "file_path", None)
        if not getattr(track, "is_audio", True) or path is None or not path.exists():
            try:
                self.service.reset()
            except Exception:
                pass
            self.play_btn.setEnabled(False)
            self._set_status("Audio preview unavailable", subdued=True)
            return
        if not getattr(self.service, "available", True):
            try:
                self.service.reset()
            except Exception:
                pass
            self.play_btn.setEnabled(False)
            msg = getattr(self.service, "unavailable_message", "Audio preview unavailable")
            self._set_status(str(msg), subdued=True)
            return

        self.play_btn.setEnabled(True)
        self._set_status(f"Loading {path.name}…")
        try:
            self.service.load(path)
        except Exception:
            self.play_btn.setEnabled(False)
            self._set_status("Audio preview unavailable", subdued=True)

    def toggle_playback(self) -> None:
        try:
            state = self.service.play_pause()
        except Exception:
            state = "stopped"
        self._apply_playback_state(state)

    def seek(self, seconds: float) -> None:
        if self.active_track is None:
            return
        try:
            self.service.seek(seconds)
        except Exception:
            pass
        try:
            self.wave.set_position(self.service.playback_position() or 0.0)
        except Exception:
            self.wave.update()

    def set_category_colors(self, colors: dict) -> None:
        self.category_colors = dict(colors or {})
        if self.active_track is not None:
            self.wave.set_color(
                preview_waveform_color(
                    getattr(self.active_row, "new_display", "") or "",
                    self.category_colors,
                )
            )

    def reset(self) -> None:
        self.active_track = None
        self.active_row = None
        try:
            self.service.reset()
        except Exception:
            pass
        self.wave.set_peaks(())
        self.wave.set_duration(0.0)
        self.wave.set_position(0.0)
        _set_transport_icon(self.play_btn, playing=False)
        self.play_btn.setEnabled(False)
        self._last_state = "stopped"
        self._set_status(_IDLE_HINT, subdued=True)

    def release_for_file_ops(self, *, settle_s: float | None = None) -> None:
        """Unload preview and drop OS file locks before rename/organize."""
        self.active_track = None
        self.active_row = None
        try:
            self.service.release_for_file_ops(settle_s=settle_s)
        except Exception:
            try:
                self.service.reset()
            except Exception:
                pass
        self.wave.set_peaks(())
        self.wave.set_duration(0.0)
        self.wave.set_position(0.0)
        _set_transport_icon(self.play_btn, playing=False)
        self.play_btn.setEnabled(False)
        self._last_state = "stopped"
        self._set_status(_IDLE_HINT, subdued=True)

    def shutdown(self) -> None:
        self._timer.stop()
        try:
            self.service.shutdown()
        except Exception:
            pass

    # ----- internals -----

    def _playback_state(self) -> str:
        try:
            return self.service.playback_state() or "stopped"
        except Exception:
            return "stopped"

    def _apply_playback_state(self, state: str) -> None:
        self._last_state = state
        _set_transport_icon(self.play_btn, playing=(state == "playing"))
        if self.active_track is None:
            return
        name = getattr(self.active_track, "display_name", "") or ""
        if state in ("playing", "paused") or self.wave._peaks:
            self._set_status(name)

    def _set_status(self, text: str, *, subdued: bool = False) -> None:
        self._status_text = text
        self._refresh_status_paint(subdued=subdued)

    def _refresh_status_paint(self, *, subdued: bool = False) -> None:
        name = ""
        if self.active_track is not None:
            name = getattr(self.active_track, "display_name", "") or ""
        # Bottom-right chip only for the selected filename; idle hint stays centered.
        filename_state = bool(name) and self._status_text == name
        idle = self.active_track is None or self._status_text == "Audio preview unavailable"
        self.wave.set_status(
            self._status_text,
            filename_state=filename_state,
            # Match idle hint fg (#4a4e62) — not brighter text_dim.
            subdued=subdued or idle or filename_state,
        )

    def _poll(self) -> None:
        try:
            events = self.service.events
        except Exception:
            return
        import queue as _q

        drained = 0
        while drained < 32:
            try:
                generation, event_type, payload = events.get_nowait()
            except _q.Empty:
                break
            drained += 1
            if generation != self.service.generation:
                continue
            if event_type == "waveform":
                self.wave.set_peaks(payload)
                if self.active_track is not None:
                    self._set_status(getattr(self.active_track, "display_name", "") or "")
            elif event_type == "duration":
                self.wave.set_duration(float(payload))
            else:
                message = str(payload).splitlines()[-1] if payload else ""
                self._set_status(message[:120])

        state = self._playback_state()
        if state != self._last_state:
            self._apply_playback_state(state)
        if state in ("playing", "paused"):
            try:
                self.wave.set_position(self.service.playback_position() or 0.0)
            except Exception:
                self.wave.update()

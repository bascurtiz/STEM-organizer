"""Status bar — idle + running layouts, progress, resource bars.

Port of stem_organizer_ui status_frame (idle: status/credit/device; running:
elapsed/progress/ETA + 5 resource bars). Resource sampling runs on a 1 s QTimer
polling the (copied verbatim) resource_monitor module.
"""
from __future__ import annotations

import time
import webbrowser
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme


def _fmt_hms(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


class MiniBar(QWidget):
    """Resource mini bar: caption + filled trough + pct label."""

    def __init__(self, caption: str) -> None:
        super().__init__()
        self._caption = caption
        self._pct = 0.0
        self.setFixedHeight(theme.RESOURCE_ROW_HEIGHT)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumWidth(theme.RESOURCE_BAR_WIDTH + 80)
        # caption + bar + pct label
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel(caption)
        lbl.setObjectName("Dim")
        lbl.setFont(theme.F_STATUS)
        lbl.setFixedWidth(40)
        # Left-align with Elapsed; fixed width keeps HDD w / ETA right edge.
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(lbl)
        self._trough = _Bar(width=theme.RESOURCE_BAR_WIDTH, height=theme.RESOURCE_BAR_HEIGHT)
        layout.addWidget(self._trough)
        self._pct_lbl = QLabel("0%")
        self._pct_lbl.setObjectName("Dim")
        self._pct_lbl.setFont(theme.F_STATUS)
        self._pct_lbl.setFixedWidth(36)
        self._pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self._pct_lbl)

    def set_value(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, pct))
        self._trough.set_fill(self._pct / 100.0)
        self._pct_lbl.setText(f"{self._pct:.0f}%")


class _Bar(QWidget):
    """Trough with a colored fill — painted in paintEvent."""

    def __init__(self, *, width: int, height: int) -> None:
        super().__init__()
        self.setFixedSize(width, height)
        self._fill = 0.0

    def set_fill(self, frac: float) -> None:
        self._fill = max(0.0, min(1.0, frac))
        self.update()

    def paintEvent(self, event):  # noqa: N802 Qt name
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(theme.COLORS["status_trough"]))
        fill_w = int(round(w * self._fill))
        if fill_w > 0:
            p.fillRect(0, 0, fill_w, h, QColor(theme.DARK["accent"]))


class StatusBar(QFrame):
    """Bottom status strip with two stacked layouts (idle ↔ running)."""

    status_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("StatusBar")
        self.setFixedHeight(theme.STATUS_FRAME_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Layout container
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.STATUS_PAD_X, theme.STATUS_PAD_TOP, theme.STATUS_PAD_X, theme.STATUS_PAD_BOTTOM)
        outer.setSpacing(0)

        # Resource row — spread across full width (CTk places by relx)
        self._res_row = QHBoxLayout()
        self._res_row.setContentsMargins(0, 0, 0, 0)
        self._res_row.setSpacing(0)
        self._bars: dict[str, MiniBar] = {}
        specs = (
            ("cpu", "CPU"),
            ("gpu", "GPU"),
            ("ram", "RAM"),
            ("disk_read", "HDD r"),
            ("disk_write", "HDD w"),
        )
        for idx, (key, caption) in enumerate(specs):
            if idx > 0:
                self._res_row.addStretch(1)
            bar = MiniBar(caption)
            self._bars[key] = bar
            self._res_row.addWidget(bar, 0, Qt.AlignVCenter)
        self._res_widget = QWidget()
        self._res_widget.setLayout(self._res_row)
        self._res_widget.setVisible(False)
        outer.addWidget(self._res_widget)

        spacer = QWidget()
        spacer.setFixedHeight(theme.STATUS_ROW_GAP)
        outer.addWidget(spacer)

        # Idle line — three equal columns so credit stays true window-center
        # even when Device text is much wider than Idle.
        self._idle = QWidget()
        idle_layout = QHBoxLayout(self._idle)
        idle_layout.setContentsMargins(0, 0, 0, 0)
        idle_layout.setSpacing(0)

        left_col = QHBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(0)
        self._status_lbl = QLabel("Idle")
        self._status_lbl.setObjectName("Dim")
        self._status_lbl.setFont(theme.F_STATUS)
        left_col.addWidget(self._status_lbl, 0, Qt.AlignLeft | Qt.AlignVCenter)
        left_col.addStretch(1)

        center_col = QHBoxLayout()
        center_col.setContentsMargins(0, 0, 0, 0)
        center_col.setSpacing(0)
        center_col.addStretch(1)
        self._credit = QLabel(f"v{theme.APP_VERSION}")
        self._credit.setObjectName("Link")
        self._credit.setFont(theme.F_STATUS)
        self._credit.setCursor(Qt.PointingHandCursor)
        self._credit.setToolTip("View source code on GitHub")
        self._credit.mousePressEvent = self._on_credit_click  # type: ignore[assignment]
        center_col.addWidget(self._credit, 0, Qt.AlignHCenter | Qt.AlignVCenter)
        center_col.addStretch(1)

        right_col = QHBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)
        right_col.addStretch(1)
        self._device_lbl = QLabel("Device: CPU")
        self._device_lbl.setObjectName("Dim")
        self._device_lbl.setFont(theme.F_STATUS)
        right_col.addWidget(self._device_lbl, 0, Qt.AlignRight | Qt.AlignVCenter)

        idle_layout.addLayout(left_col, 1)
        idle_layout.addLayout(center_col, 1)
        idle_layout.addLayout(right_col, 1)
        outer.addWidget(self._idle)

        # Running line (initially hidden)
        self._run = QWidget()
        run_layout = QHBoxLayout(self._run)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(12)
        self._elapsed_lbl = QLabel("Elapsed: 0:00:00")
        self._elapsed_lbl.setObjectName("Dim")
        self._elapsed_lbl.setFont(theme.F_STATUS)
        run_layout.addWidget(self._elapsed_lbl)
        self._progress_bar = _ProgressBar()
        run_layout.addWidget(self._progress_bar, stretch=1)
        self._eta_lbl = QLabel("ETA --:--:--")
        self._eta_lbl.setObjectName("Dim")
        self._eta_lbl.setFont(theme.F_STATUS)
        self._eta_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        run_layout.addWidget(self._eta_lbl)
        self._run.setVisible(False)
        outer.addWidget(self._run)

        # State
        self._running = False
        self._progress_start = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._tick_clock)

        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(1000)
        self._resource_timer.timeout.connect(self._tick_resources)
        self._monitor = None  # lazy import resource_monitor

    # ----- public API -----

    def set_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    def set_device_text(self, text: str) -> None:
        self._device_lbl.setText(text)

    def show_running(self) -> None:
        self._running = True
        self._idle.setVisible(False)
        self._res_widget.setVisible(True)
        self._run.setVisible(True)
        self._progress_start = time.monotonic()
        self._progress_bar.set_pct(0.0)
        self._elapsed_lbl.setText("Elapsed: 0:00:00")
        self._eta_lbl.setText("ETA --:--:--")
        self._elapsed_timer.start()
        self._resource_timer.start()
        self._tick_resources()

    def show_idle(self, status: str = "Idle") -> None:
        self._running = False
        self._elapsed_timer.stop()
        self._resource_timer.stop()
        self._res_widget.setVisible(False)
        self._run.setVisible(False)
        self._idle.setVisible(True)
        self._status_lbl.setText(status)
        # Stop the resource monitor if running
        if self._monitor is not None:
            try:
                self._monitor.close()
            except Exception:
                pass
            self._monitor = None

    def update_progress(
        self,
        pct: float,
        eta: Optional[float] = None,
        n: Optional[int] = None,
        total: Optional[int] = None,
        phase: str = "",
    ) -> None:
        """Match worker Signal(float, object, int, int, str) → pct, eta, n, total, phase."""
        self._progress_bar.set_pct(pct)
        try:
            ni = int(n) if n is not None else None
            ti = int(total) if total is not None else None
        except (TypeError, ValueError):
            ni, ti = None, None

        eta_sec: Optional[float] = None
        if eta is not None:
            try:
                eta_sec = float(eta)
            except (TypeError, ValueError):
                eta_sec = None
        # Vocal type (and any worker that sends eta=None / "?") — estimate from pace.
        if eta_sec is None and ni is not None and ti is not None and self._progress_start:
            if ni > 0 and ti > ni:
                elapsed = time.monotonic() - self._progress_start
                eta_sec = elapsed / ni * (ti - ni)
            elif ti > 0 and ni >= ti:
                eta_sec = 0.0
        if eta_sec is not None:
            self._eta_lbl.setText(f"ETA {_fmt_hms(eta_sec)}")

        phase_text = phase if isinstance(phase, str) else ""
        if phase_text:
            if ni is not None and ti is not None:
                self._status_lbl.setText(f"{phase_text} {ni:,}/{ti:,}")
            else:
                self._status_lbl.setText(phase_text)

    # ----- internal -----

    def _tick_clock(self) -> None:
        if not self._running:
            return
        elapsed = time.monotonic() - self._progress_start
        self._elapsed_lbl.setText(f"Elapsed: {_fmt_hms(elapsed)}")

    def _tick_resources(self) -> None:
        if not self._running:
            return
        if self._monitor is None:
            try:
                from resource_monitor import ResourceMonitor  # type: ignore
                self._monitor = ResourceMonitor()
            except Exception:
                self._monitor = None
                return
        try:
            snap = self._monitor.sample()
        except Exception:
            return
        mapping = {
            "cpu": getattr(snap, "cpu", 0.0),
            "gpu": getattr(snap, "gpu", 0.0),
            "ram": getattr(snap, "ram", 0.0),
            "disk_read": getattr(snap, "disk_read", 0.0),
            "disk_write": getattr(snap, "disk_write", 0.0),
        }
        for key, val in mapping.items():
            self._bars[key].set_value(float(val or 0.0))

    def _on_credit_click(self, _event) -> None:
        webbrowser.open(theme.STATUS_LINK_URL)


class _ProgressBar(QWidget):
    """Progress trough with fill + percent label, painted in paintEvent."""

    def __init__(self) -> None:
        super().__init__()
        self._pct = 0.0
        self.setMinimumHeight(theme.STATUS_PROGRESS_HEIGHT)

    def set_pct(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, pct))
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(0, 0, w, h, QColor(theme.COLORS["status_trough"]))
        fill_w = int(round(w * self._pct / 100.0))
        if fill_w > 0:
            p.fillRect(0, 0, fill_w, h, QColor(theme.DARK["accent"]))
        # Percent text on right side of fill
        p.setPen(QColor(theme.COLORS["status_pct"]))
        from PySide6.QtCore import QRect

        label = f"{self._pct:.0f}%"
        rect = QRect(max(0, fill_w - 40), 0, min(fill_w, 40), h)
        if fill_w >= 36:
            p.drawText(rect, Qt.AlignRight | Qt.AlignVCenter, label)

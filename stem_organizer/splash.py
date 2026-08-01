"""Splash screen + startup worker.

Port of stem_organizer_ui.show_splash_screen + _startup_tasks + launch.

A QSplashScreen with the logo + status text, fade in/out via QPropertyAnimation,
and a startup QThread that runs deps_bootstrap.ensure_ml_deps + classify_backend._init_ml.

Caller dismisses the splash only after MainWindow is shown (see dismiss_splash).
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import (
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSplashScreen, QVBoxLayout, QWidget

from . import theme

LOGO_PATH = Path(__file__).resolve().parent.parent / "logo.png"
SPLASH_HOLD_MS = 1800
SPLASH_FADE_IN_MS = 380
SPLASH_FADE_OUT_MS = 240


class StartupWorker(QThread):
    """Runs ensure_ml_deps + _init_ml off the UI thread."""

    status = Signal(str)
    finished_ok = Signal(object)  # passes the exception or None

    def run(self) -> None:  # noqa: N802
        error: Optional[Exception] = None
        try:
            self.status.emit("Checking ML dependencies…")
            from deps_bootstrap import ensure_ml_deps

            # show_dialog=False: on failure raises RuntimeError with package list
            ensure_ml_deps(show_dialog=False, set_status=lambda m: self.status.emit(m))
            self.status.emit("Initializing application…")
            import classify_backend

            classify_backend._init_ml()
            self.status.emit("Preparing interface…")
            time.sleep(0.15)
        except Exception as exc:
            error = exc
        self.finished_ok.emit(error)


class Splash(QSplashScreen):
    """Splash with status text below the logo, fade in/out."""

    def __init__(self) -> None:
        pixmap = self._load_logo()
        super().__init__(pixmap)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self._status = "Starting…"
        self._min_visible_until = time.monotonic() + (SPLASH_HOLD_MS / 1000.0)

        css = f"""
        QSplashScreen {{ background: {theme.COLORS['bg']}; }}
        QLabel {{ color: {theme.DARK['text']}; font-family: '{theme.FONT_FAMILY}'; font-size: 10pt; }}
        """
        self.setStyleSheet(css)

    @staticmethod
    def _load_logo() -> QPixmap:
        if LOGO_PATH.exists():
            pix = QPixmap(str(LOGO_PATH))
            if not pix.isNull():
                return pix.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # Fallback dark square
        pix = QPixmap(512, 512)
        pix.fill(QColor(theme.COLORS["bg"]))
        return pix

    def set_status(self, text: str) -> None:
        self._status = text or ""
        self.repaint()

    def drawContents(self, painter: QPainter) -> None:  # noqa: N802
        # Default draws message() at bottom — we draw a styled status string
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        font = QFont(theme.FONT_FAMILY, 10)
        painter.setFont(font)
        painter.setPen(QColor(theme.DARK["text"]))
        rect = self.contentsRect()
        painter.drawText(
            rect.adjusted(0, rect.height() - 60, 0, -16),
            Qt.AlignHCenter | Qt.AlignBottom,
            self._status,
        )

    def elapsed_ok(self) -> bool:
        return time.monotonic() >= self._min_visible_until


def dismiss_splash(splash: Splash, main_window: Optional[QWidget] = None) -> None:
    """Fade out then close splash. Prefer after ``main_window.show()`` (no blank gap).

    If ``main_window`` is given, calls ``splash.finish(main_window)`` after the fade
    so Qt raises the main window; otherwise just ``close()`` (error / no-window paths).
    """
    fade_out = QPropertyAnimation(splash, b"windowOpacity", splash)
    fade_out.setDuration(SPLASH_FADE_OUT_MS)
    fade_out.setStartValue(1.0)
    fade_out.setEndValue(0.0)

    def _close() -> None:
        if main_window is not None:
            splash.finish(main_window)
        else:
            splash.close()

    fade_out.finished.connect(_close)
    fade_out.start()
    splash._fade_out = fade_out  # type: ignore[attr-defined] keep ref


def show_splash_and_startup(
    on_ready: Callable[[Optional[Exception]], None],
) -> tuple[Splash, StartupWorker]:
    """Show splash, run startup; call ``on_ready(exc)`` when min hold + startup done.

    Splash stays visible until the caller dismisses it (after MainWindow is shown).
    """
    splash = Splash()
    splash.show()

    worker = StartupWorker()

    fade_in = QPropertyAnimation(splash, b"windowOpacity", splash)
    fade_in.setDuration(SPLASH_FADE_IN_MS)
    fade_in.setStartValue(0.0)
    fade_in.setEndValue(1.0)
    fade_in.start()
    splash._fade_in = fade_in  # type: ignore[attr-defined] keep ref

    worker.status.connect(splash.set_status)

    def check_finish(error: Optional[Exception]) -> None:
        if not splash.elapsed_ok():
            QTimer.singleShot(80, lambda: check_finish(error))
            return
        # Keep splash up while MainWindow is built/shown — caller dismisses.
        on_ready(error)

    worker.finished_ok.connect(check_finish)
    worker.start()
    return splash, worker

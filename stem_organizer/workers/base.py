"""Base QThread worker with Qt signals.

Wraps the existing threading.Thread-based workers (from classify_backend) so
they communicate via Qt signals instead of a queue + after(ms) drain loop.

The QThread owns a delegate worker object; logs/progress/sdr lines emitted by
the delegate are forwarded as Qt signals on the UI thread via Qt.QueuedConnection.
"""
from __future__ import annotations

import queue
from typing import Any, Optional

from PySide6.QtCore import QThread, Signal

from classify_backend import (
    DONE_SENTINEL,
    PROGRESS_TAG,
    SDR_LOG_TAG,
)


class BaseWorker(QThread):
    """Generic adapter around a threading.Thread-style worker.

    Subclasses build ``self._delegate`` (a threading.Thread subclass exposing
    ``stop()`` and pushing tuples onto a ``queue.Queue``).
    """

    log_line = Signal(str, str)            # (text, tag)
    progress = Signal(float, object, int, int, str)  # (pct, eta, n, total, phase)
    sdr_line = Signal(str, float, float)   # (filename, score, threshold)
    status = Signal(str)
    finished_ok = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._delegate = None
        self._q: queue.Queue = queue.Queue()
        self._drain_handle = None
        self._stop_requested = False

    def stop(self) -> None:
        """Ask the backend thread to stop (safe before or after it starts)."""
        self._stop_requested = True
        if self._delegate is not None:
            try:
                self._delegate.stop()
            except Exception:
                pass

    def run(self) -> None:  # noqa: N802 Qt name
        delegate = self._build_delegate(self._q)
        if delegate is None:
            return
        self._delegate = delegate
        if self._stop_requested:
            try:
                delegate.stop()
            except Exception:
                pass
        delegate.start()
        while True:
            try:
                msg = self._q.get(timeout=0.1)
            except queue.Empty:
                if not delegate.is_alive():
                    break
                continue
            if msg is DONE_SENTINEL:
                # Drain anything left then exit
                break
            self._dispatch(msg)
        delegate.join(timeout=2.0)
        self.finished_ok.emit()

    def _build_delegate(self, log_q: queue.Queue):
        raise NotImplementedError

    def _dispatch(self, msg: Any) -> None:
        if isinstance(msg, tuple) and msg and msg[0] == PROGRESS_TAG:
            pct = float(msg[1]) if len(msg) > 1 else 0.0
            eta = msg[2] if len(msg) > 2 else None
            n = int(msg[3]) if len(msg) > 3 and msg[3] is not None else 0
            total = int(msg[4]) if len(msg) > 4 and msg[4] is not None else 0
            phase = str(msg[5]) if len(msg) > 5 and msg[5] is not None else ""
            self.progress.emit(pct, eta, n, total, phase)
            return
        if isinstance(msg, tuple) and msg and msg[0] == SDR_LOG_TAG:
            _, filename, score, threshold = msg
            self.sdr_line.emit(str(filename), float(score), float(threshold))
            return
        if isinstance(msg, str):
            tag = self._classify_log_line(msg)
            self.log_line.emit(msg, tag)
            return
        # Unknown shape: coerce to string
        self.log_line.emit(str(msg), "")

    @staticmethod
    def _classify_log_line(line: str) -> str:
        if "[ERROR]" in line or "[error]" in line or "[delete error]" in line:
            return "err"
        if line.startswith("[deleted]") or " [deleted] " in line:
            return "deleted"
        if (
            line.startswith("[warn]")
            or "[skip existing]" in line
            or "[skip]" in line
            or "[empty]" in line
            or line.startswith("[stopping]")
        ):
            return "warn"
        # === headers / folder titles — dim gray (same as CTk info = fg_dim)
        if line.startswith("===") or line.strip().startswith("==="):
            return "info"
        if line == "DONE":
            return "ok"
        return "info"

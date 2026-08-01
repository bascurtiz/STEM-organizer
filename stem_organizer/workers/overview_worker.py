"""Overview tab QThread workers (scan / balance / compression)."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal


class OverviewWorker(QThread):
    """Runs one Overview background action."""

    log_line = Signal(str, str)
    progress = Signal(float, object, int, int, str)  # pct, eta, n, total, phase
    status = Signal(str)
    finished_ok = Signal(str)  # final status
    result_ready = Signal(object)  # payload from action (ScanResult / dict / …)

    def __init__(
        self,
        action: Callable[[Callable, Callable, threading.Event], object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._action = action
        self._stop_event = threading.Event()
        self._progress_started_at: float = 0.0
        self._final_status = "Done"

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: N802
        self._progress_started_at = time.monotonic()
        payload = None
        try:
            payload = self._action(self._on_log, self._on_progress, self._stop_event)
        except Exception as exc:
            self._on_log(traceback.format_exc(), "err")
            self._final_status = f"Failed · {exc}"
        else:
            if self._stop_event.is_set():
                self._final_status = "Stopped"
            if payload is not None:
                self.result_ready.emit(payload)
        self.finished_ok.emit(self._final_status)

    def _on_log(self, message: str, tag: str = "info") -> None:
        self.log_line.emit(message, tag)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        pct = (done / total * 100.0) if total else 0.0
        eta: Optional[float] = None
        if done > 0 and total > done and self._progress_started_at:
            elapsed = time.monotonic() - self._progress_started_at
            eta = elapsed / done * (total - done)
        self.progress.emit(pct, eta, int(done), int(total), message)
        if message:
            self.status.emit(message)

    def set_final_status(self, status: str) -> None:
        self._final_status = status

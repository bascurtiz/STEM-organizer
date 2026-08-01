"""Pair + Align worker QThread adapters.

Wraps the GUI-agnostic functions from pair_matcher / stem_align in QThread
subclasses that emit Qt signals. Each worker takes a callback config dict
(`action` → callable) so the host tab can prepare the right call without
sub-classing per action.

Emits:
  log_line(str, str)     — (message, tag) from on_log backend callbacks
  progress(float, object, int, int, str) — (pct, eta, n, total, phase)
  finished_ok()
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal


class PairWorker(QThread):
    """Runs one Match & Align action on a background thread."""

    log_line = Signal(str, str)
    progress = Signal(float, object, int, int, str)  # pct, eta, n, total, phase
    status = Signal(str)
    finished_ok = Signal(str)  # final status string

    def __init__(self, action: Callable[[Callable, Callable], None], parent=None) -> None:
        super().__init__(parent)
        self._action = action
        self._stop_event = threading.Event()
        self._progress_started_at: float = 0.0
        self._final_status = "Done"

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # noqa: N802 Qt name
        self._progress_started_at = time.monotonic()
        try:
            self._action(self._on_log, self._on_progress)
        except Exception:
            self._on_log(traceback.format_exc(), "err")
            self._final_status = "Failed"
        else:
            if self._stop_event.is_set():
                self._final_status = "Stopped"
        self.finished_ok.emit(self._final_status)

    # ---- backend callbacks ----

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

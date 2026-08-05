"""Classify (RMS) and SI-SDR worker QThread adapters.

Thin wrappers around classify_backend.Worker and classify_backend.SdrWorker that
emit Qt signals instead of pushing tuples onto a queue.

Also: ``SdrPreflightWorker`` — filesystem scan for SI-SDR process-all prompts,
kept off the GUI thread (was the Start freeze root cause).
"""
from __future__ import annotations

import queue
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from classify_backend import SdrWorker as _SdrThread
from classify_backend import Worker as _RmsThread
from classify_backend import STEM_MODES, collect_sdr_process_all_preflight, resolve_stem_mode
from .base import BaseWorker


class ClassifyWorker(BaseWorker):
    """RMS classify + mix worker."""

    def __init__(self, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._params = params

    def _build_delegate(self, log_q: queue.Queue):
        return _RmsThread(self._params, log_q)


class SdrClassifyWorker(BaseWorker):
    """SI-SDR worker."""

    def __init__(self, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._params = params

    def _build_delegate(self, log_q: queue.Queue):
        return _SdrThread(self._params, log_q)


class SdrPreflightWorker(QThread):
    """Scan the SI-SDR target tree off the GUI thread before process-all prompts.

    ``finished_result`` emits a preflight dict, or ``None`` if stopped / failed
    with no payload (errors also emit ``log_line`` then ``None``).
    """

    log_line = Signal(str, str)
    finished_result = Signal(object)

    def __init__(self, params: dict, parent=None) -> None:
        super().__init__(parent)
        self._params = params
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:  # noqa: N802 Qt name
        if self._stop_requested:
            self.finished_result.emit(None)
            return
        try:
            root = Path(self._params.get("target_dir") or "")
            if not root.is_dir():
                self.finished_result.emit(
                    {"layout": None, "pair_hint": None, "flat_hints": []}
                )
                return
            mode_cfg = STEM_MODES[resolve_stem_mode(self._params.get("stem_mode"))]
            preferred = mode_cfg["categories"]
            scan_mode = self._params.get("scan_mode") or "subfolders"
            result = collect_sdr_process_all_preflight(root, scan_mode, preferred)
            if self._stop_requested:
                self.finished_result.emit(None)
                return
            self.finished_result.emit(result)
        except Exception as exc:
            self.log_line.emit(f"[error] SI-SDR scan failed: {exc}", "err")
            self.finished_result.emit(None)

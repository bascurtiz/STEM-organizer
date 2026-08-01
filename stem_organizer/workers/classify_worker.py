"""Classify (RMS) and SI-SDR worker QThread adapters.

Thin wrappers around classify_backend.Worker and classify_backend.SdrWorker that
emit Qt signals instead of pushing tuples onto a queue.
"""
from __future__ import annotations

import queue
from typing import Any

from PySide6.QtCore import Signal

from classify_backend import SdrWorker as _SdrThread
from classify_backend import Worker as _RmsThread
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

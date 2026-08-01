"""Drip on_file / LOG callbacks (CLI fallback when stem_organizer isn't on path)."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional, Sequence

DEFAULT_LOG_PACE_S = 0.5


class PacedEmit:
    """Queue callback invocations; drip one group at a time with a delay."""

    def __init__(
        self,
        fn: Callable[..., Any],
        interval_s: float = DEFAULT_LOG_PACE_S,
        *,
        name: str = "log-pace",
    ) -> None:
        self._fn = fn
        self._interval = max(0.0, float(interval_s))
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self._q.put([(args, kwargs)])

    def put_group(self, calls: Sequence[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        if not calls:
            return
        self._q.put(list(calls))

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                for args, kwargs in item:
                    self._fn(*args, **kwargs)
            finally:
                self._q.task_done()
            if item is not None and self._interval > 0:
                time.sleep(self._interval)

    def close(self) -> None:
        self._q.put(None)
        self._q.join()
        self._thread.join(timeout=3600.0)


def paced(
    fn: Optional[Callable[..., Any]],
    interval_s: float = DEFAULT_LOG_PACE_S,
    *,
    name: str = "log-pace",
) -> Optional[PacedEmit]:
    if fn is None or interval_s <= 0:
        return None
    return PacedEmit(fn, interval_s, name=name)

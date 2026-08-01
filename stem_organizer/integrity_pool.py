"""Process vs thread pools for Integrity workers.

Frozen Windows builds re-exec the .exe for ``ProcessPoolExecutor`` children,
which hits the single-instance gate ("Another instance is already running").
Use threads when frozen; keep processes in source/dev runs.
"""
from __future__ import annotations

import sys
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor


def integrity_pool(max_workers: int) -> Executor:
    """Return a pool suitable for Compression / Corruption / Convert."""
    n = max(1, int(max_workers or 1))
    if getattr(sys, "frozen", False):
        return ThreadPoolExecutor(max_workers=n)
    return ProcessPoolExecutor(max_workers=n)

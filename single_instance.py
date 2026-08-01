"""Prevent launching more than one STEM organizer window."""
from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

_MUTEX_NAME = 'Local\\DemucsStemOrganizer_SingleInstance_v1'
_mutex_handle = None


def acquire_single_instance() -> bool:
    """Return True when this process owns the app lock."""
    global _mutex_handle
    if sys.platform == 'win32':
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ERROR_ALREADY_EXISTS = 183
        handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
        if not handle:
            return True
        already = kernel32.GetLastError() == ERROR_ALREADY_EXISTS
        if already:
            kernel32.CloseHandle(handle)
            return False
        _mutex_handle = handle

        @atexit.register
        def _release() -> None:
            global _mutex_handle
            if _mutex_handle:
                kernel32.CloseHandle(_mutex_handle)
                _mutex_handle = None

        return True

    lock_path = Path(tempfile.gettempdir()) / 'demucs_stem_organizer.lock'
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    except OSError:
        return True
    else:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

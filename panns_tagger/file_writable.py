"""Clear read-only so tag / metadata writers can save.

Explorer "Read-only" on Windows sets FILE_ATTRIBUTE_READONLY; mutagen then
fails with Errno 13. Call ``ensure_writable`` before any in-place save.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


def ensure_writable(path: Path | str) -> None:
    """Clear read-only (Windows attribute + POSIX write bit). No-op on failure."""
    p = Path(path)
    try:
        mode = p.stat().st_mode
        if not (mode & stat.S_IWRITE):
            os.chmod(p, mode | stat.S_IWRITE)
    except OSError:
        pass

    if sys.platform != "win32":
        return
    try:
        import ctypes

        fa_readonly = 0x1
        invalid = 0xFFFFFFFF
        get_attrs = ctypes.windll.kernel32.GetFileAttributesW
        set_attrs = ctypes.windll.kernel32.SetFileAttributesW
        attrs = int(get_attrs(str(p)))
        if attrs != invalid and (attrs & fa_readonly):
            set_attrs(str(p), attrs & ~fa_readonly)
    except Exception:
        pass

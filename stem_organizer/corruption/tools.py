"""Locate optional external tools (mp3val, flac) and ffmpeg."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def find_on_path(name: str, override: str = "") -> Optional[str]:
    """Return absolute path to an executable, or None."""
    if override and str(override).strip():
        p = Path(override.strip())
        if p.is_file():
            return str(p.resolve())
    found = shutil.which(name)
    if found:
        return found
    if name.lower().endswith(".exe"):
        return None
    return shutil.which(f"{name}.exe")


def find_ffmpeg() -> Optional[str]:
    try:
        from ffmpeg_bootstrap import ffmpeg_path

        p = ffmpeg_path()
        return str(p) if p else None
    except Exception:
        return find_on_path("ffmpeg")


def find_ffprobe() -> Optional[str]:
    """Prefer ffprobe next to bundled ffmpeg, else PATH."""
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(
            "ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"
        )
        if sibling.is_file():
            return str(sibling.resolve())
    return find_on_path("ffprobe")


def find_mp3val(override: str = "", *, ensure: bool = True) -> Optional[str]:
    """Locate mp3val: override → bundled ``mp3val/`` → PATH → optional download."""
    if override and str(override).strip():
        p = Path(override.strip())
        if p.is_file():
            return str(p.resolve())
    try:
        from mp3val_bootstrap import mp3val_path

        return mp3val_path(ensure=ensure)
    except Exception:
        return find_on_path("mp3val", override)


def find_flac(override: str = "", *, ensure: bool = True) -> Optional[str]:
    """Locate flac: override → bundled ``flac/`` → PATH → optional download."""
    if override and str(override).strip():
        p = Path(override.strip())
        if p.is_file():
            return str(p.resolve())
    try:
        from flac_bootstrap import flac_path

        return flac_path(ensure=ensure)
    except Exception:
        return find_on_path("flac", override)

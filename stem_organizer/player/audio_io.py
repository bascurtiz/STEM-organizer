"""Shared audio decode for Stem Player + Rename preview (soundfile / ffmpeg).

No Qt / renamer imports — safe for both callers.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .audio_engine import PLAYER_SR

SF_READ_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a", ".opus"}

_np = None
_sf = None
_ffmpeg = None
_audio_deps_ready = False


def ensure_player_audio_deps() -> None:
    global _np, _sf, _ffmpeg, _audio_deps_ready
    if _audio_deps_ready:
        return
    # Resolve a real ffmpeg (PATH scan skips the crippled Microsoft Store stub);
    # download once into <app>/ffmpeg/ when nothing usable exists. First call is
    # always from a background decode/load thread — later UI-thread calls are
    # no-ops via _audio_deps_ready.
    from ffmpeg_bootstrap import ensure_ffmpeg  # noqa: F401

    import numpy as np
    import soundfile as sf

    _np = np
    _sf = sf
    _ffmpeg = ensure_ffmpeg()
    _audio_deps_ready = True


def _normalize_player_audio(audio, file_sr: int, sr: int, ch: int):
    ensure_player_audio_deps()
    if audio.shape[0] == 1:
        audio = _np.repeat(audio, ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    if file_sr == sr:
        # Contiguous RAM copy — never keep a soundfile mmap view on the
        # mix hot path (page faults under a cold OS cache after listing
        # thousands of sibling folders cause audible underruns).
        return _np.ascontiguousarray(audio, dtype=_np.float32)
    try:
        from audio_resample import resample_audio

        audio = resample_audio(audio, file_sr, sr, axis=1)
    except ImportError:
        raise RuntimeError(
            f"Sample rate mismatch ({file_sr} Hz vs {sr} Hz) and scipy is not installed."
        )
    return _np.ascontiguousarray(audio, dtype=_np.float32)


def _read_soundfile_player(path: str, sr: int, ch: int):
    ensure_player_audio_deps()
    # Read into RAM (no mmap). Playback must not depend on page faults from
    # a library root that may contain thousands of sibling folders.
    try:
        data, file_sr = _sf.read(path, dtype="float32", always_2d=True)
        return _normalize_player_audio(data.T, file_sr, sr, ch)
    except Exception:
        return None


def _read_via_ffmpeg_player(path: str, sr: int, ch: int):
    ensure_player_audio_deps()
    if not _ffmpeg:
        return None
    from ffmpeg_bootstrap import subprocess_kwargs

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                _ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                path,
                "-ar",
                str(sr),
                "-ac",
                str(ch),
                tmp_path,
            ],
            check=True,
            capture_output=True,
            **subprocess_kwargs(),
        )
        return _read_soundfile_player(tmp_path, sr, ch)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_player_audio(path: str, sr: int = PLAYER_SR, ch: int = 2):
    """Decode to stereo float32 shape (ch, n) at ``sr`` Hz (RAM copy)."""
    ensure_player_audio_deps()
    p = Path(path)
    ext = p.suffix.lower()
    if ext in SF_READ_EXTS:
        audio = _read_soundfile_player(str(p), sr, ch)
        if audio is not None:
            return audio
    audio = _read_via_ffmpeg_player(str(p), sr, ch)
    if audio is not None:
        return audio
    try:
        from demucs.audio import AudioFile

        return (
            AudioFile(path)
            .read(streams=0, samplerate=sr, channels=ch)
            .numpy()
            .astype(_np.float32)
        )
    except Exception as exc:
        hint = (
            "Re-run install-deps.bat if packages are missing. "
            "For FLAC without ffmpeg, ensure soundfile/libsndfile supports FLAC."
        )
        raise RuntimeError(f"Could not decode audio ({p.name}): {exc}\n{hint}") from exc

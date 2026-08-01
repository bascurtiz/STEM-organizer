"""Sniff real audio container from file magic (ignore extension)."""

from __future__ import annotations

from pathlib import Path

# How far to search for a RIFF/WAVE header after a leading ID3 blob.
_RIFF_SEARCH = 256 * 1024


def is_riff_wave(path: Path | str) -> bool:
    """True when the file starts with a RIFF/WAVE header."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(12)
    except OSError:
        return False
    return len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WAVE"


def is_mpeg_audio(path: Path | str) -> bool:
    """True when content looks like MPEG/MP3 (ID3 and/or frame sync), not RIFF WAV.

    Matches foobar \"wrong extension: wav; Correct handler: MPEG Decoder\".
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(16)
            if len(head) < 4:
                return False
            if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
                return False
            if head.startswith(b"fLaC") or head.startswith(b"OggS"):
                return False
            # ID3v2 at start — either MP3, or a WAV we corrupted by prepending ID3.
            if head.startswith(b"ID3"):
                more = head + fh.read(_RIFF_SEARCH)
                idx = more.find(b"RIFF")
                if idx > 0 and more[idx + 8 : idx + 12] == b"WAVE":
                    return False  # salvageable WAV
                return True
            # MPEG frame sync (11 set bits)
            if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
                return True
    except OSError:
        return False
    return False


def wrong_extension_note(path: Path | str) -> str:
    """Human note when extension disagrees with content, else \"\"."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".wav" and is_mpeg_audio(p):
        return "wrong extension: wav (MPEG content)"
    if suffix in (".mp3", ".mp2", ".m2a") and is_riff_wave(p):
        return "wrong extension: mp3 (WAV content)"
    return ""

"""Track state + helpers for the Stem Player.

Port of stem_player._TrackState + _to_stereo (no Tk dependency).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class TrackState:
    """One stem in the player. Audio is stereo (shape (2, n_samples))."""

    __slots__ = (
        "name", "path", "audio", "peaks", "peaks_full",
        "volume", "muted", "solo", "color",
        "row_widget", "wave_widget", "solo_btn", "mute_btn", "vol_slider",
    )

    def __init__(self, name: str, path: Path, audio, color: str) -> None:
        self.name = name
        self.path = path
        self.audio = _to_stereo(audio)
        self.peaks = None
        self.peaks_full = None
        self.volume = 1.0
        self.muted = False
        self.solo = False
        self.color = color
        # UI refs (assigned when track row is built)
        self.row_widget = None
        self.wave_widget = None
        self.solo_btn = None
        self.mute_btn = None
        self.vol_slider = None


def _to_stereo(audio):
    """Force shape (2, n_samples) float32. Mono → stereo by repeat."""
    import numpy as np

    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = np.stack([a, a], axis=0)
    elif a.shape[0] == 1:
        a = np.repeat(a, 2, axis=0)
    elif a.shape[0] > 2:
        a = a[:2]
    return np.ascontiguousarray(a, dtype=np.float32)

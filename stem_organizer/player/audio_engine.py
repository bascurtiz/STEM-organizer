"""Audio engine — port of stem_player._AudioEngine (verbatim, no Tk deps).

sounddevice OutputStream + threading.Lock. The callback runs on the PortAudio
real-time thread; UI never touches widgets from inside it.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from .track_state import TrackState


PLAYER_SR = 44100


class AudioEngine:
    """Owns the audio output stream and mixes audible tracks each block."""

    def __init__(self, tracks: List[TrackState], sr: int = PLAYER_SR) -> None:
        self.tracks = tracks
        self.sr = sr
        self.duration = 0.0
        if tracks:
            self.duration = max(t.audio.shape[1] for t in tracks) / sr
        self._position = 0
        self._playing = False
        self.master_volume = 0.85
        self._lock = threading.Lock()
        self._meter = 0.0
        self._stream = None

    # ----- properties -----

    @property
    def position(self) -> float:
        with self._lock:
            return self._position / self.sr

    @position.setter
    def position(self, seconds: float) -> None:
        with self._lock:
            self._position = int(max(0.0, min(seconds, self.duration)) * self.sr)

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = playing

    def meter_level(self) -> float:
        with self._lock:
            return self._meter

    # ----- mixing -----

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        with self._lock:
            pos = self._position
            playing = self._playing
            master = self.master_volume

        out = np.zeros((frames, 2), dtype=np.float32)
        if not playing or not self.tracks:
            outdata[:] = out
            with self._lock:
                self._meter = 0.0
            return

        # Resolve solo once per block — not once per track.
        any_solo = False
        for t in self.tracks:
            if t.solo:
                any_solo = True
                break

        peak = 0.0
        for track in self.tracks:
            if track.muted:
                continue
            if any_solo and not track.solo:
                continue
            audio = track.audio
            n = audio.shape[1]
            if pos >= n:
                continue
            take = min(frames, n - pos)
            chunk = audio[:, pos:pos + take].T * (track.volume * master)
            out[:take] += chunk
            peak = max(peak, float(np.max(np.abs(out[:take]))))

        outdata[:] = out
        with self._lock:
            self._position = pos + frames
            if self._position >= int(self.duration * self.sr):
                self._playing = False
                self._position = int(self.duration * self.sr)
            self._meter = peak

    # ----- stream lifecycle -----

    def start_stream(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        self._stream = None

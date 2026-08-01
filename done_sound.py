"""Windows job-done cue (SystemAsterisk), debounced to one play."""

from __future__ import annotations

import sys
import time

_LAST_PLAY_AT = 0.0
_COOLDOWN_S = 1.5


def play_done_sound() -> None:
    """Play once; ignore repeats within cooldown (DONE line + MessageBox, etc.)."""
    global _LAST_PLAY_AT
    if sys.platform != 'win32':
        return
    now = time.monotonic()
    if now - _LAST_PLAY_AT < _COOLDOWN_S:
        return
    _LAST_PLAY_AT = now
    try:
        import winsound

        winsound.PlaySound(
            'SystemAsterisk',
            winsound.SND_ALIAS | winsound.SND_ASYNC,
        )
    except Exception:
        pass

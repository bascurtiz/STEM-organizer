"""Sample-rate conversion helper shared by the separation and player paths.

Replaces resampy (kaiser_fast) with scipy.signal.resample_poly so the main
app no longer needs resampy/its numba+cython wheels. scipy is already a
hard dependency of the core app. resample_poly is a polyphase FIR design.

Quality note (verified before the swap): on a 1 kHz sine + low noise,
steady-state max-abs difference vs resampy kaiser_fast is ~0.01 on a
0.5-peak signal (mean ~0.0015) — well below audibility for separation
feed-in and playback. Length: matches resampy exactly for downsampling and
integer ratios; for non-integer upsampling (e.g. 44100->48000) scipy yields
ceil(n*up/down) which is one sample longer than resampy. Both consumers
(separation pipeline, player ring buffer) read length dynamically, so this
1-sample difference is harmless.
"""

from __future__ import annotations

from math import gcd


def resample_audio(audio, file_sr: int, target_sr: int, axis: int = 1):
    """Resample ``audio`` from ``file_sr`` to ``target_sr`` along ``axis``.

    Lazy-imports scipy so importing this module is free when no resample is
    needed. ``up``/``down`` are reduced by their GCD, as scipy expects.
    """
    if file_sr == target_sr:
        return audio

    import scipy.signal

    up = int(target_sr)
    down = int(file_sr)
    g = gcd(up, down)
    up //= g
    down //= g
    return scipy.signal.resample_poly(audio, up, down, axis=axis)

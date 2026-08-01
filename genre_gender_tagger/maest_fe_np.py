"""Torch-free MAEST mel frontend — mirrors HF MAESTFeatureExtractor.

Produces ``input_values`` shaped (batch, 1876, 96), logC-compressed and
Discogs-normalized, ready for the MAEST ONNX classifier.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from _maest_audio_utils import mel_filter_bank, spectrogram, window_function

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
NUM_MEL_BINS = 96
MAX_LENGTH = 1876
MEAN = 2.06755686098554
STD = 1.268292820667291

_MEL_FB: np.ndarray | None = None
_WINDOW: np.ndarray | None = None


def _window() -> np.ndarray:
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = window_function(window_length=N_FFT, name="hann")
    return _WINDOW


def _mel_fb() -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = mel_filter_bank(
            num_frequency_bins=N_FFT // 2 + 1,
            num_mel_filters=NUM_MEL_BINS,
            min_frequency=0.0,
            max_frequency=SAMPLE_RATE / 2,
            sampling_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        )
    return _MEL_FB


def _extract_one(waveform: np.ndarray) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32)
    melspec = spectrogram(
        wav,
        window=_window(),
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        power=2,
        mel_filters=_mel_fb(),
        min_value=1e-30,
        mel_floor=1e-30,
        pad_mode="constant",
    ).T  # (frames, mels)
    melspec = np.log10(1.0 + melspec * 10000.0)
    n_frames = melspec.shape[0]
    if MAX_LENGTH > 0:
        diff = MAX_LENGTH - n_frames
        if diff > 0:
            melspec = np.pad(melspec, ((0, diff), (0, 0)))
        elif diff < 0:
            melspec = melspec[:MAX_LENGTH, :]
    return ((melspec - MEAN) / (STD * 2.0)).astype(np.float32, copy=False)


def maest_input_values(
    clips: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    """Waveform clip(s) → ``input_values`` (batch, 1876, 96) float32."""
    if isinstance(clips, np.ndarray) and clips.ndim == 1:
        batch = [_extract_one(clips)]
    elif isinstance(clips, np.ndarray) and clips.ndim == 2:
        batch = [_extract_one(clips[i]) for i in range(clips.shape[0])]
    else:
        batch = [_extract_one(np.asarray(c, dtype=np.float32)) for c in clips]
    return np.stack(batch, axis=0)

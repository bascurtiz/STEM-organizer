"""NumPy PaSST mel — torch-free mirror of passt_mel.PasstMelSTFT (eval path).

Used by the ONNX OpenMIC backend so hear21passt/torch are not required at runtime.
Must stay bit-close to PasstMelSTFT: preemphasis → STFT → power → mel → log norm.
"""
from __future__ import annotations

import numpy as np


def _hann_periodic_false(win_length: int) -> np.ndarray:
    """Match torch.hann_window(win_length, periodic=False)."""
    if win_length == 1:
        return np.ones(1, dtype=np.float32)
    n = np.arange(win_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (win_length - 1))).astype(np.float32)


def _torch_stft_power(
    x: np.ndarray,
    *,
    n_fft: int,
    hop: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """Match torch.stft(..., center=True, normalized=False, return_complex=True).abs()**2.

    x: (batch, samples) float32
    returns: (batch, n_freq, n_frames) power
    """
    pad = n_fft // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    batch, n = x_pad.shape
    # Number of frames: floor((n - n_fft) / hop) + 1  — torch uses this with center pad.
    n_frames = 1 + (n - n_fft) // hop
    # Zero-pad window to n_fft — torch centers win_length inside n_fft.
    win = np.zeros(n_fft, dtype=np.float32)
    pad_left = (n_fft - win_length) // 2
    win[pad_left : pad_left + win_length] = window
    n_freq = n_fft // 2 + 1
    out = np.empty((batch, n_freq, n_frames), dtype=np.float32)
    for b in range(batch):
        for t in range(n_frames):
            start = t * hop
            frame = x_pad[b, start : start + n_fft] * win
            spec = np.fft.rfft(frame, n=n_fft)
            out[b, :, t] = (spec.real * spec.real + spec.imag * spec.imag).astype(
                np.float32
            )
    return out


_MEL_BASIS: np.ndarray | None = None


def _mel_basis(
    *,
    sr: int = 32000,
    n_fft: int = 1024,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    global _MEL_BASIS
    if _MEL_BASIS is not None:
        return _MEL_BASIS
    import librosa

    if fmax is None:
        fmax = float(sr // 2 - 1000)
    mel = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm=1,
    )
    _MEL_BASIS = np.asarray(mel, dtype=np.float32)
    return _MEL_BASIS


def passt_mel_numpy(
    audio: np.ndarray,
    *,
    sr: int = 32000,
    n_mels: int = 128,
    win_length: int = 800,
    hopsize: int = 320,
    n_fft: int = 1024,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Waveform → PaSST mel.

    audio: (samples,) or (batch, samples) float32
    returns: (batch, n_mels, time) float32 — same layout as PasstMelSTFT.forward
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D, got shape {x.shape}")

    # Preemphasis: conv1d with kernel [-0.97, 1.0], no pad → drops 1 sample.
    # y[t] = -0.97 * x[t] + 1.0 * x[t+1]
    y = -0.97 * x[:, :-1] + x[:, 1:]

    window = _hann_periodic_false(win_length)
    power = _torch_stft_power(
        y, n_fft=n_fft, hop=hopsize, win_length=win_length, window=window
    )
    mel_b = _mel_basis(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # (n_mels, n_freq) @ (batch, n_freq, time) → (batch, n_mels, time)
    melspec = np.einsum("mf,bft->bmt", mel_b, power, optimize=True)
    melspec = np.log(melspec + 1e-5)
    melspec = (melspec + 4.5) / 5.0
    return melspec.astype(np.float32, copy=False)

"""PaSST-compatible mel STFT without torchaudio (Windows-friendly)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PasstMelSTFT(nn.Module):
    """Match hear21passt AugmentMelSTFT at eval (no SpecAugment)."""

    def __init__(
        self,
        n_mels: int = 128,
        sr: int = 32000,
        win_length: int = 800,
        hopsize: int = 320,
        n_fft: int = 1024,
        fmin: float = 0.0,
        fmax: float | None = None,
    ):
        super().__init__()
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.hopsize = hopsize
        # hear21passt openmic sets fmax=None → sr//2 - 1000 (=15000 @ 32k).
        self.fmin = float(fmin)
        self.fmax = float(sr // 2 - 1000) if fmax is None else float(fmax)
        self.register_buffer(
            "window",
            torch.hann_window(win_length, periodic=False),
            persistent=False,
        )
        self.register_buffer(
            "preemphasis_coefficient",
            torch.as_tensor([[[-0.97, 1.0]]]),
            persistent=False,
        )
        import librosa

        mel = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=False,
            norm=1,
        )
        self.register_buffer(
            "mel_basis",
            torch.as_tensor(mel, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, samples)
        x = F.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        spec = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hopsize,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        melspec = torch.matmul(self.mel_basis.to(power.dtype), power)
        melspec = (melspec + 0.00001).log()
        melspec = (melspec + 4.5) / 5.0
        return melspec

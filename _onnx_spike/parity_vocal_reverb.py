"""Parity: torch MelReverbNet vs ONNX runner, identical mel-crop input.

Validates that the exported ONNX reproduces the torch logits on the SAME mel
spectrograms (the mel/crop prep is identical in both paths, so this isolates
the Conv2d forward).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP / 'genre_gender_tagger'))
import onnxruntime as ort
from vocal_reverb import (MelReverbNet, DEFAULT_CONFIG, CLASS_NAMES, frames_for_clip,
                          load_mono, audio_to_logmel, crop_or_pad_logmel)

CKPT = APP / 'genre_gender_tagger' / 'models' / 'vocal_reverb.pt'
ONNX = APP / '_onnx_spike' / 'onnx_out' / 'vocal_reverb.onnx'


def build_torch_model():
    saved = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = dict(DEFAULT_CONFIG); cfg.update(saved.get('config') or {})
    channels = tuple(cfg.get('channels') or DEFAULT_CONFIG['channels'])
    m = MelReverbNet(n_mels=int(cfg['n_mels']), channels=channels,
                     n_classes=len((saved.get('classes') or CLASS_NAMES)))
    m.load_state_dict(saved['state_dict']); m.eval()
    return m, cfg


def mel_crops_from_audio(audio_path, cfg, n_crops=3):
    """Replicate VocalReverbRouter._crops_for_file deterministically (center/lin crops)."""
    audio = load_mono(audio_path, int(cfg['sample_rate']))
    logmel = audio_to_logmel(audio, sample_rate=int(cfg['sample_rate']),
                             n_mels=int(cfg['n_mels']), n_fft=int(cfg['n_fft']),
                             hop_length=int(cfg['hop_length']))
    target = frames_for_clip(cfg)
    n_frames = logmel.shape[1]
    if n_frames <= target:
        return np.stack([crop_or_pad_logmel(logmel, target)], axis=0)
    starts = np.linspace(0, n_frames - target, num=n_crops, dtype=np.int64)
    return np.stack([logmel[:, int(s):int(s)+target] for s in starts], axis=0)


def main():
    # Use a synthetic input if no real audio available — parity is about
    # torch-vs-ONNX on the SAME tensor, so any input is valid.
    rng = np.random.default_rng(42)
    # 8 random crops of shape (64, 250) — simulates a batch.
    crops = rng.standard_normal((8, 64, 250)).astype(np.float32)
    x = torch.from_numpy(crops).unsqueeze(1)  # (8,1,64,250)

    # torch reference
    model, cfg = build_torch_model()
    with torch.inference_mode():
        logits_torch = model(x).numpy()

    # ONNX (expects 4D: batch, 1 ch, n_mels, time)
    sess = ort.InferenceSession(str(ONNX), providers=['CPUExecutionProvider'])
    mel_4d = crops[:, np.newaxis, :, :]  # (8,1,64,250)
    logits_onnx = sess.run(['logits'], {'mel': mel_4d})[0]

    diff = np.abs(logits_torch - logits_onnx)
    rel = np.linalg.norm(logits_torch - logits_onnx) / (np.linalg.norm(logits_torch) + 1e-12)
    print("=" * 60)
    print(f"input  : {crops.shape}")
    print(f"torch  : {logits_torch.shape}  sample row0 = {logits_torch[0]}")
    print(f"onnx   : {logits_onnx.shape}  sample row0 = {logits_onnx[0]}")
    print(f"max |d|: {diff.max():.3e}")
    print(f"mean|d|: {diff.mean():.3e}")
    print(f"rel L2 : {rel:.3e}")
    # Agreement on argmax (dry/wet label) — the thing that actually matters
    agree = (logits_torch.argmax(1) == logits_onnx.argmax(1)).mean()
    print(f"label agreement: {agree*100:.1f}%")
    print("=" * 60)
    print("\nVerdict:")
    if rel < 1e-3 and agree == 1.0:
        print("  GO  — ONNX reproduces torch logits (rel-L2 < 1e-3, labels match).")
    else:
        print(f"  REVIEW — rel-L2 {rel:.2e}, agreement {agree*100:.1f}%.")

    # Also run on a REAL mel if test audio exists, for an end-to-end sanity point.
    raw = APP / '_onnx_spike' / 'test_track.raw'
    if raw.exists():
        # decode test.mp3 to a mono wav librosa can read — reuse the f32 raw we made
        raw_np = np.fromfile(str(raw), dtype=np.float32).reshape(-1, 2).mean(axis=1)
        logmel = audio_to_logmel(raw_np.astype(np.float32),
                                 sample_rate=int(cfg['sample_rate']),
                                 n_mels=int(cfg['n_mels']),
                                 n_fft=int(cfg['n_fft']),
                                 hop_length=int(cfg['hop_length']))
        target = frames_for_clip(cfg)
        crops_r = np.stack([crop_or_pad_logmel(logmel, target)], axis=0)
        with torch.inference_mode():
            lt = model(torch.from_numpy(crops_r).unsqueeze(1)).numpy()
        lo = sess.run(['logits'], {'mel': crops_r[:, np.newaxis, :, :]})[0]
        d = np.abs(lt - lo).max()
        print(f"\n[real audio] max|logit diff| = {d:.3e}  "
              f"torch={lt[0]}  onnx={lo[0]}")


if __name__ == '__main__':
    main()

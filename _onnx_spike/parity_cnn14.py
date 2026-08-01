"""Parity: torch Cnn14 (via panns_inference) vs ONNX runner, same waveform.

Validates the full in-graph pipeline (Spectrogram + LogMel + BN + conv blocks).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch
import onnxruntime as ort
from panns_inference import AudioTagging

ONNX = Path(__file__).resolve().parent / 'onnx_out' / 'cnn14.onnx'
CKPT = Path(r"D:/STEM-organizer-Py6/panns_tagger/models/Cnn14_mAP=0.431.pth")
SR = 32000


def main():
    # 10s random waveform (batch=4) — parity is about torch-vs-ONNX on same input.
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((4, SR * 10)).astype(np.float32)

    # torch reference (the exact path the app uses: AudioTagging.inference)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        tagger = AudioTagging(checkpoint_path=str(CKPT), device='cpu')
    clipwise, _emb = tagger.inference(audio)
    probs_torch = np.asarray(clipwise, dtype=np.float32)

    # ONNX runner
    sess = ort.InferenceSession(str(ONNX), providers=['CPUExecutionProvider'])
    probs_onnx = sess.run(['probs'], {'audio': audio})[0]

    diff = np.abs(probs_torch - probs_onnx)
    rel = np.linalg.norm(probs_torch - probs_onnx) / (np.linalg.norm(probs_torch) + 1e-12)
    # Top-5 label agreement (the thing that matters for vocal classification)
    top5_t = np.argsort(probs_torch[0])[-5:][::-1]
    top5_o = np.argsort(probs_onnx[0])[-5:][::-1]

    print("=" * 60)
    print(f"input  : {audio.shape}")
    print(f"torch  : {probs_torch.shape}  row0 max={probs_torch[0].max():.4f}")
    print(f"onnx   : {probs_onnx.shape}  row0 max={probs_onnx[0].max():.4f}")
    print(f"max |d|: {diff.max():.3e}")
    print(f"mean|d|: {diff.mean():.3e}")
    print(f"rel L2 : {rel:.3e}")
    print(f"top-5 match (row0): torch {top5_t.tolist()}  onnx {top5_o.tolist()}")
    print(f"argmax agree (all rows): {(probs_torch.argmax(1)==probs_onnx.argmax(1)).mean()*100:.0f}%")
    print("=" * 60)
    print("\nVerdict:")
    if rel < 1e-3:
        print(f"  GO  — ONNX reproduces torch Cnn14 (rel-L2 {rel:.2e}).")
    else:
        print(f"  REVIEW — rel-L2 {rel:.2e}; investigate.")


if __name__ == '__main__':
    main()

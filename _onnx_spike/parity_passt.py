"""Parity: torch PaSST (mel+net) vs ONNX net on matching mel.

1) Torch PasstMelSTFT + hear21passt net vs ORT on the *same* torch mel
   (validates the exported graph).
2) Numpy mel (passt_mel_np) vs torch mel (validates torch-free frontend).
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "instrument_tagger"))

from passt_mel import PasstMelSTFT
from passt_mel_np import passt_mel_numpy

ONNX = Path(__file__).resolve().parent / "onnx_out" / "passt_openmic.onnx"
CLIP = 997 * 320 + 1


def _load_net():
    from hear21passt.models.passt import get_model as get_model_passt
    import hear21passt.models.passt as passt_mod

    passt_mod.first_RUN = False
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = get_model_passt(arch="openmic", n_classes=20)
    net.eval()
    return net


def main() -> None:
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((2, CLIP)).astype(np.float32)
    # Peak-ish like the app path
    audio = audio / (np.max(np.abs(audio), axis=1, keepdims=True) + 1e-8)

    mel_t = PasstMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, fmin=0.0
    )
    mel_t.eval()
    net = _load_net()

    with torch.no_grad():
        x = torch.from_numpy(audio)
        mel_torch = mel_t(x)  # (B, 128, 998)
        logits_torch, _ = net(mel_torch.unsqueeze(1))
        probs_torch = torch.sigmoid(logits_torch).cpu().numpy().astype(np.float32)

    mel_np = passt_mel_numpy(audio)
    mel_diff = np.abs(mel_torch.numpy() - mel_np)
    mel_rel = np.linalg.norm(mel_torch.numpy() - mel_np) / (
        np.linalg.norm(mel_torch.numpy()) + 1e-12
    )

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    # Same torch mel → isolate net export error
    mel_in = mel_torch.unsqueeze(1).numpy().astype(np.float32)
    logits_onnx = sess.run(["logits"], {"mel": mel_in})[0]
    probs_onnx = 1.0 / (1.0 + np.exp(-logits_onnx))
    probs_onnx = probs_onnx.astype(np.float32)

    diff = np.abs(probs_torch - probs_onnx)
    rel = np.linalg.norm(probs_torch - probs_onnx) / (
        np.linalg.norm(probs_torch) + 1e-12
    )
    argmax_agree = (probs_torch.argmax(1) == probs_onnx.argmax(1)).mean() * 100

    # End-to-end: numpy mel + ORT
    mel_np_in = mel_np[:, None, :, :].astype(np.float32)
    logits_e2e = sess.run(["logits"], {"mel": mel_np_in})[0]
    probs_e2e = (1.0 / (1.0 + np.exp(-logits_e2e))).astype(np.float32)
    rel_e2e = np.linalg.norm(probs_torch - probs_e2e) / (
        np.linalg.norm(probs_torch) + 1e-12
    )
    argmax_e2e = (probs_torch.argmax(1) == probs_e2e.argmax(1)).mean() * 100

    print("=" * 60)
    print(f"audio     : {audio.shape}")
    print(f"mel torch : {tuple(mel_torch.shape)}  mel np: {mel_np.shape}")
    print(f"mel max|d|: {mel_diff.max():.3e}  mel rel-L2: {mel_rel:.3e}")
    print(f"probs torch max: {probs_torch[0].max():.4f}")
    print(f"probs onnx  max: {probs_onnx[0].max():.4f}")
    print(f"net  max|d|: {diff.max():.3e}")
    print(f"net  rel-L2: {rel:.3e}")
    print(f"net  argmax agree: {argmax_agree:.0f}%")
    print(f"e2e  rel-L2 (np mel+ORT vs torch): {rel_e2e:.3e}")
    print(f"e2e  argmax agree: {argmax_e2e:.0f}%")
    print("=" * 60)
    print("\nVerdict:")
    ok_net = rel < 1e-3
    ok_mel = mel_rel < 1e-4
    ok_e2e = rel_e2e < 1e-3 and argmax_e2e == 100
    if ok_net and ok_mel and ok_e2e:
        print(f"  GO  — net rel-L2 {rel:.2e}, mel rel-L2 {mel_rel:.2e}, e2e {rel_e2e:.2e}.")
    else:
        print(
            f"  REVIEW — net={rel:.2e} mel={mel_rel:.2e} e2e={rel_e2e:.2e} "
            f"(ok_net={ok_net} ok_mel={ok_mel} ok_e2e={ok_e2e})"
        )


if __name__ == "__main__":
    main()

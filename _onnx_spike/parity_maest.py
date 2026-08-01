"""Parity: torch MAEST (HF) vs ONNX classifier + numpy FE."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "genre_gender_tagger"))
from maest_fe_np import maest_input_values  # noqa: E402

MODEL_NAME = "mtg-upf/discogs-maest-30s-pw-129e-519l"
ONNX = Path(__file__).resolve().parent / "onnx_out" / "maest_discogs519.onnx"
LABELS = Path(__file__).resolve().parent / "onnx_out" / "maest_discogs519.id2label.json"
SR = 16000
CLIP = SR * 30


def main() -> None:
    rng = np.random.default_rng(0)
    clips = (rng.standard_normal((3, CLIP)) * 0.05).astype(np.float32)

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()

    iv_hf = fe(list(clips), sampling_rate=SR, return_tensors="pt")["input_values"]
    with torch.no_grad():
        logits_t = model(input_values=iv_hf).logits.cpu().numpy().astype(np.float32)
        probs_t = torch.softmax(torch.from_numpy(logits_t), dim=-1).numpy()

    iv_np = maest_input_values(clips)
    fe_diff = np.abs(iv_hf.numpy() - iv_np)
    fe_rel = np.linalg.norm(iv_hf.numpy() - iv_np) / (
        np.linalg.norm(iv_hf.numpy()) + 1e-12
    )

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    # Net-only parity on HF mel
    logits_o = sess.run(
        ["logits"], {"input_values": iv_hf.numpy().astype(np.float32)}
    )[0]
    probs_o = np.exp(logits_o - logits_o.max(axis=1, keepdims=True))
    probs_o = (probs_o / probs_o.sum(axis=1, keepdims=True)).astype(np.float32)
    rel = np.linalg.norm(probs_t - probs_o) / (np.linalg.norm(probs_t) + 1e-12)
    argmax = (probs_t.argmax(1) == probs_o.argmax(1)).mean() * 100

    # E2E numpy FE + ORT
    logits_e = sess.run(["logits"], {"input_values": iv_np})[0]
    probs_e = np.exp(logits_e - logits_e.max(axis=1, keepdims=True))
    probs_e = (probs_e / probs_e.sum(axis=1, keepdims=True)).astype(np.float32)
    rel_e = np.linalg.norm(probs_t - probs_e) / (np.linalg.norm(probs_t) + 1e-12)
    argmax_e = (probs_t.argmax(1) == probs_e.argmax(1)).mean() * 100

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["id2label"]
    assert labels[0] == model.config.id2label[0]

    print("=" * 60)
    print(f"clips    : {clips.shape}")
    print(f"FE max|d|: {fe_diff.max():.3e}  FE rel-L2: {fe_rel:.3e}")
    print(f"net rel-L2: {rel:.3e}  argmax: {argmax:.0f}%")
    print(f"e2e rel-L2: {rel_e:.3e}  argmax: {argmax_e:.0f}%")
    print(f"top torch: {labels[int(probs_t[0].argmax())]}")
    print(f"top onnx : {labels[int(probs_o[0].argmax())]}")
    print("=" * 60)
    print("\nVerdict:")
    if rel < 1e-3 and fe_rel < 1e-5 and rel_e < 1e-3 and argmax_e == 100:
        print(f"  GO  — net {rel:.2e}, FE {fe_rel:.2e}, e2e {rel_e:.2e}.")
    else:
        print(f"  REVIEW — net={rel:.2e} fe={fe_rel:.2e} e2e={rel_e:.2e}")


if __name__ == "__main__":
    main()

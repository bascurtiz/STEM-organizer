"""Export MAEST Discogs519 (ASTForAudioClassification) to ONNX.

Mel stays outside (MAESTFeatureExtractor → input_values). Export classifier only:
  input_values (B, 1876, 96) → logits (B, 519)

Also writes id2label sidecar JSON (needed once transformers is gone at runtime).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForAudioClassification

MODEL_NAME = "mtg-upf/discogs-maest-30s-pw-129e-519l"
OUT_DIR = Path(__file__).resolve().parent / "onnx_out"
DEST = OUT_DIR / "maest_discogs519.onnx"
LABELS = OUT_DIR / "maest_discogs519.id2label.json"
N_FRAMES = 1876
N_MELS = 96
N_LABELS = 519


class MaestLogits(nn.Module):
    """Thin wrapper: HF AST → logits only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


def main() -> None:
    print(f"Loading {MODEL_NAME} (fp32)...")
    base = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.float32
    )
    base.eval()
    id2label = {int(k): v for k, v in base.config.id2label.items()}
    assert len(id2label) == N_LABELS, len(id2label)

    model = MaestLogits(base)
    model.eval()
    dummy = torch.randn(1, N_FRAMES, N_MELS, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    print(f"  smoke: in {tuple(dummy.shape)} -> out {tuple(out.shape)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting -> {DEST} ...")
    torch.onnx.export(
        model,
        dummy,
        str(DEST),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"  done: {DEST.stat().st_size / 1e6:.1f} MB")

    # Stable list ordered by class index 0..518
    labels = [id2label[i] for i in range(N_LABELS)]
    LABELS.write_text(
        json.dumps({"id2label": labels, "model_name": MODEL_NAME}, indent=2),
        encoding="utf-8",
    )
    print(f"  labels: {LABELS} ({len(labels)} classes)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(DEST), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  IN  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  OUT {o.name}: {o.shape} {o.type}")


if __name__ == "__main__":
    main()

"""Export PaSST OpenMIC (hear21passt) to ONNX — net only, mel outside.

PasstMelSTFT uses torch.stft(return_complex=True), which is painful for ONNX.
Export the ViT net alone: mel (B,1,128,998) -> logits (B,20). Runtime mel stays
numpy (parity with instrument_tagger/passt_mel.py).

Checkpoint: torch hub openmic-passt-s-f128-10sec-p16-s10-ap.85.pt (~326 MB).
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "instrument_tagger"))

DEST = Path(__file__).resolve().parent / "onnx_out" / "passt_openmic.onnx"
N_MELS = 128
N_FRAMES = 998  # OpenMIC 10 s @ hop=320 (+ preemphasis sample)


class PasstLogits(nn.Module):
    """Thin wrapper: PaSST net -> class logits only (drop 768-d embedding)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, 128, 998) -> logits (B, 20)
        logits, _embed = self.net(mel)
        return logits


def _load_net() -> nn.Module:
    from hear21passt.models.passt import get_model as get_model_passt
    import hear21passt.models.passt as passt_mod

    passt_mod.first_RUN = False
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = get_model_passt(arch="openmic", n_classes=20)
    net.eval()
    return net


def main() -> None:
    print("Loading PaSST OpenMIC net (torch hub / cache)...")
    net = _load_net()
    model = PasstLogits(net)
    model.eval()

    dummy = torch.randn(1, 1, N_MELS, N_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    print(f"  smoke: in {tuple(dummy.shape)} -> out {tuple(out.shape)}")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting -> {DEST} ...")
    torch.onnx.export(
        model,
        dummy,
        str(DEST),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"  done: {DEST.stat().st_size / 1e6:.1f} MB")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(DEST), providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  IN  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"  OUT {o.name}: {o.shape} {o.type}")


if __name__ == "__main__":
    main()

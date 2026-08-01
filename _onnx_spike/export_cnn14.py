"""Export Cnn14 (PANNs audio tagging) to ONNX — full pipeline in one file.

Cnn14.forward includes Spectrogram + LogMel + BN + 6 conv blocks, so the whole
thing exports as one graph: raw 32kHz mono waveform in -> 527 AudioSet probs out.
No torchlibrosa / librosa needed at runtime.

The app's _PannsBackend.predict passes (batch, samples); we match that.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from panns_inference.models import Cnn14

DEST = Path(__file__).resolve().parent / 'onnx_out' / 'cnn14.onnx'

# Cnn14 params (from panns_inference.AudioTagging)
SR = 32000
WINDOW = 1024
HOP = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
SEGMENT_SAMPLES = SR * 10  # 10s clip, standard PANNs input; dynamic over time


class Cnn14Clip(nn.Module):
    """Thin wrapper: Cnn14 -> clipwise_output only (drop dict + embedding)."""

    def __init__(self):
        super().__init__()
        self.cnn14 = Cnn14(sample_rate=SR, window_size=WINDOW, hop_size=HOP,
                           mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX,
                           classes_num=527)
        self.cnn14.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (batch, samples) -> (batch, 527)
        out = self.cnn14(audio, None)
        return out['clipwise_output']


def main():
    # Locate the checkpoint the app uses
    ckpt = Path(r"D:/STEM-organizer-Py6/panns_tagger/models/Cnn14_mAP=0.431.pth")
    if not ckpt.is_file():
        # fallback to panns_data home
        ckpt = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
    print(f"checkpoint: {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)" if ckpt.is_file()
          else f"checkpoint MISSING: {ckpt}")

    model = Cnn14Clip()
    # panns_inference stores weights under a 'model' key; keys are unprefixed
    # (spectrogram_extractor..., bn0..., conv_block1...) matching Cnn14 itself.
    state = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    # Load into the inner Cnn14 (NOT the wrapper) so prefixes match.
    missing, unexpected = model.cnn14.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if 'spec_augment' not in m and 'mixup' not in m]
    print(f"  loaded: {len(sd)} params | real missing: {len(real_missing)} | "
          f"unexpected: {len(unexpected)}")
    if real_missing:
        print(f"  REAL MISSING (first 10): {real_missing[:10]}")
    model.eval()

    dummy = torch.randn(1, SEGMENT_SAMPLES, dtype=torch.float32)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting -> {DEST}  (input {tuple(dummy.shape)}) ...")
    torch.onnx.export(
        model, dummy, DEST,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['audio'], output_names=['probs'],
        dynamic_axes={'audio': {0: 'batch', 1: 'samples'}, 'probs': {0: 'batch'}},
        dynamo=False,
    )
    print(f"  done: {DEST.stat().st_size/1e6:.1f} MB")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(DEST), providers=['CPUExecutionProvider'])
    for i in sess.get_inputs():
        print(f'  IN  {i.name}: {i.shape} {i.type}')
    for o in sess.get_outputs():
        print(f'  OUT {o.name}: {o.shape} {o.type}')


if __name__ == '__main__':
    main()

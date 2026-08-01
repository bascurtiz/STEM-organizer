#!/usr/bin/env python
"""Extend GSoC convert-pth-to-onnx.py to any HTDemucs variant.
Exports a single ONNX file (STFT/iSTFT in-graph) for the named demucs model."""
import argparse, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent / 'demucs-gsoc'))
import torch
import torch.nn.functional as F
from torch.nn import functional as F
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['htdemucs','htdemucs_ft','htdemucs_6s'])
    ap.add_argument('dest_dir')
    a = ap.parse_args()
    dest = Path(a.dest_dir); dest.mkdir(parents=True, exist_ok=True)

    model = get_model(a.model)
    core = model.models[0] if hasattr(model, 'models') else model
    assert isinstance(core, HTDemucs)
    core.onnx_exportable = True
    core.eval()

    seg = int(core.segment * core.samplerate)
    dummy = F.pad(torch.randn(1, 2, seg), (0, seg - seg))  # (1,2,seg)
    out_path = dest / f"{a.model}.onnx"
    print(f"Exporting {a.model} -> {out_path} ...")
    torch.onnx.export(
        core, dummy, out_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        dynamo=False,  # legacy TorchScript tracer (GSoC script targets this; new dynamo exporter chokes on pad1d shape guard)
    )
    mb = out_path.stat().st_size / 1e6
    print(f"  done: {mb:.1f} MB")

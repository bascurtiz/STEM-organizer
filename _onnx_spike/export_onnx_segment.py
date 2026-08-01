#!/usr/bin/env python
"""Re-export htdemucs ONNX at a SHORTER fixed segment (B2 spike).

The production StemSplit weight is traced at the max 7.8 s segment (343980
samples). This script re-traces at a shorter segment (e.g. 2/3/4 s) so the
time axis of every activation is smaller — testing whether DirectML's ~31 GB
VRAM blow-up scales with segment length or is shape-independent.

Runs in the ISOLATED torch export venv (.demucs-export-venv). NEVER co-load
this with onnxruntime in the same process (see AGENTS.md §8.1 freeze).

I/O names match the StemSplit contract (mix -> stems) so the same raw bench
harness and runner work; only SEGMENT_LENGTH differs.

Usage (from repo root, in .demucs-export-venv):
  python _onnx_spike/export_onnx_segment.py --segment 2.0 --out _smoke/htdemucs_s2.onnx
  python _onnx_spike/export_onnx_segment.py --segment 3.0 --out _smoke/htdemucs_s3.onnx
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent / "demucs-gsoc"))

import torch
import torch.nn.functional as F
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs

SAMPLERATE = 44100


def export_segment(segment_seconds: float, out_path: Path, *, fp16: bool = False) -> None:
    model = get_model("htdemucs")
    core = model.models[0] if hasattr(model, "models") else model
    assert isinstance(core, HTDemucs), f"expected HTDemucs, got {type(core)}"
    core.onnx_exportable = True
    core.eval()

    # Override the segment length BEFORE tracing. The export dummy input must
    # match; the traced graph then has a static time axis = segment_samples.
    core.segment = segment_seconds
    seg = int(segment_seconds * SAMPLERATE)
    print(f"[export] segment={segment_seconds}s -> {seg} samples")

    if fp16:
        core = core.half()
        dummy = torch.randn(1, 2, seg, dtype=torch.float16)
        print("[export] fp16 trace")
    else:
        dummy = torch.randn(1, 2, seg, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] tracing -> {out_path} ...")
    torch.onnx.export(
        core, dummy, str(out_path),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["mix"], output_names=["stems"],
        dynamic_axes={"mix": {0: "batch"}, "stems": {0: "batch"}},
        dynamo=False,  # legacy tracer (pad1d shape guard breaks dynamo)
    )
    mb = out_path.stat().st_size / 1e6
    print(f"[export] done: {mb:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", type=float, required=True,
                    help="segment length in seconds (e.g. 2.0, 3.0, 4.0)")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--fp16", action="store_true", help="trace in half precision (B3)")
    a = ap.parse_args()
    export_segment(a.segment, Path(a.out), fp16=a.fp16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

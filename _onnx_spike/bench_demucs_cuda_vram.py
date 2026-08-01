"""Bench StemSplit htdemucs on CUDA EP; print peak VRAM.

Usage:
  _onnx_spike\\.venv-cuda\\Scripts\\python.exe _onnx_spike\\bench_demucs_cuda_vram.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KILL_GB = float(os.environ.get("STEM_CUDA_KILL_GB", "20"))
CLIP = ROOT / "_smoke" / "in" / "clip.wav"
MEM_LIMIT = int(float(os.environ.get("STEM_CUDA_MEM_LIMIT_GB", "8")) * (1024**3))


def _nvidia_used_gb() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(vals) / 1024.0 if vals else None
    except Exception as e:
        print("nvidia-smi unavailable:", e, file=sys.stderr)
        return None


def main() -> int:
    import onnxruntime as ort
    from demucs_onnx import resolve_htdemucs_onnx, separate

    path = resolve_htdemucs_onnx()
    if path is None or not CLIP.is_file():
        print("ERROR: model or clip missing", file=sys.stderr)
        return 2

    print(f"ort={ort.__version__} providers_avail={ort.get_available_providers()}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("ERROR: CUDAExecutionProvider not available", file=sys.stderr)
        return 2

    baseline = _nvidia_used_gb()
    print(f"model={path} baseline_vram_gb={baseline} kill_gb={KILL_GB}")

    peak = {"gb": baseline or 0.0}
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.3):
            used = _nvidia_used_gb()
            if used is None:
                continue
            if used > peak["gb"]:
                peak["gb"] = used
                print(f"  peak_vram_gb={used:.2f}", flush=True)
            if used >= KILL_GB:
                print(f"ABORT: VRAM {used:.2f} >= {KILL_GB}", flush=True)
                os._exit(99)

    threading.Thread(target=monitor, daemon=True).start()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": MEM_LIMIT,
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        ),
        "CPUExecutionProvider",
    ]
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        print(f"session_providers={sess.get_providers()}")
        audio, sr = sf.read(str(CLIP), dtype="float32", always_2d=True)
        mix = audio.T
        if mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)
        elif mix.shape[0] > 2:
            mix = mix[:2]
        out = separate(sess, mix)
        dt = time.perf_counter() - t0
        print(f"out={out.shape} finite={bool(np.isfinite(out).all())} sec={dt:.2f}")
        print(f"peak_vram_gb={peak['gb']:.2f}")
        print("CUDA_BENCH_OK" if peak["gb"] <= 12.0 else "CUDA_BENCH_HIGH")
        return 0 if peak["gb"] <= 12.0 else 3
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())

"""Bench StemSplit htdemucs on DirectML; print peak VRAM; abort if runaway.

Usage (from repo root):
  set STEM_DEMUCS_DML=1
  .build-venv\\Scripts\\python.exe _onnx_spike\\bench_demucs_dml_vram.py
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

# Default kill only for true runaway (prior ~31 GB blow-ups). 12 GB is fine.
KILL_GB = float(os.environ.get("STEM_DML_KILL_GB", "40"))
CLIP = ROOT / "_smoke" / "in" / "clip.wav"


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
        if not vals:
            return None
        return max(vals) / 1024.0  # MiB → GiB
    except Exception as e:
        print("nvidia-smi unavailable:", e, file=sys.stderr)
        return None


def main() -> int:
    os.environ["STEM_DEMUCS_DML"] = "1"
    from demucs_onnx import DemucsOnnxModel, resolve_htdemucs_onnx

    path = resolve_htdemucs_onnx()
    if path is None:
        print("ERROR: htdemucs.onnx not found", file=sys.stderr)
        return 2
    if not CLIP.is_file():
        print(f"ERROR: missing {CLIP}", file=sys.stderr)
        return 2

    print(f"model={path}")
    print(f"kill_if_vram_gb>{KILL_GB}")
    baseline = _nvidia_used_gb()
    print(f"baseline_vram_gb={baseline}")

    peak = {"gb": baseline or 0.0, "abort": False}
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.4):
            used = _nvidia_used_gb()
            if used is None:
                continue
            if used > peak["gb"]:
                peak["gb"] = used
                print(f"  peak_vram_gb={used:.2f}", flush=True)
            if used >= KILL_GB:
                peak["abort"] = True
                print(f"ABORT: VRAM {used:.2f} GB >= {KILL_GB}", flush=True)
                stop.set()
                # Hard-kill this process — DML often won't unwind cleanly.
                os._exit(99)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    t0 = time.perf_counter()
    try:
        m = DemucsOnnxModel(path, prefer_gpu=True)
        print(f"providers={m.session.get_providers()} device={m._device}")
        audio, sr = sf.read(str(CLIP), dtype="float32", always_2d=True)
        mix = audio.T
        if mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)
        elif mix.shape[0] > 2:
            mix = mix[:2]
        print(f"clip_shape={mix.shape} sr={sr}", flush=True)
        out = m.separate_numpy(mix)
        dt = time.perf_counter() - t0
        print(f"out={out.shape} finite={bool(np.isfinite(out).all())} sec={dt:.2f}")
        print(f"peak_vram_gb={peak['gb']:.2f}")
        # 12 GB peak is acceptable on modern GPUs; only flag extreme spikes.
        if peak["gb"] <= 12.0:
            print("DML_BENCH_OK")
            return 0
        if peak["gb"] <= 20.0:
            print("DML_BENCH_OK_HIGH")
            return 0
        print("DML_BENCH_HIGH_VRAM")
        return 3
    finally:
        stop.set()


if __name__ == "__main__":
    raise SystemExit(main())

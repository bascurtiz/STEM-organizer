"""Raw ORT harness for Demucs (no demucs_onnx import) — Phase B spike tooling.

Runs a single htdemucs ONNX forward on a chosen provider and reports:
  - peak dedicated VRAM (nvidia-smi) with a kill-watchdog
  - wall-clock per segment
  - stem validity (finite, sum≈mix) and optional SI-SDR vs a reference

Designed to run in the ISOLATED DirectML venv (.demucs-dml-venv) OR the CUDA
build venv. It never imports demucs_onnx / ort_util / classify_backend, so it is
safe to point at any ONNX variant (simplified / short-segment / fp16 / split).

Usage:
  python _onnx_spike/bench_demucs_raw.py --onnx models/htdemucs.onnx \
      --provider dml --input-name mix --output-name stems

VRAM watchdog: polls nvidia-smi every 0.5s; if dedicated VRAM exceeds
--kill-gb (default 28), hard-kills the process so a recurrence of the ~31 GB
blow-up cannot stall the 5090. Set --kill-gb 0 to disable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEGMENT_LENGTH = 343980  # int(7.8 * 44100); override with --segment
SAMPLERATE = 44100


def _nvidia_used_gb() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(vals) / 1024.0 if vals else None  # MiB -> GiB
    except Exception as e:
        print(f"[watchdog] nvidia-smi unavailable: {e}", file=sys.stderr)
        return None


def _ensure_cuda_dlls():
    """Add nvidia-*-cu12 bin dirs to PATH / add_dll_directory for the CUDA EP.

    The production path does this via ort_util.ensure_nvidia_cuda_dlls(); we
    replicate the minimum here so the raw harness works standalone (no demucs_onnx
    import) — needed to load cudnn64_9.dll / cublas*64_12.dll in the build venv.
    """
    import importlib.metadata as md
    import os
    added = False
    for pkg in ("nvidia-cudnn-cu12", "nvidia-cublas-cu12",
                "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12"):
        try:
            files = md.distribution(pkg).files or []
        except Exception:
            continue
        for f in files:
            s = str(f).replace("\\", "/")
            if s.endswith("/bin") or (s.endswith(".dll") and "/bin/" in s):
                d = os.path.dirname(s) if s.endswith(".dll") else s
                # Resolve against site-packages.
                d = os.path.join(md.distribution(pkg).locate_file("."), d)
                d = os.path.abspath(d)
                if os.path.isdir(d):
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
                    added = True
    if not added:
        # Fallback: scan the venv site-packages for nvidia/*/bin
        import sys
        for sp in sys.path:
            base = os.path.join(sp, "nvidia")
            if os.path.isdir(base):
                for pkg in os.listdir(base):
                    bind = os.path.join(base, pkg, "bin")
                    if os.path.isdir(bind):
                        os.environ["PATH"] = bind + os.pathsep + os.environ.get("PATH", "")
                        try:
                            os.add_dll_directory(bind)
                        except Exception:
                            pass


def _make_session(onnx_path: Path, provider: str):
    import onnxruntime as ort
    if provider == "cuda":
        _ensure_cuda_dlls()
    so = ort.SessionOptions()
    so.enable_mem_pattern = False  # per ORT DirectML guidance
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider == "dml":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [("CUDAExecutionProvider", {
            "device_id": 0,
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GiB soft cap (matches prod)
            "arena_extend_strategy": "kNextPowerOfTwo",
        }), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    return sess, sess.get_providers()


def _si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SDR between two (C, T) float arrays. Higher = better."""
    ref = ref.astype(np.float64).reshape(ref.shape[0], -1)
    est = est.astype(np.float64).reshape(est.shape[0], -1)
    # zero-mean
    ref = ref - ref.mean(axis=1, keepdims=True)
    est = est - est.mean(axis=1, keepdims=True)
    dot = np.sum(ref * est, axis=1)
    proj = (dot / (np.sum(ref * ref, axis=1) + eps)) * ref
    noise = est - proj
    ratio = (np.sum(proj * proj, axis=1) + eps) / (np.sum(noise * noise, axis=1) + eps)
    return float(10.0 * np.log10(np.maximum(ratio, eps)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to the .onnx file")
    ap.add_argument("--provider", default="dml", choices=["dml", "cuda", "cpu"])
    ap.add_argument("--input-name", default="mix", help="graph input tensor name")
    ap.add_argument("--output-name", default="stems", help="graph output tensor name")
    ap.add_argument("--segment", type=int, default=SEGMENT_LENGTH,
                    help="segment length in samples (must match the export)")
    ap.add_argument("--clip", default=str(ROOT / "_smoke" / "in" / "clip.wav"),
                    help="audio clip (stereo wav); omit to use zeros")
    ap.add_argument("--kill-gb", type=float, default=28.0,
                    help="watchdog kill threshold in GiB; 0 disables")
    ap.add_argument("--reference", default="",
                    help="optional .npy of a reference stems array for SI-SDR")
    ap.add_argument("--save", default="",
                    help="optional path to save the output stems (4,2,T) as .npy")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"ERROR: {onnx_path} not found", file=sys.stderr)
        return 2

    # Build input chunk (1, 2, segment)
    clip = Path(args.clip)
    if clip.is_file():
        try:
            import soundfile as sf
            audio, sr = sf.read(str(clip), dtype="float32", always_2d=True)
            mix = audio.T  # (C, T)
            if mix.shape[0] == 1:
                mix = np.repeat(mix, 2, axis=0)
            elif mix.shape[0] > 2:
                mix = mix[:2]
            seg = args.segment
            chunk = np.zeros((1, 2, seg), np.float32)
            t = min(seg, mix.shape[1])
            chunk[0, :, :t] = mix[:, :t]
            print(f"[input] clip {clip.name} sr={sr} -> chunk {chunk.shape} ({t} real samples)")
        except Exception as e:
            print(f"[input] soundfile load failed ({e}); using zeros")
            chunk = np.zeros((1, 2, args.segment), np.float32)
    else:
        print(f"[input] no clip at {clip}; using zeros")
        chunk = np.zeros((1, 2, args.segment), np.float32)

    # Watchdog
    kill_gb = args.kill_gb
    peak = {"gb": 0.0}
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.5):
            used = _nvidia_used_gb()
            if used is None:
                continue
            if used > peak["gb"]:
                peak["gb"] = used
            if kill_gb and used >= kill_gb:
                print(f"[watchdog] ABORT: VRAM {used:.2f} GB >= {kill_gb} — killing process",
                      flush=True)
                stop.set()
                os._exit(99)

    if kill_gb and (args.provider in ("dml", "cuda")):
        threading.Thread(target=monitor, daemon=True).start()

    print(f"[bench] onnx={onnx_path.name} provider={args.provider} in={args.input_name} out={args.output_name}")
    print(f"[bench] kill_gb={kill_gb} baseline_vram_gb={_nvidia_used_gb()}")

    t0 = time.perf_counter()
    sess, active = _make_session(onnx_path, args.provider)
    print(f"[bench] session created in {time.perf_counter()-t0:.2f}s; active providers={active}")

    t1 = time.perf_counter()
    out = sess.run([args.output_name], {args.input_name: chunk})[0]
    dt = time.perf_counter() - t1
    finite = bool(np.isfinite(out).all())
    print(f"[bench] forward done in {dt:.3f}s; out={out.shape} finite={finite}")
    print(f"[bench] peak_vram_gb={peak['gb']:.2f}")

    # Validity: stems should sum back to ~mix (Demucs masking reconstruction)
    if out.ndim == 4 and out.shape[1] == 4:
        recon = out[0].sum(axis=0)  # (2, T)
        t = min(recon.shape[1], chunk.shape[2])
        rel = np.linalg.norm(recon[:, :t] - chunk[0, :, :t]) / (np.linalg.norm(chunk[0, :, :t]) + 1e-8)
        print(f"[bench] sum(stems)≈mix rel-L2 = {rel:.4f} (expect small for valid Demucs)")
    else:
        print(f"[bench] unexpected output ndim {out.ndim}; skipping reconstruction check")

    # Optional SI-SDR vs a reference (e.g. the CUDA baseline stems)
    if args.reference:
        ref = np.load(args.reference)
        # Compare per-stem, mean SI-SDR
        sdrs = [_si_sdr(ref[i, :, :], out[0, i, :, :]) for i in range(min(ref.shape[0], out.shape[1]))]
        print(f"[bench] SI-SDR vs reference: mean={np.mean(sdrs):.2f} dB per-stem={[f'{s:.1f}' for s in sdrs]}")

    # Optional save of the output stems (4, 2, T) for use as a downstream reference
    if args.save:
        np.save(args.save, out[0].astype(np.float32))
        print(f"[bench] saved stems {out[0].shape} -> {args.save}")

    print("[bench] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

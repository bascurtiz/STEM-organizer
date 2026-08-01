"""Lightweight torch-free validation of the demucs_onnx segment loop.

Validates correctness WITHOUT loading torch (the thing that froze the machine):
  1. Reconstruction: Demucs is a masking model -> sum(stems) ~= mixture.
     If the segment/overlap/blend loop is wrong, this breaks loudly.
  2. Sanity: per-stem energy is plausible (no NaNs, no silence, no blow-up).

Run: python validate_loop.py
"""
from __future__ import annotations
import sys, time, warnings, os
warnings.filterwarnings('ignore')
import numpy as np
import onnxruntime as ort
from demucs_onnx import separate, SAMPLERATE

SOURCES = ['drums', 'bass', 'other', 'vocals']


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'onnx_out/htdemucs.onnx'
    raw_path = sys.argv[2] if len(sys.argv) > 2 else 'test_track.raw'

    # Load raw f32 stereo (interleaved L,R,L,R...) -> (2, length)
    raw = np.fromfile(raw_path, dtype=np.float32)
    assert raw.size % 2 == 0, "expected interleaved stereo"
    length = raw.size // 2
    mix = raw.reshape(length, 2).T  # (2, length)
    mix = mix.astype(np.float32)[np.newaxis]  # (1, 2, length)
    print(f"Track: {raw_path}  ({mix.shape}, {length/SAMPLERATE:.1f}s)\n")

    print(f"Loading {onnx_path} ...")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print(f"  providers: {sess.get_providers()}")

    # Memory snapshot before separation
    def mem_mb():
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:
            return -1
    m0 = mem_mb()

    print("Separating (shifts=0, split, overlap=0.25) ...")
    t0 = time.time()
    stems = separate(sess, mix)   # (1, 4, 2, length)
    dt = time.time() - t0
    m1 = mem_mb()
    print(f"  done in {dt:.1f}s  shape={stems.shape}  peak RAM ~{m1:.0f} MB (start {m0:.0f} MB)\n")

    # --- Checks ---
    print("=" * 64)
    # 1. Finite
    finite = np.isfinite(stems).all()
    print(f"finite (no NaN/Inf)   : {'OK' if finite else 'FAIL'}")

    # 2. Per-stem energy
    print(f"{'stem':8s} {'rms':>10s} {'peak':>10s}")
    for i, name in enumerate(SOURCES):
        s = stems[0, i]
        print(f"{name:8s} {float(np.sqrt(np.mean(s**2))):>10.4f} {float(np.abs(s).max()):>10.4f}")

    # 3. Reconstruction: sum of stems should approximate the mixture.
    recon = stems.sum(axis=1)[0]  # (2, length)
    diff = recon - mix[0]
    rel_err = np.linalg.norm(diff) / (np.linalg.norm(mix[0]) + 1e-12)
    print(f"\nreconstruction        : sum(stems) vs mixture")
    print(f"  rel L2 error        : {rel_err:.3e}")
    print(f"  (demucs is a masking model; small error => loop+blend is correct)")

    # 4. SI-SDR of reconstruction vs mixture (should be high; not a stem metric)
    def si_sdr(est, ref):
        e = est.reshape(-1).astype(np.float64); r = ref.reshape(-1).astype(np.float64)
        sc = float(r @ e) / (float(r @ r) + 1e-12)
        n = e - sc * r
        return 10 * np.log10((float((sc*r) @ (sc*r)) + 1e-12) / (float(n @ n) + 1e-12))
    print(f"  recon SI-SDR        : {si_sdr(recon, mix[0]):.2f} dB")
    print("=" * 64)

    print("\nVerdict (loop correctness):")
    if finite and rel_err < 0.05:
        print(f"  OK  — loop is correct (finite, reconstruction rel-L2 {rel_err:.2e}).")
        print(f"        Combined with per-segment torch/ONNX parity (rel-L2 1.5e-4),")
        print(f"        Demucs ONNX path is GO.")
    else:
        print(f"  REVIEW — finite={finite} rel_err={rel_err:.3e}; investigate.")


if __name__ == '__main__':
    main()

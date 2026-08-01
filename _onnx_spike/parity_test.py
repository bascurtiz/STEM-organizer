"""Phase 0 go/no-go: compare torch Demucs vs our pure-numpy ONNX runner on real audio.

Both paths use the SAME segment params (shifts=0, split=True, overlap=0.25).
Metric: SI-SDR per stem + max/mean abs diff. SI-SDR > ~10 dB = excellent.
"""
from __future__ import annotations
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'demucs-gsoc')

import numpy as np
import torch
import onnxruntime as ort
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

SAMPLERATE = 44100
SOURCES = ['drums', 'bass', 'other', 'vocals']


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR (dB). Higher = better. ∞ if est==ref."""
    est = est.astype(np.float64).reshape(-1)
    ref = ref.astype(np.float64).reshape(-1)
    ref_energy = float(ref @ ref) + 1e-12
    scale = float(ref @ est) / ref_energy
    target = scale * ref
    noise = est - target
    ratio = float(target @ target) / (float(noise @ noise) + 1e-12)
    return 10.0 * np.log10(ratio + 1e-12)


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else 'onnx_out/htdemucs.onnx'
    audio_path = sys.argv[2] if len(sys.argv) > 2 else 'demucs-gsoc/test.mp3'

    # --- Decode once ---
    wav = AudioFile(audio_path).read(samplerate=SAMPLERATE, channels=2)
    wav = wav.numpy() if hasattr(wav, 'numpy') else np.asarray(wav)
    wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav[np.newaxis]
    length = wav.shape[-1]
    print(f"Track: {audio_path}  ({wav.shape}, {length/SAMPLERATE:.1f}s)\n")

    # --- torch reference (demucs apply_model, exact same params as classify_backend SDR probe) ---
    print("torch reference (apply_model shifts=0 split overlap=0.25) ...")
    model = get_model('htdemucs')
    core = model.models[0] if hasattr(model, 'models') else model
    core.onnx_exportable = True
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out_torch = apply_model(
            model, torch.from_numpy(wav),
            shifts=0, split=True, overlap=0.25, progress=False,
        )
    stems_torch = out_torch.numpy().astype(np.float32)
    print(f"  {time.time()-t0:.1f}s  shape={stems_torch.shape}\n")

    # --- ONNX runner (pure numpy) ---
    from demucs_onnx import separate
    sess = ort.InferenceSession(onnx_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    print(f"ONNX runner (providers={sess.get_providers()}) ...")
    t0 = time.time()
    stems_onnx = separate(sess, wav)
    print(f"  {time.time()-t0:.1f}s  shape={stems_onnx.shape}\n")

    # --- Compare ---
    print("=" * 60)
    print(f"{'stem':8s} {'SI-SDR(dB)':>11s} {'max|d|':>10s} {'mean|d|':>10s}")
    print("-" * 60)
    all_sisdr = []
    for i, name in enumerate(SOURCES):
        est = stems_onnx[0, i]   # (2, length)
        ref = stems_torch[0, i]
        s = si_sdr(est, ref)
        all_sisdr.append(s)
        diff = np.abs(ref - est)
        print(f"{name:8s} {s:>11.3f} {diff.max():>10.2e} {diff.mean():>10.2e}")
    print("-" * 60)
    print(f"{'MEAN':8s} {np.mean(all_sisdr):>11.3f}")
    print("=" * 60)
    print("\nVerdict:")
    mean_sisdr = float(np.mean(all_sisdr))
    if mean_sisdr > 20:
        print(f"  GO  (mean SI-SDR {mean_sisdr:.1f} dB — effectively identical)")
    elif mean_sisdr > 10:
        print(f"  GO  (mean SI-SDR {mean_sisdr:.1f} dB — inaudible diff; matches GSoC blog ~0.1 dB)")
    else:
        print(f"  REVIEW  (mean SI-SDR {mean_sisdr:.1f} dB — investigate before proceeding)")


if __name__ == '__main__':
    main()

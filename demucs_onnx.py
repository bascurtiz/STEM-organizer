"""Pure numpy + onnxruntime Demucs stem separator (production runner).

Uses StemSplitio/htdemucs-onnx weights + overlap-add contract (MIT):
  https://huggingface.co/StemSplitio/htdemucs-onnx
  mix (1,2,343980) → stems (1,4,2,343980) — drums, bass, other, vocals.

GPU path: CUDAExecutionProvider (onnxruntime-gpu). DirectML is unsuitable for
htdemucs (~31 GB VRAM on RTX 5090 even with StemSplit weights).

No torch / demucs package at runtime. Duck-types HTDemucs for classify_backend.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SEGMENT = 7.8  # seconds (StemSplit / htdemucs default)
SAMPLERATE = 44100
SEGMENT_LENGTH = int(SEGMENT * SAMPLERATE)  # 343980
N_SOURCES = 4
# StemSplit / official htdemucs source order.
SOURCES = ("drums", "bass", "other", "vocals")
ONNX_NAME = "htdemucs.onnx"
# StemSplit infer.py: overlap = N_SAMPLES // 4 → 25% overlap, linear fade window.
DEFAULT_OVERLAP = 0.25


def resolve_htdemucs_onnx(app_root: Path | None = None) -> Path | None:
    """Locate bundled StemSplit htdemucs.onnx."""
    roots: list[Path] = []
    if app_root is not None:
        roots.append(Path(app_root))
    here = Path(__file__).resolve().parent
    roots.extend([here, here / "models", here / "_onnx_spike" / "onnx_out"])
    try:
        from deps_bootstrap import app_dir

        roots.insert(0, app_dir())
        roots.insert(1, app_dir() / "models")
    except Exception:
        pass
    seen: set[str] = set()
    for root in roots:
        candidates = [root / ONNX_NAME, root / "models" / ONNX_NAME]
        if root.name == "onnx_out":
            candidates = [root / ONNX_NAME]
        for cand in candidates:
            key = str(cand.resolve()) if cand.exists() else str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                return cand
    return None


def _make_window(n: int, overlap: int) -> np.ndarray:
    """StemSplit linear fade window (infer.py)."""
    w = np.ones(n, dtype=np.float32)
    if overlap <= 0:
        return w
    fade = np.linspace(0, 1, overlap, dtype=np.float32)
    w[:overlap] = fade
    w[-overlap:] = fade[::-1]
    return w


def separate_segment(session, chunk: np.ndarray) -> np.ndarray:
    """chunk: (B,2,SEGMENT_LENGTH) → (B, N_SOURCES, 2, SEGMENT_LENGTH).

    StemSplit I/O names: mix → stems.
    """
    return session.run(["stems"], {"mix": chunk})[0]


def separate(
    session,
    mix: np.ndarray,
    *,
    overlap: float = DEFAULT_OVERLAP,
) -> np.ndarray:
    """Separate a full track. mix: (B, 2, length) or (2, length) float32.

    Returns (B, N_SOURCES, 2, length). Chunking matches StemSplit infer.py
    (overlap samples = SEGMENT_LENGTH * overlap, linear fade OLA).
    """
    if mix.ndim == 2:
        mix = mix[np.newaxis]
    batch, channels, length = mix.shape
    if channels != 2:
        raise ValueError(f"expected stereo, got {channels}ch")
    if batch != 1:
        parts = [separate(session, mix[i], overlap=overlap) for i in range(batch)]
        return np.concatenate(parts, axis=0)

    overlap_n = int(SEGMENT_LENGTH * overlap)
    if overlap_n < 0:
        overlap_n = 0
    if overlap_n >= SEGMENT_LENGTH:
        overlap_n = SEGMENT_LENGTH // 4
    stride = SEGMENT_LENGTH - overlap_n
    if stride < 1:
        stride = SEGMENT_LENGTH
    window = _make_window(SEGMENT_LENGTH, overlap_n)

    out = np.zeros((1, N_SOURCES, 2, length), dtype=np.float32)
    sum_weight = np.zeros(length, dtype=np.float32)

    for start in range(0, max(length, 1), stride):
        end = min(start + SEGMENT_LENGTH, length)
        chunk = np.zeros((1, 2, SEGMENT_LENGTH), dtype=np.float32)
        clen = end - start
        if clen <= 0:
            break
        chunk[0, :, :clen] = mix[0, :, start:end]

        stems = separate_segment(session, chunk)  # (1,4,2,T)
        w = window[:clen]
        out[0, :, :, start:end] += stems[0, :, :, :clen] * w
        sum_weight[start:end] += w

        if end >= length:
            break

    out /= np.maximum(sum_weight, 1e-8)
    return out


def _cuda_available() -> bool:
    try:
        from ort_util import cuda_ep_usable

        return bool(cuda_ep_usable())
    except Exception:
        return False


def _gpu_mem_limit_bytes() -> int:
    """Soft CUDA arena cap (bytes). Override with STEM_ORT_CUDA_MEM_LIMIT_GB."""
    raw = os.environ.get("STEM_ORT_CUDA_MEM_LIMIT_GB", "8").strip()
    try:
        gb = float(raw)
    except ValueError:
        gb = 8.0
    return max(1, int(gb * (1024**3)))


def _demucs_session_options():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so


def _demucs_providers(*, want_gpu: bool):
    """Return ORT providers list (may include CUDA provider options tuple)."""
    if want_gpu and _cuda_available():
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "gpu_mem_limit": _gpu_mem_limit_bytes(),
                    "arena_extend_strategy": "kNextPowerOfTwo",
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def _open_demucs_session(onnx_path: Path, *, want_gpu: bool):
    import onnxruntime as ort

    if want_gpu and _cuda_available():
        try:
            from ort_util import ensure_nvidia_cuda_dlls

            ensure_nvidia_cuda_dlls()
        except Exception:
            pass
    providers = _demucs_providers(want_gpu=want_gpu)
    so = _demucs_session_options()
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)


class DemucsOnnxModel:
    """Duck-types demucs HTDemucs for classify_backend call sites."""

    sources = SOURCES
    samplerate = SAMPLERATE
    audio_channels = 2
    segment = SEGMENT
    backend = "onnx"

    def __init__(self, onnx_path: Path, *, prefer_gpu: bool = False):
        self.onnx_path = Path(onnx_path)
        self.session = _open_demucs_session(self.onnx_path, want_gpu=bool(prefer_gpu))
        self.prefer_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        self._device = "cuda" if self.prefer_gpu else "cpu"

    def eval(self):
        return self

    def to(self, device: str):
        want_gpu = (device or "").strip().lower() not in ("cpu", "")
        active = "CUDAExecutionProvider" in self.session.get_providers()
        if want_gpu != active:
            self.session = _open_demucs_session(self.onnx_path, want_gpu=want_gpu)
            self.prefer_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        self._device = "cuda" if self.prefer_gpu else "cpu"
        return self

    def cpu(self):
        return self.to("cpu")

    def separate_numpy(self, mix: np.ndarray, *, overlap: float = DEFAULT_OVERLAP) -> np.ndarray:
        """mix (B,2,T) or (2,T) → (B,S,2,T)."""
        return separate(self.session, np.asarray(mix, dtype=np.float32), overlap=overlap)


def stem_onnx_enabled() -> bool:
    return os.environ.get("STEM_ONNX", "1").strip() != "0"

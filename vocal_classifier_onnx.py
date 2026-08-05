"""Vocal/instrumental CNN6 classifier ONNX runner.

Duck-types the stem-separator interface (DemucsOnnxModel) so
``classify_backend.classify_batch`` can consume it without changes. Instead of
real stem separation, the model produces a synthetic 4-stem output whose *RMS
ratios* encode the vocal/instrumental probability:

    vocals RMS ∝ prob
    drums/bass/other RMS ∝ (1-prob)/3 each

This lets ``classify_to_category`` see ``vocal_share ≈ prob`` and
``instrumental_share ≈ (1-prob)``, producing the same classification decisions
as the existing pipeline.

The actual audio stems are discarded after RMS computation — the pipeline
mixes *original* files, not separated stems. The synthetic audio exists only
to carry RMS values.

Input: 32 kHz mono ONNX model  →  output: (B, 1) float32 probability [0,1].
Runner converts stereo → mono internally.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

SAMPLERATE = 32000
CLIP_SECONDS = 10.0
CLIP_SAMPLES = int(SAMPLERATE * CLIP_SECONDS)  # 320000 — training segment length
CHUNK_OVERLAP = 0.5  # 50% overlap between chunks for long audio
# 4 sources for compatibility with 2-stem/4-stem classify modes.
SOURCES = ("drums", "bass", "other", "vocals")
ONNX_FILENAME = "vocal_classifier.onnx"


def resolve_vocal_classifier_onnx(app_root: Path | None = None) -> Path | None:
    """Find vocal_classifier.onnx in the usual model search paths."""
    candidates: list[Path] = []
    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "models" / ONNX_FILENAME,
        here / ONNX_FILENAME,
    ])
    if app_root is not None:
        candidates.insert(0, Path(app_root) / "models" / ONNX_FILENAME)
        candidates.insert(1, Path(app_root) / ONNX_FILENAME)
    try:
        from deps_bootstrap import app_dir
        ad = app_dir()
        candidates.insert(0, ad / "models" / ONNX_FILENAME)
        candidates.insert(1, ad / ONNX_FILENAME)
    except Exception:
        pass
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def vocal_classifier_installed(app_root: Path | None = None) -> bool:
    return resolve_vocal_classifier_onnx(app_root) is not None


def _providers(*, want_gpu: bool) -> list:
    """Prefer CUDA, then DirectML, then CPU."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    force_cpu = os.environ.get("STEM_ORT_FORCE_CPU", "").strip() == "1"
    if force_cpu or not want_gpu:
        return ["CPUExecutionProvider"]

    out: list = []
    if "CUDAExecutionProvider" in available:
        try:
            from ort_util import cuda_ep_usable, ensure_nvidia_cuda_dlls

            if cuda_ep_usable():
                ensure_nvidia_cuda_dlls()
                out.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "gpu_mem_limit": int(
                                float(
                                    os.environ.get(
                                        "STEM_ORT_CUDA_MEM_LIMIT_GB", "8"
                                    )
                                )
                                * (1024**3)
                            ),
                            "arena_extend_strategy": "kNextPowerOfTwo",
                        },
                    )
                )
        except Exception:
            if "CUDAExecutionProvider" in available:
                out.append("CUDAExecutionProvider")
    if "DmlExecutionProvider" in available:
        out.append("DmlExecutionProvider")
    out.append("CPUExecutionProvider")
    seen: set[str] = set()
    uniq: list = []
    for p in out:
        name = p[0] if isinstance(p, tuple) else p
        if name in seen:
            continue
        seen.add(name)
        uniq.append(p)
    return uniq


def _session_options():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    except Exception:
        pass
    return so


class VocalClassifierOnnxModel:
    """Duck-types the stem-separator surface for classify_backend.

    ``classify_batch`` calls ``model.samplerate``, ``model.sources``,
    ``model.separate_numpy(mix, overlap=...)``, ``model.to(device)``,
    and ``model.cpu()``.  All are implemented here.
    """

    sources = SOURCES
    samplerate = SAMPLERATE
    audio_channels = 2  # classify_batch loads stereo; we downmix to mono
    backend = "onnx"

    def __init__(self, onnx_path: Path, *, prefer_gpu: bool = False):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self._prefer_gpu = bool(prefer_gpu)
        providers = _providers(want_gpu=self._prefer_gpu)
        so = _session_options()
        self._session = ort.InferenceSession(
            str(self.onnx_path), sess_options=so, providers=providers
        )
        active = self._session.get_providers()
        if active:
            first = active[0]
            if "DmlExecutionProvider" in first:
                self._device = "dml"
            elif "CUDAExecutionProvider" in first:
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = "cpu"

    def eval(self):
        return self

    def to(self, device: str):
        """Switch ORT session between CPU and GPU providers."""
        import onnxruntime as ort

        want_gpu = (device or "").strip().lower() not in ("cpu", "")
        if want_gpu == (self._device != "cpu"):
            return self
        providers = _providers(want_gpu=want_gpu)
        so = _session_options()
        self._session = ort.InferenceSession(
            str(self.onnx_path), sess_options=so, providers=providers
        )
        active = self._session.get_providers()
        if active:
            first = active[0]
            if "DmlExecutionProvider" in first:
                self._device = "dml"
            elif "CUDAExecutionProvider" in first:
                self._device = "cuda"
            else:
                self._device = "cpu"
        else:
            self._device = "cpu"
        self._prefer_gpu = self._device != "cpu"
        return self

    def cpu(self):
        return self.to("cpu")

    def _classify_mono(self, mono: np.ndarray) -> float:
        """Run the ONNX model on mono audio, chunking if needed.

        The model was trained on 10 s clips (320k samples @ 32 kHz). Feeding
        entire songs (3–5 min) blows up intermediate CNN activations to
        several GB, causing CUDA OOM.  Instead, chunk long audio into 10 s
        segments with 50 % overlap, run each through ONNX, and return the
        mean probability across all chunks.

        ``mono``: (T,) float32.
        """
        t_len = mono.shape[0]
        if t_len <= CLIP_SAMPLES:
            # Short clip: zero-pad to CLIP_SAMPLES (matches training).
            if t_len < CLIP_SAMPLES:
                padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
                padded[:t_len] = mono
            else:
                padded = mono.astype(np.float32)
            inp = padded[np.newaxis, :]  # (1, T)
            out = self._session.run(["vocal_prob"], {"audio": inp})[0]
            return float(np.clip(out[0, 0], 0.0, 1.0))

        # Long audio: sliding-window chunks.
        stride = int(CLIP_SAMPLES * (1.0 - CHUNK_OVERLAP))
        stride = max(1, stride)
        n_chunks = (t_len - CLIP_SAMPLES) // stride + 1
        # Ensure we cover the tail.
        if (n_chunks - 1) * stride + CLIP_SAMPLES < t_len:
            n_chunks += 1

        probs: list[float] = []
        for ci in range(n_chunks):
            start = ci * stride
            end = min(start + CLIP_SAMPLES, t_len)
            chunk = mono[start:end].astype(np.float32)
            clen = chunk.shape[0]
            if clen < CLIP_SAMPLES:
                padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
                padded[:clen] = chunk
                chunk = padded
            inp = chunk[np.newaxis, :]
            out = self._session.run(["vocal_prob"], {"audio": inp})[0]
            probs.append(float(np.clip(out[0, 0], 0.0, 1.0)))

        return float(np.mean(probs))

    def separate_numpy(
        self, mix: np.ndarray, *, overlap: float = 0.0
    ) -> np.ndarray:
        """Synthetic 4-stem output where RMS ratios encode vocal probability.

        ``mix``: (B, 2, T) or (2, T) float32 stereo audio @ 32 kHz.

        Returns (B, 4, 2, T) where:
          - stem 3 (vocals) RMS ∝ prob
          - stems 0-2 (drums/bass/other) RMS ∝ (1-prob)/3 each

        The audio content is meaningless (scaled copy of input) — only RMS
        values matter for the downstream ``classify_to_category`` call.

        Long audio (> 10 s) is chunked into 10 s segments (50 % overlap)
        to stay within CUDA arena limits; the mean probability across all
        chunks is used.
        """
        single = mix.ndim == 2
        if single:
            mix = mix[np.newaxis]  # (1, 2, T)
        b, ch, t_len = mix.shape

        # Stereo → mono, then classify each item (with chunking if needed)
        prob = np.zeros((b, 1), dtype=np.float32)
        for i in range(b):
            mono = mix[i].mean(axis=0).astype(np.float32)  # (T,)
            prob[i, 0] = self._classify_mono(mono)

        # Build synthetic 4-stem output.
        # stems[3] = vocals, stems[0:3] = drums/bass/other
        out = np.zeros((b, 4, 2, t_len), dtype=np.float32)
        for i in range(b):
            p = float(prob[i, 0])
            # vocals gets full energy scaled by sqrt(p)
            vocal_scale = math.sqrt(max(p, 1e-12))
            # other 3 share (1-p)
            inst_scale = math.sqrt(max(1.0 - p, 1e-12) / 3.0)
            out[i, 3] = mix[i] * vocal_scale  # vocals
            for j in range(3):
                out[i, j] = mix[i] * inst_scale  # drums/bass/other

        return out

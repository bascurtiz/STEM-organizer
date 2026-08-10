"""Stem CNN6 11-class instrument classifier ONNX runner.

Duck-types the stem-separator interface (DemucsOnnxModel) so
``classify_backend.classify_batch`` can consume it without changes. Instead of
real stem separation, the model produces a synthetic 4-stem output whose *RMS
ratios* encode a 4-way stem share derived from the underlying 11-class
instrument prediction:

    vocals share ∝ P(VOCALS)
    bass    share ∝ P(BASS)
    drums   share ∝ P(DRUMS)
    other   share ∝ Σ P(FLUTE/FX/GUITAR/KEYS/ORGAN/STRINGS/SYNTH/WINDS)

This lets ``classify_to_category`` see those 4 shares and reach the same
2-stem (vocals/instrumental) or 4-stem (vocals/bass/drums/other) bucketing
decisions as the existing pipeline, while the *fine* 11-class prediction is
surfaced in the log line (e.g. ``FLUTE``, ``VOCALS``, ``KEYS``) via
``classify_batch`` reading ``self._last_fine_labels``.

The actual audio stems are discarded after RMS computation — the pipeline
mixes *original* files, not separated stems. The synthetic audio exists only
to carry RMS values.

Input: 32 kHz mono ONNX model → output: (B, 11) float32 probabilities.
Runner converts stereo → mono internally.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SAMPLERATE = 32000
CLIP_SECONDS = 10.0
CLIP_SAMPLES = int(SAMPLERATE * CLIP_SECONDS)  # 320000 — training segment length
CHUNK_OVERLAP = 0.5  # 50% overlap between chunks for long audio
SILENCE_RMS_FLOOR = 1e-4  # chunks quieter than this contribute ~0 weight to the average
MAX_ONNX_BATCH = 16  # cap chunks per Session.run() call to bound peak memory
# 4 sources for compatibility with 2-stem/4-stem classify modes.
SOURCES = ("drums", "bass", "other", "vocals")
ONNX_FILENAME = "stem_cnn6.onnx"

# Fine 11-class label order — MUST match train_vocal_classifier.CLASSES and the
# ONNX output column order (column i == FINE_CLASSES[i]).
FINE_CLASSES = (
    "BASS",
    "DRUMS",
    "FLUTE",
    "FX",
    "GUITAR",
    "KEYS",
    "ORGAN",
    "STRINGS",
    "SYNTH",
    "VOCALS",
    "WINDS",
)
N_FINE = len(FINE_CLASSES)

# Collapse from the 11 fine classes onto the 4 synthetic-stem buckets.
# Index into FINE_CLASSES: vocals/bass/drums map 1:1, everything else → other.
_VOCALS_IDX = FINE_CLASSES.index("VOCALS")
_BASS_IDX = FINE_CLASSES.index("BASS")
_DRUMS_IDX = FINE_CLASSES.index("DRUMS")
_OTHER_IDX = tuple(
    i for i, c in enumerate(FINE_CLASSES)
    if i not in (_VOCALS_IDX, _BASS_IDX, _DRUMS_IDX)
)

# Index into SOURCES for each synthetic stem.
_SRC_DRUMS = SOURCES.index("drums")
_SRC_BASS = SOURCES.index("bass")
_SRC_OTHER = SOURCES.index("other")
_SRC_VOCALS = SOURCES.index("vocals")


def resolve_stem_cnn6_onnx(app_root: Path | None = None) -> Path | None:
    """Find stem_cnn6.onnx in the root models/ folder (single source)."""
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


def stem_cnn6_installed(app_root: Path | None = None) -> bool:
    return resolve_stem_cnn6_onnx(app_root) is not None


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


class StemCnn6OnnxModel:
    """Duck-types the stem-separator surface for classify_backend.

    ``classify_batch`` calls ``model.samplerate``, ``model.sources``,
    ``model.separate_numpy(mix, overlap=...)``, ``model.to(device)``,
    and ``model.cpu()``.  All are implemented here.

    After each ``separate_numpy`` call, ``self._last_fine_labels`` holds the
    fine 11-class prediction string (e.g. ``"FLUTE"``) for each batch item,
    so the caller can surface it in the log.
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
        # Fine label per batch item, populated by separate_numpy. classify_batch
        # reads this to thread the 11-class label into the log line.
        self._last_fine_labels: list[str | None] = []

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

    def _chunk_mono(self, mono: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Split mono audio into CLIP_SAMPLES-length chunks + per-chunk RMS weights.

        The model was trained on 10 s clips (320k samples @ 32 kHz), so every
        chunk here is exactly that length (zero-padded if the tail is
        shorter). Each chunk's RMS (computed on the *real*, unpadded audio —
        not the zero-padded tail) is returned alongside it so the caller can
        weight/skip near-silent chunks: a chunk that's mostly zero-amplitude
        input is audio the model never saw in training, and averaging its
        (essentially arbitrary) prediction in at full weight dilutes the
        chunks that actually contain signal.

        Returns:
            (chunks, weights) — parallel lists. ``chunks[i]`` is a
            ``(CLIP_SAMPLES,)`` float32 array; ``weights[i]`` is that chunk's
            RMS on the real (pre-padding) audio, floored at 0 below
            ``SILENCE_RMS_FLOOR``.
        """
        t_len = mono.shape[0]
        if t_len <= CLIP_SAMPLES:
            starts = [0]
        else:
            stride = max(1, int(CLIP_SAMPLES * (1.0 - CHUNK_OVERLAP)))
            n_chunks = (t_len - CLIP_SAMPLES) // stride + 1
            if (n_chunks - 1) * stride + CLIP_SAMPLES < t_len:
                n_chunks += 1
            starts = [ci * stride for ci in range(n_chunks)]

        chunks: list[np.ndarray] = []
        weights: list[float] = []
        for start in starts:
            end = min(start + CLIP_SAMPLES, t_len)
            real = mono[start:end].astype(np.float32)
            rms = float(np.sqrt(np.mean(real**2) + 1e-12)) if real.size else 0.0
            weights.append(rms if rms >= SILENCE_RMS_FLOOR else 0.0)
            if real.shape[0] < CLIP_SAMPLES:
                padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
                padded[: real.shape[0]] = real
                chunks.append(padded)
            else:
                chunks.append(real)
        return chunks, weights

    def _run_onnx_batch(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Run all chunks through ONNX, sub-batched to bound peak memory.

        ``chunks``: list of ``(CLIP_SAMPLES,)`` float32 arrays (any length,
        from one or many audio files pooled together).
        Returns: ``(len(chunks), N_FINE)`` float32 probabilities.
        """
        if not chunks:
            return np.zeros((0, N_FINE), dtype=np.float32)
        out_rows = []
        for start in range(0, len(chunks), MAX_ONNX_BATCH):
            sub = chunks[start : start + MAX_ONNX_BATCH]
            inp = np.stack(sub, axis=0)  # (n, CLIP_SAMPLES)
            out = self._session.run(["probs"], {"audio": inp})[0]
            out_rows.append(np.clip(out, 0.0, 1.0))
        return np.concatenate(out_rows, axis=0)

    def separate_numpy(
        self, mix: np.ndarray, *, overlap: float = 0.0
    ) -> np.ndarray:
        """Synthetic 4-stem output whose RMS ratios encode the stem shares.

        ``mix``: (B, 2, T) or (2, T) float32 stereo audio @ 32 kHz.

        The fine 11-class probabilities are first collapsed to 4 stem shares:

            vocals ∝ P(VOCALS)
            bass   ∝ P(BASS)
            drums  ∝ P(DRUMS)
            other  ∝ Σ P(remaining classes)

        Returns (B, 4, 2, T) where each stem is the input scaled by ``share``
        directly (linear, not sqrt) so that the RMS *ratio* the downstream
        ``classify_to_category`` computes — a linear ratio of RMS values —
        actually equals the predicted share. (Scaling by ``sqrt(share)``
        would make RMS scale as ``sqrt(share)``, which compresses confident
        predictions toward uniform once normalised.) The audio content is
        meaningless — only RMS values matter for the downstream call.

        Long audio (> 10 s) is chunked into 10 s segments (50 % overlap) to
        stay within CUDA arena limits. Near-silent chunks (RMS below
        ``SILENCE_RMS_FLOOR``) are weighted to ~0 in the average instead of
        contributing an arbitrary prediction at full weight — the model
        never saw near-silent input during training. All chunks from all
        items in the batch are pooled into shared ONNX calls (sub-batched by
        ``MAX_ONNX_BATCH``) rather than one call per chunk per item.

        Side effect: populates ``self._last_fine_labels`` (one entry per batch
        item) with the fine 11-class prediction string.
        """
        single = mix.ndim == 2
        if single:
            mix = mix[np.newaxis]  # (1, 2, T)
        b, ch, t_len = mix.shape

        # Build every item's chunks + weights, then pool all chunks from the
        # whole batch into shared ONNX calls instead of one call per item.
        per_item_chunks: list[list[np.ndarray]] = []
        per_item_weights: list[list[float]] = []
        pooled_chunks: list[np.ndarray] = []
        for i in range(b):
            mono = mix[i].mean(axis=0).astype(np.float32)  # (T,)
            chunks, weights = self._chunk_mono(mono)
            per_item_chunks.append(chunks)
            per_item_weights.append(weights)
            pooled_chunks.extend(chunks)

        pooled_probs = self._run_onnx_batch(pooled_chunks)  # (total_chunks, N_FINE)

        # Slice pooled predictions back out per item and weight-average them.
        probs = np.zeros((b, N_FINE), dtype=np.float32)
        fine_labels: list[str | None] = []
        offset = 0
        for i in range(b):
            n = len(per_item_chunks[i])
            item_probs = pooled_probs[offset : offset + n]  # (n, N_FINE)
            offset += n
            w = np.asarray(per_item_weights[i], dtype=np.float32)
            wsum = float(w.sum())
            if wsum > 0:
                p = (item_probs * w[:, np.newaxis]).sum(axis=0) / wsum
            else:
                # Every chunk was near-silent (e.g. an empty/blank stem) —
                # fall back to a plain mean so we still return *something*
                # rather than an all-zero vector.
                p = item_probs.mean(axis=0)
            probs[i] = p
            fine_labels.append(FINE_CLASSES[int(np.argmax(p))])
        self._last_fine_labels = fine_labels

        # Collapse fine probs → 4 stem shares.
        shares = np.zeros((b, len(SOURCES)), dtype=np.float32)
        shares[:, _SRC_VOCALS] = probs[:, _VOCALS_IDX]
        shares[:, _SRC_BASS] = probs[:, _BASS_IDX]
        shares[:, _SRC_DRUMS] = probs[:, _DRUMS_IDX]
        shares[:, _SRC_OTHER] = probs[:, _OTHER_IDX].sum(axis=1)
        # Renormalise so the 4 shares sum to 1 (handles the rare all-other case).
        denom = shares.sum(axis=1, keepdims=True) + 1e-12
        shares = shares / denom

        # Build synthetic 4-stem output. Linear scale (not sqrt) — see
        # docstring above for why this matters for the downstream RMS ratio.
        out = np.zeros((b, len(SOURCES), 2, t_len), dtype=np.float32)
        for i in range(b):
            for j in range(len(SOURCES)):
                scale = max(float(shares[i, j]), 1e-12)
                out[i, j] = mix[i] * scale

        return out

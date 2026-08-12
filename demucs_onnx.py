"""Pure numpy + onnxruntime Demucs stem separator (production runner).

Uses StemSplitio/htdemucs-onnx weights + overlap-add contract (MIT):
  https://huggingface.co/StemSplitio/htdemucs-onnx
  mix (B,2,343980) → stems (B,4,2,343980) — drums, bass, other, vocals.

StemSplit ships a fixed batch=1 graph; at load we promote dim-0 to a dynamic
``batch`` axis (cached sibling ``*.batch.onnx``) so Classify can run true
multi-file / multi-category ORT forwards. When file-batch ``B`` is below
``STEM_DEMUCS_MAX_BATCH``, multiple time offsets are packed into one forward
(same OLA math, fewer launches).

GPU path: CUDAExecutionProvider (onnxruntime-gpu). DirectML is unsuitable for
htdemucs (~31 GB VRAM on RTX 5090 even with StemSplit weights).

No torch / demucs package at runtime. Duck-types HTDemucs for classify_backend.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

SEGMENT = 7.8  # seconds (StemSplit / htdemucs default)
SAMPLERATE = 44100
SEGMENT_LENGTH = int(SEGMENT * SAMPLERATE)  # 343980
N_SOURCES = 4
# StemSplit / official htdemucs source order.
SOURCES = ("drums", "bass", "other", "vocals")
ONNX_NAME = "htdemucs.onnx"
# StemSplit infer.py used N_SAMPLES // 4 (25%); Classify analysis path defaults to 10%.
DEFAULT_OVERLAP = 0.1


def resolve_htdemucs_onnx(app_root: Path | None = None) -> Path | None:
    """Locate bundled StemSplit htdemucs.onnx (or htdemucs.batch.onnx).

    Installer DestName is ``htdemucs.onnx`` (content may already be the
    dynamic-batch rewrite). Older/dev trees may only have ``htdemucs.batch.onnx``.
    """
    names = (ONNX_NAME, "htdemucs.batch.onnx")
    roots: list[Path] = []
    if app_root is not None:
        roots.append(Path(app_root))
    here = Path(__file__).resolve().parent
    roots.extend([here, here / "models"])
    try:
        from deps_bootstrap import app_dir

        roots.insert(0, app_dir())
        roots.insert(1, app_dir() / "models")
    except Exception:
        pass
    seen: set[str] = set()
    for root in roots:
        candidates: list[Path] = []
        for name in names:
            candidates.append(root / name)
            candidates.append(root / "models" / name)
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


class _OlaSegmentRunner:
    """Reusable host chunk buffer for identical-shape OLA segment forwards.

    Avoids per-segment ``np.zeros`` allocs. (CUDA IOBinding was measured and did
    not beat ``session.run`` here — OLA fade still needs a host copy each seg.)
    """

    def __init__(self, session, batch: int, *, use_cuda: bool = False):
        self.session = session
        self.batch = int(batch)
        self.chunk = np.zeros((self.batch, 2, SEGMENT_LENGTH), dtype=np.float32)
        self._use_cuda = bool(use_cuda)  # reserved; inference uses session providers

    def infer(self) -> np.ndarray:
        """Run on ``self.chunk`` (must be C-contiguous float32)."""
        return separate_segment(self.session, self.chunk)


_WINDOW_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _cached_window(n: int, overlap: int) -> np.ndarray:
    key = (int(n), int(overlap))
    w = _WINDOW_CACHE.get(key)
    if w is None:
        w = _make_window(n, overlap)
        _WINDOW_CACHE[key] = w
    return w


def demucs_max_batch() -> int:
    """Soft cap for ONNX batch dim. Override with STEM_DEMUCS_MAX_BATCH (default 4)."""
    raw = os.environ.get("STEM_DEMUCS_MAX_BATCH", "4").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 4
    return max(1, n)


def separate(
    session,
    mix: np.ndarray,
    *,
    overlap: float = DEFAULT_OVERLAP,
    runner_cache: dict[int, _OlaSegmentRunner] | None = None,
) -> np.ndarray:
    """Separate full track(s). mix: (B, 2, length) or (2, length) float32.

    Returns (B, N_SOURCES, 2, length). One ORT forward per time-offset pack
    (StemSplit OLA). Requires a dynamic batch axis (see
    ``ensure_batch_dynamic_onnx``); falls back to per-item runs if the graph is B=1.
    """
    if mix.ndim == 2:
        mix = mix[np.newaxis]
    batch, channels, length = mix.shape
    if channels != 2:
        raise ValueError(f"expected stereo, got {channels}ch")

    if batch > 1:
        try:
            return _separate_ola(session, mix, overlap=overlap, runner_cache=runner_cache)
        except Exception as e:
            if _is_batch_shape_error(e):
                parts = [
                    _separate_ola(
                        session, mix[i : i + 1], overlap=overlap, runner_cache=runner_cache
                    )
                    for i in range(batch)
                ]
                return np.concatenate(parts, axis=0)
            raise
    return _separate_ola(session, mix, overlap=overlap, runner_cache=runner_cache)


def _separate_ola(
    session,
    mix: np.ndarray,
    *,
    overlap: float,
    runner_cache: dict[int, _OlaSegmentRunner] | None = None,
) -> np.ndarray:
    """OLA loop. mix must be (B, 2, length).

    When ``B < demucs_max_batch()``, packs multiple time offsets into one ORT
    forward (same segment math; fewer launches). Multi-file batches with B at
    the cap stay one-offset-per-forward.
    """
    batch, _channels, length = mix.shape

    overlap_n = int(SEGMENT_LENGTH * overlap)
    if overlap_n < 0:
        overlap_n = 0
    if overlap_n >= SEGMENT_LENGTH:
        overlap_n = SEGMENT_LENGTH // 4
    stride = SEGMENT_LENGTH - overlap_n
    if stride < 1:
        stride = SEGMENT_LENGTH
    window = _cached_window(SEGMENT_LENGTH, overlap_n)

    offsets = [s for s in range(0, max(length, 1), stride) if s < length]
    if not offsets:
        offsets = [0]

    max_pack = demucs_max_batch()
    k_pack = max(1, max_pack // batch)
    use_cuda = "CUDAExecutionProvider" in session.get_providers()

    def _runner_for(packed_b: int) -> _OlaSegmentRunner:
        if runner_cache is not None:
            r = runner_cache.get(packed_b)
            if r is None:
                r = _OlaSegmentRunner(session, packed_b, use_cuda=use_cuda)
                runner_cache[packed_b] = r
            return r
        return _OlaSegmentRunner(session, packed_b, use_cuda=use_cuda)

    out = np.zeros((batch, N_SOURCES, 2, length), dtype=np.float32)
    sum_weight = np.zeros(length, dtype=np.float32)

    for pack_i in range(0, len(offsets), k_pack):
        pack = offsets[pack_i : pack_i + k_pack]
        n_off = len(pack)
        packed_b = batch * n_off
        seg_runner = _runner_for(packed_b)

        chunk = seg_runner.chunk
        chunk.fill(0)
        clens: list[int] = []
        for oi, start in enumerate(pack):
            end = min(start + SEGMENT_LENGTH, length)
            clen = end - start
            clens.append(clen)
            base = oi * batch
            chunk[base : base + batch, :, :clen] = mix[:, :, start:end]

        stems = seg_runner.infer()  # (packed_b, 4, 2, T)

        for oi, start in enumerate(pack):
            clen = clens[oi]
            w = window[:clen]
            base = oi * batch
            out[:, :, :, start : start + clen] += stems[base : base + batch, :, :, :clen] * w
            sum_weight[start : start + clen] += w

    out /= np.maximum(sum_weight, 1e-8)
    return out


def _cuda_available() -> bool:
    try:
        from ort_util import cuda_ep_usable

        return bool(cuda_ep_usable())
    except Exception:
        return False


def _total_vram_bytes() -> int:
    """Total VRAM of device 0 (bytes) via nvidia-smi; 0 if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = (out.stdout or "").strip().splitlines()[0].strip()
        return int(line) * 1024 * 1024
    except Exception:
        return 0


def _gpu_mem_limit_bytes() -> int:
    """CUDA arena cap (bytes).

    Default: ~75% of device VRAM, clamped to [8 GB, 24 GB].  The old fixed
    8 GB cap OOMs HTDemucs on big cards (measured ~9 GB peak for one batched
    forward on an RTX 5090).  Override with STEM_ORT_CUDA_MEM_LIMIT_GB.
    """
    raw = os.environ.get("STEM_ORT_CUDA_MEM_LIMIT_GB", "").strip()
    if raw:
        try:
            return max(1, int(float(raw) * (1024**3)))
        except ValueError:
            pass
    total = _total_vram_bytes()
    if total > 0:
        cap = int(total * 0.75)
        cap = max(8 * 1024**3, min(cap, 24 * 1024**3))
        return cap
    return 8 * 1024**3


def _demucs_session_options(*, want_gpu: bool = False):
    """ORT session options aligned with ort_util tagger hygiene."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
    except Exception:
        pass
    env_intra = os.environ.get("STEM_ORT_INTRA_OP", "").strip()
    if env_intra:
        try:
            so.intra_op_num_threads = max(1, int(env_intra))
        except ValueError:
            so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    elif want_gpu:
        # CUDA EP: keep host thread pool modest; avoid oversubscription with UI.
        so.intra_op_num_threads = 1
    else:
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
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


def _batch_dim_is_dynamic(onnx_path: Path) -> bool:
    """True if graph input ``mix`` already has a symbolic / unbound batch dim."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    for inp in sess.get_inputs():
        if inp.name != "mix":
            continue
        dim0 = inp.shape[0] if inp.shape else None
        # Dynamic: str name, None, or unbound; fixed StemSplit: int 1.
        return not isinstance(dim0, int)
    return False


def ensure_batch_dynamic_onnx(onnx_path: Path) -> Path:
    """Return an ONNX path whose mix/stems batch axis is dynamic.

    StemSplit ``htdemucs.onnx`` is traced at B=1. We write a sibling
    ``<stem>.batch.onnx`` once (invalidated when the source mtime/size changes)
    so Classify batch_size>1 can issue a single ``session.run`` per OLA offset.
    Requires the ``onnx`` package; on failure returns the original path (B=1 only).
    """
    src = Path(onnx_path)
    if not src.is_file():
        return src

    cache = src.with_name(f"{src.stem}.batch{src.suffix}")
    # Already the batched sibling (or DestName was the batch graph): do not nest
    # ``htdemucs.batch.batch.onnx``.
    if src.name.endswith(".batch.onnx") or src.stem.endswith(".batch"):
        return src
    meta = cache.with_suffix(cache.suffix + ".srcmeta")
    try:
        st = src.stat()
        token = f"{st.st_mtime_ns}:{st.st_size}"
        if cache.is_file() and meta.is_file() and meta.read_text(encoding="utf-8").strip() == token:
            return cache
    except Exception:
        token = None

    # Already-dynamic verdict cache: the CPU probe below opens a full 316 MB
    # session (~17 s, GIL held). Persist the verdict so it runs once per model
    # version, not on every app launch.
    verdict = src.with_name(f"{src.stem}.dynamic")
    try:
        st = src.stat()
        token = f"{st.st_mtime_ns}:{st.st_size}"
        if verdict.is_file() and verdict.read_text(encoding="utf-8").strip() == token:
            return src
    except Exception:
        pass

    try:
        if _batch_dim_is_dynamic(src):
            try:
                st = src.stat()
                verdict.write_text(f"{st.st_mtime_ns}:{st.st_size}", encoding="utf-8")
            except Exception:
                pass
            return src
    except Exception:
        pass

    try:
        import onnx
    except ImportError:
        return src

    try:
        model = onnx.load(str(src))
        for inp in model.graph.input:
            if inp.name == "mix" and inp.type.tensor_type.HasField("shape"):
                d0 = inp.type.tensor_type.shape.dim[0]
                d0.dim_param = "batch"
                d0.ClearField("dim_value")
        for out in model.graph.output:
            if out.name == "stems" and out.type.tensor_type.HasField("shape"):
                d0 = out.type.tensor_type.shape.dim[0]
                d0.dim_param = "batch"
                d0.ClearField("dim_value")
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        onnx.save(model, str(tmp))
        tmp.replace(cache)
        st = src.stat()
        meta.write_text(f"{st.st_mtime_ns}:{st.st_size}", encoding="utf-8")
        return cache
    except Exception:
        return src


def _is_batch_shape_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "invalid dimensions" in msg or "expected: 1" in msg


# Process-wide ORT session reuse. Creating a CUDA session for the ~316 MB
# Demucs graph takes ~17 s with the GIL held (freezes the Qt UI during the
# load); the app otherwise builds a fresh session on every run (RMS classify
# + SI-SDR). Keyed by resolved model path + provider names, so a GPU/CPU
# switch or model swap gets its own session.
_DEMUCS_SESSION_CACHE: dict[tuple, object] = {}
# Serialize creation: the background warm-up and a run's first load can race;
# both would otherwise see an empty cache and build two sessions back-to-back.
_DEMUCS_SESSION_LOCK: object = None  # set to threading.Lock() on first use
# Sessions that already ran DemucsOnnxModel._warmup (skip on later wrappers).
_DEMUCS_WARMUP_DONE: set[int] = set()


def _session_lock():
    global _DEMUCS_SESSION_LOCK
    if _DEMUCS_SESSION_LOCK is None:
        import threading

        _DEMUCS_SESSION_LOCK = threading.Lock()
    return _DEMUCS_SESSION_LOCK


def _session_cache_key(path: Path, providers: list) -> tuple:
    names = tuple(p[0] if isinstance(p, tuple) else p for p in providers)
    return (str(path), names)


def _cache_path_candidates(onnx_path: Path) -> list[Path]:
    """Paths that ``ensure_batch_dynamic_onnx`` may use as the session key."""
    src = Path(onnx_path)
    out = [src]
    if not (src.name.endswith(".batch.onnx") or src.stem.endswith(".batch")):
        out.append(src.with_name(f"{src.stem}.batch{src.suffix}"))
    return out


def demucs_session_ready(*, prefer_gpu: bool = True) -> bool:
    """True when the HTDemucs ORT session is already in the process cache.

    Cheap check — does not open a session or rewrite the ONNX graph.
    """
    onnx_path = resolve_htdemucs_onnx()
    if onnx_path is None:
        return False
    providers = _demucs_providers(want_gpu=bool(prefer_gpu))
    for path in _cache_path_candidates(onnx_path):
        key = _session_cache_key(path, providers)
        if _DEMUCS_SESSION_CACHE.get(key) is not None:
            return True
    return False


def warm_demucs_session(*, prefer_gpu: bool = True) -> bool:
    """Deprecated in-process warm — would freeze the GUI (ORT holds the GIL).

    Use ``demucs_sidecar_client.start_demucs_sidecar_warm`` instead. This stub
    only reports whether a session is already cached in *this* process.
    """
    if demucs_session_ready(prefer_gpu=bool(prefer_gpu)):
        return True
    return False


def _open_demucs_session(onnx_path: Path, *, want_gpu: bool):
    import onnxruntime as ort

    if want_gpu and _cuda_available():
        try:
            from ort_util import ensure_nvidia_cuda_dlls

            ensure_nvidia_cuda_dlls()
        except Exception:
            pass
    path = ensure_batch_dynamic_onnx(Path(onnx_path))
    providers = _demucs_providers(want_gpu=want_gpu)
    key = _session_cache_key(path, providers)
    cached = _DEMUCS_SESSION_CACHE.get(key)
    if cached is not None:
        return cached
    with _session_lock():
        cached = _DEMUCS_SESSION_CACHE.get(key)
        if cached is not None:
            return cached
        so = _demucs_session_options(want_gpu=want_gpu and _cuda_available())
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        _DEMUCS_SESSION_CACHE[key] = sess
        return sess


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
        self._runners: dict[int, _OlaSegmentRunner] = {}
        if self.prefer_gpu:
            self._warmup()

    def _warmup(self) -> None:
        """Dummy segments for common packed batch sizes so cuDNN/ORT allocate early."""
        sid = id(self.session)
        if sid in _DEMUCS_WARMUP_DONE:
            return
        try:
            for b in (1, demucs_max_batch()):
                runner = self._runners.get(b)
                if runner is None:
                    runner = _OlaSegmentRunner(self.session, b, use_cuda=self.prefer_gpu)
                    self._runners[b] = runner
                runner.chunk.fill(0)
                runner.infer()
            _DEMUCS_WARMUP_DONE.add(sid)
        except Exception:
            pass

    def eval(self):
        return self

    def to(self, device: str):
        want_gpu = (device or "").strip().lower() not in ("cpu", "")
        active = "CUDAExecutionProvider" in self.session.get_providers()
        if want_gpu != active:
            self.session = _open_demucs_session(self.onnx_path, want_gpu=want_gpu)
            self.prefer_gpu = "CUDAExecutionProvider" in self.session.get_providers()
            self._runners.clear()
            if self.prefer_gpu:
                self._warmup()
        self._device = "cuda" if self.prefer_gpu else "cpu"
        return self

    def cpu(self):
        return self.to("cpu")

    def separate_numpy(self, mix: np.ndarray, *, overlap: float = DEFAULT_OVERLAP) -> np.ndarray:
        """mix (B,2,T) or (2,T) → (B,S,2,T)."""
        arr = np.asarray(mix, dtype=np.float32)
        return separate(self.session, arr, overlap=overlap, runner_cache=self._runners)


def stem_onnx_enabled() -> bool:
    return os.environ.get("STEM_ONNX", "1").strip() != "0"

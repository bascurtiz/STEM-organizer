"""MusicalKeyCNN inference — CQT preprocess + KeyNet forward.

Batch mode: multiprocess CQT (predict3-style) + chunk batches.
Per-file mode: one full-spectrogram forward (live log, sequential).

ONNX path (default, STEM_ONNX): numpy + onnxruntime only — no torch.
Torch fallback: STEM_ONNX=0 with a .pt checkpoint.
"""

from __future__ import annotations

# Pin BLAS/OMP before numpy/librosa import so pool workers don't oversubscribe.
import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Optional

import librosa
import numpy as np

from keys import CAMELOT_TO_SHORT_KEY

try:
    from stem_organizer.log_pace import DEFAULT_LOG_PACE_S, paced as _paced_wrap
except ImportError:
    from log_pace import DEFAULT_LOG_PACE_S, paced as _paced_wrap

N_BINS = 136
SAMPLE_RATE = 44100
HOP_LENGTH = 11025
MIN_DURATION = 8
FMIN = 40
BATCH_SIZE = 320
MODEL_NF = 50
MODEL_P = 0.5
LOG_PACE_S = 0.5

AUDIO_EXTENSIONS = frozenset(
    {
        ".flac",
        ".wav",
        ".mp3",
        ".ogg",
        ".opus",
        ".m4a",
        ".mp4",
        ".aac",
        ".aif",
        ".aiff",
        ".ape",
    }
)

OnFileFn = Callable[[str, Optional[str], Optional[float], Optional[str]], None]
StopFn = Callable[[], bool]


def resolve_device(preferred: str = "") -> str:
    """Pick cuda|cpu. ONNX path uses ORT CUDA EP (no torch required)."""
    pref = (preferred or "").strip().lower()
    onnx_mode = os.environ.get("STEM_ONNX", "1").strip() != "0"

    def _ort_cuda() -> bool:
        try:
            from ort_util import cuda_ep_usable

            return bool(cuda_ep_usable())
        except Exception:
            return False

    def _torch_cuda() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if onnx_mode:
            return "cuda" if _ort_cuda() else "cpu"
        return "cuda" if _torch_cuda() else "cpu"
    if onnx_mode:
        return "cuda" if _ort_cuda() else "cpu"
    return "cuda" if _torch_cuda() else "cpu"


def load_model(checkpoint: Path | str, device: str) -> Any:
    """Load KeyNet — ONNX by default (STEM_ONNX env), torch fallback.

    ONNX returns logits as numpy ``(B, 24)``. Torch returns a module whose
    forward yields torch tensors (converted to numpy in process_*).
    """
    checkpoint = Path(checkpoint)
    if os.environ.get("STEM_ONNX", "1").strip() != "0":
        onnx_path = checkpoint.with_suffix(".onnx")
        if not onnx_path.is_file() and checkpoint.suffix.lower() == ".onnx":
            onnx_path = checkpoint
        if onnx_path.is_file():
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                pass
            else:
                return KeyNetOnnx(onnx_path, device)

    import torch
    from model import KeyNet

    model = KeyNet(num_classes=24, in_channels=1, Nf=MODEL_NF, p=MODEL_P).to(device)
    try:
        state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


class KeyNetOnnx:
    """ONNX Runtime backend — numpy in, numpy logits out (torch-free)."""

    def __init__(self, onnx_path: Path, device: str = ""):
        from ort_util import create_ort_session

        self.session = create_ort_session(onnx_path, device=device or "")
        providers = list(self.session.get_providers())
        self.device = (
            "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        )
        self.backend = "onnx"

    def __call__(self, x: np.ndarray | Any) -> np.ndarray:
        if hasattr(x, "detach"):
            np_in = x.detach().cpu().numpy()
        else:
            np_in = np.asarray(x)
        np_in = np.ascontiguousarray(np_in, dtype=np.float32)
        return self.session.run(["logits"], {"cqt": np_in})[0]

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


def _load_mono(path_s: str) -> tuple[np.ndarray, int]:
    """Decode via soundfile + soxr; fall back to librosa for exotic codecs."""
    try:
        import soundfile as sf

        data, sr = sf.read(path_s, always_2d=True, dtype="float32")
        mono = data.mean(axis=1)
        if int(sr) != SAMPLE_RATE:
            mono = librosa.resample(
                mono,
                orig_sr=int(sr),
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
        return np.ascontiguousarray(mono, dtype=np.float32), SAMPLE_RATE
    except Exception:
        waveform, sr = librosa.load(path_s, sr=SAMPLE_RATE, mono=True)
        return np.asarray(waveform, dtype=np.float32), int(sr)


def preproc(path: Path | str) -> tuple[str, Optional[np.ndarray], Optional[str]]:
    """Return (path, cqt_or_None, error_or_None)."""
    path_s = str(path)
    try:
        waveform, sr = _load_mono(path_s)
    except Exception as exc:
        return path_s, None, f"decode failed: {exc}"
    if waveform is None or len(waveform) == 0:
        return path_s, None, "decode failed: empty"
    if len(waveform) / float(sr) < MIN_DURATION:
        return path_s, None, "too short (< 8 s)"
    try:
        cqt = librosa.cqt(
            waveform.astype(np.float32, copy=False),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=24,
            fmin=FMIN,
        )
    except Exception as exc:
        return path_s, None, f"CQT failed: {exc}"
    data = np.log1p(np.abs(cqt)).astype(np.float32)
    return path_s, data, None


def _preproc_worker(path_s: str) -> tuple[str, Optional[np.ndarray], Optional[str]]:
    """Pool entry — top-level for Windows spawn pickling."""
    return preproc(path_s)


def _pool_worker_init() -> None:
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[k] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def default_cqt_workers() -> int:
    """Parallel CQT workers. ``KEY_CQT_WORKERS`` overrides (from io_tune)."""
    raw = os.environ.get("KEY_CQT_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpus = os.cpu_count() or 4
    return max(2, min(8, cpus // 2))


def _chunk_frames() -> int:
    return (MIN_DURATION * SAMPLE_RATE) // HOP_LENGTH


def logits_to_key(logits: Any) -> tuple[str, float]:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    arr = arr - arr.max()
    e = np.exp(arr)
    probs = e / e.sum()
    pred = int(probs.argmax())
    conf = float(probs[pred])
    return CAMELOT_TO_SHORT_KEY.get(pred, "Unknown"), conf


def _as_numpy_logits(out: Any) -> np.ndarray:
    if hasattr(out, "detach"):
        return out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


def collect_audio_files(folder: Path, *, recursive: bool = True) -> list[Path]:
    out: list[Path] = []
    if recursive:
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            if any(part.lower() == "_backup_before_align" for part in p.parts):
                continue
            out.append(p)
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                out.append(p)
    return out


def process_batched(
    model: Any,
    paths: list[Path],
    device: str,
    *,
    on_file: Optional[OnFileFn] = None,
    stop_check: Optional[StopFn] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
    workers: Optional[int] = None,
    log_pace_s: float = LOG_PACE_S,
) -> dict[str, tuple[str, float]]:
    """Chunk-averaged inference; parallel CQT + batched forwards (numpy/ONNX)."""
    chunk = _chunk_frames()
    accum: dict[str, list[np.ndarray]] = {}
    expected_chunks: dict[str, int] = {}
    emitted: set[str] = set()
    results: dict[str, tuple[str, float]] = {}
    batch_paths: list[str] = []
    batch_chunks: list[np.ndarray] = []
    total = max(1, len(paths))
    n_workers = default_cqt_workers() if workers is None else max(1, int(workers))
    use_torch = getattr(model, "backend", None) != "onnx"

    paced = _paced_wrap(on_file, log_pace_s, name="key-log-pace")
    emit: Optional[OnFileFn] = paced if paced is not None else on_file

    def maybe_emit(pth: str) -> None:
        if pth in emitted:
            return
        parts = accum.get(pth)
        if not parts or len(parts) < expected_chunks.get(pth, 0):
            return
        mean = np.mean(np.stack(parts, axis=0), axis=0)
        key, conf = logits_to_key(mean)
        results[pth] = (key, conf)
        emitted.add(pth)
        if emit:
            emit(pth, key, conf, None)

    def flush() -> None:
        nonlocal batch_paths, batch_chunks
        if not batch_chunks:
            return
        batch = np.stack(batch_chunks, axis=0)  # (B, 1, 136, T)
        if use_torch:
            import torch

            tensor = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                out = _as_numpy_logits(model(tensor))
        else:
            out = _as_numpy_logits(model(batch))
        for i, pth in enumerate(batch_paths):
            accum.setdefault(pth, []).append(np.asarray(out[i], dtype=np.float32))
            maybe_emit(pth)
        batch_paths = []
        batch_chunks = []

    def handle_preproc(
        path_s: str, data: Optional[np.ndarray], err: Optional[str]
    ) -> None:
        if err or data is None:
            if emit:
                emit(path_s, None, None, err or "decode failed")
            return
        n_chunks = int(data.shape[1]) // chunk
        if n_chunks <= 0:
            if emit:
                emit(path_s, None, None, "too short (< 8 s)")
            return
        expected_chunks[path_s] = n_chunks
        for i in range(n_chunks):
            sl = data[:, i * chunk : (i + 1) * chunk]
            batch_paths.append(path_s)
            batch_chunks.append(sl[np.newaxis, ...].astype(np.float32, copy=False))
            if len(batch_chunks) >= BATCH_SIZE:
                flush()

    path_strs = [str(p) for p in paths]
    try:
        if len(path_strs) <= 1 or n_workers <= 1:
            for idx, path_s in enumerate(path_strs, start=1):
                if stop_check and stop_check():
                    break
                if on_progress:
                    on_progress(idx, total)
                handle_preproc(*preproc(path_s))
            flush()
            for pth in list(expected_chunks):
                maybe_emit(pth)
            return results

        # ProcessPool: librosa CQT is GIL-bound — threads don't scale.
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_pool_worker_init,
        )
        try:
            done = 0
            for path_s, data, err in pool.imap_unordered(
                _preproc_worker, path_strs, chunksize=4
            ):
                if stop_check and stop_check():
                    break
                done += 1
                if on_progress:
                    on_progress(done, total)
                handle_preproc(path_s, data, err)
            flush()
            for pth in list(expected_chunks):
                maybe_emit(pth)
        finally:
            pool.terminate()
            pool.join()
        return results
    finally:
        if paced is not None:
            paced.close()


def process_per_file(
    model: Any,
    paths: list[Path],
    device: str,
    *,
    on_file: Optional[OnFileFn] = None,
    stop_check: Optional[StopFn] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, tuple[str, float]]:
    """One full-spectrogram forward per file (live log)."""
    results: dict[str, tuple[str, float]] = {}
    total = max(1, len(paths))
    use_torch = getattr(model, "backend", None) != "onnx"
    for idx, path in enumerate(paths, start=1):
        if stop_check and stop_check():
            break
        if on_progress:
            on_progress(idx, total)
        path_s, data, err = preproc(path)
        if err or data is None:
            if on_file:
                on_file(path_s, None, None, err or "decode failed")
            continue
        x = data[np.newaxis, np.newaxis, ...].astype(np.float32, copy=False)
        if use_torch:
            import torch

            t = torch.from_numpy(x).to(device)
            with torch.no_grad():
                out = _as_numpy_logits(model(t)[0])
        else:
            out = _as_numpy_logits(model(x)[0])
        key, conf = logits_to_key(out)
        results[path_s] = (key, conf)
        if on_file:
            on_file(path_s, key, conf, None)
    return results

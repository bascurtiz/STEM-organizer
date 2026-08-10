from __future__ import annotations

import gc
import json
import logging
import shutil
import ssl
import time
import os
import sys
import urllib.request
import warnings

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Harmless HF Hub noise on Windows VMs (no symlink support / no token).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")


APP_NAME = "Genre / Gender Tagger"
APP_VERSION = "1.0"

# Classify-style startup/config indent; === file / Summary headers stay flush-left.
LOG_INDENT = "  "

# Avoid Windows cp1252 crashes on non-ASCII filenames in print/tqdm.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _fmt_summary_elapsed(seconds: float) -> str:
    """SI-SDR-style elapsed for unified summaries (m:ss or h:mm:ss)."""
    total = max(0, int(round(float(seconds or 0))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def print_feature_summary(
    feature: str,
    *,
    elapsed: float,
    files: int,
    tagged: int | None = None,
    skipped: int | None = None,
    peak_vram_gb: float | None = None,
    results_path=None,
    extra_lines: list | None = None,
) -> None:
    """Unified LOG footer: === Feature Summary === … DONE."""
    n = max(0, int(files or 0))
    minutes = max(float(elapsed or 0) / 60.0, 1e-9)
    print(flush=True)
    print(f"=== {feature} Summary ===", flush=True)
    print(f"  Total time: {_fmt_summary_elapsed(elapsed)}", flush=True)
    print(f"  Files: {n:,}", flush=True)
    if n > 0:
        print(f"  Sec/file: {float(elapsed) / n:.3f}", flush=True)
        print(f"  Files/min: {n / minutes:.2f}", flush=True)
    if tagged is not None:
        if skipped is not None:
            print(f"  Tagged: {tagged:,} | Skipped: {skipped:,}", flush=True)
        else:
            print(f"  Tagged: {tagged:,}", flush=True)
    if peak_vram_gb is not None:
        print(f"  Peak VRAM: {float(peak_vram_gb):.2f} GB", flush=True)
    if extra_lines:
        for line in extra_lines:
            text = str(line).strip()
            if text:
                print(f"  {text}", flush=True)
    if results_path:
        print(f"  Results: {results_path}", flush=True)
    print(flush=True)
    print("DONE", flush=True)


def _status(msg):
    """Immediate console feedback during slow startup imports.

    Indents like Classify startup/config lines. Section headers (=== … ===)
    and empty lines stay flush-left.
    """
    s = str(msg)
    if not s.strip():
        print(flush=True)
        return
    stripped = s.lstrip()
    if stripped.startswith("===") or s.startswith("__"):
        print(s, flush=True)
        return
    print(LOG_INDENT + s, flush=True)


def _gender_model_status(msg) -> None:
    """Drop routine model-path / EP chatter from Gender LOG (keep downloads)."""
    low = str(msg or "").strip().lower()
    if low.startswith("using bundled"):
        return
    if low.startswith("onnx runtime providers:"):
        return
    if "for discogs-effnet" in low:
        return
    if low.startswith("directml unavailable"):
        return
    _status(msg)


def _log_intro(msg: str) -> None:
    """Startup/config line (Classify indent). Prefer over bare print()."""
    print(f"{LOG_INDENT}{msg}", flush=True)

def _write_results_csv(path, rows) -> None:
    """Write list[dict] to CSV via stdlib (no pandas)."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    # Stable column order: keys from first row, then any extras.
    fieldnames: list[str] = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def emit_gg_processed(n, total, *, force: bool = False) -> None:
    """One-line batch progress for STEM LOG: __gg_processed__ n total."""
    global _last_gg_processed_emit, _last_gg_processed_n
    total_i = int(total or 0)
    if total_i <= 0:
        return
    n_i = int(max(0, min(total_i, int(n or 0))))
    now = time.monotonic()
    if (
        not force
        and n_i != total_i
        and n_i == _last_gg_processed_n
    ):
        return
    if (
        not force
        and n_i != total_i
        and (now - _last_gg_processed_emit) < _STEM_PROGRESS_INTERVAL
    ):
        return
    _last_gg_processed_emit = now
    _last_gg_processed_n = n_i
    print(f"__gg_processed__\t{n_i}\t{total_i}", flush=True)

def emit_stem_progress(
    n,
    total,
    phase="",
    *,
    pct=None,
    eta="",
    force=False,
    display_n=None,
    display_total=None,
):
    """Machine progress for STEM host: __progress__ pct eta n total phase.

    ``n`` may be fractional for smooth bars; ``display_n`` / ``display_total``
    control the human count shown in the status bar (defaults from n/total).
    """
    global _last_stem_progress_emit
    total_i = int(total or 0)
    if total_i <= 0:
        return
    n_f = float(n or 0)
    show_total = int(display_total if display_total is not None else total_i)
    if display_n is not None:
        n_i = int(max(0, min(show_total, int(display_n))))
    else:
        n_i = int(max(0, min(show_total, round(n_f))))
    now = time.monotonic()
    if (
        not force
        and n_f < float(total_i)
        and (now - _last_stem_progress_emit) < _STEM_PROGRESS_INTERVAL
    ):
        return
    _last_stem_progress_emit = now
    if pct is None:
        pct = 100.0 * min(n_f, float(total_i)) / float(total_i)
    eta_s = "" if eta is None else str(eta)
    phase_s = (phase or "").replace("\t", " ").strip()
    print(
        f"__progress__\t{float(pct):.2f}\t{eta_s}\t{n_i}\t{show_total}\t{phase_s}",
        flush=True,
    )

def _fmt_confidence_pct(score) -> str:
    """Fraction 0–1 → whole percent for LOG, e.g. 0.723 → '72%'."""
    try:
        return f"{int(round(float(score) * 100.0))}%"
    except (TypeError, ValueError):
        return "0%"

def _log_gender_result(row, *, index: int | None = None, total: int | None = None):
    """Classify-style LOG: === [i/n] file === + badge + dim pct (e.g. female 72%).

    Blank line after the block only when 2+ badge chips (e.g. gender + reverb).
    """
    name = Path(row.get("file") or "").name
    gender = str(row.get("gender") or "").strip().lower()
    reverb = str(row.get("reverb") or "").strip().lower()
    gconf = _fmt_confidence_pct(row.get("confidence", 0))
    rconf = _fmt_confidence_pct(row.get("reverb_confidence", 0))
    if index is not None and total is not None and total > 0:
        print(f"=== [{int(index)}/{int(total)}] {name} ===", flush=True)
    else:
        print(f"=== {name} ===", flush=True)
    n_badges = 0
    if gender in ("female", "male"):
        print(f"  {gender} {gconf}", flush=True)
        n_badges += 1
    elif gender:
        print(f"GENDER: {gender}", flush=True)
        print(f"  {gconf}", flush=True)
        n_badges += 1
    if reverb in ("dry", "wet"):
        print(f"  {reverb} {rconf}", flush=True)
        n_badges += 1
    elif reverb and reverb != "?":
        print(f"REVERB: {reverb}", flush=True)
        print(f"  {rconf}", flush=True)
        n_badges += 1
    if n_badges >= 2:
        print(flush=True)

def _log_genre_result(
    path, genre, style, conf, *, index: int | None = None, total: int | None = None
):
    """LOG block: === [i/n] file ===, GENRE / STYLE, then dim pct."""
    name = Path(path).name
    if index is not None and total is not None and total > 0:
        print(f"=== [{int(index)}/{int(total)}] {name} ===", flush=True)
    else:
        print(f"=== {name} ===", flush=True)
    print("GENRE:", genre, flush=True)
    print("STYLE:", style or "", flush=True)
    print(f"  {_fmt_confidence_pct(conf)}", flush=True)

def _paths_from_files_list(path: str) -> list[str]:
    """One audio path per line (STEM organizer filtered tag list)."""
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.lstrip("\ufeff").strip().strip('"')
        if not line:
            continue
        candidate = Path(line)
        if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
            out.append(str(candidate.resolve(strict=False)))
    return out

def iter_audio_files(folder):
    """Yield audio files under folder (recursive when INCLUDE_SUBFOLDERS)."""
    files_from = os.environ.get("GG_FILES_FROM", "").strip()
    if files_from:
        for path_str in _paths_from_files_list(files_from):
            yield Path(path_str)
        return
    root = Path(folder)
    it = root.rglob("*") if INCLUDE_SUBFOLDERS else root.iterdir()
    for path in it:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path

def list_audio_files(folder):
    """Collect audio paths with periodic scan feedback for large libraries."""
    _log_intro("Scanning for audio files...")
    files = []
    for i, path in enumerate(iter_audio_files(folder), 1):
        files.append(str(path))
        if i % 5000 == 0:
            print(f"{LOG_INDENT}  …found {i:,} audio files", flush=True)
    return files

def _tagger_package_dir() -> Path:
    return Path(__file__).resolve().parent

def _gender_model_dir_candidates() -> list[Path]:
    """Single model source: the root models/ folder (repo root / app root)."""
    script_dir = _tagger_package_dir()
    parent = script_dir.parent
    # Frozen: taggers may sit under _internal/ — root models/ is beside the exe.
    if parent.name == "_internal":
        parent = parent.parent
    return [parent / "models"]

def _writable_gender_model_dir() -> Path:
    """Prefer a non-_internal folder so downloads work in frozen builds."""
    for path in _gender_model_dir_candidates():
        if "_internal" not in path.parts:
            return path
    return _gender_model_dir_candidates()[0]

def _probe_ort_cuda() -> tuple[bool, str]:
    """Return (ok, detail) for ORT CUDA EP + real NVIDIA GPU."""
    if os.environ.get("STEM_ORT_CUDA", "1").strip() == "0":
        return False, "STEM_ORT_CUDA=0"
    try:
        import onnxruntime as ort
        from ort_util import cuda_ep_usable, nvidia_gpu_present

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False, "CUDA EP not in this onnxruntime build"
        if not nvidia_gpu_present():
            return False, "no NVIDIA GPU detected"
        if not cuda_ep_usable():
            return False, "CUDA EP not usable"
        return True, "CUDAExecutionProvider"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

# ==========================================================
# VOICE-GENDER HELPERS (gender-discogs-effnet via ONNX Runtime)
# ==========================================================

def _model_file_ready(path, min_bytes: int = 1000) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= int(min_bytes)
    except OSError:
        return False

def _find_ready_model(name: str, min_bytes: int, model_dir=None) -> Path | None:
    """Return an existing complete model path, or None."""
    search = (
        [Path(model_dir)]
        if model_dir is not None
        else list(_gender_model_dir_candidates())
    )
    if model_dir is None:
        search.append(_INSTRUMENT_MODELS)
    for folder in search:
        path = Path(folder) / name
        if _model_file_ready(path, min_bytes):
            return path
    return None

def _ssl_download_context():
    """Prefer certifi CA bundle (fixes Win11/VM CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _download_model_file(path, url, status=print, min_bytes: int = 1000):
    """Download to a .part file, verify size, then replace the destination."""
    status(f"  downloading {path.name} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        ctx = _ssl_download_context()
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp, open(
            tmp, "wb"
        ) as out:
            shutil.copyfileobj(resp, out)
        size = tmp.stat().st_size
        if size < int(min_bytes):
            try:
                tmp.unlink()
            except OSError:
                pass
            raise SystemExit(
                f"\nERROR: download incomplete for {path.name}\n"
                f"  got {size:,} bytes (need at least {int(min_bytes):,})\n"
                f"  url: {url}\n\n"
                f"Retry Tag gender, or place the full file in:\n  {path.parent}\n"
            )
        tmp.replace(path)
        status(f"  downloaded {path.name} ({size:,} bytes)")
    except SystemExit:
        raise
    except Exception as exc:
        for victim in (tmp, path):
            if victim.exists():
                try:
                    victim.unlink()
                except OSError:
                    pass
        raise SystemExit(
            f"\nERROR: could not download {path.name}\n"
            f"  reason: {exc}\n"
            f"  url:    {url}\n\n"
            f"Offline fix: place the file in:\n  {path.parent}\n"
        ) from exc

def ensure_gender_onnx_models(model_dir=None, status=print):
    """Download EffNet + gender .onnx files if missing or truncated."""

    download_dir = Path(model_dir or _writable_gender_model_dir())
    download_dir.mkdir(parents=True, exist_ok=True)

    effnet = _find_ready_model(
        GENDER_EFFNET_ONNX_NAME,
        GENDER_EFFNET_ONNX_MIN_BYTES,
        model_dir=model_dir,
    )
    if effnet is not None:
        if model_dir is None and effnet.parent != download_dir:
            status(f"  using bundled {effnet}")
    else:
        effnet = download_dir / GENDER_EFFNET_ONNX_NAME
        if effnet.exists():
            status(
                f"  replacing incomplete {effnet.name} "
                f"({effnet.stat().st_size:,} bytes)"
            )
            try:
                effnet.unlink()
            except OSError:
                pass
        _download_model_file(
            effnet,
            GENDER_EFFNET_ONNX_URL,
            status=status,
            min_bytes=GENDER_EFFNET_ONNX_MIN_BYTES,
        )

    gender = _find_ready_model(
        GENDER_HEAD_ONNX_NAME,
        GENDER_HEAD_ONNX_MIN_BYTES,
        model_dir=model_dir,
    )
    if gender is not None:
        if model_dir is None and gender.parent != download_dir:
            status(f"  using bundled {gender}")
    else:
        gender = download_dir / GENDER_HEAD_ONNX_NAME
        if gender.exists():
            status(
                f"  replacing incomplete {gender.name} "
                f"({gender.stat().st_size:,} bytes)"
            )
            try:
                gender.unlink()
            except OSError:
                pass
        _download_model_file(
            gender,
            GENDER_HEAD_ONNX_URL,
            status=status,
            min_bytes=GENDER_HEAD_ONNX_MIN_BYTES,
        )

    return effnet, gender

class _GenderOrtBackend:
    def __init__(self, effnet_sess, head_sess, provider):
        self.effnet = effnet_sess
        self.head = head_sess
        self.name = f"onnxruntime:{provider}"
        self._mel_in = effnet_sess.get_inputs()[0].name
        out_names = [o.name for o in effnet_sess.get_outputs()]
        self._emb_out = "embeddings" if "embeddings" in out_names else out_names[-1]
        self._head_in = head_sess.get_inputs()[0].name
        self._head_out = head_sess.get_outputs()[0].name

    def predict_batch(self, chunk):
        """chunk [N,128,96] -> probs [N,2] (dynamic batch, no pad required)."""
        emb = self.effnet.run([self._emb_out], {self._mel_in: chunk})[0]
        return self.head.run([self._head_out], {self._head_in: emb})[0]

def _ort_providers():
    """Prefer CUDA EP when available — do not require torch.cuda (frozen ONNX)."""
    try:
        from ort_util import ort_providers as _shared

        return _shared("")
    except Exception:
        pass
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        try:
            from ort_util import nvidia_gpu_present

            if nvidia_gpu_present():
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _quiet_ort_session(model_path, sess_options, providers):
    """Create InferenceSession; mute C-level EP Error spam if a GPU EP fails."""
    import onnxruntime as ort

    # ORT writes EP fallback diagnostics to fd 2 (not always Python sys.stderr).
    try:
        devnull = open(os.devnull, "w")
        old_fd = os.dup(2)
    except Exception:
        return ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
    try:
        os.dup2(devnull.fileno(), 2)
        return ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )
    finally:
        try:
            os.dup2(old_fd, 2)
            os.close(old_fd)
        except Exception:
            pass
        try:
            devnull.close()
        except Exception:
            pass

def load_gender_ort_backend(model_dir=None, status=print):
    import onnxruntime as ort

    effnet_path, head_path = ensure_gender_onnx_models(model_dir, status=status)
    providers = _ort_providers()
    status(f"  ONNX Runtime providers: {', '.join(providers)}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if providers[0] == "CPUExecutionProvider":
            cpu_n = max(1, os.cpu_count() or 4)
            env_intra = os.environ.get("STEM_ORT_INTRA_OP", "").strip()
            so.intra_op_num_threads = (
                max(1, int(env_intra)) if env_intra else max(1, cpu_n // 2)
            )
    except Exception:
        pass
    # Quiet create when a GPU EP is first — ORT may still probe-and-fallback.
    if providers[0] != "CPUExecutionProvider":
        effnet = _quiet_ort_session(effnet_path, so, providers)
        head = _quiet_ort_session(head_path, so, providers)
    else:
        effnet = ort.InferenceSession(
            str(effnet_path), sess_options=so, providers=providers
        )
        head = ort.InferenceSession(
            str(head_path), sess_options=so, providers=providers
        )
    active = effnet.get_providers()[0]
    if (
        providers[0] != "CPUExecutionProvider"
        and active == "CPUExecutionProvider"
    ):
        status(f"  GPU EP unavailable — fell back to CPU ({ORT_CUDA_DETAIL})")
    else:
        status(f"  using {active} for discogs-effnet + gender head")
    return _GenderOrtBackend(effnet, head, active)

def load_gender_models(model_dir=None, status=print):
    """
    Load the ONNX Runtime gender backend (DirectML/CUDA/CPU).

    Returns a backend with .predict_batch(chunk) -> probs [N,2] and .name.
    """

    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "\nERROR: ONNX Runtime is not installed.\n"
            "  Gender tagging needs onnxruntime-directml (Windows) or "
            "onnxruntime.\n"
            f"  Reason: {exc}\n\n"
            "Fix: re-run install-deps.bat, then retry Tag gender.\n"
        ) from exc

    return load_gender_ort_backend(model_dir, status=status)

def load_mono_16k(filename):
    """Decode to mono float32 @ 16 kHz (Essentia MonoLoader target)."""
    import librosa

    try:
        data, sr = sf.read(filename, always_2d=True, dtype="float32")
        audio = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
    except Exception:
        # sf can't decode some codecs (m4a/aac/opus) — fall back to librosa,
        # which uses audioread/ffmpeg (bundled app ffmpeg is on PATH).
        audio, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)

    return audio.astype(np.float32, copy=False)

def _hz2mel_slaney(hz):
    """Essentia hz2melSlaney (Auditory Toolbox)."""

    hz = np.asarray(hz, dtype=np.float64)
    min_log_hz = 1000.0
    lin_slope = 3.0 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0

    mel = np.empty_like(hz)
    linear = hz < min_log_hz
    mel[linear] = hz[linear] * lin_slope
    mel[~linear] = min_log_mel + np.log(hz[~linear] / min_log_hz) / log_step
    return mel

def _mel2hz_slaney(mel):
    """Essentia mel2hzSlaney."""

    mel = np.asarray(mel, dtype=np.float64)
    min_log_hz = 1000.0
    lin_slope = 3.0 / 200.0
    min_log_mel = min_log_hz * lin_slope
    log_step = np.log(6.4) / 27.0

    hz = np.empty_like(mel)
    linear = mel < min_log_mel
    hz[linear] = mel[linear] / lin_slope
    hz[~linear] = min_log_hz * np.exp((mel[~linear] - min_log_mel) * log_step)
    return hz

def _essentia_hann(size):
    """Essentia Windowing hann, symmetric=True, normalized=False."""

    i = np.arange(size, dtype=np.float64)
    return (0.5 - 0.5 * np.cos((2.0 * np.pi * i) / (size - 1.0))).astype(
        np.float32
    )

def _frame_cutter_essentia(signal, frame_size=None, hop_size=None):
    """
    Essentia FrameCutter: startFromZero=False, validFrameThresholdRatio=0.
    Returns [n_frames, frame_size].
    """

    if frame_size is None:
        frame_size = GENDER_FRAME_SIZE
    if hop_size is None:
        hop_size = GENDER_HOP_SIZE

    signal = np.asarray(signal, dtype=np.float32)
    n = int(signal.shape[0])
    start0 = -((frame_size + 1) // 2)

    starts = []
    start = start0
    while True:
        starts.append(start)
        if start + frame_size // 2 >= n:
            break
        start += hop_size

    starts = np.asarray(starts, dtype=np.int64)
    left = max(0, -int(starts[0]))
    right = max(0, int(starts[-1] + frame_size - n))
    padded = np.pad(signal, (left, right), mode="constant")
    starts_p = starts + left
    idx = starts_p[:, None] + np.arange(frame_size, dtype=np.int64)[None, :]
    return padded[idx]

def _build_essentia_mel_filterbank(
    spectrum_size=None,
    n_mels=None,
    sample_rate=None,
):
    """
    Essentia MelBands filterbank:
      warpingFormula=slaneyMel, weighting=linear, normalize=unit_tri.
    Returns [n_mels, spectrum_size].
    """

    if spectrum_size is None:
        spectrum_size = GENDER_FRAME_SIZE // 2 + 1
    if n_mels is None:
        n_mels = GENDER_N_MELS
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    low_hz = 0.0
    high_hz = sample_rate / 2.0

    low_mel = float(_hz2mel_slaney(np.array([low_hz]))[0])
    high_mel = float(_hz2mel_slaney(np.array([high_hz]))[0])
    mel_step = (high_mel - low_mel) / (n_mels + 1)

    mel_points = low_mel + mel_step * np.arange(n_mels + 2, dtype=np.float64)
    band_hz = _mel2hz_slaney(mel_points)

    frequency_scale = (sample_rate / 2.0) / (spectrum_size - 1)
    filters = np.zeros((n_mels, spectrum_size), dtype=np.float64)

    for i in range(n_mels):
        f_left, f_center, f_right = band_hz[i], band_hz[i + 1], band_hz[i + 2]
        fstep1 = f_center - f_left
        fstep2 = f_right - f_center

        jbegin = int(np.ceil(f_left / frequency_scale))
        jend = int(np.floor(f_right / frequency_scale))
        jend = min(jend, spectrum_size - 1)

        for j in range(jbegin, jend + 1):
            binfreq = j * frequency_scale
            if binfreq < f_center:
                coeff = (binfreq - f_left) / fstep1 if fstep1 else 0.0
            else:
                coeff = (f_right - binfreq) / fstep2 if fstep2 else 0.0
            filters[i, j] = max(coeff, 0.0)

        area = (fstep1 + fstep2) / 2.0
        if area > 0:
            filters[i] /= area

    return filters.astype(np.float32)

def _get_mel_filterbank():
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = _build_essentia_mel_filterbank()
    return _MEL_FILTERBANK

def musicnn_logmel(audio):
    """Essentia TensorflowInputMusiCNN equivalent. Returns [n_frames, 96]."""

    frames = _frame_cutter_essentia(audio)
    window = _essentia_hann(GENDER_FRAME_SIZE)
    mel_fb = _get_mel_filterbank()

    n = GENDER_FRAME_SIZE
    half = n // 2
    windowed = frames * window[np.newaxis, :]
    zp = np.empty_like(windowed)
    zp[:, : n - half] = windowed[:, half:]
    zp[:, n - half :] = windowed[:, :half]

    magnitude = np.abs(np.fft.rfft(zp, n=GENDER_FRAME_SIZE, axis=1)).astype(
        np.float32
    )
    power = magnitude * magnitude
    mel = power @ mel_fb.T

    return np.log10(1.0 + mel * 10000.0).astype(np.float32)

def mel_patches(mel):
    """
    Cut [n_frames, 96] into patches of 128 frames, hop 62.
    lastPatchMode=discard. Returns [n_patches, 128, 96].
    """

    n_frames = mel.shape[0]

    if n_frames < GENDER_PATCH_SIZE:
        pad = np.zeros(
            (GENDER_PATCH_SIZE - n_frames, GENDER_N_MELS),
            dtype=np.float32,
        )
        mel = np.concatenate([mel, pad], axis=0)
        return mel[np.newaxis, ...]

    patches = []
    start = 0
    while start + GENDER_PATCH_SIZE <= n_frames:
        patches.append(mel[start : start + GENDER_PATCH_SIZE])
        start += GENDER_PATCH_HOP

    if not patches:
        pad = np.zeros(
            (GENDER_PATCH_SIZE - n_frames, GENDER_N_MELS),
            dtype=np.float32,
        )
        mel = np.concatenate([mel, pad], axis=0)
        return mel[np.newaxis, ...]

    return np.stack(patches, axis=0).astype(np.float32)

def extract_patches(filename):
    """Decode + Mel + patch cut. Returns [n_patches, 128, 96]."""

    audio = load_mono_16k(filename)
    mel = musicnn_logmel(audio)
    return mel_patches(mel)

def predict_patches(patches, backend):
    """Run EffNet + gender head on patches [n, 128, 96] -> probs [n, 2]."""

    batch_size = (
        GENDER_ORT_BATCH
        if getattr(backend, "name", "").startswith("onnxruntime")
        else GENDER_BATCH_SIZE
    )
    n_patches = patches.shape[0]
    probs_all = []

    for batch_start in range(0, n_patches, batch_size):
        chunk = patches[batch_start : batch_start + batch_size]
        probs_all.append(backend.predict_batch(chunk))

    return np.concatenate(probs_all, axis=0)

def predict_fixed_batch(chunk64, backend):
    """One forward pass on a stacked patch batch. backend is a gender backend."""

    return backend.predict_batch(np.asarray(chunk64, dtype=np.float32))

def probs_to_result(probs):
    """Average patch probs -> gender label + confidence."""

    mean_prob = probs.mean(axis=0)
    best = int(mean_prob.argmax())

    return {
        "gender": GENDER_LABELS[best],
        "confidence": float(mean_prob[best]),
        "female": float(mean_prob[0]),
        "male": float(mean_prob[1]),
        "n_patches": int(probs.shape[0]),
    }

def calibrate_multiclass_confidence(top1, top2=None):
    """Map ~519-way top-1 softmax to a binary-like score.

    Renormalize top-1 vs top-2: p1/(p1+p2). Same reading as gender/reverb:
    ~0.5 = barely beats runner-up, higher = clearer win. Raw top-1 stays in top5.
    """
    t1 = float(top1)
    if top2 is None:
        return t1
    t2 = float(top2)
    denom = t1 + t2
    if denom <= 0.0:
        return 0.0
    return t1 / denom

def classify_gender_file(filename, backend):
    """Run gender-discogs-effnet on one file."""

    patches = extract_patches(filename)
    probs = predict_patches(patches, backend)
    return probs_to_result(probs)

def _id3_set_text(tags: ID3, frame_id: str, value: str | None, *, txxx_desc: str | None = None) -> None:
    if txxx_desc is not None:
        key = f"TXXX:{txxx_desc}"
        tags.delall(key)
        if value:
            tags.add(TXXX(encoding=3, desc=txxx_desc, text=[value]))
        return

    tags.delall(frame_id)
    if not value:
        return
    if frame_id == "TCON":
        tags.add(TCON(encoding=3, text=[value]))
    elif frame_id == "COMM":
        tags.add(COMM(encoding=3, lang="eng", desc="", text=[value]))

def _apply_id3_updates(tags: ID3, updates: dict) -> None:
    if "genre" in updates:
        _id3_set_text(tags, "TCON", updates["genre"])
    if "style" in updates:
        _id3_set_text(tags, "TXXX", updates["style"], txxx_desc="STYLE")
    if "comment" in updates:
        _id3_set_text(tags, "COMM", updates["comment"])
    if "gender" in updates:
        _id3_set_text(tags, "TXXX", updates["gender"], txxx_desc="GENDER")
    if "reverb" in updates:
        _id3_set_text(tags, "TXXX", updates["reverb"], txxx_desc="REVERB")

def _apply_mp4_updates(audio: MP4, updates: dict) -> None:
    for field, value in updates.items():
        if field in _MP4_STD:
            key = _MP4_STD[field]
            if value:
                audio[key] = [value]
            else:
                audio.pop(key, None)
        elif field in _MP4_FREEFORM:
            key = _MP4_FREEFORM[field]
            if value:
                audio[key] = [MP4FreeForm(value.encode("utf-8"))]
            else:
                audio.pop(key, None)

def apply_audio_tags(filename, updates: dict) -> bool:
    """Apply logical tag updates. None clears a field. Returns True on success."""

    from file_writable import ensure_writable

    ext = Path(filename).suffix.lower()
    ensure_writable(filename)

    try:
        if ext == ".flac":
            audio = FLAC(filename)
            for field, value in updates.items():
                if value:
                    audio[field] = value
                else:
                    audio.pop(field, None)
            audio.save()
            return True

        if ext == ".mp3":
            audio = MP3(filename)
            if audio.tags is None:
                audio.add_tags()
            _apply_id3_updates(audio.tags, updates)
            audio.save()
            return True

        if ext == ".wav":
            audio = WAVE(filename)
            if audio.tags is None:
                audio.add_tags()
            _apply_id3_updates(audio.tags, updates)
            audio.save()
            return True

        if ext in (".ogg", ".opus"):
            audio = (OggOpus(filename) if ext == ".opus" else OggVorbis(filename))
            for field, value in updates.items():
                if value:
                    audio[field] = value
                else:
                    audio.pop(field, None)
            audio.save()
            return True

        if ext in (".m4a", ".mp4"):
            audio = MP4(filename)
            _apply_mp4_updates(audio, updates)
            audio.save()
            return True

        print("SKIP (unsupported for tags):", filename)
        return False

    except Exception as exc:
        print(f"SKIP (tag error): {filename} ({exc})")
        return False

def _first_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(value).strip()

def read_tag_field(filename, field: str) -> str:
    """Read one logical tag field (genre/style/comment/gender/reverb)."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".flac":
            audio = FLAC(filename)
            return _first_str(audio.get(field))

        if ext in (".ogg", ".opus"):
            audio = (OggOpus(filename) if ext == ".opus" else OggVorbis(filename))
            return _first_str(audio.get(field))

        if ext in (".mp3", ".wav"):
            audio = MP3(filename) if ext == ".mp3" else WAVE(filename)
            tags = audio.tags
            if tags is None:
                return ""
            if field == "genre":
                frame = tags.get("TCON")
                return _first_str(getattr(frame, "text", None))
            if field == "comment":
                frames = tags.getall("COMM")
                return _first_str(frames[0].text if frames else "")
            txxx = {
                "style": "TXXX:STYLE",
                "gender": "TXXX:GENDER",
                "reverb": "TXXX:REVERB",
            }.get(field)
            if txxx:
                frames = tags.getall(txxx)
                return _first_str(frames[0].text if frames else "")
            return ""

        if ext in (".m4a", ".mp4"):
            audio = MP4(filename)
            if field in _MP4_STD:
                return _first_str(audio.get(_MP4_STD[field]))
            if field in _MP4_FREEFORM:
                raw = audio.get(_MP4_FREEFORM[field])
                if not raw:
                    return ""
                item = raw[0]
                if isinstance(item, (bytes, bytearray, MP4FreeForm)):
                    return bytes(item).decode("utf-8", errors="replace").strip()
                return _first_str(item)
            return ""

    except Exception:
        return ""
    return ""

def has_genre_tags(filename) -> bool:
    """True when GENRE already has a non-empty value."""
    return bool(read_tag_field(filename, "genre"))

def has_gender_tags(filename) -> bool:
    """True when target gender field already looks like female/male[+reverb]."""
    field = (
        GENDER_TAG_FIELD
        if GENDER_TAG_FIELD in ("comment", "gender")
        else "comment"
    )
    val = read_tag_field(filename, field).lower()
    if not val:
        return False
    head = val.split("/", 1)[0].strip()
    return head in ("female", "male")

def filter_untagged_files(files, *, kind: str):
    """Drop already-tagged files unless OVERWRITE_TAGS. Returns (kept, skipped)."""
    if OVERWRITE_TAGS:
        _log_intro(
            "Skip if already tagged is off — tagging all files "
            "(not checking for existing tags)."
        )
        return list(files), 0
    label = "gender" if kind == "gender" else "genre"
    _log_intro(
        f"Checking existing {label} tags "
        f"({len(files):,} file(s)) — already tagged files will be skipped…"
    )
    check = has_gender_tags if kind == "gender" else has_genre_tags
    kept = []
    skipped = 0
    iterator = files
    if len(files) >= 200:
        iterator = tqdm(files, desc="Checking tags")
    for path in iterator:
        if check(path):
            skipped += 1
        else:
            kept.append(path)
    return kept, skipped

def write_metadata(filename, genre, style):
    """Write genre/style tags. Returns True if written."""

    if TAG_WRITE_MODE == "split":
        updates = {
            "genre": genre or None,
            "style": style or None,
        }
    else:
        updates = {
            "genre": f"{genre}/{style}" if style else (genre or None),
            "style": None,
        }

    return apply_audio_tags(filename, updates)

def write_gender_metadata(filename, gender_value, reverb_value=None):
    """Write voice-gender (+ optional reverb) per GENDER_TAG_FIELD / REVERB_TAG_MODE."""

    field = (
        GENDER_TAG_FIELD
        if GENDER_TAG_FIELD in ("comment", "gender")
        else "comment"
    )
    reverb_value = (reverb_value or "").strip().lower() or None

    if REVERB_TAG_MODE == "combined" and reverb_value:
        return apply_audio_tags(
            filename,
            {field: f"{gender_value}/{reverb_value}"},
        )

    updates = {field: gender_value}
    if reverb_value:
        updates["reverb"] = reverb_value
    return apply_audio_tags(filename, updates)

# ==========================================================
# AUDIO FUNCTIONS
# ==========================================================
#
# v0.7 change: feature extraction has moved OUT of the main
# thread and INTO the worker. Each worker now returns ready
# input_values tensors, so the GPU never waits on serial
# feature extraction anymore.

def _softmax_np(logits, axis=-1):
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _scores_to_numpy(scores):
    """Clip scores -> float32 ndarray [n_clips, n_labels]."""
    if isinstance(scores, np.ndarray):
        return np.asarray(scores, dtype=np.float32)
    if hasattr(scores, "detach"):
        return scores.detach().float().cpu().numpy()
    # list of 1d tensors/arrays
    rows = []
    for row in scores:
        if hasattr(row, "detach"):
            rows.append(row.detach().float().cpu().numpy())
        else:
            rows.append(np.asarray(row, dtype=np.float32))
    return np.stack(rows, axis=0)

def _avg_topk(scores, k):
    """Mean over clips, then top-k. Returns (values, indices) numpy 1d."""
    arr = _scores_to_numpy(scores)
    if arr.ndim == 1:
        avg = arr
    else:
        avg = arr.mean(axis=0)
    k = max(1, min(int(k), int(avg.shape[0])))
    # partial sort: largest k
    idx = np.argpartition(-avg, kth=k - 1)[:k]
    idx = idx[np.argsort(-avg[idx])]
    return avg[idx], idx

def load_audio(filename, *, use_maest_onnx=None, torch_mod=None):
    """
    Decode audio via SoundFile (faster than librosa load),
    downmix to mono, resample to SAMPLE_RATE via librosa
    (avoids brittle torchaudio native wheels on Windows).
    Returns 1D float32 numpy (ONNX) or torch tensor (HF).
    """
    if use_maest_onnx is None:
        use_maest_onnx = globals().get("USE_MAEST_ONNX")
    if torch_mod is None:
        torch_mod = globals().get("torch")

    try:
        data, sr = sf.read(filename, always_2d=True, dtype="float32")

        # stereo -> mono (numpy), then optional resample
        audio_np = data.mean(axis=1)

        if sr != SAMPLE_RATE:
            audio_np = librosa.resample(
                audio_np,
                orig_sr=sr,
                target_sr=SAMPLE_RATE,
                res_type="soxr_hq",
            )
    except Exception:
        # sf can't decode some codecs (m4a/aac/opus) — fall back to librosa,
        # which uses audioread/ffmpeg (bundled app ffmpeg is on PATH).
        audio_np, _ = librosa.load(filename, sr=SAMPLE_RATE, mono=True)

    audio_np = np.ascontiguousarray(audio_np, dtype=np.float32)
    if use_maest_onnx or torch_mod is None:
        return audio_np
    return torch_mod.from_numpy(audio_np)

def create_clips(audio):

    """
    Slice into NUMBER_OF_CLIPS equal windows of CLIP_LENGTH.
    Short tracks are zero-padded to one full clip.
    Every clip returned here is exactly CLIP_LENGTH * SAMPLE_RATE
    samples long, so downstream stacking needs no padding.
    """

    clip_samples = (
        CLIP_LENGTH
        *
        SAMPLE_RATE
    )

    # Normalize to numpy for slicing (ONNX path / torch-free).
    if hasattr(audio, "detach"):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    length = int(audio_np.shape[0])

    if length <= clip_samples:
        audio_np = np.pad(audio_np, (0, clip_samples - length))
        return [audio_np]

    clips = []
    step = (length - clip_samples) / (NUMBER_OF_CLIPS - 1)

    for i in range(NUMBER_OF_CLIPS):
        start = int(i * step)
        clips.append(audio_np[start : start + clip_samples])

    return clips

def load_and_extract(
    args,
    *,
    feature_extractor=None,
    use_maest_onnx=None,
    torch_mod=None,
    sample_rate=None,
):
    """
    Worker function. Runs on a CPU thread.

    Decodes the file, creates clips AND runs the feature
    extractor, so the main thread receives ready tensors.

    This is the key v0.7 change: the heavy CPU work that used
    to block the GPU (feature extraction in run_gpu_batch) now
    happens in parallel across AUDIO_WORKERS threads.
    """

    if feature_extractor is None:
        feature_extractor = globals().get("feature_extractor")
    if use_maest_onnx is None:
        use_maest_onnx = globals().get("USE_MAEST_ONNX")
    if torch_mod is None:
        torch_mod = globals().get("torch")
    if sample_rate is None:
        sample_rate = SAMPLE_RATE

    index, filename = args

    audio = load_audio(
        filename,
        use_maest_onnx=use_maest_onnx,
        torch_mod=torch_mod,
    )

    clips = create_clips(
        audio
    )


    # All clips are identical length, so padding is a no-op.
    # We do not need the attention_mask (it would be all ones),
    # so it is omitted to save transfer bandwidth.

    inputs = feature_extractor(
        clips,
        sampling_rate=sample_rate,
        return_tensors="np" if use_maest_onnx else "pt",
    )


    return (
        index,
        inputs
    )

def _genre_extract_worker(args, **kwargs):
    """Thread-pool worker: decode + extract, or return error without crashing."""
    index, filename = args
    try:
        return load_and_extract((index, filename), **kwargs) + (None,)
    except Exception as exc:
        return index, None, str(exc)

# ==========================================================
# GPU BATCH
# ==========================================================

def run_gpu_batch(
    batch,
    *,
    model=None,
    device=None,
    is_gpu=None,
    model_dtype=None,
    use_maest_onnx=None,
    measure_gpu_time=None,
):
    """
    batch: list of (index, inputs_dict)
    where inputs_dict has input_values [n_clips, ...] on CPU.

    All feature extraction is already done by the workers,
    so this function only collates, pins memory, transfers
    asynchronously and runs the model.

    State params (model/device/is_gpu/...) default to the module
    globals the CLI populates, so standalone callers can pass
    them explicitly for testing/embedding.
    """

    if model is None:
        model = globals().get("model")
    if device is None:
        device = globals().get("device")
    if is_gpu is None:
        is_gpu = globals().get("IS_GPU")
    if model_dtype is None:
        model_dtype = globals().get("MODEL_DTYPE")
    if use_maest_onnx is None:
        use_maest_onnx = globals().get("USE_MAEST_ONNX")
    if measure_gpu_time is None:
        measure_gpu_time = globals().get("MEASURE_GPU_TIME")

    def _maest_onnx_forward(sub_batch):
        input_values_list = []
        mapping = []
        for index, inputs in sub_batch:
            iv = inputs["input_values"]
            input_values_list.append(iv)
            n_clips = iv.shape[0]
            mapping.extend([index] * n_clips)
        all_iv = np.concatenate(
            [np.asarray(iv, dtype=np.float32) for iv in input_values_list],
            axis=0,
        )
        output = model(input_values=all_iv)
        logits = np.asarray(output.logits, dtype=np.float32)
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        probs = _softmax_np(logits)
        return probs, mapping

    def _is_ort_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "allocate memory" in msg
            or "out of memory" in msg
            or "oom" in msg
            or "bfcarena" in msg
            or "failed to allocate" in msg
        )

    # ONNX path: numpy concat + softmax (no torch). Split on VRAM OOM.
    if use_maest_onnx:
        try:
            return _maest_onnx_forward(batch)
        except Exception as exc:
            if len(batch) <= 1 or not _is_ort_oom(exc):
                raise
            mid = max(1, len(batch) // 2)
            print(
                f"  MAEST ORT OOM at batch={len(batch)} clips-files — "
                f"retrying as {mid}+{len(batch) - mid}",
                flush=True,
            )
            p0, m0 = run_gpu_batch(
                batch[:mid],
                model=model,
                device=device,
                is_gpu=is_gpu,
                model_dtype=model_dtype,
                use_maest_onnx=use_maest_onnx,
                measure_gpu_time=measure_gpu_time,
            )
            p1, m1 = run_gpu_batch(
                batch[mid:],
                model=model,
                device=device,
                is_gpu=is_gpu,
                model_dtype=model_dtype,
                use_maest_onnx=use_maest_onnx,
                measure_gpu_time=measure_gpu_time,
            )
            return np.concatenate([p0, p1], axis=0), m0 + m1

    input_values_list = []
    mapping = []
    for index, inputs in batch:
        iv = inputs["input_values"]
        input_values_list.append(iv)
        n_clips = iv.shape[0]
        mapping.extend([index] * n_clips)

    # All clips are equal length -> plain cat, no padding needed.
    all_iv = torch.cat(input_values_list, dim=0)

    # Pinned memory + non_blocking transfer (GPU only).
    # Requires a pinned source to actually run async.
    # fp16 dtype matches v0.4 (proven). On CPU we keep fp32 and
    # a plain transfer.

    if is_gpu:

        all_iv = (
            all_iv
            .pin_memory()
            .to(
                device,
                dtype=model_dtype,
                non_blocking=True
            )
        )

    else:

        all_iv = all_iv.to(
            device,
            dtype=model_dtype
        )



    if measure_gpu_time and is_gpu:

        torch.cuda.synchronize()

        gpu_start = time.perf_counter()


    with torch.inference_mode():

        # autocast is CUDA-only; on CPU we run straight fp32.

        if is_gpu:

            with torch.autocast(
                device_type="cuda",
                dtype=model_dtype
            ):

                output = model(
                    input_values=all_iv
                )

        else:

            output = model(
                input_values=all_iv
            )


    if measure_gpu_time and device == "cuda":

        torch.cuda.synchronize()

        gpu_time = (
            time.perf_counter()
            -
            gpu_start
        )

        print(
            "GPU batch:",
            round(
                gpu_time,
                1
            ),
            "sec | clips:",
            len(mapping)
        )



    probs = torch.nn.functional.softmax(
        output.logits,
        dim=-1
    )


    del all_iv
    del output

    # No empty_cache: 6 GB / 32 GB leaves plenty of headroom,
    # and empty_cache forces a synchronize that kills throughput.


    return (
        probs.cpu(),
        mapping
    )

# ==========================================================
# PER-FILE INFERENCE (v0.1 style)
# ==========================================================
#
# Same audio pipeline as the batch path (load_audio + create_clips
# + feature_extractor), but one file at a time. Each file's top
# result is printed live.

def classify_one_file(
    filename,
    *,
    model=None,
    device=None,
    is_gpu=None,
    model_dtype=None,
    use_maest_onnx=None,
    feature_extractor=None,
    torch_mod=None,
):
    """
    Returns (scores_cpu, n_clips) where scores_cpu is a
    [n_clips, n_labels] array/tensor on CPU.
    """

    if model is None:
        model = globals().get("model")
    if device is None:
        device = globals().get("device")
    if is_gpu is None:
        is_gpu = globals().get("IS_GPU")
    if model_dtype is None:
        model_dtype = globals().get("MODEL_DTYPE")
    if use_maest_onnx is None:
        use_maest_onnx = globals().get("USE_MAEST_ONNX")
    if feature_extractor is None:
        feature_extractor = globals().get("feature_extractor")
    if torch_mod is None:
        torch_mod = globals().get("torch")

    audio = load_audio(
        filename,
        use_maest_onnx=use_maest_onnx,
        torch_mod=torch_mod,
    )

    clips = create_clips(
        audio
    )

    inputs = feature_extractor(
        clips,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np" if use_maest_onnx else "pt",
    )

    iv = inputs["input_values"]

    if use_maest_onnx:
        output = model(input_values=np.asarray(iv, dtype=np.float32))
        logits = np.asarray(output.logits, dtype=np.float32)
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        scores = _softmax_np(logits)
        return scores, int(scores.shape[0])

    if is_gpu:

        iv = (
            iv
            .pin_memory()
            .to(
                device,
                dtype=model_dtype,
                non_blocking=True
            )
        )

    else:

        iv = iv.to(
            device,
            dtype=model_dtype
        )


    with torch.inference_mode():

        if is_gpu:

            with torch.autocast(
                device_type="cuda",
                dtype=model_dtype
            ):

                output = model(
                    input_values=iv
                )

        else:

            output = model(
                input_values=iv
            )


    scores = torch.nn.functional.softmax(
        output.logits,
        dim=-1
    )


    return (
        scores.cpu(),
        scores.shape[0]
    )

def _log_genre_from_scores(index, scores_tensor, *, model=None, files=None):
    """Compact LOG line from clip scores for one file."""
    if model is None:
        model = globals().get("model")
    if files is None:
        files = globals().get("files")
    values, indices = _avg_topk(scores_tensor, 2)
    best_label = model.config.id2label[int(indices[0])]
    top1 = float(values[0])
    top2 = float(values[1]) if len(values) > 1 else None
    best_score = round(calibrate_multiclass_confidence(top1, top2), 4)
    parts = best_label.split("---")
    genre = parts[0]
    style = parts[1] if len(parts) > 1 else ""
    _log_genre_result(
        files[index], genre, style, best_score, index=index + 1, total=len(files)
    )

def _store_gpu_scores(scores, mapping, *, progress=None, score_storage=None):
    """Accumulate clip scores (batch mode: no per-file LOG spam)."""
    global total_clips
    if score_storage is None:
        score_storage = globals().get("score_storage")
    total_clips += len(mapping)
    for score, idx in zip(scores, mapping):
        score_storage.setdefault(idx, []).append(score)
    if progress is not None:
        _emit_genre_progress(progress, score_storage=score_storage)

def _emit_genre_progress(progress, *, score_storage=None, files=None):
    if score_storage is None:
        score_storage = globals().get("score_storage")
    if files is None:
        files = globals().get("files")
    emit_gg_processed(
        max(progress["decoded"], len(score_storage)),
        len(files),
    )

torch = None  # placeholder; main() imports torch when available
ORT_CUDA_DETAIL = ""  # populated by _probe_ort_cuda() during CLI run
total_clips = 0  # clip counter shared with the batch pipeline (_store_gpu_scores)

import csv

import librosa

import numpy as np

import soundfile as sf

try:
    from tqdm import tqdm as _tqdm_cls
except ImportError:
    # Frozen ONNX builds may omit tqdm; keep a thin progress shim.
    class _tqdm_cls:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc="", disable=False, **_kwargs):
            self.iterable = iterable
            self.total = total
            if self.total is None and iterable is not None:
                try:
                    self.total = len(iterable)
                except TypeError:
                    self.total = None
            self.n = 0
            self.desc = desc or ""
            self.disable = bool(disable)
            self.format_dict = {"rate": None}

        def __iter__(self):
            it = self.iterable
            if it is None:
                return
            for item in it:
                yield item
                self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()
            return False

        def update(self, n=1):
            self.n = int(self.n or 0) + int(n or 0)
            try:
                self.display()
            except Exception:
                pass

        def display(self, *args, **kwargs):
            return None

        def close(self):
            return None

from mutagen.flac import FLAC
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis

from mutagen.id3 import COMM, ID3, TCON, TXXX

# Optional — not required for decode/resample (librosa). May be absent or
# mismatched after other packages (e.g. hear21passt) pull a CUDA wheel.
try:
    import torchaudio  # noqa: F401
except Exception:
    torchaudio = None

_STEM_PROGRESS_INTERVAL = 0.1

_last_stem_progress_emit = 0.0

_last_gg_processed_emit = 0.0

_last_gg_processed_n = -1

class _UiTqdm(_tqdm_cls):
    """tqdm that drives STEM-organizer progress bar when stdout is piped."""

    def __init__(self, *args, **kwargs):
        # Must exist before super().__init__ — tqdm refreshes/display during init.
        self._last_ui_emit = 0.0
        self._ui_piped = not sys.stdout.isatty()
        self._stem_phase = kwargs.pop("stem_phase", None)
        self._stem_pct_scale = float(kwargs.pop("stem_pct_scale", 1.0) or 1.0)
        self._stem_pct_offset = float(kwargs.pop("stem_pct_offset", 0.0) or 0.0)
        self._stem_global_total = int(kwargs.pop("stem_global_total", 0) or 0)
        self._stem_global_offset = int(kwargs.pop("stem_global_offset", 0) or 0)
        self._stem_global_started = float(kwargs.pop("stem_global_started", 0.0) or 0.0)
        kwargs.setdefault("file", sys.stdout)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault(
            "mininterval",
            _STEM_PROGRESS_INTERVAL if self._ui_piped else 0.35,
        )
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ncols", 88)
        super().__init__(*args, **kwargs)

    def display(self, *args, **kwargs):
        if self._ui_piped:
            # Skip tqdm's \\r bar; only emit machine progress for the host.
            try:
                self._emit_stem_progress()
            except Exception:
                pass
            return None
        out = super().display(*args, **kwargs)
        try:
            self._emit_stem_progress()
        except Exception:
            pass
        return out

    def close(self):
        try:
            self._emit_stem_progress(force=True)
        except Exception:
            pass
        if self._ui_piped:
            # Avoid final bar line in the STEM LOG.
            try:
                self.disable = True
            except Exception:
                pass
        return super().close()

    def _emit_stem_progress(self, force: bool = False) -> None:
        total = getattr(self, "total", None) or 0
        if total <= 0:
            return
        n = int(getattr(self, "n", 0) or 0)
        now = time.monotonic()
        last = float(getattr(self, "_last_ui_emit", 0.0) or 0.0)
        if (
            not force
            and n < total
            and (now - last) < _STEM_PROGRESS_INTERVAL
        ):
            return
        self._last_ui_emit = now
        global_total = int(getattr(self, "_stem_global_total", 0) or 0)
        global_offset = int(getattr(self, "_stem_global_offset", 0) or 0)
        if global_total > 0:
            global_n = global_offset + n
            frac = float(global_n) / float(global_total)
            pct = 100.0 * self._stem_pct_scale * frac
            display_n = global_n
            display_total = global_total
        else:
            frac = float(n) / float(total)
            pct = 100.0 * (
                self._stem_pct_offset + self._stem_pct_scale * frac
            )
            display_n = None
            display_total = None
        eta = ""
        try:
            if global_total > 0 and global_n > 0:
                started = float(getattr(self, "_stem_global_started", 0.0) or 0.0)
                if started > 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    g_rate = global_n / elapsed
                    if g_rate > 0:
                        eta = f"{(global_total - global_n) / g_rate:.1f}"
            else:
                rate = self.format_dict.get("rate")
                if rate:
                    eta = f"{(total - n) / float(rate):.1f}"
        except Exception:
            eta = ""
        phase = self._stem_phase
        if not phase:
            try:
                phase = (self.desc or "").strip()
            except Exception:
                phase = ""
        emit_stem_progress(
            n if global_total <= 0 else global_n,
            total if global_total <= 0 else global_total,
            phase or "",
            pct=pct,
            eta=eta,
            force=True,
            display_n=display_n,
            display_total=display_total,
        )

tqdm = _UiTqdm

from mutagen.mp3 import MP3

from mutagen.mp4 import MP4, MP4FreeForm

from mutagen.wave import WAVE

# ==========================================================
# SETTINGS
# ==========================================================

INPUT_FOLDER = ""  # asked at runtime (see prompt below)

# CONTENT_TYPE selects the classifier path at runtime (see prompt).
#   "instrumental" -> Discogs MAEST genre/style -> GENRE(/STYLE) tags
#   "acapella"     -> gender + vocal mel-CNN reverb -> COMMENT/GENDER (+ REVERB)
CONTENT_TYPE = "instrumental"

OUTPUT_CSV = "genre_gender_results.csv"

OUTPUT_CSV_GENDER = "genre_gender_voice_results.csv"

# Metadata toggles.
# WRITE_METADATA=True  -> write tags at all. Set False to only export
#                        the CSV and leave every file untouched.
# OVERWRITE_TAGS=False -> skip files that already have genre/gender tags
#                        (resume-friendly). True forces re-tag everything.
# DRY_RUN              -> kept for parity with older versions, now only
#                        controls the final banner message.
WRITE_METADATA = True

OVERWRITE_TAGS = False

DRY_RUN = False

# TAG_WRITE_MODE controls HOW genre+style are stored in FLAC tags.
# Overridden at runtime by the prompt below, unless RUNTIME_PROMPTS
# is False.
#
#   "combined" -> GENRE field only, as "Genre/Style"
#                 e.g. GENRE = "Rock/Surf"
#   "split"    -> GENRE field = genre, STYLE field = style
#                 e.g. GENRE = "Rock", STYLE = "Surf"
#
# Both modes overwrite whatever is currently in those fields.
TAG_WRITE_MODE = "combined"

# GENDER_TAG_FIELD controls WHERE voice-gender is stored (acapella).
# Overridden at runtime by the prompt below when WRITE_METADATA is True.
#
#   "comment" -> COMMENT field, e.g. COMMENT = "female"
#   "gender"  -> GENDER field,  e.g. GENDER  = "female"
GENDER_TAG_FIELD = "comment"

# REVERB_TAG_MODE controls how vocal dry/wet reverb is stored with gender.
#
#   "combined" -> gender field only, as "gender/reverb"
#                 e.g. COMMENT = "female/wet"
#   "split"    -> gender field = gender, REVERB field = dry|wet
#                 e.g. COMMENT = "female", REVERB = "wet"
REVERB_TAG_MODE = "combined"

# BATCH_MODE toggles the two run styles.
# Overridden at runtime by the prompt below, unless RUNTIME_PROMPTS
# is False.
#
#   True  -> fast batched pipeline. Files are decoded in parallel
#           across AUDIO_WORKERS threads and pushed to the GPU in
#           BATCH_SIZE chunks. Fastest, but no per-file output.
#
#   False -> per-file style. Files are processed one at a time and each
#           result is printed live (GENRE / STYLE / CONF). Slower
#           (no batching), but you see every file's outcome as it
#           finishes. Also the recommended mode on CPU-only machines / VMs.
BATCH_MODE = True

# RUNTIME_PROMPTS=True  -> ask at startup which mode to run in and
#                          how to write tags. The answers override
#                          BATCH_MODE and TAG_WRITE_MODE above.
# RUNTIME_PROMPTS=False -> use BATCH_MODE and TAG_WRITE_MODE as-is,
#                          no questions asked.
RUNTIME_PROMPTS = True

MODEL_NAME = (
    "mtg-upf/"
    "discogs-maest-30s-pw-129e-519l"
)

SAMPLE_RATE = 16000

CLIP_LENGTH = 30

NUMBER_OF_CLIPS = 3

# Defaults tuned for large libraries on a single HDD/NAS (RTX-class GPU).
# Workers/chunk come from Quick tune; GPU batch aligned with Gender/Reverb (64).
BATCH_SIZE = 4  # MAEST files/batch (×3 clips); 64 OOMs Softmax ~26 GB on CUDA

# CPU workers — parallel decode + feature extract (batch mode only).
AUDIO_WORKERS = 4

# Genre batch: process files in waves so the thread pool does not queue
# the entire library and so HDD read concurrency stays bounded.
GENRE_FILE_CHUNK = 128

# Gender batch: never queue the whole library at once. Each Future
# caches its mel-patch result until destroyed — with 80k files that
# OOMs Windows (ACCESS_VIOLATION / taskkill 0xc000012d). Process and
# tag in waves so RAM stays bounded and a crash keeps earlier tags.
GENDER_FILE_CHUNK = 256

# GPU timing debug
# OFF in v0.7: the per-batch synchronize blocks the CPU->GPU
# overlap that this version relies on. Total wall-clock timing
# at the end stays accurate without it.
MEASURE_GPU_TIME = False

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aif",
    ".aiff"
}

# Overridden by GG_RECURSIVE in non-interactive (STEM organizer) mode.
INCLUDE_SUBFOLDERS = True

# ----------------------------------------------------------
# Voice-gender / EffNet Discogs (acapella path)
# Essentia has no Windows wheels; we run official .pb weights
# with a MusiCNN-style mel front-end matching Essentia.
# ----------------------------------------------------------

GENDER_FRAME_SIZE = 512

GENDER_HOP_SIZE = 256

GENDER_N_MELS = 96

GENDER_PATCH_SIZE = 128

GENDER_PATCH_HOP = 62

GENDER_BATCH_SIZE = 64  # discogs-effnet-bs64 fixed batch

# Reverb (vocal mel-CNN): files decoded in parallel, crops stacked on GPU.
REVERB_FILE_CHUNK = 64

REVERB_GPU_BATCH = 64  # align with EffNet wave rhythm; dynamic batch OK

GENDER_LABELS = ("female", "male")

GENDER_MODEL_DIR = _writable_gender_model_dir()

# Gender classifier runs on ONNX Runtime only (TF frozen-graph fallback was
# removed). ONNX is GPU-capable via DirectML/CUDA and CPU on every platform.
GENDER_EFFNET_ONNX_NAME = "discogs-effnet-bsdynamic-1.onnx"

GENDER_HEAD_ONNX_NAME = "gender-discogs-effnet-1.onnx"

GENDER_EFFNET_ONNX_URL = (
    "https://essentia.upf.edu/models/feature-extractors/"
    "discogs-effnet/discogs-effnet-bsdynamic-1.onnx"
)

GENDER_HEAD_ONNX_URL = (
    "https://essentia.upf.edu/models/classification-heads/"
    "gender/gender-discogs-effnet-1.onnx"
)

# Full bsdynamic ONNX is ~18 MB; truncated downloads (~4 MB) look "present"
# but ORT fails with a vague system error and tagging appears stuck/dead.
GENDER_EFFNET_ONNX_MIN_BYTES = 15_000_000

GENDER_HEAD_ONNX_MIN_BYTES = 100_000

GENDER_ORT_BATCH = 128

_INSTRUMENT_MODELS = (
    _tagger_package_dir().parent / "instrument_tagger" / "models"
)

_MEL_FILTERBANK = None

# ==========================================================
# METADATA WRITER (FLAC / MP3 / M4A / WAV)
# ==========================================================
#
# Genre TAG_WRITE_MODE:
#   "combined" -> GENRE only as "Genre/Style"
#   "split"    -> GENRE + STYLE
#
# Storage by format:
#   FLAC -> Vorbis comments (genre / style / comment / gender / reverb)
#   MP3 / WAV -> ID3 (TCON, COMM, TXXX:STYLE, TXXX:GENDER, TXXX:REVERB)
#   M4A -> MP4 atoms (©gen, ©cmt, iTunes freeform STYLE/GENDER/REVERB)

_MP4_STD = {
    "genre": "\xa9gen",
    "comment": "\xa9cmt",
}

_MP4_FREEFORM = {
    "style": "----:com.apple.iTunes:STYLE",
    "gender": "----:com.apple.iTunes:GENDER",
    "reverb": "----:com.apple.iTunes:REVERB",
}

def main():
    global AUDIO_WORKERS, AutoFeatureExtractor, AutoModelForAudioClassification, BATCH_MODE, BATCH_SIZE, CONTENT_TYPE, GENDER_FILE_CHUNK, GENDER_TAG_FIELD, GENRE_FILE_CHUNK, INCLUDE_SUBFOLDERS, INPUT_FOLDER, IS_GPU, MODEL_DTYPE, ORT_CUDA_DETAIL, ORT_CUDA_OK, OUTPUT_CSV, OUTPUT_CSV_GENDER, OVERWRITE_TAGS, REVERB_FILE_CHUNK, REVERB_GPU_BATCH, REVERB_TAG_MODE, RUNTIME_PROMPTS, TAG_WRITE_MODE, USE_MAEST_ONNX, WRITE_METADATA, _GENRE_AUDIO_WEIGHT, _GENRE_TAG_WEIGHT, _GG_MODE, _cuda_ok, _gg_batch, _gg_batch_sz, _gg_csv, _gg_file_chunk, _gg_files_from, _gg_gender_fld, _gg_input, _gg_overwrite, _gg_recursive, _gg_reverb, _gg_tag_style, _gg_workers, _gg_write_meta, _has_audio_env, _input_path, _maest_exc, _maest_onnx, _maest_onnx_early, _maest_ort_device, _try_maest, args, batch, best, best_label, best_score, candidate, chunk, conf, content_in, cuda_ep_usable, device, done_n, elapsed, err, error, errors, exc, executor, extract_done, extras, feature_extractor, filename, files, files_total, finalize_state, finalized_in_wave, force, future, futures, gender_backend, gender_started, gender_tag_in, genre, genre_audio_started, genre_progress, gpu_name, has_audio, i, idx, idxs, index, indices, inputs, item, label_index, load_vocal_reverb, m, mapping, mode_in, model, n_clips, nvidia_gpu_present, ok, out_csv, outs, pad, parts, patch, patches, paths, pbar, peak, pending_map, pending_patches, phase_name, pred, predictions, prob, probs, rate, results, rev, rev_chunk, rev_start, reverb_dir, reverb_jobs, reverb_mode_label, reverb_router, reverb_tag_in, row, runner, score, score_storage, scores, skipped, skipped_tagged, soft_n, stacked, start_time, style, tag_field_label, tag_in, tag_total, take, top1, top2, torch, total_clips, total_found, try_load_maest_onnx, user_input, values, wave_end, wave_files, wave_len, wave_start, written

    _status(f"{APP_NAME} v{APP_VERSION}")

    _status("Starting up...")

    _status("  loading torch...")

    try:
        import torch
    except ImportError:
        torch = None
        _status("  torch not installed (ONNX-only mode)")

    _status("  loading audio / data helpers...")

    _status("Core libraries ready.")

    # ==========================================================
    # DEVICE / DTYPE
    # ==========================================================
    #
    # Torch CUDA is optional (frozen ONNX builds ship without torch).
    # ONNX Runtime can still use CUDAExecutionProvider on NVIDIA — Genre/Gender
    # must not treat "no torch" as "CPU only".

    _status("Detecting compute device...")

    ORT_CUDA_OK, ORT_CUDA_DETAIL = _probe_ort_cuda()

    if torch is not None:
        IS_GPU = torch.cuda.is_available()
        device = "cuda" if IS_GPU else ("cuda" if ORT_CUDA_OK else "cpu")
        # fp16/autocast only when torch CUDA exists; ORT-CUDA uses fp32 tensors.
        MODEL_DTYPE = torch.float16 if IS_GPU else torch.float32
        torch.set_grad_enabled(False)
        if IS_GPU:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0)
            _status(f"Device: CUDA (torch) — {gpu_name}")
        elif ORT_CUDA_OK:
            _status(f"Device: ONNX Runtime CUDA EP — {ORT_CUDA_DETAIL}")
        else:
            _status(f"Device: CPU (no torch CUDA; ORT: {ORT_CUDA_DETAIL})")
    else:
        IS_GPU = False
        MODEL_DTYPE = None
        if ORT_CUDA_OK:
            device = "cuda"  # GPU via ORT — not torch
            _status(f"Device: ONNX Runtime CUDA EP — {ORT_CUDA_DETAIL}")
        else:
            device = "cpu"
            _status(f"Device: CPU (ONNX) — {ORT_CUDA_DETAIL}")

    _status("Startup complete.\n")

    # ==========================================================
    # NON-INTERACTIVE MODE (env-driven, e.g. from STEM organizer)
    # ==========================================================
    #
    # Set GG_MODE=genre or GG_MODE=gender plus the other GG_* vars
    # to run without any interactive prompts.  When GG_MODE is
    # empty the classic interactive CLI starts as usual.

    _GG_MODE = os.environ.get("GG_MODE", "").strip().lower()

    if _GG_MODE:
        _gg_input      = os.environ.get("GG_INPUT",        "").strip()
        _gg_batch      = os.environ.get("GG_BATCH",        "1").strip()
        _gg_tag_style  = os.environ.get("GG_TAG_STYLE",    "combined").strip().lower()
        _gg_gender_fld = os.environ.get("GG_GENDER_FIELD", "comment").strip().lower()
        _gg_reverb     = os.environ.get("GG_REVERB_MODE",  "combined").strip().lower()
        _gg_write_meta = os.environ.get("GG_WRITE_META",   "1").strip()
        _gg_overwrite  = os.environ.get("GG_OVERWRITE",    "0").strip()
        _gg_recursive  = os.environ.get("GG_RECURSIVE",    "1").strip()
        _gg_csv        = os.environ.get("GG_CSV",          "").strip()
        _gg_files_from = os.environ.get("GG_FILES_FROM",   "").strip()

        CONTENT_TYPE     = "acapella" if _GG_MODE == "gender" else "instrumental"
        INPUT_FOLDER     = _gg_input
        BATCH_MODE       = (_gg_batch != "0")
        TAG_WRITE_MODE   = _gg_tag_style if _gg_tag_style in ("combined", "split") else "combined"
        GENDER_TAG_FIELD = _gg_gender_fld if _gg_gender_fld in ("comment", "gender") else "comment"
        REVERB_TAG_MODE  = _gg_reverb if _gg_reverb in ("combined", "split") else "combined"
        WRITE_METADATA   = (_gg_write_meta != "0")
        OVERWRITE_TAGS   = (_gg_overwrite == "1")
        INCLUDE_SUBFOLDERS = (_gg_recursive != "0")
        RUNTIME_PROMPTS  = False

        if _gg_csv:
            OUTPUT_CSV        = _gg_csv
            OUTPUT_CSV_GENDER = _gg_csv

        _gg_workers = os.environ.get("GG_AUDIO_WORKERS", "").strip()
        _gg_batch_sz = os.environ.get("GG_BATCH_SIZE", "").strip()
        _gg_file_chunk = os.environ.get("GG_FILE_CHUNK", "").strip()
        if _gg_workers:
            AUDIO_WORKERS = max(1, int(_gg_workers))
        if _gg_batch_sz:
            BATCH_SIZE = max(1, int(_gg_batch_sz))
            # Genre MAEST only — reverb stays aligned with EffNet (64).
        if _gg_file_chunk:
            GENRE_FILE_CHUNK = max(BATCH_SIZE, int(_gg_file_chunk))
            GENDER_FILE_CHUNK = max(128, int(_gg_file_chunk))
            REVERB_FILE_CHUNK = GENDER_FILE_CHUNK
            REVERB_GPU_BATCH = GENDER_BATCH_SIZE

        # Safety nets after UI Quick-tune env:
        # - CPU: avoid ORT×ThreadPool desktop freeze
        # - Genre MAEST (GPU or CPU): never batch 64 files (×3 clips → ~26 GB Softmax OOM)
        def _ort_cuda_ready() -> bool:
            if os.environ.get("STEM_ORT_CUDA", "1").strip() == "0":
                return False
            try:
                from ort_util import cuda_ep_usable, nvidia_gpu_present

                return bool(cuda_ep_usable() and nvidia_gpu_present())
            except Exception:
                return False

        _cuda_ok = _ort_cuda_ready()
        if _GG_MODE == "genre":
            # Caps even when GG_BATCH_SIZE=64 is injected from an old hint/cache.
            BATCH_SIZE = min(BATCH_SIZE, 4)
            GENRE_FILE_CHUNK = min(GENRE_FILE_CHUNK, max(BATCH_SIZE, 32))
            AUDIO_WORKERS = min(AUDIO_WORKERS, 4 if _cuda_ok else 2)
        if not _cuda_ok:
            AUDIO_WORKERS = min(AUDIO_WORKERS, 2)
            if _GG_MODE == "genre":
                BATCH_SIZE = min(BATCH_SIZE, 4)
                GENRE_FILE_CHUNK = min(GENRE_FILE_CHUNK, max(BATCH_SIZE, 32))
            GENDER_FILE_CHUNK = min(GENDER_FILE_CHUNK, 64)
            REVERB_FILE_CHUNK = GENDER_FILE_CHUNK
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        _input_path = Path(INPUT_FOLDER) if INPUT_FOLDER else None
        if not _input_path or not _input_path.is_dir():
            print(
                f"GG_MODE error: INPUT_FOLDER does not exist: {INPUT_FOLDER!r}",
                file=sys.stderr,
            )
            sys.exit(1)

        _has_audio_env = bool(_paths_from_files_list(_gg_files_from)) if _gg_files_from else any(
            True for _ in iter_audio_files(_input_path)
        )
        if not _has_audio_env:
            print(
                f"GG_MODE error: no supported audio files in {INPUT_FOLDER!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ==========================================================
    # ASK CONTENT TYPE (instrumental vs acapella)
    # ==========================================================

    if not _GG_MODE:

        print("==============================")
        print("Content type")
        print("==============================")
        print()
        print("  1 = Instrumental - genre/style (discogs-maest-30s-pw-129e-519l)")
        print("  2 = Acapella     - voice gender + dry/wet reverb (vocal mel-CNN)")
        print()

        while True:

            content_in = input(
                "Content type [1/2] (default 1): "
            ).strip()

            if content_in == "":

                content_in = "1"

            if content_in in ("1", "2"):

                CONTENT_TYPE = (
                    "instrumental"
                    if content_in == "1"
                    else "acapella"
                )

                break

            print("Enter 1 or 2.")

        print()

        if CONTENT_TYPE == "instrumental":

            print("Selected: INSTRUMENTAL (discogs-maest-30s-pw-129e-519l)")

        else:

            print("Selected: ACAPELLA (gender-discogs-effnet)")

        print()

    # ==========================================================
    # ASK INPUT FOLDER
    # ==========================================================

    if not _GG_MODE:

        while True:

            user_input = input(
                "Enter input folder (drag in or paste path): "
            ).strip()


            # Strip surrounding quotes that Windows drag-and-drop adds.

            if (
                len(user_input) >= 2
                and user_input[0] == user_input[-1]
                and user_input[0] in "\"'"
            ):

                user_input = user_input[1:-1]


            candidate = Path(user_input)


            if not user_input:

                print("Empty input, try again.")

                continue


            if not candidate.exists():

                print(
                    "Path does not exist:",
                    user_input
                )

                continue


            if not candidate.is_dir():

                print(
                    "Not a folder:",
                    user_input
                )

                continue


            # Quick sanity: any supported audio in this tree?

            has_audio = any(True for _ in iter_audio_files(candidate))

            if not has_audio:

                print(
                    "No supported audio files found in tree. Try again."
                )

                continue


            INPUT_FOLDER = str(candidate)

            break


        print()
        print(
            "Input folder:",
            INPUT_FOLDER
        )

    # ==========================================================
    # RUNTIME PROMPTS (mode + tag style)
    # ==========================================================

    if RUNTIME_PROMPTS and CONTENT_TYPE == "instrumental":

        # ---- Mode: batch vs per-file ----

        print()
        print("==============================")
        print("Run mode")
        print("==============================")
        print()
        print("  1 = Batch    - much faster, but no per-file overview")
        print("  2 = Per-file - slower, prints GENRE/STYLE/CONF per file")

        while True:

            mode_in = input(
                "Run mode [1/2] (default 1): "
            ).strip()

            if mode_in == "":

                mode_in = "1"

            if mode_in in ("1", "2"):

                BATCH_MODE = (mode_in == "1")

                break

            print("Enter 1 or 2.")

        print()

        if BATCH_MODE:

            print("Selected: BATCH")

        else:

            print("Selected: PER-FILE")


        # ---- Tag write style ----

        print()
        print("==============================")
        print("Tag writing")
        print("==============================")
        print()

        if WRITE_METADATA:

            print("  1 = GENRE field only, as \"Genre/Style\"")
            print("       e.g. GENRE = Rock/Metal")
            print("  2 = GENRE + STYLE fields, separated")
            print("       e.g. GENRE = Rock , STYLE = Metal")

            while True:

                tag_in = input(
                    "Tag style [1/2] (default 1): "
                ).strip()

                if tag_in == "":

                    tag_in = "1"

                if tag_in in ("1", "2"):

                    TAG_WRITE_MODE = (
                        "combined"
                        if tag_in == "1"
                        else "split"
                    )

                    break

                print("Enter 1 or 2.")

        else:

            print("Tag writing is OFF (WRITE_METADATA=False).")
            print("Predictions will only be exported to CSV.")

        print()

        if WRITE_METADATA:

            if TAG_WRITE_MODE == "combined":

                print("Selected: GENRE field = \"Genre/Style\"")

            else:

                print("Selected: GENRE + STYLE fields, separated")

        print()


    elif RUNTIME_PROMPTS and CONTENT_TYPE == "acapella":

        # ---- Mode: batch vs per-file ----

        if RUNTIME_PROMPTS:

            print()
            print("==============================")
            print("Run mode")
            print("==============================")
            print()
            print("  1 = Batch    - much faster, but no per-file overview")
            print("  2 = Per-file - slower, prints GENDER/CONF per file")

            while True:

                mode_in = input(
                    "Run mode [1/2] (default 1): "
                ).strip()

                if mode_in == "":

                    mode_in = "1"

                if mode_in in ("1", "2"):

                    BATCH_MODE = (mode_in == "1")

                    break

                print("Enter 1 or 2.")

            print()

            if BATCH_MODE:

                print("Selected: BATCH")

            else:

                print("Selected: PER-FILE")

        print()
        print("==============================")
        print("Tag writing")
        print("==============================")
        print()

        if WRITE_METADATA:

            print("  1 = COMMENT field")
            print("       e.g. COMMENT = female")
            print("  2 = GENDER field")
            print("       e.g. GENDER = female")

            while True:

                gender_tag_in = input(
                    "Tag field [1/2] (default 1): "
                ).strip()

                if gender_tag_in == "":

                    gender_tag_in = "1"

                if gender_tag_in in ("1", "2"):

                    GENDER_TAG_FIELD = (
                        "comment"
                        if gender_tag_in == "1"
                        else "gender"
                    )

                    break

                print("Enter 1 or 2.")

            print()

            if GENDER_TAG_FIELD == "comment":

                print("Selected: COMMENT field")

            else:

                print("Selected: GENDER field")

            print()
            print("==============================")
            print("Reverb tagging (vocal mel-CNN)")
            print("==============================")
            print()
            print("  1 = Combined with gender")
            print("       e.g. COMMENT = female/wet")
            print("  2 = Separate REVERB field")
            print("       e.g. COMMENT = female, REVERB = wet")

            while True:

                reverb_tag_in = input(
                    "Reverb mode [1/2] (default 1): "
                ).strip()

                if reverb_tag_in == "":

                    reverb_tag_in = "1"

                if reverb_tag_in in ("1", "2"):

                    REVERB_TAG_MODE = (
                        "combined"
                        if reverb_tag_in == "1"
                        else "split"
                    )

                    break

                print("Enter 1 or 2.")

            print()

            if REVERB_TAG_MODE == "combined":

                print("Selected: combined gender/reverb")

            else:

                print("Selected: split gender + REVERB")

        else:

            print("Tag writing is OFF (WRITE_METADATA=False).")
            print("Predictions will only be exported to CSV.")

        print()

    # ==========================================================
    # ACAPELLA PATH (gender-discogs-effnet + vocal mel-CNN reverb)
    # ==========================================================

    if CONTENT_TYPE == "acapella":

        from vocal_reverb import load_vocal_reverb

        tag_field_label = (
            "COMMENT"
            if GENDER_TAG_FIELD == "comment"
            else "GENDER"
        )
        reverb_mode_label = (
            f"{tag_field_label}=gender/reverb"
            if REVERB_TAG_MODE == "combined"
            else f"{tag_field_label}=gender + REVERB"
        )

        print()
        print("==============================")
        _log_intro(f"{APP_NAME} v{APP_VERSION}")
        _log_intro("Acapella / voice-gender + reverb")
        if BATCH_MODE:
            _log_intro("Batched pipeline")
        else:
            _log_intro("Per-file mode")
        _log_intro("gender-discogs-effnet + vocal_reverb.pt (mel-CNN)")
        _log_intro(f"Recursive + {reverb_mode_label} metadata")
        print("==============================")
        print()

        if BATCH_MODE:

            _log_intro(f"Audio workers: {AUDIO_WORKERS}")
            print()

        gender_backend = load_gender_models(status=_gender_model_status)

        _log_intro("Loading vocal reverb classifier...")
        reverb_dir = GENDER_MODEL_DIR
        for candidate in _gender_model_dir_candidates():
            if (candidate / "vocal_reverb.pt").is_file():
                reverb_dir = candidate
                break
        reverb_router = load_vocal_reverb(reverb_dir, status=_status)
        _log_intro("Reverb classifier loaded")
        print()

        files = list_audio_files(INPUT_FOLDER)
        total_found = len(files)
        files, skipped_tagged = filter_untagged_files(files, kind="gender")

        _log_intro(
            f"Found {total_found} "
            f"{'files (recursive)' if INCLUDE_SUBFOLDERS else 'files (top-level only)'}"
        )
        if skipped_tagged:
            _log_intro(
                f"Skipping {skipped_tagged} already tagged "
                "(Skip if already tagged is on)"
            )
        _log_intro(f"To process: {len(files)}")
        print()

        if not files:

            _log_intro("No untagged audio files left.")
            _log_intro(
                "Turn off Skip if already tagged to re-tag, or pick another folder."
            )
            sys.exit(0)

        start_time = time.perf_counter()
        results = [None] * len(files)

        def _empty_gender_row(filename, error=""):

            return {
                "file": filename,
                "gender": "",
                "confidence": 0.0,
                "female": 0.0,
                "male": 0.0,
                "reverb": "",
                "reverb_confidence": 0.0,
                "wet": 0.0,
                "dry": 0.0,
                "n_patches": 0,
                "error": error,
            }

        def _merge_reverb(row, filename):

            try:

                rev = reverb_router.predict(filename)

            except Exception as exc:

                row["error"] = (
                    (row.get("error") or "") + f" | reverb: {exc}"
                ).strip(" |")
                return row

            row["reverb"] = rev["reverb"]
            row["reverb_confidence"] = round(rev["reverb_confidence"], 4)
            row["wet"] = round(rev["wet"], 4)
            row["dry"] = round(rev["dry"], 4)
            return row

        written = 0
        skipped = 0

        if BATCH_MODE:

            # Parallel mel/patch extract in bounded waves. Never submit the
            # whole library — Future objects cache patch arrays until GC.

            print(
                f"  workers={AUDIO_WORKERS}  "
                f"file_chunk={GENDER_FILE_CHUNK}  "
                f"gpu_batch={GENDER_BATCH_SIZE}"
            )
            print(
                f"  reverb workers={AUDIO_WORKERS}  "
                f"file_chunk={REVERB_FILE_CHUNK}  "
                f"gpu_batch={REVERB_GPU_BATCH}  "
                f"device={getattr(reverb_router, 'device', '?')}"
            )
            print("Processing gender (batch)...")
            emit_gg_processed(0, len(files), force=True)

            def _extract_worker(args):

                index, filename = args

                try:

                    patches = extract_patches(filename)

                    return index, patches, None

                except Exception as exc:

                    return index, None, str(exc)

            files_total = len(files)
            gender_started = time.perf_counter()

            def _gender_eta(done_n):
                if done_n <= 0:
                    return ""
                rate = done_n / max(time.perf_counter() - gender_started, 1e-9)
                if rate <= 0:
                    return ""
                return f"{(files_total - done_n) / rate:.1f}"

            with tqdm(total=files_total, desc="Gender") as pbar:

                for wave_start in range(0, files_total, GENDER_FILE_CHUNK):

                    wave_end = min(wave_start + GENDER_FILE_CHUNK, files_total)
                    wave_files = files[wave_start:wave_end]
                    wave_len = wave_end - wave_start

                    score_storage = {}
                    errors = {}
                    pending_patches = []
                    pending_map = []

                    def _flush_gender_batch(force=False):

                        while pending_patches and (
                            force
                            or len(pending_patches) >= GENDER_BATCH_SIZE
                        ):

                            take = min(
                                GENDER_BATCH_SIZE, len(pending_patches)
                            )
                            chunk = pending_patches[:take]
                            mapping = pending_map[:take]
                            del pending_patches[:take]
                            del pending_map[:take]

                            stacked = np.stack(chunk, axis=0)

                            if take < GENDER_BATCH_SIZE:

                                pad = np.zeros(
                                    (
                                        GENDER_BATCH_SIZE - take,
                                        GENDER_PATCH_SIZE,
                                        GENDER_N_MELS,
                                    ),
                                    dtype=np.float32,
                                )
                                stacked = np.concatenate(
                                    [stacked, pad], axis=0
                                )

                            probs = predict_fixed_batch(
                                stacked,
                                gender_backend,
                            )

                            for prob, idx in zip(probs[:take], mapping):

                                score_storage.setdefault(
                                    idx, []
                                ).append(prob)

                    extract_done = 0

                    with ThreadPoolExecutor(
                        max_workers=AUDIO_WORKERS
                    ) as executor:

                        futures = [
                            executor.submit(
                                _extract_worker,
                                (wave_start + i, filename),
                            )
                            for i, filename in enumerate(wave_files)
                        ]

                        for future in as_completed(futures):

                            index, patches, err = future.result()
                            extract_done += 1
                            # Soft bar during extract (40% of this wave).
                            soft_n = (
                                wave_start
                                + (extract_done / wave_len) * wave_len * 0.4
                            )
                            emit_stem_progress(
                                soft_n,
                                files_total,
                                "extracting",
                                eta=_gender_eta(wave_start),
                                display_n=wave_start + extract_done,
                            )

                            if err is not None:

                                errors[index] = err
                                continue

                            for patch in patches:

                                pending_patches.append(patch)
                                pending_map.append(index)

                            del patches

                            if len(pending_patches) >= GENDER_BATCH_SIZE:

                                _flush_gender_batch()

                        del futures

                    _flush_gender_batch(force=True)

                    for index in range(wave_start, wave_end):

                        filename = files[index]

                        if index in errors:

                            print("ERROR:", filename, flush=True)
                            print(" ", errors[index], flush=True)
                            results[index] = _empty_gender_row(
                                filename,
                                errors[index],
                            )
                            continue

                        probs = np.stack(score_storage[index], axis=0)
                        pred = probs_to_result(probs)

                        results[index] = {
                            "file": filename,
                            "gender": pred["gender"],
                            "confidence": round(pred["confidence"], 4),
                            "female": round(pred["female"], 4),
                            "male": round(pred["male"], 4),
                            "reverb": "",
                            "reverb_confidence": 0.0,
                            "wet": 0.0,
                            "dry": 0.0,
                            "n_patches": pred["n_patches"],
                            "error": "",
                        }

                    reverb_jobs = [
                        (index, files[index])
                        for index in range(wave_start, wave_end)
                        if results[index] is not None
                        and results[index].get("gender")
                    ]

                    for rev_start in range(
                        0, len(reverb_jobs), REVERB_FILE_CHUNK
                    ):

                        rev_chunk = reverb_jobs[
                            rev_start : rev_start + REVERB_FILE_CHUNK
                        ]
                        idxs = [item[0] for item in rev_chunk]
                        paths = [item[1] for item in rev_chunk]
                        outs = reverb_router.predict_many(
                            paths,
                            gpu_batch_size=REVERB_GPU_BATCH,
                            num_workers=AUDIO_WORKERS,
                        )

                        for index, rev in zip(idxs, outs):

                            row = results[index]

                            if isinstance(rev, BaseException):

                                row["error"] = (
                                    (row.get("error") or "")
                                    + f" | reverb: {rev}"
                                ).strip(" |")

                            else:

                                row["reverb"] = rev["reverb"]
                                row["reverb_confidence"] = round(
                                    rev["reverb_confidence"], 4
                                )
                                row["wet"] = round(rev["wet"], 4)
                                row["dry"] = round(rev["dry"], 4)

                    # Finalize per file: LOG + optional write + progress (60% of wave).
                    # Use a list so nested helper can mutate without nonlocal
                    # (this path runs at module scope, not inside a function).
                    finalize_state = [0]

                    def _advance_finalize(phase_name):
                        finalize_state[0] += 1
                        finalized_in_wave = finalize_state[0]
                        done_n = wave_start + finalized_in_wave
                        soft_n = (
                            wave_start
                            + wave_len * 0.4
                            + finalized_in_wave * 0.6
                        )
                        emit_stem_progress(
                            soft_n,
                            files_total,
                            phase_name,
                            eta=_gender_eta(done_n),
                            force=True,
                            display_n=done_n,
                        )
                        emit_gg_processed(done_n, files_total)
                        pbar.n = done_n
                        # Piped: host already got emit_stem_progress — avoid
                        # refresh overwriting soft pct with integer n/total.
                        if not getattr(pbar, "_ui_piped", False):
                            pbar.refresh()

                    for index in range(wave_start, wave_end):

                        row = results[index]

                        if index in errors:

                            _advance_finalize("tagging")
                            continue

                        if WRITE_METADATA:

                            if not row or not row.get("gender"):

                                skipped += 1

                            else:

                                ok = write_gender_metadata(
                                    row["file"],
                                    row["gender"],
                                    row.get("reverb") or None,
                                )

                                if ok:
                                    written += 1
                                else:
                                    skipped += 1

                        _advance_finalize(
                            "writing" if WRITE_METADATA else "tagging"
                        )

                    del score_storage, errors, pending_patches, pending_map
                    gc.collect()

            emit_gg_processed(files_total, files_total, force=True)

            print()

        else:

            # Per-file: live GENDER / REVERB / CONF

            print("Processing (one file at a time)...")
            print()

            for index, filename in enumerate(
                tqdm(files, desc="Gender")
            ):

                try:

                    pred = classify_gender_file(
                        filename,
                        gender_backend,
                    )

                except Exception as exc:

                    print()
                    print("ERROR:", filename)
                    print(" ", exc)
                    results[index] = _empty_gender_row(
                        filename,
                        str(exc),
                    )
                    continue

                row = {
                    "file": filename,
                    "gender": pred["gender"],
                    "confidence": round(pred["confidence"], 4),
                    "female": round(pred["female"], 4),
                    "male": round(pred["male"], 4),
                    "reverb": "",
                    "reverb_confidence": 0.0,
                    "wet": 0.0,
                    "dry": 0.0,
                    "n_patches": pred["n_patches"],
                    "error": "",
                }
                row = _merge_reverb(row, filename)
                results[index] = row

                _log_gender_result(row, index=index + 1, total=len(files))

                if WRITE_METADATA and row.get("gender"):

                    ok = write_gender_metadata(
                        row["file"],
                        row["gender"],
                        row.get("reverb") or None,
                    )

                    if ok:
                        written += 1
                    else:
                        skipped += 1

                elif WRITE_METADATA:

                    skipped += 1

            print()

        elapsed = time.perf_counter() - start_time
        out_csv = OUTPUT_CSV_GENDER
        _write_results_csv(out_csv, results)

        peak = None
        if device == "cuda" and torch is not None:
            try:
                peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            except Exception:
                peak = None

        extras = []
        if WRITE_METADATA:
            extras.append(f"{reverb_mode_label} tags written: {written}")
        else:
            extras.append("METADATA UNTOUCHED (WRITE_METADATA=False)")

        print_feature_summary(
            "Gender",
            elapsed=elapsed,
            files=len(files),
            tagged=written if WRITE_METADATA else None,
            skipped=skipped if WRITE_METADATA else None,
            peak_vram_gb=peak,
            results_path=out_csv,
            extra_lines=extras,
        )

        sys.exit(0)

    # ==========================================================
    # INSTRUMENTAL PATH (MAEST Discogs519)
    # ==========================================================

    # Prefer ONNX (STEM_ONNX default) — skip transformers import when assets exist.
    # Prefer ORT CUDA when available — frozen builds have no torch.cuda.
    _maest_ort_device = (
        "cpu" if os.environ.get("STEM_ORT_CUDA", "1").strip() == "0" else ""
    )

    if not ORT_CUDA_OK:
        _maest_ort_device = "cpu"

    _maest_onnx_early = None

    try:
        from maest_onnx import try_load_maest_onnx as _try_maest

        _maest_onnx_early = _try_maest(
            device=_maest_ort_device,
            status=lambda m: None,
        )
    except Exception:
        _maest_onnx_early = None

    if _maest_onnx_early is None:
        _status("Loading transformers (MAEST)...")
        # Quiet HF Hub "unauthenticated" / remote-code download chatter.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message=r".*cache-system uses symlinks.*",
            category=UserWarning,
            module=r"huggingface_hub.*",
        )
        from transformers import (  # noqa: F401 — used in LOAD MODEL fallback
            AutoModelForAudioClassification,
            AutoFeatureExtractor,
        )
        _status("Transformers ready.")
    else:
        _status("MAEST ONNX path (transformers not required).")

    print()

    print()

    print("==============================")

    _log_intro(f"{APP_NAME} v{APP_VERSION}")

    if BATCH_MODE:
        _log_intro("Batched GPU pipeline")
    else:
        _log_intro("Per-file mode")

    _log_intro("Instrumental / Discogs genre")

    _log_intro("Recursive + metadata write")

    print("==============================")

    print()

    _log_intro(
        f"Torch: {torch.__version__ if torch is not None else 'not installed (ONNX)'}"
    )

    _log_intro(
        "Torchaudio: "
        f"{getattr(torchaudio, '__version__', None) or 'not installed (librosa resample)'}"
    )

    if torch is not None and getattr(torch.version, "cuda", None):
        _log_intro(
            f"CUDA (torch): {torch.version.cuda} "
            f"(available={torch.cuda.is_available()})"
        )
    elif ORT_CUDA_OK:
        _log_intro(f"CUDA (ONNX Runtime): {ORT_CUDA_DETAIL}")
    else:
        _log_intro(f"CUDA: not available ({ORT_CUDA_DETAIL})")

    if device == "cuda" and IS_GPU:
        _log_intro(f"Device: cuda (torch)")
        _log_intro(f"GPU: {torch.cuda.get_device_name(0)}")
        _log_intro(
            "VRAM: "
            f"{round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB"
        )
    elif device == "cuda" and ORT_CUDA_OK:
        _log_intro("Device: cuda (ONNX Runtime CUDA EP)")
    else:
        _log_intro(f"Device: {device}")

    print()

    if BATCH_MODE:

        _log_intro(f"Batch size: {BATCH_SIZE}")

        _log_intro(f"Audio workers: {AUDIO_WORKERS}")

        _log_intro(f"File chunk: {GENRE_FILE_CHUNK}")

    _log_intro(f"Clips/song: {NUMBER_OF_CLIPS}")

    _log_intro(f"Write metadata: {WRITE_METADATA}")

    if WRITE_METADATA:

        _log_intro(
            "Tag style: "
            + (
                "Genre/Style combined"
                if TAG_WRITE_MODE == "combined"
                else "GENRE + STYLE separated"
            )
        )

    print()

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    _log_intro("Loading model...")

    USE_MAEST_ONNX = False

    _maest_onnx = _maest_onnx_early

    if _maest_onnx is None:
        try:
            from maest_onnx import try_load_maest_onnx

            _maest_onnx = try_load_maest_onnx(
                device=_maest_ort_device,
                status=lambda m: _log_intro(m),
            )
        except Exception as _maest_exc:
            _log_intro(
                f"  MAEST ONNX unavailable ({type(_maest_exc).__name__}); torch fallback"
            )
            _maest_onnx = None

    if _maest_onnx is not None:
        feature_extractor, model = _maest_onnx
        USE_MAEST_ONNX = True
        _log_intro(f"  MAEST onnxruntime ready ({getattr(model, 'device', 'onnx')})")
    else:
        if torch is None:
            raise SystemExit(
                "MAEST ONNX assets missing and torch is not installed.\n"
                "Place maest_discogs519.onnx (+ id2label) under models/"
            )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=MODEL_DTYPE,
        )

        model.to(device)

        model.eval()

    _log_intro("Model loaded" + (" (ONNX)" if USE_MAEST_ONNX else ""))

    print()

    # ==========================================================
    # FIND FILES
    # ==========================================================

    files = list_audio_files(INPUT_FOLDER)

    total_found = len(files)

    files, skipped_tagged = filter_untagged_files(files, kind="genre")

    _log_intro(
        f"Found {total_found} "
        f"{'files (recursive)' if INCLUDE_SUBFOLDERS else 'files (top-level only)'}"
    )

    if skipped_tagged:
        _log_intro(
            f"Skipping {skipped_tagged} already tagged "
            "(Skip if already tagged is on)"
        )

    _log_intro(f"To process: {len(files)}")

    print()

    if not files:
        _log_intro("No untagged audio files left.")
        _log_intro(
            "Turn off Skip if already tagged to re-tag, or pick another folder."
        )
        sys.exit(0)

    # ==========================================================
    # TIMER START
    # ==========================================================

    start_time = time.perf_counter()

    # ==========================================================
    # PROCESS
    # ==========================================================

    score_storage = {}

    genre_errors: dict[int, str] = {}

    total_clips = 0

    # When metadata write follows, audio owns 85% of the bar so Tagging
    # never resets percent back to 0.
    _GENRE_AUDIO_WEIGHT = 0.85 if WRITE_METADATA else 1.0

    _GENRE_TAG_WEIGHT = 1.0 - _GENRE_AUDIO_WEIGHT

    if BATCH_MODE:

        # ------------------------------------------------
        # BATCH PATH (streaming pipeline, bounded waves)
        # ------------------------------------------------

        batch = []
        genre_progress = {"decoded": 0}
        genre_errors.clear()
        files_total = len(files)

        print(
            "Processing..."
        )
        if files_total > GENRE_FILE_CHUNK:
            print(
                f"  workers={AUDIO_WORKERS}  "
                f"file_chunk={GENRE_FILE_CHUNK}  "
                f"gpu_batch={BATCH_SIZE}"
            )

        emit_gg_processed(0, files_total, force=True)

        genre_audio_started = time.monotonic()

        for wave_start in range(0, files_total, GENRE_FILE_CHUNK):

            wave_end = min(wave_start + GENRE_FILE_CHUNK, files_total)

            with ThreadPoolExecutor(
                max_workers=AUDIO_WORKERS
            ) as executor:

                futures = {
                    executor.submit(
                        _genre_extract_worker,
                        (i, files[i]),
                    ):
                    i
                    for i in range(wave_start, wave_end)
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Audio",
                    stem_phase="audio",
                    stem_pct_scale=_GENRE_AUDIO_WEIGHT,
                    stem_global_total=files_total,
                    stem_global_offset=wave_start,
                    stem_global_started=genre_audio_started,
                ):

                    index, inputs, err = future.result()

                    genre_progress["decoded"] += 1
                    _emit_genre_progress(genre_progress)

                    if err is not None:
                        genre_errors[index] = err
                        print("ERROR:", files[index], flush=True)
                        print(" ", err, flush=True)
                        continue

                    batch.append(
                        (
                            index,
                            inputs
                        )
                    )

                    if len(batch) >= BATCH_SIZE:

                        scores, mapping = run_gpu_batch(
                            batch
                        )
                        _store_gpu_scores(scores, mapping, progress=genre_progress)
                        batch = []

        # remaining batch

        if batch:

            scores, mapping = run_gpu_batch(
                batch
            )
            _store_gpu_scores(scores, mapping, progress=genre_progress)

        emit_gg_processed(files_total, files_total, force=True)

        if genre_errors:
            print(
                f"{LOG_INDENT}Decode errors: {len(genre_errors)}",
                flush=True,
            )

        print()

        print(
            "Total inference clips:",
            total_clips
        )



    else:

        # ------------------------------------------------
        # PER-FILE PATH (v0.1 style, live output)
        # ------------------------------------------------

        print(
            "Processing (one file at a time)..."
        )

        print()


        for index, filename in enumerate(
            tqdm(
                files,
                desc="Analyzing",
                stem_phase="analyzing",
                stem_pct_scale=_GENRE_AUDIO_WEIGHT,
                stem_pct_offset=0.0,
            )
        ):

            scores, n_clips = classify_one_file(
                filename
            )

            total_clips += n_clips

            score_storage[index] = [
                scores[i]
                for i in range(n_clips)
            ]


            # Average the clips, then show the top hit, exactly like
            # v0.1's GENRE / STYLE / CONF block.

            values, indices = _avg_topk(scores, 2)

            best_label = model.config.id2label[
                int(indices[0])
            ]

            top1 = float(values[0])
            top2 = (
                float(values[1])
                if len(values) > 1
                else None
            )
            best_score = calibrate_multiclass_confidence(
                top1,
                top2,
            )

            parts = best_label.split(
                "---"
            )

            genre = parts[0]

            style = (
                parts[1]
                if len(parts) > 1
                else ""
            )


            _log_genre_result(
                filename,
                genre,
                style,
                round(best_score, 4),
                index=index + 1,
                total=len(files),
            )



        print()

        print(
            "Total inference clips:",
            total_clips
        )

    # ==========================================================
    # RESULTS
    # ==========================================================

    results = []

    for index, filename in enumerate(files):

        if index not in score_storage:
            err = genre_errors.get(index, "no scores (decode/inference failed)")
            results.append(
                {
                    "file": filename,
                    "genre": "",
                    "style": "",
                    "confidence": 0.0,
                    "top5": "[]",
                    "error": err,
                }
            )
            continue

        values, indices = _avg_topk(score_storage[index], 5)

        predictions = []

        for score, label_index in zip(values, indices):

            predictions.append(
                {
                    "label":
                        model.config.id2label[
                            int(label_index)
                        ],

                    "score":
                        round(
                            float(score),
                            4
                        )
                }
            )


        best = predictions[0]
        runner = predictions[1] if len(predictions) > 1 else None

        parts = best["label"].split(
            "---"
        )

        genre = parts[0]

        style = (
            parts[1]
            if len(parts) > 1
            else ""
        )

        conf = calibrate_multiclass_confidence(
            best["score"],
            runner["score"] if runner else None,
        )

        results.append(
            {
                "file": filename,
                "genre": genre,
                "style": style,
                "confidence": round(conf, 4),
                "top5": json.dumps(predictions)
            }
        )

    # ==========================================================
    # WRITE METADATA
    # ==========================================================

    written = 0

    skipped = 0

    if WRITE_METADATA:

        print()
        print(
            "Writing metadata to",
            len(results),
            "files..."
        )

        tag_total = len(results)
        emit_gg_processed(0, tag_total, force=True)

        for i, row in enumerate(
            tqdm(
                results,
                desc="Tagging",
                stem_phase="tagging",
                stem_pct_scale=_GENRE_TAG_WEIGHT,
                stem_pct_offset=_GENRE_AUDIO_WEIGHT,
            ),
            start=1,
        ):

            if row.get("error") or not row.get("genre"):
                skipped += 1
                emit_gg_processed(i, tag_total)
                continue

            ok = write_metadata(
                row["file"],
                row["genre"],
                row["style"]
            )

            if ok:

                written += 1

            else:

                skipped += 1

            emit_gg_processed(i, tag_total)

        emit_gg_processed(tag_total, tag_total, force=True)


    else:

        print()
        print(
            "Metadata writing OFF"
        )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    elapsed = time.perf_counter() - start_time

    peak = None

    if device == "cuda" and torch is not None:
        try:
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        except Exception:
            peak = None

    _write_results_csv(OUTPUT_CSV, results)

    extras = []

    if DRY_RUN:
        extras.append("DRY RUN flag set (banner only)")

    if not WRITE_METADATA:
        extras.append("METADATA UNTOUCHED (WRITE_METADATA=False)")

    print_feature_summary(
        "Genre",
        elapsed=elapsed,
        files=len(files),
        tagged=written if WRITE_METADATA else None,
        skipped=skipped if WRITE_METADATA else None,
        peak_vram_gb=peak,
        results_path=OUTPUT_CSV,
        extra_lines=extras,
    )


if __name__ == "__main__":
    main()

"""Tk-free Classify (RMS) + SI-SDR backend — verbatim port of the worker layer
from D:\\STEM-organizer\\stem_organizer_ui.py (lines 1–2530 plus the small helpers
_skip_reason_label / format_skip_summary / FOLDER_OUTCOME_LABELS / FOLDER_OUTCOME_ORDER).

Everything below is GUI-agnostic and only depends on numpy / soundfile (always)
and either onnxruntime Demucs (default, STEM_ONNX) or torch / demucs (fallback).
Loaded lazily via :func:`_init_ml`. The original threading.Thread classes are
preserved so callers can use them as-is on plain ``threading`` if they prefer;
the PySide6 port wraps them in ``stem_organizer.workers.classify_worker``
QThread adapters.

Source line refs in D:\\STEM-organizer\\stem_organizer_ui.py: 1–2530, 2829–2865.
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from ffmpeg_bootstrap import subprocess_kwargs

# ---------------------------------------------------------------------------
# ML stack — set lazily in _init_ml()
# ---------------------------------------------------------------------------

np = None
sf = None
torch = None
get_model = None
apply_model = None
AudioFile = None
_ML_INITIALIZED = False


# ---------------------------------------------------------------------------
# Constants & dicts
# ---------------------------------------------------------------------------

AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.aif', '.aiff', '.ogg', '.m4a', '.opus')

MODELS = {
    # Analysis-first separators + CUDA HTDemucs (AGENTS.md / CUDA ship path).
    # Keys = Model dropdown display labels; values = internal ids (unchanged).
    'Vocal CNN6': 'vocal_cnn6',
    'HTdemucs (demucs)': 'htdemucs',
}

STEM_MODES = {
    '2 (instrumental/vocals)': {
        'categories': ('instrumental', 'vocals'),
        'mapping':    {'vocals': 'vocals'},
        'fallback':   'instrumental',
    },
    '4 (bass/drums/other/vocals)': {
        'categories': ('bass', 'drums', 'other', 'vocals'),
        'mapping':    {n: n for n in ('bass', 'drums', 'other', 'vocals')},
        'fallback':   'other',
    },
    '6 (b/d/o/v+guitar/piano)': {
        'categories': ('bass', 'drums', 'other', 'vocals', 'guitar', 'piano'),
        'mapping':    {n: n for n in ('bass', 'drums', 'other', 'vocals', 'guitar', 'piano')},
        'fallback':   'other',
    },
}

# Older settings / UI labels → current STEM_MODES key
STEM_MODE_ALIASES = {
    '2-way (instrumental/vocals)': '2 (instrumental/vocals)',
    '4-way (bass/drums/other/vocals)': '4 (bass/drums/other/vocals)',
    '4-way (drums/bass/other/vocals)': '4 (bass/drums/other/vocals)',
    'Vocals + Instrumental': '2 (instrumental/vocals)',
    '6 (bass/drums/other/vocals/guitar/piano)': '6 (b/d/o/v+guitar/piano)',
}


def resolve_stem_mode(label: str) -> str:
    """Map saved/legacy stem labels onto a STEM_MODES key."""
    if label in STEM_MODES:
        return label
    return STEM_MODE_ALIASES.get(label, next(iter(STEM_MODES)))


QUALITY_PRESETS = {
    'FLAC 16-bit':      {'ext': '.flac', 'subtype': 'PCM_16'},
    'FLAC 24-bit':      {'ext': '.flac', 'subtype': 'PCM_24'},
    'WAV 16-bit':       {'ext': '.wav',  'subtype': 'PCM_16'},
    'WAV 24-bit':       {'ext': '.wav',  'subtype': 'PCM_24'},
    'WAV 32-bit float': {'ext': '.wav',  'subtype': 'FLOAT'},
}

AMBIG_MODES = {
    'Skip ambiguous stem only': 'skip_stem',
    'Skip the entire song':     'skip_song',
}

STEM_FILE_EXTS = ('.flac', '.wav', '.mp3')
SDR_STEM_EXT_LABEL = 'flac/wav/mp3'

SDR_DEFAULT_THRESHOLDS = {
    'bass': 25,
    'drums': 20,
    'other': 20,
    'vocals': 30,
    'instrumental': 30,
    'guitar': 20,
    'piano': 20,
}

SCAN_MODES = {
    'Each subfolder (one level)':    'subfolders',
    'Each leaf folder (recursive)':  'recursive',
}

NAMING_MODES = {
    'Original folder name':              'preserve',
    'Folder name (simplified)':          'slug',
    'Sequential (song_0000, 0001, …)':   'sequential',
}

MANIFEST_FILENAME = 'index.json'

# Sentinels / IPC tags (kept for parity with the Tk version's queue protocol).
DONE_SENTINEL = object()
PROGRESS_TAG = '__progress__'
PAIR_LOG_TAG = '__pair_log__'
SDR_LOG_TAG = '__sdr_line__'
GG_PROCESSED_TAG = '__gg_processed__'


# ---------------------------------------------------------------------------
# App paths & init
# ---------------------------------------------------------------------------

def _is_frozen() -> bool:
    return getattr(sys, 'frozen', False)


def _resource_dir() -> Path:
    if _is_frozen():
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _app_dir() -> Path:
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = _app_dir()
RESOURCE_DIR = _resource_dir()

FFMPEG: str | None = None
_SEQ_RE = re.compile(r'^song_(\d+)$')
SF_READ_EXTS = {'.wav', '.flac', '.aif', '.aiff', '.ogg', '.mp3', '.m4a', '.opus'}
_ALLOWED_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789')


def _configure_torch_home() -> None:
    for base in (APP_DIR, RESOURCE_DIR):
        torch_home = base / 'torch_home'
        checkpoints = torch_home / 'hub' / 'checkpoints'
        if checkpoints.is_dir() and any(checkpoints.glob('*.th')):
            os.environ['TORCH_HOME'] = str(torch_home)
            return


def _init_ml() -> None:
    """Initialize numpy / soundfile (+ Demucs ONNX or torch). Idempotent."""
    global np, sf, torch, get_model, apply_model, AudioFile, FFMPEG, _ML_INITIALIZED
    if _ML_INITIALIZED:
        return
    from deps_bootstrap import init_external_deps, load_ml_deps

    init_external_deps()
    np, sf, torch, get_model, apply_model, AudioFile = load_ml_deps()
    if torch is not None:
        _configure_torch_home()
    from ffmpeg_bootstrap import ffmpeg_path

    FFMPEG = ffmpeg_path()
    _ML_INITIALIZED = True


def _onnx_demucs_wanted() -> bool:
    """True when any ONNX stem-separation weight is present."""
    try:
        from vocal_classifier_onnx import vocal_classifier_installed

        if vocal_classifier_installed():
            return True
    except Exception:
        pass
    try:
        from demucs_onnx import resolve_htdemucs_onnx

        return resolve_htdemucs_onnx() is not None
    except Exception:
        return False


def load_demucs_model(model_id: str, *, prefer_gpu: bool = False):
    """Load stem separator — Vocal CNN6 or HTDemucs.

    Legacy ids (umxl/xumxl/scnet_tran/bsroformer/demucs) map to HTDemucs; the
    UMX-L / X-UMXL / SCNet Tran / BS-RoFormer separators were retired (not
    faster than HTDemucs on the 3060-class target).
    prefer_gpu: CUDA EP when usable, else DirectML if present, else CPU.
    HTDemucs is CUDA-or-CPU only (DirectML historically rejected for Demucs).
    Vocal CNN6 is a classifier (not separator) — produces synthetic stems for
    RMS classification; SI-SDR should use HTDemucs or similar.
    """
    mid = (model_id or "htdemucs").strip().lower()
    # Legacy/retired separator ids → HTDemucs.
    if mid in ("umxl", "xumxl", "scnet_tran", "scnet", "scnet-tran",
               "bsroformer", "demucs", ""):
        mid = "htdemucs"

    if mid == "htdemucs":
        from demucs_onnx import DemucsOnnxModel, resolve_htdemucs_onnx

        onnx_path = resolve_htdemucs_onnx()
        if onnx_path is None:
            raise RuntimeError(
                "HTDemucs ONNX weight missing. Place models/htdemucs.onnx "
                "(or htdemucs.batch.onnx) from StemSplitio/htdemucs-onnx."
            )
        return DemucsOnnxModel(onnx_path, prefer_gpu=bool(prefer_gpu))

    if mid == "vocal_cnn6":
        from vocal_classifier_onnx import VocalClassifierOnnxModel, resolve_vocal_classifier_onnx

        onnx_path = resolve_vocal_classifier_onnx()
        if onnx_path is None:
            raise RuntimeError(
                "Vocal CNN6 ONNX weight missing. Place models/vocal_classifier.onnx "
                "beside the app."
            )
        return VocalClassifierOnnxModel(onnx_path, prefer_gpu=bool(prefer_gpu))

    raise RuntimeError(
        f"Unknown model id {model_id!r} "
        "(expected 'vocal_cnn6' or 'htdemucs')."
    )


def _is_onnx_model(model) -> bool:
    return getattr(model, "backend", None) == "onnx"


def _is_gpu_oom(exc: BaseException) -> bool:
    """True for torch CUDA OOM or ORT CUDA allocation failures."""
    if torch is not None and hasattr(torch, "cuda") and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "oom",
            "failed to allocate",
            "cudamalloc",
            "cuda error: out of memory",
            "resource exhausted",
            "memory allocation failure",
        )
    )


def _release_gpu(model, device: str) -> None:
    """Drop torch CUDA cache after a run; no-op for ONNX / CPU."""
    if _is_onnx_model(model) or device != "cuda" or torch is None:
        return
    try:
        model.cpu()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# Classify OLA fraction (chunk hop = segment * (1 - overlap)). Analysis-first
# default 0.1; override with STEM_CLASSIFY_OVERLAP (float, clamped 0.0–0.5).
DEFAULT_CLASSIFY_OVERLAP = 0.1


def _classify_overlap() -> float:
    raw = os.environ.get("STEM_CLASSIFY_OVERLAP", "").strip()
    if not raw:
        return DEFAULT_CLASSIFY_OVERLAP
    try:
        v = float(raw)
    except ValueError:
        return DEFAULT_CLASSIFY_OVERLAP
    return max(0.0, min(0.5, v))


def _demucs_apply(model, mix_np: "np.ndarray", device: str, *, overlap: float | None = None):
    """Run separation. mix_np: (B,2,T) float32 → (B,S,2,T) float32."""
    if overlap is None:
        overlap = _classify_overlap()
    if _is_onnx_model(model):
        want = "cpu" if device == "cpu" else (device or "dml")
        if getattr(model, "_device", None) != want:
            model.to(want)
        return model.separate_numpy(mix_np, overlap=overlap)
    batch = torch.from_numpy(mix_np).to(device)
    with torch.no_grad():
        out = apply_model(
            model,
            batch,
            device=device,
            progress=False,
            shifts=0,
            split=True,
            overlap=overlap,
        )
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# CUDA helpers
# ---------------------------------------------------------------------------

def torch_cuda_built() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.backends.cuda.is_built())
    except Exception:
        return False


def cuda_device_name() -> str | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


_CUDA_USABLE: bool | None = None


def cuda_compute_capability() -> tuple[int, int] | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def cuda_arch_tag() -> str | None:
    cap = cuda_compute_capability()
    if not cap:
        return None
    major, minor = cap
    return f'sm_{major}{minor}'


def cuda_arch_listed() -> bool | None:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return None
    if not arch_list:
        return None
    tag = cuda_arch_tag()
    return bool(tag and tag in arch_list)


def cuda_usable(*, force: bool = False) -> bool:
    global _CUDA_USABLE
    if not force and _CUDA_USABLE is not None:
        return _CUDA_USABLE
    if torch is None or not torch.cuda.is_available():
        _CUDA_USABLE = False
        return False
    listed = cuda_arch_listed()
    if listed is False:
        _CUDA_USABLE = False
        return False
    try:
        probe = torch.zeros(1, device='cuda')
        probe.add_(1)
        torch.cuda.synchronize()
        _CUDA_USABLE = True
    except RuntimeError:
        _CUDA_USABLE = False
    return _CUDA_USABLE


def cuda_effective() -> bool:
    return cuda_usable()


def cuda_incompatibility_hint() -> str | None:
    if torch is None or not torch.cuda.is_available() or cuda_usable():
        return None
    name = cuda_device_name() or 'NVIDIA GPU'
    tag = cuda_arch_tag()
    cap = cuda_compute_capability()
    if cap and cap[0] >= 12:
        tag_text = f' ({tag})' if tag else ''
        return (
            f'{name}{tag_text} needs PyTorch with CUDA 12.8 (cu128). '
            'Re-run install-deps.bat and choose option 3 (RTX 50-series).'
        )
    if tag:
        return (
            f'{name} ({tag}) is not supported by the installed PyTorch build. '
            'Re-run install-deps.bat or disable CUDA to use CPU.'
        )
    return (
        f'{name} is visible to PyTorch but CUDA kernels failed. '
        'Re-run install-deps.bat or disable CUDA to use CPU.'
    )


def resolve_processing_device(use_cuda: bool) -> tuple[str, list[str]]:
    """Return device id for the stem separator.

    Prefer ``cuda`` when a real NVIDIA GPU + CUDA EP are usable, else ``dml``
    when DirectML is exposed, else ``cpu``. ``use_cuda`` is the UI "Use GPU"
    flag (name is historical).
    """
    warnings: list[str] = []
    if not use_cuda:
        return 'cpu', warnings
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        warnings.append("onnxruntime unavailable — using CPU.")
        return 'cpu', warnings

    if "CUDAExecutionProvider" in available:
        try:
            from ort_util import cuda_ep_usable, nvidia_gpu_present

            if nvidia_gpu_present() and cuda_ep_usable():
                return 'cuda', warnings
            if not nvidia_gpu_present():
                warnings.append("No NVIDIA GPU — trying DirectML/CPU.")
        except Exception:
            warnings.append("CUDA probe failed — trying DirectML/CPU.")

    if "DmlExecutionProvider" in available:
        return 'dml', warnings

    warnings.append("No CUDA/DirectML EP in this ORT build — using CPU.")
    return 'cpu', warnings


def is_cuda_kernel_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return 'no kernel image is available' in msg or 'cuda error' in msg


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = ''.join(c for c in name.lower() if c in _ALLOWED_NAME_CHARS)
    return s or 'folder'


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_duration_log(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f'{hours}:{mins:02d}:{secs:02d}:{ms:03d}'
    return f'{mins:02d}:{secs:02d}:{ms:03d}'


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


class PhaseTimer:
    def __init__(self) -> None:
        self._times: dict[str, float] = {}

    def add(self, phase: str, seconds: float) -> None:
        if seconds > 0:
            self._times[phase] = self._times.get(phase, 0.0) + seconds

    def get(self, phase: str) -> float:
        return self._times.get(phase, 0.0)

    def log_summary(self, log_fn, labels: dict[str, str], *, title: str = 'Phase timing', prefix: str = '  ') -> None:
        if not self._times:
            return
        log_fn(f'{prefix}{title}:')
        for phase, sec in sorted(self._times.items(), key=lambda kv: (-kv[1], kv[0])):
            label = labels.get(phase, phase)
            log_fn(f'{prefix}  {label}: {format_duration_log(sec)}')


ORGANIZE_PHASE_LABELS = {
    'model_load': 'Model load',
    'input_scan': 'Input scan',
    'dedup': 'De-dupe',
    'prescan': 'Pre-scan',
    'classification': 'RMS classification',
    'mixing': 'Mix stems',
    'export': 'Write output',
}

SDR_PHASE_LABELS = {
    'model_load': 'Model load',
    'target_scan': 'Target scan',
    'audio_load': 'Load audio',
    'separation': 'Separation',  # filled with active model via sdr_phase_labels()
    'sdr_compute': 'SDR computation',
}


def model_display_label(model_id: str | None) -> str:
    """Dropdown-facing label for an internal model id (e.g. htdemucs → HTdemucs)."""
    mid = (model_id or '').strip().lower()
    for label, mid_v in MODELS.items():
        if mid_v == mid:
            return label
    return mid or 'unknown'


def sdr_phase_labels(model_id: str | None = None) -> dict[str, str]:
    """SI-SDR phase labels with Separation (Model) matching the Model dropdown."""
    labels = dict(SDR_PHASE_LABELS)
    labels['separation'] = f'Separation ({model_display_label(model_id)})'
    return labels


def folder_name_with_duration(name: str, duration_sec: float | None, append: bool) -> str:
    if append and duration_sec and duration_sec > 0:
        return f"{name} [{format_duration(duration_sec)}]"
    return name


def folder_has_outputs(path: Path, categories: tuple[str, ...], ext: str) -> bool:
    if not path.is_dir():
        return False
    exts = (ext, '.flac', '.wav', '.mp3', '.ogg')
    seen_exts: list[str] = []
    for e in exts:
        if e not in seen_exts:
            seen_exts.append(e)
    for cat in categories:
        for e in seen_exts:
            if (path / f"{cat}{e}").is_file():
                return True
    return False


def find_existing_output_dir(
    out_dir: Path, rel: Path, naming_mode: str,
    categories: tuple[str, ...], ext: str, manifest: dict | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if naming_mode == 'sequential' and manifest:
        rel_s = str(rel).replace('\\', '/')
        for folder_name, orig in manifest.items():
            if orig == rel_s:
                candidates.append(out_dir / folder_name)
    elif str(rel) == '.':
        candidates.append(out_dir)
    elif naming_mode == 'preserve':
        base_name = rel.name
        parent = out_dir.joinpath(*rel.parts[:-1]) if len(rel.parts) > 1 else out_dir
        candidates.append(out_dir / rel)
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (child.name == base_name or child.name.startswith(base_name + ' [')):
                    candidates.append(child)
    else:
        slug_parts = [slugify(pp) for pp in rel.parts if pp not in ('', '.')]
        if slug_parts:
            base = slug_parts[-1]
            parent = out_dir.joinpath(*slug_parts[:-1]) if len(slug_parts) > 1 else out_dir
            candidates.append(out_dir.joinpath(*slug_parts))
            if parent.is_dir():
                for child in sorted(parent.iterdir()):
                    if child.is_dir() and (child.name == base or child.name.startswith(base + ' [')):
                        candidates.append(child)

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if folder_has_outputs(candidate, categories, ext):
            return candidate
    return None


def collect_song_groups(in_dir: Path, scan_mode: str) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    if scan_mode == 'subfolders':
        for sub in sorted(in_dir.iterdir()):
            if not sub.is_dir():
                continue
            stems = [f for f in sub.rglob('*') if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
            if stems:
                groups[sub] = sorted(stems)
        return groups
    for f in in_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            groups.setdefault(f.parent, []).append(f)
    for folder in groups:
        groups[folder] = sorted(groups[folder])
    return groups


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (MANIFEST_FILENAME + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(out_dir / MANIFEST_FILENAME)


# ---------------------------------------------------------------------------
# Recycle bin / cleanup
# ---------------------------------------------------------------------------

def send_to_recycle_bin(path: Path) -> None:
    target = Path(path).resolve()
    if not target.exists():
        return
    if sys.platform == 'win32':
        from ctypes import wintypes

        FO_DELETE = 0x0003
        FOF_ALLOWUNDO = 0x0040
        FOF_NOCONFIRMATION = 0x0010
        FOF_SILENT = 0x0004

        class _SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ('hwnd', wintypes.HWND),
                ('wFunc', ctypes.c_uint),
                ('pFrom', wintypes.LPCWSTR),
                ('pTo', wintypes.LPCWSTR),
                ('fFlags', ctypes.c_ushort),
                ('fAnyOperationsAborted', wintypes.BOOL),
                ('hNameMappings', wintypes.LPVOID),
                ('lpszProgressTitle', wintypes.LPCWSTR),
            ]

        op = _SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = str(target) + '\0\0'
        op.pTo = None
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
        op.hNameMappings = None
        op.lpszProgressTitle = None
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if result != 0:
            raise OSError(f'recycle bin delete failed (code {result})')
        if op.fAnyOperationsAborted:
            raise OSError('recycle bin delete was aborted')
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return not any(path.iterdir())
    except OSError:
        return False


def remove_empty_parent_dirs(start: Path, root: Path) -> list[Path]:
    removed: list[Path] = []
    root = root.resolve()
    try:
        current = Path(start).resolve().parent
    except OSError:
        return removed
    while current != root:
        try:
            current.relative_to(root)
        except ValueError:
            break
        if not _is_empty_dir(current):
            break
        try:
            send_to_recycle_bin(current)
            removed.append(current)
        except OSError:
            break
        current = current.parent
    return removed


def prune_empty_dirs_under(root: Path) -> list[Path]:
    root = root.resolve()
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        current = Path(dirpath)
        if current == root or not _is_empty_dir(current):
            continue
        try:
            send_to_recycle_bin(current)
            removed.append(current)
        except OSError:
            pass
    return removed


def cleanup_empty_dirs_after_delete(deleted_paths: list[Path], root: Path) -> list[Path]:
    removed: list[Path] = []
    seen: set[Path] = set()
    for path in deleted_paths:
        for parent in remove_empty_parent_dirs(path, root):
            key = parent.resolve()
            if key not in seen:
                seen.add(key)
                removed.append(parent)
    return removed


def next_sequence_number(out_dir: Path, manifest: dict) -> int:
    n_max = -1
    if out_dir.exists():
        for d in out_dir.iterdir():
            if d.is_dir():
                m = _SEQ_RE.match(d.name)
                if m:
                    n_max = max(n_max, int(m.group(1)))
    for k in manifest:
        m = _SEQ_RE.match(k)
        if m:
            n_max = max(n_max, int(m.group(1)))
    return n_max + 1


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def _normalize_audio(audio, file_sr: int, sr: int, ch: int):
    if audio.shape[0] == 1:
        audio = np.repeat(audio, ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    if file_sr != sr:
        from audio_resample import resample_audio

        try:
            audio = resample_audio(audio, file_sr, sr, axis=1)
        except ImportError:
            raise RuntimeError(
                f"Sample rate mismatch ({file_sr} Hz vs expected {sr} Hz) "
                "and scipy is not installed. Install it with: pip install scipy"
            )
    return audio.astype(np.float32)


def _read_soundfile(path: str, sr: int, ch: int):
    try:
        data, file_sr = sf.read(path, dtype='float32', always_2d=True)
        return _normalize_audio(data.T, file_sr, sr, ch)
    except Exception:
        return None


def _read_via_ffmpeg(path: str, sr: int, ch: int):
    if not FFMPEG:
        return None
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [FFMPEG, '-y', '-loglevel', 'error', '-i', path,
             '-ar', str(sr), '-ac', str(ch), tmp_path],
            check=True, capture_output=True,
            **subprocess_kwargs(),
        )
        return _read_soundfile(tmp_path, sr, ch)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def format_load_error(exc: Exception) -> str:
    msg = str(exc)
    if 'CalledProcessError' in type(exc).__name__ or 'ffmpeg' in msg.lower():
        return 'decode failed'
    if len(msg) > 100:
        return msg[:97] + '...'
    return msg


def load_audio(path: str, sr: int, ch: int = 2):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in SF_READ_EXTS:
        audio = _read_soundfile(str(p), sr, ch)
        if audio is not None:
            return audio
    audio = _read_via_ffmpeg(str(p), sr, ch)
    if audio is not None:
        return audio
    if AudioFile is None:
        raise RuntimeError(
            f'Could not decode {p.name}. soundfile/ffmpeg failed and audio libraries are not initialized.'
        )
    return AudioFile(path).read(streams=0, samplerate=sr, channels=ch).numpy().astype(np.float32)


def write_audio(path: str, audio, sr: int, subtype: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith('.flac') and FFMPEG:
        bps = 16 if subtype == 'PCM_16' else 24
        sample_fmt = 's16' if subtype == 'PCM_16' else 's32'
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio.T, sr, subtype='FLOAT')
            subprocess.run(
                [FFMPEG, '-y', '-loglevel', 'error', '-i', tmp_path,
                 '-c:a', 'flac', '-compression_level', '12',
                 '-sample_fmt', sample_fmt, '-bits_per_raw_sample', str(bps), path],
                check=True,
                **subprocess_kwargs(),
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        sf.write(path, audio.T, sr, subtype=subtype)


def _rms(a) -> float:
    return float(np.sqrt(np.mean(a ** 2) + 1e-12))


def _tensor_rms(t) -> float:
    return float(torch.sqrt(torch.mean(t * t) + 1e-12).item())


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

_DEDUP_COMPARE_SR = 11025
_DEDUP_MIN_GPU_VRAM_BYTES = 8 * 1024 ** 3
_DEDUP_GPU_STACK_VRAM_FRACTION = 0.40


def _dedup_gpu_stack_limit_bytes() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    vram = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    if vram < _DEDUP_MIN_GPU_VRAM_BYTES:
        return None
    return int(vram * _DEDUP_GPU_STACK_VRAM_FRACTION)


class _UnionFind:
    def __init__(self, n: int):
        self._p = list(range(n))

    def find(self, i: int) -> int:
        while self._p[i] != i:
            self._p[i] = self._p[self._p[i]]
            i = self._p[i]
        return i

    def union(self, i: int, j: int) -> None:
        self._p[self.find(i)] = self.find(j)

    def groups(self, items: list) -> dict[int, list]:
        result: dict[int, list] = {}
        for i, item in enumerate(items):
            result.setdefault(self.find(i), []).append(item)
        return result


def _dedup_pairs_union(items, uf, threshold, *, device='cpu', log_fn=None) -> None:
    n = len(items)
    use_cuda = device == 'cuda' and torch is not None and torch.cuda.is_available()
    gpu_stack = None
    lengths: list[int] = []
    if use_cuda:
        lengths = [a.shape[1] for _, a in items]
        max_len = max(lengths)
        n_ch = max(a.shape[0] for _, a in items)
        stack_bytes = n * n_ch * max_len * np.dtype(np.float32).itemsize
        stack_limit = _dedup_gpu_stack_limit_bytes()
        if stack_limit is None:
            use_cuda = False
            if log_fn:
                vram_gib = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
                log_fn(f"  [dedup] GPU has {vram_gib:.1f} GiB VRAM (< 8 GiB); using CPU compare for {n} stems")
        elif stack_bytes > stack_limit:
            use_cuda = False
            if log_fn:
                log_fn(f"  [dedup] GPU stack needs {stack_bytes / (1024 ** 3):.1f} GiB; budget {stack_limit / (1024 ** 3):.1f} GiB; using CPU compare for {n} stems")
        else:
            try:
                stack = np.zeros((n, n_ch, max_len), dtype=np.float32)
                for idx, (_, audio) in enumerate(items):
                    stack[idx, :audio.shape[0], :audio.shape[1]] = audio
                gpu_stack = torch.from_numpy(stack).to(device=device, dtype=torch.float32)
            except (MemoryError, RuntimeError):
                use_cuda = False
                gpu_stack = None
                if log_fn:
                    log_fn(f"  [dedup] GPU staging failed; using CPU compare for {n} stems")

    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            if use_cuda and gpu_stack is not None:
                length = min(lengths[i], lengths[j])
                ai = gpu_stack[i, :, :length]
                aj = gpu_stack[j, :, :length]
                diff = ai - aj
                denom = max(_tensor_rms(ai), _tensor_rms(aj), 1e-12)
                residual = _tensor_rms(diff) / denom
            else:
                ai, aj = items[i][1], items[j][1]
                length = min(ai.shape[1], aj.shape[1])
                denom = max(_rms(ai[:, :length]), _rms(aj[:, :length]), 1e-12)
                residual = _rms(ai[:, :length] - aj[:, :length]) / denom
            if residual < threshold:
                uf.union(i, j)

    if gpu_stack is not None:
        del gpu_stack
        if use_cuda:
            torch.cuda.empty_cache()


def find_duplicates(paths, sr: int, log_fn=None, threshold: float = 0.05, *, device: str = 'cpu'):
    if len(paths) < 2:
        return list(paths)
    compare_sr = min(sr, _DEDUP_COMPARE_SR)
    min_samples = compare_sr
    audios = {}
    for p in paths:
        try:
            audios[p] = load_audio(str(p), sr=compare_sr, ch=1)
        except Exception:
            pass
    items = [(p, a) for p, a in audios.items() if a.shape[1] >= min_samples]
    n = len(items)
    if n < 2:
        return list(paths)
    uf = _UnionFind(n)
    _dedup_pairs_union(items, uf, threshold, device=device, log_fn=log_fn)
    keep = []
    for grp in uf.groups(items).values():
        if len(grp) == 1:
            keep.append(grp[0][0])
            continue
        best_path, _ = min(grp, key=lambda pa: float(np.max(np.abs(pa[1]))))
        keep.append(best_path)
        if log_fn:
            others = [g[0].name for g in grp if g[0] != best_path]
            log_fn(f"  [dedup] kept {best_path.name}; removed duplicates: {', '.join(others)}")
    keep.extend(p for p in paths if p not in audios)
    return keep


# ---------------------------------------------------------------------------
# Classify (RMS) helpers
# ---------------------------------------------------------------------------

def prescan_stems(paths):
    issues = []
    ok = []
    for p in paths:
        path = Path(p)
        try:
            if path.stat().st_size == 0:
                issues.append((path, 'empty file (0 bytes)'))
                continue
        except OSError as exc:
            issues.append((path, f'unreadable: {exc}'))
            continue
        if path.suffix.lower() in SF_READ_EXTS and sf is not None:
            try:
                if sf.info(str(path)).frames == 0:
                    issues.append((path, 'empty audio (0 frames)'))
                    continue
            except Exception:
                pass
        ok.append(path)
    return issues, ok


def _load_classify_chunk(file_paths, sr: int):
    """Load a classify batch chunk. Returns (audios, lengths, valid, errors).

    errors: list of (path, err_str).
    """
    audios, lengths, valid = [], [], []
    errors = []
    for fp in file_paths:
        try:
            a = load_audio(str(fp), sr=sr)
        except Exception as e:
            errors.append((fp, f'load failed: {format_load_error(e)}'))
            continue
        n_samples = int(a.shape[-1])
        if n_samples == 0:
            errors.append((fp, 'empty audio (0 samples)'))
            continue
        audios.append(a)
        lengths.append(n_samples)
        valid.append(fp)
    return audios, lengths, valid, errors


def classify_batch(model, file_paths, device: str, batch_size: int = 4, stop_event=None):
    from concurrent.futures import ThreadPoolExecutor

    sr = model.samplerate
    sources = list(model.sources)
    onnx = _is_onnx_model(model)
    effective_bs = max(1, int(batch_size))
    if onnx:
        from demucs_onnx import demucs_max_batch

        effective_bs = max(1, min(effective_bs, demucs_max_batch()))
        # BS-RoFormer is not zero-pad invariant and runs STFT outside the graph:
        # stacking unequal stems to max_len wastes work and corrupts tails. Always
        # one file per forward (OLA still packs multiple time-offsets internally).
        if getattr(model, "sources", None) is not None and "guitar" in list(model.sources):
            effective_bs = 1

    starts = list(range(0, len(file_paths), effective_bs))
    # Prefetch next decode while Demucs runs (no quality change; overlaps I/O with GPU).
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_fut = None
        for i, start in enumerate(starts):
            if stop_event and stop_event.is_set():
                if not onnx and device == 'cuda':
                    model.cpu()
                    torch.cuda.empty_cache()
                return
            chunk = file_paths[start:start + effective_bs]
            if next_fut is not None:
                audios, lengths, valid, errors = next_fut.result()
                next_fut = None
            else:
                audios, lengths, valid, errors = _load_classify_chunk(chunk, sr)
            for fp, err in errors:
                yield (fp, None, err)
            if i + 1 < len(starts) and not (stop_event and stop_event.is_set()):
                next_chunk = file_paths[starts[i + 1]:starts[i + 1] + effective_bs]
                next_fut = pool.submit(_load_classify_chunk, next_chunk, sr)
            if not audios:
                continue
            max_len = max(lengths)
            batch = np.zeros((len(audios), 2, max_len), dtype=np.float32)
            for j, a in enumerate(audios):
                batch[j, :, :lengths[j]] = a
            try:
                out_np = _demucs_apply(model, batch, device)
            except Exception as e:
                if device == 'cuda' and _is_gpu_oom(e):
                    if not onnx and torch is not None:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    if len(valid) == 1:
                        yield (valid[0], None, 'cuda OOM')
                        continue
                    for fp in valid:
                        yield from classify_batch(model, [fp], device, batch_size=1)
                    continue
                if (
                    not onnx
                    and device == 'cuda'
                    and isinstance(e, RuntimeError)
                    and is_cuda_kernel_error(e)
                ):
                    torch.cuda.empty_cache()
                    for fp in valid:
                        yield (
                            fp,
                            None,
                            'CUDA kernels incompatible with this GPU (use cu128 for RTX 50-series)',
                        )
                    continue
                if isinstance(e, AssertionError):
                    if len(valid) == 1:
                        yield (valid[0], None, 'audio too short for model')
                        continue
                    for fp in valid:
                        yield from classify_batch(model, [fp], device, batch_size=1)
                    continue
                for fp in valid:
                    yield (fp, None, str(e))
                continue

            for j, fp in enumerate(valid):
                energies = {n: _rms(out_np[j, k, :, :lengths[j]]) for k, n in enumerate(sources)}
                yield (fp, energies, None)
            if not onnx and device == 'cuda':
                torch.cuda.empty_cache()


def classify_to_category(energies: dict, mode_cfg: dict, threshold: float, min_margin: float):
    total = sum(energies.values()) + 1e-12
    cat_shares = {c: 0.0 for c in mode_cfg['categories']}
    for src, e in energies.items():
        cat = mode_cfg['mapping'].get(src, mode_cfg['fallback'])
        if cat in cat_shares:
            cat_shares[cat] += e / total
    ranked = sorted(cat_shares, key=cat_shares.get, reverse=True)
    top_cat = ranked[0]
    runner_share = cat_shares[ranked[1]] if len(ranked) > 1 else 0.0
    top_share = cat_shares[top_cat]
    margin = top_share - runner_share
    if top_share < threshold and margin < min_margin:
        return "skip", top_cat, top_share, margin, 'both'
    if top_share < threshold:
        return "skip", top_cat, top_share, margin, 'confidence'
    if margin < min_margin:
        return "skip", top_cat, top_share, margin, 'margin'
    return (top_cat, top_cat, top_share, margin, None)


def mix_originals(paths, sr: int):
    tracks = []
    for p in paths:
        try:
            tracks.append(load_audio(str(p), sr=sr))
        except Exception:
            pass
    if not tracks:
        return np.zeros((2, 0), dtype=np.float32)
    cut = min(t.shape[1] for t in tracks)
    mixed = np.zeros((2, cut), dtype=np.float32)
    for t in tracks:
        mixed += t[:, :cut]
    return mixed


# ---------------------------------------------------------------------------
# SDR layout detection + helpers
# ---------------------------------------------------------------------------

def list_folder_sdr_audio(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in STEM_FILE_EXTS
    )


def _is_named_category_stem_file(path: Path, category: str) -> bool:
    """True when filename looks like this stem category (exact or keyword)."""
    if path.suffix.lower() not in STEM_FILE_EXTS:
        return False
    cat = category.lower()
    if cat == 'vocals':
        return is_vocals_only_stem_file(path)
    if cat == 'instrumental':
        return is_instrumental_only_stem_file(path)
    stem = path.stem.lower()
    if stem == cat or stem.endswith(f'_{cat}') or stem.endswith(f'-{cat}'):
        return True
    if f'({cat})' in stem:
        return True
    # Avoid loose 'other' / 'bass' substring hits on unrelated titles.
    return False


def _pick_best_category_match(matches: list[Path], category: str) -> Path:
    cat = category.lower()

    def score(path: Path) -> tuple:
        s = path.stem.lower()
        if s == cat:
            return (0, len(s), s)
        if s.endswith(f'_{cat}') or s.endswith(f'-{cat}'):
            return (1, len(s), s)
        if f'({cat})' in s:
            return (2, len(s), s)
        return (3, len(s), s)

    return sorted(matches, key=score)[0]


def find_category_stem(folder: Path, category: str) -> Path | None:
    """Exact ``{category}.ext`` first, then keyword / role names in the folder."""
    for ext in STEM_FILE_EXTS:
        p = folder / f'{category}{ext}'
        if p.is_file():
            return p
    matches = [
        f for f in list_folder_sdr_audio(folder)
        if _is_named_category_stem_file(f, category)
    ]
    if not matches:
        return None
    return _pick_best_category_match(matches, category)


def assign_two_file_pair(files: list[Path]) -> dict[str, Path] | None:
    """Map exactly two audio files to instrumental + vocals when possible."""
    if len(files) != 2:
        return None
    inst = voc = None
    for path in files:
        if is_instrumental_only_stem_file(path) and inst is None:
            inst = path
        elif is_vocals_only_stem_file(path) and voc is None:
            voc = path
    remaining = [p for p in files if p not in {inst, voc}]
    if inst is not None and voc is not None and inst != voc:
        return {'instrumental': inst, 'vocals': voc}
    if inst is not None and len(remaining) == 1:
        return {'instrumental': inst, 'vocals': remaining[0]}
    if voc is not None and len(remaining) == 1:
        return {'instrumental': remaining[0], 'vocals': voc}

    # Filename classifier (Align roles) when keywords miss both.
    try:
        from stem_align import classify_audio_file
    except Exception:
        classify_audio_file = None  # type: ignore
    if classify_audio_file is not None:
        roles: dict[str, Path] = {}
        for path in files:
            try:
                role = classify_audio_file(path)
            except Exception:
                role = None
            if role == 'instrumental' and 'instrumental' not in roles:
                roles['instrumental'] = path
            elif role == 'acapella' and 'vocals' not in roles:
                roles['vocals'] = path
        if len(roles) == 2:
            return roles
        if 'instrumental' in roles:
            other = next(p for p in files if p != roles['instrumental'])
            return {'instrumental': roles['instrumental'], 'vocals': other}
        if 'vocals' in roles:
            other = next(p for p in files if p != roles['vocals'])
            return {'instrumental': other, 'vocals': roles['vocals']}
    return None


def resolve_musdb_folder_stems(
    folder: Path,
    categories: tuple[str, ...],
    *,
    force_two_file: bool = False,
) -> dict[str, Path] | None:
    """Resolve stem paths in a song folder.

    For 2-stem (instrumental/vocals), any folder with exactly two audio files is
    treated as a pair — keywords assign roles when present; otherwise the other
    file is the complement, or size is used when both names are unmarked.
    ``force_two_file`` is kept for API compatibility (same behavior for 2-stem).
    """
    stems: dict[str, Path] = {}
    for cat in categories:
        p = find_category_stem(folder, cat)
        if p is not None and p not in stems.values():
            stems[cat] = p
    if all(c in stems for c in categories) and len(stems) == len(categories):
        return stems

    if set(categories) != {'instrumental', 'vocals'}:
        return None

    audio = list_folder_sdr_audio(folder)
    if len(audio) != 2:
        return None

    assigned = assign_two_file_pair(audio)
    if assigned is not None:
        return assigned

    # Both unmarked (e.g. track_a.flac + track_b.flac): still a pair.
    # Larger file → instrumental (typical stem dump); stable tie-break by name.
    try:
        a, b = sorted(audio, key=lambda p: (-p.stat().st_size, p.name.lower()))
    except OSError:
        a, b = sorted(audio, key=lambda p: p.name.lower())
    return {'instrumental': a, 'vocals': b}


def collect_sdr_song_folders(root: Path, scan_mode: str) -> list[Path]:
    if not root.is_dir():
        return []
    if scan_mode == 'subfolders':
        return sorted(p for p in root.iterdir() if p.is_dir())
    folders: set[Path] = set()
    for f in root.rglob('*'):
        if f.is_file() and f.suffix.lower() in STEM_FILE_EXTS:
            folders.add(f.parent)
    return sorted(folders)


def folder_has_all_stems(folder: Path, categories: tuple[str, ...]) -> bool:
    return resolve_musdb_folder_stems(folder, categories, force_two_file=False) is not None


def count_two_file_song_folders(root: Path, scan_mode: str) -> tuple[int, int, int]:
    """Return (two_file_total, keyword_complete, two_file_without_keywords)."""
    cats = SDR_MIXED_FLAT_CATEGORIES
    two_file = 0
    complete = 0
    unmatched_two = 0
    for folder in collect_sdr_song_folders(root, scan_mode):
        n = len(list_folder_sdr_audio(folder))
        has = folder_has_all_stems(folder, cats)
        if has:
            complete += 1
        if n == 2:
            two_file += 1
            if not has:
                unmatched_two += 1
    return two_file, complete, unmatched_two


def build_pair_folder_process_all_hint(root: Path, scan_mode: str) -> dict | None:
    """Ask when many 2-file song folders lack instrumental/vocals filename markers."""
    two_file, complete, unmatched = count_two_file_song_folders(root, scan_mode)
    if two_file < SDR_SINGLE_STEM_ASK_MIN_MATCHES:
        return None
    should_ask = (
        unmatched >= SDR_SINGLE_STEM_ASK_MIN_MATCHES
        or (complete == 0 and two_file >= SDR_SINGLE_STEM_ASK_MIN_MATCHES)
    )
    if not should_ask:
        return None
    return {
        'kind': 'pairs',
        'two_file': two_file,
        'complete': complete,
        'unmatched': unmatched if unmatched > 0 else two_file,
        'should_ask_process_all': True,
    }


def pair_folder_process_all_message(hint: dict) -> str:
    two_file = int(hint['two_file'])
    complete = int(hint['complete'])
    unmatched = int(hint['unmatched'])
    return (
        f'Found {two_file:,} song folder(s) with exactly 2 audio files.\n'
        f'{complete:,} already match instrumental + vocals by filename '
        f'(vocals / acapella / instrumental / …).\n'
        f'{unmatched:,} still need to be treated as pairs without clear names.\n\n'
        f'Process all {two_file:,} two-file folders as instrumental + vocals pairs?'
    )


SDR_LAYOUT_MUSDB = 'musdb'
SDR_LAYOUT_STEMS = 'stems'
SDR_LAYOUT_SINGLE_FLAT = 'single_flat'
SDR_LAYOUT_MIXED_FLAT = 'mixed_flat'

SDR_VOCALS_ONLY_CATEGORIES = ('vocals',)
SDR_INSTRUMENTAL_ONLY_CATEGORIES = ('instrumental',)
SDR_MIXED_FLAT_CATEGORIES = ('instrumental', 'vocals')

SDR_SINGLE_FLAT_LAYOUTS = frozenset({SDR_LAYOUT_SINGLE_FLAT, SDR_LAYOUT_MIXED_FLAT})

SDR_STEM_PICK_ORDER = ('instrumental', 'vocals', 'bass', 'drums', 'other')

SDR_SINGLE_STEM_ASK_MIN_MATCHES = 10

_SDR_NON_VOCAL_MARKERS = (
    '_instrumental', '-instrumental', '_drums', '-drums',
    '_bass', '-bass', '_other', '-other',
    '_no_vocals', '_novocal', '_no-vocals',
)

_SDR_NON_INSTRUMENTAL_MARKERS = (
    '_vocals', '-vocals', '_drums', '-drums',
    '_bass', '-bass', '_other', '-other',
    'acapella', 'acappella', 'a capella',
)


def is_sdr_audio_file(path: Path) -> bool:
    return path.suffix.lower() in STEM_FILE_EXTS


def iter_sdr_audio_files(root: Path, scan_mode: str):
    if not root.is_dir():
        return
    if scan_mode == 'subfolders':
        for f in root.iterdir():
            if f.is_file() and is_sdr_audio_file(f):
                yield f
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sub.iterdir():
                if f.is_file() and is_sdr_audio_file(f):
                    yield f
    else:
        for f in root.rglob('*'):
            if f.is_file() and is_sdr_audio_file(f):
                yield f


def is_vocals_only_stem_file(path: Path) -> bool:
    if path.suffix.lower() not in STEM_FILE_EXTS:
        return False
    stem = path.stem.lower()
    if any(m in stem for m in _SDR_NON_VOCAL_MARKERS):
        return False
    if 'instrumental' in stem or '(instrumental)' in stem or 'inst.' in stem or '-inst' in stem:
        return False
    if stem == 'vocals' or stem.endswith('_vocals') or stem.endswith('-vocals'):
        return True
    if stem == 'vocal' or stem.endswith('_vocal') or stem.endswith('-vocal'):
        return True
    if any(
        m in stem
        for m in (
            'acapella',
            'a capella',
            'acappella',
            'acapela',
            'accapella',
            '(vocals)',
            '(vocal)',
            '(acapella)',
        )
    ):
        return True
    if 'vocals' in stem or 'vocal' in stem:
        return True
    return False


def is_instrumental_only_stem_file(path: Path) -> bool:
    if path.suffix.lower() not in STEM_FILE_EXTS:
        return False
    stem = path.stem.lower()
    if any(m in stem for m in _SDR_NON_INSTRUMENTAL_MARKERS):
        return False
    if 'vocal' in stem and 'instrumental' not in stem and '(instrumental)' not in stem:
        return False
    if stem == 'instrumental' or stem.endswith('_instrumental') or stem.endswith('-instrumental'):
        return True
    if '(instrumental)' in stem:
        return True
    # Common typo: "(Instrumenta)l" — paren closed early, so "instrumental" never appears.
    if 'instrumenta' in stem:
        return True
    if 'inst.' in stem or '-inst' in stem:
        return True
    if 'instrumental' in stem:
        return True
    # Stem dumps often use "Beat" / "beat 1" instead of instrumental.
    if re.search(r'(^|[\s_\-\[(])beat(\s|$|[\d\)\]])', stem):
        return True
    return False


def _collect_single_stem_targets(root: Path, scan_mode: str, category: str, predicate):
    files = sorted(f for f in iter_sdr_audio_files(root, scan_mode) if predicate(f))
    return [{category: f} for f in files]


def collect_all_audio_targets(root: Path, scan_mode: str, category: str):
    files = sorted(iter_sdr_audio_files(root, scan_mode))
    return [{category: f} for f in files]


def sdr_single_flat_keyword_predicate(category: str):
    if category == 'vocals':
        return is_vocals_only_stem_file
    if category == 'instrumental':
        return is_instrumental_only_stem_file
    return None


def single_flat_category_patterns(category: str) -> str:
    return {
        'vocals': 'vocals, vocal, acapella, a capella, (vocals), …',
        'instrumental': '*_instrumental, instrumental, inst., -inst, (instrumental), …',
        'bass': '*_bass or bass in name',
        'drums': '*_drums or drums in name',
        'other': '*_other or other in name',
    }.get(category, category)


def build_single_stem_folder_hint(root: Path, scan_mode: str, category: str):
    predicate = sdr_single_flat_keyword_predicate(category)
    if predicate is None:
        return None
    patterns = single_flat_category_patterns(category)
    all_files = list(iter_sdr_audio_files(root, scan_mode))
    total = len(all_files)
    if total == 0:
        return None
    keyword_matches = sum(1 for f in all_files if predicate(f))
    unmatched = total - keyword_matches
    return {
        'kind': category,
        'patterns': patterns,
        'keyword_matches': keyword_matches,
        'total_audio': total,
        'should_ask_process_all': (keyword_matches >= SDR_SINGLE_STEM_ASK_MIN_MATCHES and unmatched > 0),
    }


def collect_sdr_process_all_preflight(
    root: Path,
    scan_mode: str,
    preferred_categories: tuple[str, ...],
) -> dict:
    """Filesystem-only SI-SDR process-all preflight (safe off the GUI thread).

    Walks the output tree to detect layout and gather prompt payloads. Does not
    mutate params or show dialogs — the UI applies those on the main thread.
    """
    _cats, layout = resolve_sdr_layout_and_categories(root, scan_mode, preferred_categories)
    pair_hint = None
    if set(preferred_categories) == {'instrumental', 'vocals'}:
        pair_hint = build_pair_folder_process_all_hint(root, scan_mode)
        if pair_hint and not pair_hint.get('should_ask_process_all'):
            pair_hint = None
    flat_hints: list[dict] = []
    if layout in (SDR_LAYOUT_SINGLE_FLAT, SDR_LAYOUT_MIXED_FLAT, None):
        for kind in ('instrumental', 'vocals'):
            hint = build_single_stem_folder_hint(root, scan_mode, kind)
            if hint and hint.get('should_ask_process_all'):
                flat_hints.append(hint)
    return {
        'layout': layout,
        'pair_hint': pair_hint,
        'flat_hints': flat_hints,
    }


def single_stem_process_all_message(hint: dict) -> str:
    kind = str(hint['kind'])
    return (
        f'Found {int(hint["keyword_matches"]):,} files with {kind} keywords\n'
        f'({hint["patterns"]})\n'
        f'out of {int(hint["total_audio"]):,} audio files in this folder.\n\n'
        f'Process all {int(hint["total_audio"]):,} files as {kind}?'
    )


def collect_vocals_only_targets(root: Path, scan_mode: str):
    return _collect_single_stem_targets(root, scan_mode, 'vocals', is_vocals_only_stem_file)


def collect_instrumental_only_targets(root: Path, scan_mode: str):
    return _collect_single_stem_targets(root, scan_mode, 'instrumental', is_instrumental_only_stem_file)


def collect_mixed_flat_targets(root: Path, scan_mode: str):
    targets: list[dict[str, Path]] = []
    for path in sorted(iter_sdr_audio_files(root, scan_mode)):
        is_vocals = is_vocals_only_stem_file(path)
        is_instrumental = is_instrumental_only_stem_file(path)
        if is_vocals and not is_instrumental:
            targets.append({'vocals': path})
        elif is_instrumental and not is_vocals:
            targets.append({'instrumental': path})
    return targets


def find_category_folder(root: Path, category: str) -> Path | None:
    direct = root / category
    if direct.is_dir():
        return direct
    cat_lower = category.lower()
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == cat_lower:
            return p
    return None


def stem_file_song_key(path: Path, category: str) -> str:
    stem = path.stem
    cat = category.lower()
    lower = stem.lower()
    for prefix in (f'{cat}_', f'{cat}-', f'{cat} '):
        if lower.startswith(prefix):
            return stem[len(prefix):]
    for suffix in (f'_{cat}', f'-{cat}', f' {cat}'):
        if lower.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def collect_sdr_stem_songs(root: Path, categories: tuple[str, ...]):
    if not root.is_dir():
        return []
    cat_dirs: dict[str, Path] = {}
    for cat in categories:
        folder = find_category_folder(root, cat)
        if folder is not None:
            cat_dirs[cat] = folder
    if len(cat_dirs) != len(categories):
        return []
    songs: dict[str, dict[str, Path]] = {}
    for cat, folder in cat_dirs.items():
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in STEM_FILE_EXTS:
                key = stem_file_song_key(f, cat)
                songs.setdefault(key, {})[cat] = f
    return [paths for paths in songs.values() if all(c in paths for c in categories)]


def detect_sdr_layout(root: Path, categories: tuple[str, ...], scan_mode: str) -> str | None:
    type1 = [f for f in collect_sdr_song_folders(root, scan_mode) if folder_has_all_stems(f, categories)]
    type2 = collect_sdr_stem_songs(root, categories)
    if type1 and not type2:
        return SDR_LAYOUT_MUSDB
    if type2 and not type1:
        return SDR_LAYOUT_STEMS
    if type1 and type2:
        return SDR_LAYOUT_MUSDB if len(type1) >= len(type2) else SDR_LAYOUT_STEMS
    return None


_SDR_CATEGORY_ALTERNATES: dict[tuple[str, ...], tuple[str, ...]] = {
    STEM_MODES['2 (instrumental/vocals)']['categories']: (
        STEM_MODES['4 (bass/drums/other/vocals)']['categories']
    ),
    STEM_MODES['4 (bass/drums/other/vocals)']['categories']: (
        STEM_MODES['2 (instrumental/vocals)']['categories']
    ),
    STEM_MODES['6 (b/d/o/v+guitar/piano)']['categories']: (
        STEM_MODES['4 (bass/drums/other/vocals)']['categories']
    ),
}

_ALL_SDR_CATEGORIES = ('bass', 'drums', 'other', 'vocals', 'instrumental', 'guitar', 'piano')


def resolve_sdr_layout_and_categories(root: Path, scan_mode: str, preferred_categories: tuple[str, ...]):
    layout = detect_sdr_layout(root, preferred_categories, scan_mode)
    if layout is not None:
        return preferred_categories, layout
    alt = _SDR_CATEGORY_ALTERNATES.get(preferred_categories)
    if alt is not None:
        layout = detect_sdr_layout(root, alt, scan_mode)
        if layout is not None:
            return alt, layout
    vocals = collect_vocals_only_targets(root, scan_mode)
    instrumental = collect_instrumental_only_targets(root, scan_mode)
    if vocals and not instrumental:
        return SDR_VOCALS_ONLY_CATEGORIES, SDR_LAYOUT_SINGLE_FLAT
    if instrumental and not vocals:
        return SDR_INSTRUMENTAL_ONLY_CATEGORIES, SDR_LAYOUT_SINGLE_FLAT
    if vocals and instrumental:
        return SDR_MIXED_FLAT_CATEGORIES, SDR_LAYOUT_MIXED_FLAT
    type1_vocals = [f for f in collect_sdr_song_folders(root, scan_mode) if folder_has_all_stems(f, SDR_VOCALS_ONLY_CATEGORIES)]
    if type1_vocals:
        return SDR_VOCALS_ONLY_CATEGORIES, SDR_LAYOUT_MUSDB
    type1_inst = [f for f in collect_sdr_song_folders(root, scan_mode) if folder_has_all_stems(f, SDR_INSTRUMENTAL_ONLY_CATEGORIES)]
    if type1_inst:
        return SDR_INSTRUMENTAL_ONLY_CATEGORIES, SDR_LAYOUT_MUSDB
    return None, None


def sdr_thresholds_for_categories(categories: tuple[str, ...], known: dict[str, float]):
    return {cat: float(known.get(cat, SDR_DEFAULT_THRESHOLDS.get(cat, 30))) for cat in categories}


def describe_sdr_scan_failure(root: Path, scan_mode: str, preferred_categories: tuple[str, ...]) -> str:
    folders = collect_sdr_song_folders(root, scan_mode)
    lines = ['No complete stem sets found.\n']
    if not folders:
        lines.append(f'No stem files ({SDR_STEM_EXT_LABEL}) found under:\n{root}')
        return '\n'.join(lines)
    lines.append(f'Scanned {len(folders)} folder(s) under:\n{root}\n')
    for label, cats in (
        ('2-stem (instrumental + vocals)', STEM_MODES['2 (instrumental/vocals)']['categories']),
        ('4-stem (bass/drums/other/vocals)', STEM_MODES['4 (bass/drums/other/vocals)']['categories']),
        ('6-stem (bass/drums/other/vocals/guitar/piano)', STEM_MODES['6 (b/d/o/v+guitar/piano)']['categories']),
    ):
        n = sum(1 for f in folders if folder_has_all_stems(f, cats))
        if n:
            lines.append(f'  • {n} folder(s) with complete {label} sets')
    n_vocals = len(collect_vocals_only_targets(root, scan_mode))
    n_inst = len(collect_instrumental_only_targets(root, scan_mode))
    n_audio = sum(1 for _ in iter_sdr_audio_files(root, scan_mode))
    if n_audio:
        lines.append(f'  • {n_audio:,} audio file(s) total ({SDR_STEM_EXT_LABEL})')
    if n_vocals:
        lines.append(f'  • {n_vocals:,} vocals keyword match(es) (*_vocals, acapella, or vocal in name)')
    if n_inst:
        lines.append(f'  • {n_inst:,} instrumental keyword match(es) (*_instrumental, instrumental, inst., -inst, or (instrumental) in name)')
    sample = folders[0]
    found = [c for c in _ALL_SDR_CATEGORIES if find_category_stem(sample, c)]
    if found:
        lines.append(f'\nExample ({sample.name}): {", ".join(found)}')
    missing = [c for c in preferred_categories if c not in found]
    if missing and found:
        lines.append(f'\nStem mode expects {", ".join(preferred_categories)} — missing in example: {", ".join(missing)}')
    expected = ', '.join(f'{c}{SDR_STEM_EXT_LABEL}' for c in preferred_categories)
    lines.append(
        f'\nType 1 (MUSDB): each song folder contains:\n{expected}\n\n'
        f'Type 2 (Stems): each stem has its own folder:\n'
        f'{", ".join(preferred_categories)}/'
    )
    return '\n'.join(lines)


def sdr_process_order(categories: tuple[str, ...]) -> tuple[str, ...]:
    if set(categories) == {'vocals', 'instrumental'}:
        return ('instrumental', 'vocals')
    return categories


def collect_sdr_targets(root: Path, categories: tuple[str, ...], scan_mode: str, layout: str, *, process_all: bool = False):
    if layout == SDR_LAYOUT_MIXED_FLAT:
        return collect_mixed_flat_targets(root, scan_mode)
    if layout == SDR_LAYOUT_SINGLE_FLAT:
        cat = categories[0]
        if process_all:
            return collect_all_audio_targets(root, scan_mode, cat)
        predicate = sdr_single_flat_keyword_predicate(cat)
        if predicate is not None:
            return _collect_single_stem_targets(root, scan_mode, cat, predicate)
        return collect_all_audio_targets(root, scan_mode, cat)
    if layout == SDR_LAYOUT_STEMS:
        return collect_sdr_stem_songs(root, categories)
    # MUSDB song folders
    force = bool(process_all) and set(categories) == {'instrumental', 'vocals'}
    out: list[Path] = []
    for folder in collect_sdr_song_folders(root, scan_mode):
        if resolve_musdb_folder_stems(folder, categories, force_two_file=force) is not None:
            out.append(folder)
    return out


def _instrumental_tag_categories(categories: tuple[str, ...]) -> tuple[str, ...]:
    if set(categories) == {'instrumental', 'vocals'}:
        return ('instrumental',)
    return tuple(c for c in categories if c != 'vocals')


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve(strict=False))
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def collect_instrumental_tag_paths(
    root: Path,
    categories: tuple[str, ...],
    scan_mode: str,
    layout: str,
    *,
    flat_process_all: bool = False,
    pair_process_all: bool = False,
) -> list[Path]:
    """Audio files for genre/style tagging (instrumental side of pairs, not vocals)."""
    process_all = bool(flat_process_all or pair_process_all)
    targets = collect_sdr_targets(
        root, categories, scan_mode, layout, process_all=process_all
    )
    inst_cats = _instrumental_tag_categories(categories)
    paths: list[Path] = []
    if layout == SDR_LAYOUT_MUSDB:
        for folder in targets:
            stems = resolve_musdb_folder_stems(
                folder, categories, force_two_file=pair_process_all
            )
            if not stems:
                continue
            for cat in inst_cats:
                if cat in stems:
                    paths.append(stems[cat])
    else:
        for target in targets:
            if not isinstance(target, dict):
                continue
            for cat in inst_cats:
                if cat in target:
                    paths.append(target[cat])
    return _dedupe_paths(paths)


def collect_vocal_tag_paths(
    root: Path,
    categories: tuple[str, ...],
    scan_mode: str,
    layout: str,
    *,
    flat_process_all: bool = False,
    pair_process_all: bool = False,
) -> list[Path]:
    """Audio files for gender / reverb / vocal-type tagging (vocals only)."""
    process_all = bool(flat_process_all or pair_process_all)
    targets = collect_sdr_targets(
        root, categories, scan_mode, layout, process_all=process_all
    )
    paths: list[Path] = []
    if layout == SDR_LAYOUT_MUSDB:
        for folder in targets:
            stems = resolve_musdb_folder_stems(
                folder, categories, force_two_file=pair_process_all
            )
            if stems and 'vocals' in stems:
                paths.append(stems['vocals'])
    else:
        for target in targets:
            if isinstance(target, dict) and 'vocals' in target:
                paths.append(target['vocals'])
    return _dedupe_paths(paths)


def describe_tag_layout_label(
    categories: tuple[str, ...],
    layout: str,
    *,
    user_picked_category: bool = False,
) -> str:
    if layout == SDR_LAYOUT_MUSDB:
        return 'Type 1 (song folders)'
    if layout == SDR_LAYOUT_STEMS:
        return 'Type 2 (stem category folders)'
    if layout == SDR_LAYOUT_SINGLE_FLAT:
        if user_picked_category:
            return f'Type 3 ({categories[0]}-only, user confirmed)'
        return f'Type 3 ({categories[0]}-only files)'
    if layout == SDR_LAYOUT_MIXED_FLAT:
        return 'Type 3 (mixed vocals/instrumental files)'
    return str(layout)


def _audio_to_mono(audio):
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[0] == 1:
            return a[0]
        return a.mean(axis=0)
    raise ValueError(f'unsupported audio shape: {a.shape}')


def compute_si_sdr(reference, estimate) -> float:
    delta = 1e-7
    ref = _audio_to_mono(reference)
    est = _audio_to_mono(estimate)
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = np.dot(ref, ref)
    if ref_energy < delta:
        return float(-np.inf)
    alpha = np.dot(est, ref) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + delta) / (np.sum(noise ** 2) + delta)))


def model_estimate_for_category(out_np, sources, category, mixture=None):
    if category in sources:
        return out_np[sources.index(category)]
    if category == 'instrumental':
        if mixture is not None and 'vocals' in sources:
            vocals = out_np[sources.index('vocals')]
            n = min(mixture.shape[1], vocals.shape[1])
            return mixture[:, :n] - vocals[:, :n]
        # Sum all non-vocal sources (handles 4-stem drums/bass/other and 6-stem
        # +guitar/+piano uniformly — never assumes a fixed source list).
        parts = [out_np[sources.index(s)] for s in sources if s != 'vocals']
        if not parts:
            raise ValueError('instrumental: no non-vocal sources in model')
        est = np.zeros_like(parts[0])
        for p in parts:
            est += p
        return est
    raise ValueError(f'unknown category: {category}')


def separate_mixture(model, mixture, device: str):
    mix = mixture.astype(np.float32)
    ref = mix.mean(axis=0)
    mean = float(ref.mean())
    std = float(ref.std()) + 1e-8
    normalized = (mix - mean) / std
    out_np = _demucs_apply(model, normalized[np.newaxis], device)
    return out_np[0] * std + mean


def separate_mixtures_batch(model, mixtures: list, device: str) -> list:
    """Per-stem mean/std normalize, one batched Demucs apply, denormalize.

    mixtures: list of (2, T) arrays (may differ in T — padded internally).
    Returns list of (S, 2, T_i) arrays matching each input length.
    On GPU OOM with B>1, falls back to per-stem separate_mixture.
    """
    if not mixtures:
        return []
    if len(mixtures) == 1:
        return [separate_mixture(model, mixtures[0], device)]

    norms: list[tuple[float, float]] = []
    arrays: list = []
    lengths: list[int] = []
    for mixture in mixtures:
        mix = mixture.astype(np.float32)
        ref = mix.mean(axis=0)
        mean = float(ref.mean())
        std = float(ref.std()) + 1e-8
        norms.append((mean, std))
        arrays.append((mix - mean) / std)
        lengths.append(int(mix.shape[1]))

    max_len = max(lengths)
    batch = np.zeros((len(arrays), 2, max_len), dtype=np.float32)
    for i, a in enumerate(arrays):
        batch[i, :, : lengths[i]] = a

    try:
        out_np = _demucs_apply(model, batch, device)
    except Exception as e:
        if device == "cuda" and _is_gpu_oom(e) and len(mixtures) > 1:
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return [separate_mixture(model, m, device) for m in mixtures]
        raise

    results = []
    for i, (mean, std) in enumerate(norms):
        results.append(out_np[i, :, :, : lengths[i]] * std + mean)
    return results


def _play_done_sound() -> None:
    from done_sound import play_done_sound
    play_done_sound()


# ---------------------------------------------------------------------------
# Outcome / skip labels
# ---------------------------------------------------------------------------

def _skip_reason_label(reason: str) -> str:
    return {
        'confidence': "didn't match confidence",
        'margin': "didn't match min. margin",
        'both': "didn't match confidence + min. margin",
        'error': 'error',
    }.get(reason, reason)


def format_skip_summary(counts: dict[str, int]) -> str:
    active = [(k, counts[k]) for k in ('confidence', 'margin', 'both', 'error') if counts.get(k)]
    total = sum(counts.values())
    if not total:
        return ''
    if len(active) == 1:
        detail = _skip_reason_label(active[0][0])
    else:
        detail = ', '.join(f"{n} {_skip_reason_label(k)}" for k, n in active)
    return f"  ({total} stem(s) skipped: {detail})"


FOLDER_OUTCOME_LABELS = {
    'success': 'Successful',
    'success_with_errors': 'Successful (with errors)',
    'skip_existing': 'Skipped - output already exists',
    'skip_song': 'Skipped - ambiguous song',
    'skip_no_stems': 'Skipped - no stems to export',
    'deleted_short': 'Deleted - shorter than minimum duration',
    'deleted_incomplete': 'Deleted - missing expected stem(s)',
    'deleted_both': 'Deleted - too short and incomplete',
    'delete_failed': 'Delete failed',
}

FOLDER_OUTCOME_ORDER = (
    'success', 'success_with_errors', 'skip_existing', 'skip_song', 'skip_no_stems',
    'deleted_short', 'deleted_incomplete', 'deleted_both', 'delete_failed',
)


# ---------------------------------------------------------------------------
# Worker classes — verbatim port (threading.Thread, daemon=True).
# Callers can wrap these in a QThread adapter, see stem_organizer.workers.classify_worker.
# ---------------------------------------------------------------------------

class Worker(threading.Thread):
    def __init__(self, params: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.p = params
        self.q = log_q
        self._stop_event = threading.Event()
        self._total_stems = 0
        self._completed_stems = 0
        self._run_started_at = 0.0
        self._stats: dict = {}
        self._phase_timer = PhaseTimer()

    def _reset_stats(self, folders_total: int) -> None:
        self._stats = {
            'folders_total': folders_total,
            'folder_outcomes': {},
            'stems_skipped': {'confidence': 0, 'margin': 0, 'both': 0, 'error': 0},
            'folder_names': {
                'deleted_short': [],
                'deleted_incomplete': [],
                'deleted_both': [],
            },
            'stem_skip_details': [],
        }

    @staticmethod
    def _folder_display(rel: Path) -> str:
        s = str(rel).replace('\\', '/')
        return s if s and s != '.' else '.'

    def _record_folder_outcome(self, outcome: str, folder_name: str | None = None) -> None:
        oc = self._stats['folder_outcomes']
        oc[outcome] = oc.get(outcome, 0) + 1
        if not folder_name:
            return
        names = self._stats['folder_names']
        if outcome == 'deleted_both':
            names['deleted_short'].append(folder_name)
            names['deleted_incomplete'].append(folder_name)
            names['deleted_both'].append(folder_name)
        elif outcome in names:
            names[outcome].append(folder_name)

    def _record_stem_skip(self, folder_rel: Path, stem_path: Path, reason: str) -> None:
        self._stats['stems_skipped'][reason] = self._stats['stems_skipped'].get(reason, 0) + 1
        folder = self._folder_display(folder_rel)
        stem = stem_path.name
        self._stats['stem_skip_details'].append({'folder': folder, 'stem': stem, 'reason': reason})

    def _log_run_summary(self, elapsed: float) -> None:
        oc = self._stats['folder_outcomes']
        folders_total = self._stats['folders_total']
        processed = sum(oc.values())
        self.log('')
        self.log('=== RMS Summary ===')
        self.log(f'  Total time: {format_elapsed(elapsed)}')
        if folders_total:
            self.log(f'  Avg per folder: {format_elapsed(elapsed / folders_total)} ({folders_total} folder(s))')
        for key in FOLDER_OUTCOME_ORDER:
            if key in ('deleted_short', 'deleted_incomplete', 'deleted_both'):
                continue
            n = oc.get(key, 0)
            if n:
                self.log(f'  {FOLDER_OUTCOME_LABELS[key]}: {n}')
        names = self._stats['folder_names']
        short_names = names['deleted_short']
        if short_names:
            self.log(f'  {FOLDER_OUTCOME_LABELS["deleted_short"]}: {len(short_names)}')
            for name in short_names:
                self.log(f'    {name}')
        incomplete_names = names['deleted_incomplete']
        if incomplete_names:
            self.log(f'  {FOLDER_OUTCOME_LABELS["deleted_incomplete"]}: {len(incomplete_names)}')
            for name in incomplete_names:
                self.log(f'    {name}')
        if processed < folders_total:
            self.log(f'  Not processed: {folders_total - processed}')
        details = self._stats['stem_skip_details']
        if details:
            self.log('  Stems skipped during classification:')
            by_reason: dict[str, list[str]] = {}
            for item in details:
                folder, stem = item['folder'], item['stem']
                label = stem if folder == '.' else f'{folder}/{stem}'
                by_reason.setdefault(item['reason'], []).append(label)
            for reason in ('confidence', 'margin', 'both', 'error'):
                items = by_reason.get(reason, [])
                if items:
                    self.log(f'    {_skip_reason_label(reason)}: {len(items)}')
                    for label in items:
                        self.log(f'      {label}')
        self._phase_timer.log_summary(self.log, ORGANIZE_PHASE_LABELS)
        self.log('')
        self.log('DONE')

    def stop(self):
        self._stop_event.set()

    def log(self, msg: str):
        self.q.put(msg)

    def _report_progress(self) -> None:
        total = self._total_stems
        done = self._completed_stems
        if total <= 0:
            pct, eta = 0.0, None
        else:
            pct = min(100.0, done / total * 100.0)
            elapsed = time.monotonic() - self._run_started_at
            eta = (elapsed / done * (total - done)) if done > 0 else None
        self.q.put((PROGRESS_TAG, pct, eta))

    def _mark_stems_done(self, count: int) -> None:
        if count <= 0:
            return
        self._completed_stems = min(self._total_stems, self._completed_stems + count)
        self._report_progress()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log(f'[ERROR] {e}')
            self.log(traceback.format_exc())
        finally:
            self.q.put(DONE_SENTINEL)

    def _resolve_output_dir(self, out_dir: Path, rel: Path, manifest: dict, next_n_ref: list, duration_sec: float | None = None):
        append_duration = self.p.get('append_duration', False)
        naming_mode = self.p['naming_mode']
        if naming_mode == 'sequential':
            name = folder_name_with_duration(f"song_{next_n_ref[0]:04d}", duration_sec, append_duration)
            target_dir = out_dir / name
            manifest[name] = str(rel).replace('\\', '/')
            save_manifest(out_dir, manifest)
            next_n_ref[0] += 1
            self.log(f"  -> {name}  (original: {rel})")
        elif naming_mode == 'preserve':
            if str(rel) == '.':
                target_dir = out_dir
                display = '.'
            else:
                parts = list(rel.parts)
                parts[-1] = folder_name_with_duration(parts[-1], duration_sec, append_duration)
                target_dir = out_dir.joinpath(*parts)
                display = Path(*parts)
            self.log(f"  -> {display}")
        else:
            slug_parts = [slugify(pp) for pp in rel.parts if pp not in ('', '.')]
            if slug_parts:
                slug_parts[-1] = folder_name_with_duration(slug_parts[-1], duration_sec, append_duration)
            target_dir = out_dir.joinpath(*slug_parts) if slug_parts else out_dir
            self.log(f"  -> {Path(*slug_parts) if slug_parts else '.'}")
        return target_dir, manifest, next_n_ref[0]

    def _compute_gain(self, mixes: dict, cut: int) -> float:
        if not self.p['peak_norm'] or cut == 0:
            return 1.0
        total = sum(m[:, :cut] for m in mixes.values())
        peak = float(np.max(np.abs(total)))
        target_lin = 10 ** (-1.0 / 20.0)
        return target_lin / peak if peak > 0 else 1.0

    def _write_category_mixes(self, mixes, buckets, mode_cfg, target_dir, ext, subtype, sr, gain, cut):
        written: set[str] = set()
        had_errors = False
        for cat in mode_cfg['categories']:
            if cat not in mixes:
                self.log(f"  ({cat}: no stems)")
                continue
            scaled = mixes[cat][:, :cut] * gain
            out_path = target_dir / f"{cat}{ext}"
            try:
                write_audio(str(out_path), scaled, sr, subtype)
                written.add(cat)
                self.log(f"  wrote {cat}{ext}  ({len(buckets[cat])} stems, {format_duration_log(cut / sr)})")
            except Exception as e:
                had_errors = True
                self.log(f"  [export error] {cat}: {e}")
        return written, had_errors

    def _maybe_cleanup_output_folder(self, target_dir, duration_sec, written_cats, mode_cfg, manifest, out_dir):
        reasons: list[str] = []
        short = False
        incomplete = False
        if self.p.get('delete_if_short'):
            min_sec = float(self.p.get('min_duration_sec', 8))
            if duration_sec < min_sec:
                short = True
                reasons.append(f'duration {format_duration(duration_sec)} < {format_duration(min_sec)}')
        if self.p.get('delete_if_incomplete'):
            expected = set(mode_cfg['categories'])
            missing = sorted(expected - written_cats)
            if missing:
                incomplete = True
                reasons.append(f'missing: {", ".join(missing)}')
        if not reasons:
            return manifest, None
        folder_name = target_dir.name
        if target_dir.exists():
            try:
                send_to_recycle_bin(target_dir)
                self.log(f"  [deleted] {folder_name}: {'; '.join(reasons)}")
            except OSError as e:
                self.log(f"  [delete error] {folder_name}: {e}")
                return manifest, 'delete_failed'
        if self.p['naming_mode'] == 'sequential' and folder_name in manifest:
            del manifest[folder_name]
            save_manifest(out_dir, manifest)
        if short and incomplete:
            return manifest, 'deleted_both'
        if short:
            return manifest, 'deleted_short'
        return manifest, 'deleted_incomplete'

    def _process_folder(self, folder, stems, model, device, mode_cfg, ext, subtype, sr, out_dir, manifest, next_n_ref):
        in_dir = Path(self.p['input_dir'])
        rel = folder.relative_to(in_dir) if folder != in_dir else Path('.')
        if self.p.get('skip_existing'):
            existing = find_existing_output_dir(out_dir, rel, self.p['naming_mode'], mode_cfg['categories'], ext, manifest)
            if existing:
                try:
                    display = existing.relative_to(out_dir)
                except ValueError:
                    display = existing
                self.log(f"  [skip existing] {display}")
                self._mark_stems_done(len(stems))
                self._record_folder_outcome('skip_existing')
                return manifest, next_n_ref[0]
        if self.p['dedup']:
            self.log('Starting de-duping...')
            t0 = time.monotonic()
            before = len(stems)
            stems = find_duplicates(stems, sr=sr, log_fn=self.log, device=device)
            dedup_dt = time.monotonic() - t0
            self._phase_timer.add('dedup', dedup_dt)
            if len(stems) < before:
                self.log(f"  [dedup] {before} -> {len(stems)} stems after deduplication")

        buckets = {c: [] for c in mode_cfg['categories']}
        skipped = 0
        had_ambig = False
        folder_had_errors = False
        skip_reasons: dict[str, int] = {'confidence': 0, 'margin': 0, 'both': 0, 'error': 0}

        self.log('Scanning stems...')
        t0 = time.monotonic()
        prescan_issues, stems = prescan_stems(stems)
        prescan_dt = time.monotonic() - t0
        self._phase_timer.add('prescan', prescan_dt)
        for path, reason in prescan_issues:
            tag = '[empty]' if reason.startswith('empty') else '[skip]'
            self.log(f"  {tag} {path.name} — {reason}")
            self._mark_stems_done(1)
            skipped += 1
            skip_reasons['error'] += 1
            folder_had_errors = True
            self._record_stem_skip(rel, path, 'error')

        self.log('Starting RMS classification...')
        t0 = time.monotonic()
        for path, energies, err in classify_batch(model, stems, device, batch_size=int(self.p['batch_size']), stop_event=self._stop_event):
            self._mark_stems_done(1)
            if self._stop_event.is_set():
                # Outer folder loop logs "Stopped by user." once
                _release_gpu(model, device)
                return manifest, next_n_ref[0]
            if err:
                self.log(f"  [skip] {path.name}: {err}")
                skipped += 1
                skip_reasons['error'] += 1
                folder_had_errors = True
                self._record_stem_skip(rel, path, 'error')
                continue
            label, _, top_share, _margin, skip_reason = classify_to_category(
                energies, mode_cfg, float(self.p['threshold']), float(self.p['min_margin']))
            self.log(f"  {label} {top_share:.0%}  →  {path.name}")
            if label == 'skip':
                skipped += 1
                had_ambig = True
                if skip_reason:
                    skip_reasons[skip_reason] += 1
                    self._record_stem_skip(rel, path, skip_reason)
            else:
                buckets[label].append(path)
        class_dt = time.monotonic() - t0
        self._phase_timer.add('classification', class_dt)

        if had_ambig and self.p['ambig_mode'] == 'skip_song':
            self.log('  [skip song] ambiguous stem(s) detected; skipping entire song')
            self._record_folder_outcome('skip_song')
            return manifest, next_n_ref[0]

        t0 = time.monotonic()
        mixes = {}
        for cat, paths in buckets.items():
            if not paths:
                continue
            m = mix_originals(paths, sr=sr)
            if m.shape[1] == 0:
                self.log(f"  [error] {cat}: all stems failed to load")
                folder_had_errors = True
                continue
            mixes[cat] = m
        mix_dt = time.monotonic() - t0
        if mixes:
            self._phase_timer.add('mixing', mix_dt)

        cut = min((m.shape[1] for m in mixes.values()), default=0)
        if cut == 0:
            self.log('  [skip] no stems to export')
            self._record_folder_outcome('skip_no_stems')
            return manifest, next_n_ref[0]

        duration_sec = cut / sr
        target_dir, manifest, _ = self._resolve_output_dir(out_dir, rel, manifest, next_n_ref, duration_sec=duration_sec)
        self.log('Starting to write output...')
        t0 = time.monotonic()
        target_dir.mkdir(parents=True, exist_ok=True)
        gain = self._compute_gain(mixes, cut)

        written_cats, export_errors = self._write_category_mixes(mixes, buckets, mode_cfg, target_dir, ext, subtype, sr, gain, cut)
        folder_had_errors = folder_had_errors or export_errors

        if self.p['make_mixture'] and ext == '.wav' and cut > 0:
            total = sum(m[:, :cut] for m in mixes.values()) * gain
            mix_path = target_dir / f"mixture{ext}"
            try:
                write_audio(str(mix_path), total, sr, subtype)
                n = sum(len(buckets[c]) for c in mixes)
                peak_db = 20 * np.log10(max(float(np.max(np.abs(total))), 1e-12))
                self.log(f"  wrote mixture{ext}  ({n} stems, peak {peak_db:+.2f} dBFS)")
            except Exception as e:
                folder_had_errors = True
                self.log(f"  [export error] mixture: {e}")
        export_dt = time.monotonic() - t0
        self._phase_timer.add('export', export_dt)

        manifest, delete_outcome = self._maybe_cleanup_output_folder(target_dir, duration_sec, written_cats, mode_cfg, manifest, out_dir)
        folder_name = self._folder_display(rel)
        if delete_outcome:
            self._record_folder_outcome(delete_outcome, folder_name)
        elif folder_had_errors:
            self._record_folder_outcome('success_with_errors')
        else:
            self._record_folder_outcome('success')

        if skipped:
            summary = format_skip_summary(skip_reasons)
            if summary:
                self.log(summary)

        return manifest, next_n_ref[0]

    def _run(self):
        model = None
        device = 'cpu'
        self._phase_timer = PhaseTimer()
        try:
            p = self.p
            # Log before CUDA probe / weight load so the UI isn't blank.
            self.log(f"  Preparing '{p['model_id']}' …")
            device, device_warnings = resolve_processing_device(bool(p['use_cuda']))
            for warning in device_warnings:
                self.log(f'  [warn] {warning}')
            if FFMPEG:
                from ffmpeg_bootstrap import ffmpeg_folder_path
                self.log(f'  ffmpeg: {ffmpeg_folder_path()}')
            else:
                from ffmpeg_bootstrap import ffmpeg_missing_message
                self.log(f'  [warn] {ffmpeg_missing_message()}')
            self.log(f"  Device: {device}")
            from deps_bootstrap import demucs_model_cached

            if not demucs_model_cached(p['model_id']):
                mid = str(p.get('model_id') or '')
                if mid == 'vocal_cnn6':
                    self.log(
                        '  [warn] Vocal CNN6 ONNX weight missing (~24 MB). '
                        "Re-run the installer's models component, or place "
                        'models/vocal_classifier.onnx beside the app.'
                    )
                else:
                    self.log(
                        '  [warn] HTDemucs ONNX weight missing (~302 MB). '
                        "Re-run the installer's models component, or place "
                        'models/htdemucs.onnx beside the app.'
                    )
            self.log(f"  Loading model '{p['model_id']}' ...")
            t0 = time.monotonic()
            model = load_demucs_model(
                p['model_id'], prefer_gpu=(device != 'cpu')
            ).eval().to(device)
            # Yield GIL so queued "Loading…" logs / Qt UI can breathe.
            time.sleep(0)
            model_load_dt = time.monotonic() - t0
            self._phase_timer.add('model_load', model_load_dt)
            backend = getattr(model, 'backend', 'torch')
            self.log(
                f"  Model sources: {list(model.sources)}  "
                f"(sr={model.samplerate}, backend={backend})"
            )

            in_dir, out_dir = Path(p['input_dir']), Path(p['output_dir'])
            mode_cfg = STEM_MODES[resolve_stem_mode(p['stem_mode'])]
            ext, subtype = QUALITY_PRESETS[p['quality']].values()
            sr = model.samplerate

            self.log('  Scanning input folders...')
            t0 = time.monotonic()
            groups = collect_song_groups(in_dir, p['scan_mode'])
            input_scan_dt = time.monotonic() - t0
            self._phase_timer.add('input_scan', input_scan_dt)

            if not groups:
                self.log(f"[skip] no audio files found under {in_dir}")
                return

            self._total_stems = sum(len(v) for v in groups.values())
            self._completed_stems = 0
            self._run_started_at = time.monotonic()
            self._reset_stats(len(groups))
            self._report_progress()
            self.log(f"  Found {sum(len(v) for v in groups.values())} stem(s) across {len(groups)} folder(s).")

            manifest: dict = {}
            next_n_ref = [0]
            if p['naming_mode'] == 'sequential' or p.get('skip_existing'):
                manifest = load_manifest(out_dir)
            if p['naming_mode'] == 'sequential':
                next_n_ref[0] = next_sequence_number(out_dir, manifest)
                self.log(f"  Naming: sequential; resuming at song_{next_n_ref[0]:04d}")
            elif p['naming_mode'] == 'preserve':
                self.log('  Naming: original folder name')
            else:
                self.log('  Naming: simplified folder name')

            scan_label = next(k for k, v in SCAN_MODES.items() if v == p['scan_mode'])
            self.log(f"  Scan: {scan_label}")
            if p.get('skip_existing'):
                self.log('  Resume: skipping songs that already have output stems')

            for fi, (folder, stems) in enumerate(sorted(groups.items()), 1):
                if self._stop_event.is_set():
                    self.log('Stopped by user.')
                    _release_gpu(model, device)
                    return
                rel = folder.relative_to(in_dir) if folder != in_dir else Path('.')
                self.log('')
                self.log(f"=== [{fi}/{len(groups)}] {rel}  ({len(stems)} stems) ===")
                manifest, next_n_ref[0] = self._process_folder(folder, stems, model, device, mode_cfg, ext, subtype, sr, out_dir, manifest, next_n_ref)
                if self._stop_event.is_set():
                    self.log('Stopped by user.')
                    _release_gpu(model, device)
                    return

            elapsed = time.monotonic() - self._run_started_at
            self._log_run_summary(elapsed)
            self._completed_stems = self._total_stems
            self._report_progress()
        finally:
            if model is not None:
                _release_gpu(model, device)


class SdrWorker(threading.Thread):
    def __init__(self, params: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.p = params
        self.q = log_q
        self._stop_event = threading.Event()
        self._total_folders = 0
        self._completed_folders = 0
        self._run_started_at = 0.0
        self._stats: dict = {}
        self._phase_timer = PhaseTimer()

    def _reset_stats(self, folders_total: int) -> None:
        self._stats = {
            'folders_total': folders_total,
            'passed': 0,
            'skipped_incomplete': 0,
            'skipped_existing': 0,
            'deleted_whole_folder': [],
            'deleted_stem_files': [],
        }

    def stop(self):
        self._stop_event.set()

    def log(self, msg):
        self.q.put(msg)

    def _log_sdr_line(self, filename: str, score: float, threshold: float):
        self.q.put((SDR_LOG_TAG, filename, score, threshold))

    def _report_progress(self) -> None:
        total = self._total_folders
        done = self._completed_folders
        if total <= 0:
            pct, eta = 0.0, None
        else:
            pct = min(100.0, done / total * 100.0)
            elapsed = time.monotonic() - self._run_started_at
            eta = (elapsed / done * (total - done)) if done > 0 else None
        self.q.put((PROGRESS_TAG, pct, eta))

    def _mark_folder_done(self) -> None:
        self._completed_folders = min(self._total_folders, self._completed_folders + 1)
        self._report_progress()

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log(f'[ERROR] {e}')
            self.log(traceback.format_exc())
        finally:
            self.q.put(DONE_SENTINEL)

    def _folder_display(self, folder: Path, root: Path) -> str:
        try:
            rel = folder.relative_to(root)
            s = str(rel).replace('\\', '/')
            return s if s and s != '.' else folder.name
        except ValueError:
            return folder.name

    def _cleanup_empty_dirs_after_delete(self, deleted_paths: list[Path]) -> None:
        root = getattr(self, '_sdr_root', None)
        if root is None or not deleted_paths:
            return
        cleanup_empty_dirs_after_delete(deleted_paths, root)

    def _prune_empty_sdr_dirs(self) -> None:
        root = getattr(self, '_sdr_root', None)
        if root is None:
            return
        pruned = prune_empty_dirs_under(root)
        if pruned:
            self.log(f'  [cleanup] Removed {len(pruned)} empty folder(s).')

    def _log_sdr_summary(self, elapsed: float) -> None:
        st = self._stats
        self.log('')
        self.log('=== SI-SDR Summary ===')
        self.log(f'  Total time: {format_elapsed(elapsed)}')
        if st['folders_total']:
            self.log(f'  Avg per folder: {format_elapsed(elapsed / st["folders_total"])} ({st["folders_total"]} folder(s))')
        if st['passed']:
            self.log(f'  Passed: {st["passed"]}')
        if st.get('skipped_existing'):
            self.log(f'  Skipped (SDR tag already exists): {st["skipped_existing"]}')
        if st['skipped_incomplete']:
            self.log(f'  Skipped (incomplete stems): {st["skipped_incomplete"]}')
        whole = st['deleted_whole_folder']
        if whole:
            self.log(f'  Deleted (whole folder): {len(whole)}')
            for name in whole:
                self.log(f'    {name}')
        stems = st['deleted_stem_files']
        if stems:
            self.log(f'  Deleted (stem file): {len(stems)}')
            for name in stems:
                self.log(f'    {name}')
        self._phase_timer.log_summary(
            self.log, sdr_phase_labels(self.p.get('model_id')),
        )
        self.log('')
        self.log('DONE')

    def _process_song(self, stem_paths, display, categories, thresholds, model, device, sources, sr, fi, total, *, delete_whole=None, layout=SDR_LAYOUT_MUSDB):
        self.log('')
        self.log(f'=== [{fi:02d}/{total}] {display} ===')
        missing = [cat for cat in categories if cat not in stem_paths]
        if missing:
            self.log(f'  [skip] missing expected stem(s): {", ".join(missing)}')
            self._stats['skipped_incomplete'] += 1
            return

        # Default ON (no GUI toggle): skip songs whose stems already have SDR tags.
        if self.p.get('sdr_skip_existing', True):
            try:
                from stem_organizer.meta_tags import read_custom_tag
            except Exception:
                read_custom_tag = None  # type: ignore[assignment]
            if read_custom_tag is not None:
                def _has_sdr(path: Path) -> bool:
                    raw = read_custom_tag(path, "SDR").strip()
                    if not raw:
                        return False
                    try:
                        float(raw.replace(",", ".").split()[0])
                        return True
                    except ValueError:
                        return False

                if all(_has_sdr(stem_paths[cat]) for cat in categories):
                    self.log('  [skip] SDR tag already exists')
                    self._stats['skipped_existing'] += 1
                    return

        audios: dict = {}
        t0 = time.monotonic()
        for cat, path in stem_paths.items():
            try:
                audios[cat] = load_audio(str(path), sr=sr)
            except Exception as e:
                self.log(f'  [skip] {path.name}: load failed: {format_load_error(e)}')
                self._stats['skipped_incomplete'] += 1
                return
        load_dt = time.monotonic() - t0
        self._phase_timer.add('audio_load', load_dt)

        cut = min(a.shape[1] for a in audios.values())
        if cut == 0:
            self.log('  [skip] empty audio')
            self._stats['skipped_incomplete'] += 1
            return

        scores: dict[str, float] = {}
        sep_dt = 0.0
        sdr_dt = 0.0
        order = list(sdr_process_order(categories))
        refs = [audios[cat][:, :cut] for cat in order]
        t0 = time.monotonic()
        try:
            outs = separate_mixtures_batch(model, refs, device)
        except Exception as e:
            if device == 'cuda' and _is_gpu_oom(e):
                if torch is not None:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                self.log('  [error] cuda OOM during separation')
                self._stats['skipped_incomplete'] += 1
                return
            self.log(f'  [error] separation failed: {e}')
            self._stats['skipped_incomplete'] += 1
            return
        sep_dt += time.monotonic() - t0
        if device == 'cuda' and torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        for cat, out_np, ref in zip(order, outs, refs):
            t0 = time.monotonic()
            try:
                est = model_estimate_for_category(out_np[:, :, :cut], sources, cat, mixture=ref)
            except ValueError as e:
                self.log(f'  [error] {cat}: {e}')
                self._stats['skipped_incomplete'] += 1
                return
            scores[cat] = compute_si_sdr(ref, est)
            sdr_dt += time.monotonic() - t0
            self._log_sdr_line(stem_paths[cat].name, scores[cat], thresholds[cat])
            if self.p.get('write_sdr_tags'):
                try:
                    from stem_organizer.meta_tags import write_sdr_tag

                    if not write_sdr_tag(stem_paths[cat], scores[cat]):
                        self.log(f'  [warn] could not write SDR tag: {stem_paths[cat].name}')
                except Exception as tag_exc:
                    self.log(f'  [warn] SDR tag failed ({stem_paths[cat].name}): {tag_exc}')
        self._phase_timer.add('separation', sep_dt)
        self._phase_timer.add('sdr_compute', sdr_dt)

        failed = [(cat, stem_paths[cat], scores[cat]) for cat in categories if scores[cat] < thresholds[cat]]
        if not failed:
            self._stats['passed'] += 1
            return

        delete_folder = self.p.get('sdr_delete_folder', True)
        deleted_paths: list[Path] = []
        if delete_folder:
            try:
                if layout == SDR_LAYOUT_MUSDB and delete_whole is not None:
                    send_to_recycle_bin(delete_whole)
                    deleted_paths.append(delete_whole)
                else:
                    for path in stem_paths.values():
                        send_to_recycle_bin(path)
                        deleted_paths.append(path)
                names = ', '.join(p.name for _, p, _ in failed)
                self.log(f'[deleted] {display}: Threshold not met: {names}')
                if layout == SDR_LAYOUT_MUSDB and delete_whole is not None:
                    self._stats['deleted_whole_folder'].append(display)
                elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                    self._stats['deleted_stem_files'].extend(p.name for _, p, _ in failed)
                else:
                    self._stats['deleted_whole_folder'].append(display)
            except OSError as e:
                self.log(f'[delete error] {display}: {e}')
            else:
                self._cleanup_empty_dirs_after_delete(deleted_paths)
        else:
            for cat, path, _score in failed:
                try:
                    send_to_recycle_bin(path)
                    deleted_paths.append(path)
                    if layout == SDR_LAYOUT_STEMS:
                        loc = f'{path.parent.name}/{path.name}'
                    elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                        loc = path.name
                    else:
                        loc = f'{display}\\{path.name}'
                    self.log(f'[deleted] {loc}: Threshold not met.')
                    self._stats['deleted_stem_files'].append(loc)
                except OSError as e:
                    self.log(f'[delete error] {path}: {e}')
            if deleted_paths:
                self._cleanup_empty_dirs_after_delete(deleted_paths)

    def _process_folder(self, folder, root, categories, thresholds, model, device, sources, sr, fi, total, *, force_two_file=False):
        display = self._folder_display(folder, root)
        stem_paths = resolve_musdb_folder_stems(folder, categories, force_two_file=force_two_file)
        if stem_paths is None:
            missing = [c for c in categories if find_category_stem(folder, c) is None]
            self.log('')
            self.log(f'=== [{fi:02d}/{total}] {display} ===')
            self.log(f'  [skip] missing expected stem(s): {", ".join(missing) or ", ".join(categories)}')
            self._stats['skipped_incomplete'] += 1
            return
        self._process_song(stem_paths, display, categories, thresholds, model, device, sources, sr, fi, total, delete_whole=folder, layout=SDR_LAYOUT_MUSDB)

    def _run(self):
        model = None
        device = 'cpu'
        self._phase_timer = PhaseTimer()
        try:
            p = self.p
            self.log(f"  Preparing '{p['model_id']}' …")
            device, device_warnings = resolve_processing_device(bool(p['use_cuda']))
            for warning in device_warnings:
                self.log(f'  [warn] {warning}')
            if FFMPEG:
                from ffmpeg_bootstrap import ffmpeg_folder_path
                self.log(f'  ffmpeg: {ffmpeg_folder_path()}')
            else:
                from ffmpeg_bootstrap import ffmpeg_missing_message
                self.log(f'  [warn] {ffmpeg_missing_message()}')
            self.log(f"  Device: {device}")
            from deps_bootstrap import demucs_model_cached
            if not demucs_model_cached(p['model_id']):
                mid = str(p.get('model_id') or '')
                if mid == 'vocal_cnn6':
                    self.log(
                        '  [warn] Vocal CNN6 ONNX weight missing (~24 MB). '
                        "Re-run the installer's models component, or place "
                        'models/vocal_classifier.onnx beside the app.'
                    )
                else:
                    self.log(
                        '  [warn] HTDemucs ONNX weight missing (~302 MB). '
                        "Re-run the installer's models component, or place "
                        'models/htdemucs.onnx beside the app.'
                    )
            self.log(f"  Loading model '{p['model_id']}' ...")
            t0 = time.monotonic()
            model = load_demucs_model(
                p['model_id'], prefer_gpu=(device != 'cpu')
            ).eval().to(device)
            # Yield GIL so queued "Loading…" logs / Qt UI can breathe.
            time.sleep(0)
            model_load_dt = time.monotonic() - t0
            self._phase_timer.add('model_load', model_load_dt)
            sources = list(model.sources)
            sr = model.samplerate
            backend = getattr(model, 'backend', 'torch')
            self.log(f'  Model sources: {sources}  (sr={sr}, backend={backend})')

            mode_cfg = STEM_MODES[resolve_stem_mode(p['stem_mode'])]
            preferred = mode_cfg['categories']
            root = Path(p['target_dir'])
            self._sdr_root = root.resolve()
            scan_mode = p['scan_mode']

            self.log('  Starting SI-SDR determination...')
            t0 = time.monotonic()
            if 'sdr_categories' in p and 'sdr_layout' in p:
                categories = tuple(p['sdr_categories'])
                layout = p['sdr_layout']
            else:
                categories, layout = resolve_sdr_layout_and_categories(root, scan_mode, preferred)
            if layout is None or categories is None:
                self.log('[error] ' + describe_sdr_scan_failure(root, scan_mode, preferred))
                return

            if layout == SDR_LAYOUT_MIXED_FLAT:
                self.log('[info] Mixed loose files detected; classifying each filename as instrumental or vocals.')
            elif categories != preferred:
                self.log(f'[info] Folders contain {len(categories)}-stem sets ({", ".join(categories)}); Stem mode is {p["stem_mode"]}.')
            elif layout == SDR_LAYOUT_SINGLE_FLAT:
                if p.get('sdr_user_picked_category'):
                    self.log(f'[info] User-selected {categories[0]}-only folder; processing all audio files as {categories[0]}.')
                else:
                    patterns = single_flat_category_patterns(categories[0])
                    self.log(f'[info] {categories[0]}-only files detected ({patterns}).')

            process_all = bool(p.get('sdr_flat_process_all', False))
            pair_process_all = bool(p.get('sdr_pair_process_all', False))
            if process_all and layout in SDR_SINGLE_FLAT_LAYOUTS:
                self.log(f'[info] Processing all audio files in folder as {categories[0]}.')
            if pair_process_all and layout == SDR_LAYOUT_MUSDB:
                self.log('[info] Processing all two-file song folders as instrumental + vocals pairs.')

            thresholds = sdr_thresholds_for_categories(categories, p['sdr_thresholds'])
            targets = collect_sdr_targets(
                root, categories, scan_mode, layout,
                process_all=process_all or pair_process_all,
            )
            if layout == SDR_LAYOUT_MUSDB:
                layout_label = 'Type 1 (MUSDB)'
            elif layout == SDR_LAYOUT_STEMS:
                layout_label = 'Type 2 (Stems)'
            elif layout == SDR_LAYOUT_SINGLE_FLAT:
                layout_label = f'Type 3 ({categories[0]}-only files)'
            elif layout == SDR_LAYOUT_MIXED_FLAT:
                layout_label = 'Type 3 (mixed vocals/instrumental files)'
            else:
                layout_label = layout
            self.log(f'  Detected layout: {layout_label}')

            if not targets:
                expected = ', '.join(f'{c}{SDR_STEM_EXT_LABEL}' for c in categories)
                self.log(f'[error] No folders with all expected stems found under {root}.\n  Expected each song folder to contain: {expected}')
                return

            self._phase_timer.add('target_scan', time.monotonic() - t0)

            self._total_folders = len(targets)
            self._completed_folders = 0
            self._run_started_at = time.monotonic()
            self._reset_stats(len(targets))
            self._report_progress()

            scanned = len(collect_sdr_song_folders(root, scan_mode)) if layout == SDR_LAYOUT_MUSDB else len(targets)
            if layout == SDR_LAYOUT_SINGLE_FLAT:
                self.log(f'  Found {len(targets)} {categories[0]} file(s) to check.')
            elif layout == SDR_LAYOUT_MIXED_FLAT:
                counts = {category: sum(category in target for target in targets) for category in categories}
                self.log(f'  Found {counts["instrumental"]} instrumental and {counts["vocals"]} vocals file(s) to check.')
                unmatched = sum(1 for _ in iter_sdr_audio_files(root, scan_mode)) - len(targets)
                if unmatched:
                    self.log(f'  [skip] {unmatched} file(s) had no recognizable vocals/instrumental filename marker.')
            else:
                unit = 'folder(s)' if layout == SDR_LAYOUT_MUSDB else 'song(s)'
                self.log(f'  Found {len(targets)} {unit} with complete {len(categories)}-stem sets (of {scanned} scanned).')

            try:
                for fi, target in enumerate(targets, 1):
                    if self._stop_event.is_set():
                        self.log('Stopped by user.')
                        _release_gpu(model, device)
                        return
                    if layout == SDR_LAYOUT_MUSDB:
                        self._process_folder(
                            target, root, categories, thresholds, model, device, sources, sr, fi, len(targets),
                            force_two_file=pair_process_all,
                        )
                    elif layout in SDR_SINGLE_FLAT_LAYOUTS:
                        stem_paths: dict[str, Path] = target
                        cat = next(iter(stem_paths))
                        display = stem_paths[cat].stem
                        target_categories = (cat,) if layout == SDR_LAYOUT_MIXED_FLAT else categories
                        self._process_song(stem_paths, display, target_categories, thresholds, model, device, sources, sr, fi, len(targets), layout=layout)
                    else:
                        stem_paths: dict[str, Path] = target
                        cat = next(iter(stem_paths))
                        display = stem_file_song_key(stem_paths[cat], cat)
                        self._process_song(stem_paths, display, categories, thresholds, model, device, sources, sr, fi, len(targets), layout=SDR_LAYOUT_STEMS)
                    self._mark_folder_done()
            finally:
                self._prune_empty_sdr_dirs()

            elapsed = time.monotonic() - self._run_started_at
            self._log_sdr_summary(elapsed)
            self._completed_folders = self._total_folders
            self._report_progress()
        finally:
            if model is not None:
                _release_gpu(model, device)
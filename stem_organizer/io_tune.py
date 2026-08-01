"""Per-volume I/O quick-tune: probe decode concurrency and cache results.

Used by Genre/Gender (thread workers), Compression, and Corruption (process workers).
Cache key is the volume root (drive letter or UNC share), not the full folder path.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Optional

from .settings_store import SettingsStore, load_settings, save_settings

SETTINGS_KEY = "io_tune_by_volume"
_CACHE_VERSION = 2  # bump when probe ladder / selection rules change
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg", ".opus"}
_MIN_SAMPLES = 6
_DEFAULT_WORKERS = 4
_PROBE_FRAMES = 48000  # ~1–3 s of audio depending on sample rate
# Ladder of worker counts to try (capped by cpu_count; early-stop on plateau).
_CANDIDATE_LADDER = (2, 4, 6, 8, 12, 16)
_MAX_WORKERS = 16
# Keep probing higher counts only while throughput rises by at least this fraction.
_IMPROVE_FRAC = 0.03


def _candidate_workers(cpu: int) -> tuple[int, ...]:
    """Worker counts to probe: ladder up to cpu, never above _MAX_WORKERS."""
    cpu_n = max(1, min(int(cpu), _MAX_WORKERS))
    out: list[int] = []
    for n in _CANDIDATE_LADDER:
        if n <= cpu_n:
            out.append(n)
    if cpu_n not in out:
        out.append(cpu_n)
    return tuple(sorted(set(max(1, n) for n in out)))


LogFn = Callable[[str, str], None]
Workload = Literal["genre", "gender", "compression", "corruption"]


@dataclass(frozen=True)
class ProbeResult:
    workers: int
    files_per_sec: float
    tried: dict[int, float]
    sample_count: int
    probed: bool  # False = fallback (too few samples / probe failed)


@dataclass(frozen=True)
class ParallelismHint:
    audio_workers: int
    gpu_batch_size: int
    file_chunk: int
    process_workers: int
    volume: str
    files_per_sec: float
    probed: bool
    message: str


def volume_key(path: str | Path) -> str:
    """Stable cache key for a drive letter or UNC share root."""
    raw = str(path).strip()
    if os.name == "nt":
        # UNC before resolve — resolve() can rewrite missing shares oddly.
        norm = raw.replace("/", "\\")
        if norm.startswith("\\\\"):
            parts = [p for p in norm.lstrip("\\").split("\\") if p]
            if len(parts) >= 2:
                return f"\\\\{parts[0]}\\{parts[1]}"
            return norm.rstrip("\\") + "\\"
        # Drive letter without requiring the path to exist.
        if len(norm) >= 2 and norm[1] == ":":
            return norm[0].upper() + ":\\"

    p = Path(raw).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(str(p)))
    s = str(p)
    if os.name == "nt":
        drive = getattr(p, "drive", "") or ""
        if drive:
            return drive.upper().rstrip("\\") + "\\"
    parts = p.parts
    if len(parts) >= 1:
        return parts[0] if parts[0] == "/" else str(Path(*parts[:2]))
    return s


def _collect_samples(folder: Path, sample_n: int) -> list[Path]:
    found: list[Path] = []
    try:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if Path(name).suffix.lower() in _AUDIO_EXTS:
                    found.append(Path(root) / name)
                    if len(found) >= sample_n:
                        return found
    except OSError:
        pass
    return found


def _read_one(path: Path) -> bool:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        frames = min(int(info.frames or 0), _PROBE_FRAMES)
        if frames <= 0:
            frames = _PROBE_FRAMES
        sf.read(str(path), frames=frames, dtype="float32", always_2d=True)
        return True
    except Exception:
        return False


def probe_decode_workers(
    folder: str | Path,
    *,
    candidates: tuple[int, ...] | None = None,
    sample_n: int = 24,
) -> ProbeResult:
    """Time short soundfile reads at each worker count; pick the fastest.

    Tries an ascending ladder (2…16, capped by CPU). Stops early when a higher
    count does not beat the previous step by ~3% (plateau / thrash).
    """
    root = Path(folder)
    samples = _collect_samples(root, sample_n) if root.is_dir() else []
    cpu = max(1, os.cpu_count() or 4)
    if candidates is None:
        capped = _candidate_workers(cpu)
    else:
        capped = tuple(sorted({max(1, min(int(c), cpu, _MAX_WORKERS)) for c in candidates}))

    if len(samples) < _MIN_SAMPLES:
        return ProbeResult(
            workers=_DEFAULT_WORKERS,
            files_per_sec=0.0,
            tried={},
            sample_count=len(samples),
            probed=False,
        )

    tried: dict[int, float] = {}
    prev_fps = 0.0
    for n_workers in capped:
        t0 = time.perf_counter()
        ok = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_read_one, p) for p in samples]
            for fut in as_completed(futures):
                try:
                    if fut.result():
                        ok += 1
                except Exception:
                    pass
        elapsed = max(time.perf_counter() - t0, 1e-6)
        if ok <= 0:
            continue
        fps = ok / elapsed
        tried[n_workers] = fps
        # Plateau: this step did not meaningfully beat the previous → stop.
        if prev_fps > 0 and fps < prev_fps * (1.0 + _IMPROVE_FRAC):
            break
        prev_fps = fps

    if not tried:
        return ProbeResult(
            workers=_DEFAULT_WORKERS,
            files_per_sec=0.0,
            tried={},
            sample_count=len(samples),
            probed=False,
        )

    best_workers = max(tried, key=lambda w: (tried[w], -w))
    return ProbeResult(
        workers=int(best_workers),
        files_per_sec=float(tried[best_workers]),
        tried={int(k): float(v) for k, v in tried.items()},
        sample_count=len(samples),
        probed=True,
    )


def hint_for_workload(workers: int, workload: Workload, *, volume: str = "", fps: float = 0.0, probed: bool = True) -> ParallelismHint:
    w = max(1, int(workers))
    file_chunk = max(64, min(256, w * 32))
    if workload == "gender":
        file_chunk = max(file_chunk, 128)
    process_workers = max(2, min(_MAX_WORKERS, w))
    return ParallelismHint(
        audio_workers=w,
        gpu_batch_size=64,
        file_chunk=file_chunk,
        process_workers=process_workers,
        volume=volume,
        files_per_sec=fps,
        probed=probed,
        message="",
    )


def _read_cache() -> dict:
    data = load_settings()
    cache = data.get(SETTINGS_KEY)
    return dict(cache) if isinstance(cache, dict) else {}


def _write_cache(cache: dict, settings: Optional[SettingsStore] = None) -> None:
    data = load_settings()
    data[SETTINGS_KEY] = cache
    save_settings(data)
    if settings is not None:
        settings.set(SETTINGS_KEY, cache)


def clear_volume_cache(
    path: str | Path,
    settings: Optional[SettingsStore] = None,
) -> str:
    """Remove cached tune for this path's volume. Returns the volume key."""
    key = volume_key(path)
    cache = _read_cache()
    if key in cache:
        del cache[key]
        _write_cache(cache, settings)
    return key


def ensure_tuned(
    folder: str | Path,
    settings: Optional[SettingsStore] = None,
    *,
    workload: Workload = "genre",
    force: bool = False,
    log: Optional[LogFn] = None,
) -> ParallelismHint:
    """Return parallelism hint for folder's volume; probe + cache on miss."""
    key = volume_key(folder)
    cache = _read_cache()

    def _emit(msg: str, tag: str = "info") -> None:
        if log is not None:
            log(msg, tag)

    if not force and key in cache:
        entry = cache[key]
        if (
            isinstance(entry, dict)
            and entry.get("workers")
            and int(entry.get("v") or 0) == _CACHE_VERSION
        ):
            try:
                workers = max(1, int(entry["workers"]))
            except (TypeError, ValueError):
                workers = _DEFAULT_WORKERS
            fps = float(entry.get("fps") or 0.0)
            hint = hint_for_workload(
                workers, workload, volume=key, fps=fps, probed=True
            )
            _emit(
                f"  Quick tune ({key}): workers={hint.audio_workers} (cached)",
                "info",
            )
            return hint

    _emit(f"  Quick tune ({key}): probing decode workers…", "info")
    result = probe_decode_workers(folder)
    hint = hint_for_workload(
        result.workers,
        workload,
        volume=key,
        fps=result.files_per_sec,
        probed=result.probed,
    )

    if result.probed:
        parts = " ".join(
            f"{w}→{result.tried[w]:.1f}/s" for w in sorted(result.tried)
        )
        msg = (
            f"  Quick tune ({key}): workers={hint.audio_workers}"
            f" ({parts})"
        )
        cache[key] = {
            "v": _CACHE_VERSION,
            "workers": hint.audio_workers,
            "fps": round(result.files_per_sec, 3),
            "tried": {str(k): round(v, 3) for k, v in result.tried.items()},
            "sample_count": result.sample_count,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write_cache(cache, settings)
    else:
        msg = (
            f"  Quick tune ({key}): workers={hint.audio_workers} "
            f"(fallback — only {result.sample_count} readable sample"
            f"{'' if result.sample_count == 1 else 's'})"
        )
    _emit(msg, "info")
    return hint

"""Fill Track.instrument from PaSST OpenMIC (in-process preferred, subprocess fallback)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from tagger_launch import (
    STEM_SITE_PACKAGES_ENV,
    build_tagger_command,
    instrument_tagger_dir,
    instrument_tagger_script,
    missing_tagger_python_hint,
    resolve_tagger_python,
    tagger_subprocess_env,
)
from track_renamer.engine.defaults import DEFAULT_CATEGORY_SOURCE, map_instrument_to_category
from track_renamer.engine.models import OpRule, Rule, Track


def _tagger_dir() -> Path:
    return instrument_tagger_dir()


def _tagger_script() -> Path:
    return instrument_tagger_script()


# Bump when model/label set / primary-pick policy changes so stale cache dies.
_CACHE_MODEL = "passt-openmic-nosynth-g35"
_CACHE_FILE = "instrument_passt_cache.json"
_CACHE_VERSION = 1
_CACHE_MAX_ENTRIES = 100_000

# path → (mtime_ns, label, score, second_score, model_id)
_CACHE: dict[str, tuple] = {}
_DISK_LOADED = False
_DISK_DIRTY = False

# Reused ONNX/torch backend across Analyze runs in this process.
_INPROC_BACKEND = None
_INPROC_LOCK = threading.Lock()

ResultCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[[int, int], None]
ProcessCallback = Callable[[subprocess.Popen], None]


def terminate_tagger_process(proc: subprocess.Popen | None) -> None:
    """Kill the instrument tagger and any children (Cancel / cleanup)."""
    if proc is None:
        return
    try:
        import psutil

        try:
            parent = psutil.Process(proc.pid)
        except (psutil.Error, OSError):
            parent = None
        targets: list = []
        if parent is not None:
            try:
                targets.extend(parent.children(recursive=True))
            except (psutil.Error, OSError):
                pass
            targets.append(parent)
        for child in targets:
            try:
                child.kill()
            except (psutil.Error, OSError):
                pass
        if targets:
            psutil.wait_procs(targets, timeout=1.5)
            return
    except Exception:
        pass
    try:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def rules_need_instrument_ml(rules: list[Rule]) -> bool:
    for rule in rules:
        if isinstance(rule, OpRule) and rule.op == "categoryBundle":
            source = str(rule.params.get("source", DEFAULT_CATEGORY_SOURCE)).lower()
            if source in ("model", "combo"):
                return True
    return False


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _cache_path() -> Path:
    try:
        from stem_organizer.settings_store import app_dir

        return app_dir() / _CACHE_FILE
    except Exception:
        return Path(__file__).resolve().parent.parent / _CACHE_FILE


def _ensure_disk_loaded() -> None:
    global _DISK_LOADED
    if _DISK_LOADED:
        return
    _DISK_LOADED = True
    path = _cache_path()
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict) or int(raw.get("v") or 0) != _CACHE_VERSION:
        return
    if str(raw.get("model") or "") != _CACHE_MODEL:
        return
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return
    for key, val in entries.items():
        if not isinstance(val, (list, tuple)) or len(val) < 4:
            continue
        try:
            _CACHE[str(key)] = (
                int(val[0]),
                str(val[1]),
                float(val[2]),
                float(val[3]),
                _CACHE_MODEL,
            )
        except (TypeError, ValueError):
            continue


def _flush_disk_cache() -> None:
    global _DISK_DIRTY
    if not _DISK_DIRTY:
        return
    path = _cache_path()
    entries: dict[str, list] = {}
    items = list(_CACHE.items())
    if len(items) > _CACHE_MAX_ENTRIES:
        items = items[-_CACHE_MAX_ENTRIES:]
    for key, cached in items:
        unpacked = _unpack_cache(cached)
        if unpacked is None:
            continue
        mtime, label, score, second = unpacked
        entries[key] = [mtime, label, score, second]
    payload = {"v": _CACHE_VERSION, "model": _CACHE_MODEL, "entries": entries}
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        _DISK_DIRTY = False
    except OSError:
        pass


def classify_decision(
    label: str,
    score: float = 0.0,
    *,
    second_score: float = 0.0,
) -> tuple[str, str]:
    """Return (action, category_name). action: 'apply' | 'skip_unmap'."""
    _ = (score, second_score)
    category = map_instrument_to_category(label)
    if not category:
        return "skip_unmap", category
    return "apply", category


def _unpack_cache(cached: tuple) -> tuple[int, str, float, float] | None:
    """Return (mtime, label, score, second) if entry matches current model."""
    if len(cached) < 5:
        return None
    mtime, label, score, second, model_id = cached[:5]
    if model_id != _CACHE_MODEL:
        return None
    return int(mtime), str(label), float(score), float(second)


def apply_cached_labels(tracks: list[Track]) -> int:
    """Apply cache hits onto tracks. Returns number filled from cache."""
    _ensure_disk_loaded()
    filled = 0
    for track in tracks:
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        cached = _CACHE.get(key)
        if not cached:
            continue
        unpacked = _unpack_cache(cached)
        if not unpacked:
            continue
        mtime, label, score, second = unpacked
        if mtime != _mtime_ns(path):
            continue
        track.instrument = label
        track.instrument_score = score
        track.instrument_second = float(second)
        track.category = map_instrument_to_category(label)
        filled += 1
    return filled


def _paths_needing_infer(tracks: list[Track]) -> list[Path]:
    _ensure_disk_loaded()
    needed: list[Path] = []
    for track in tracks:
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        cached = _CACHE.get(key)
        unpacked = _unpack_cache(cached) if cached else None
        if unpacked and unpacked[0] == _mtime_ns(path):
            continue
        needed.append(path)
    return needed


def _second_from_row(row: dict) -> float:
    """Runner-up share. Worker score is calibrated p1/(p1+p2) → second = 1-score."""
    try:
        score = float(row.get("score") or 0.0)
        if 0.0 < score <= 1.0:
            return max(0.0, 1.0 - score)
        top = row.get("top") or []
        if isinstance(top, list) and len(top) >= 2:
            return float(top[1][1])
    except (TypeError, ValueError, IndexError):
        pass
    return 0.0


def _emit_result(
    on_result: ResultCallback | None,
    *,
    path: Path,
    label: str,
    score: float,
    second_score: float,
    error: str = "",
    index: int | None = None,
    total: int | None = None,
) -> None:
    if on_result is None:
        return
    category = map_instrument_to_category(label) if not error else ""
    payload = {
        "path": path,
        "name": path.name,
        "label": label,
        "score": score,
        "second_score": second_score,
        "category": category,
        "error": error,
    }
    if index is not None:
        payload["index"] = int(index)
    if total is not None:
        payload["total"] = int(total)
    on_result(payload)


def _force_subprocess() -> bool:
    return os.environ.get("STEM_PASST_SUBPROCESS", "0").strip() == "1"


def _import_instrument_tagger():
    tagger_dir = _tagger_dir()
    script = _tagger_script()
    if not script.is_file():
        raise FileNotFoundError(f"instrument_tagger missing: {script}")
    entry = str(tagger_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    import instrument_tagger as it  # type: ignore

    return it


def _get_inproc_backend(status: Callable[[str], None]):
    global _INPROC_BACKEND
    with _INPROC_LOCK:
        if _INPROC_BACKEND is not None:
            return _INPROC_BACKEND
        it = _import_instrument_tagger()
        status("  loading PaSST OpenMIC (in-process)…")
        _INPROC_BACKEND = it.load_backend(status=status)
        status(f"  backend: {_INPROC_BACKEND.name}  device: {_INPROC_BACKEND.device}")
        return _INPROC_BACKEND


def _passt_env_defaults(status: Callable[[str], None], pending: list[Path]) -> None:
    """Set PASST_* / ORT thread caps in this process (in-proc path)."""
    try:
        from ort_util import cuda_ep_usable, nvidia_gpu_present
        from stem_organizer.io_tune import ensure_tuned

        on_gpu = bool(
            os.environ.get("STEM_ORT_CUDA", "1").strip() != "0"
            and cuda_ep_usable()
            and nvidia_gpu_present()
        )
        probe_dir = pending[0].parent if pending else Path(".")
        hint = ensure_tuned(
            probe_dir,
            workload="gender",
            log=lambda msg, _tag="info": status(msg),
            inference_on_gpu=on_gpu,
        )
        os.environ.setdefault(
            "PASST_AUDIO_WORKERS",
            str(max(1, min(4, int(hint.audio_workers)))),
        )
        os.environ.setdefault("PASST_BATCH_SIZE", "8" if on_gpu else "4")
        if not on_gpu:
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            os.environ.setdefault("STEM_ORT_INTRA_OP", "2")
    except Exception as tune_exc:
        status(f"  [warn] quick-tune skipped: {tune_exc}")
        os.environ.setdefault("PASST_AUDIO_WORKERS", "2")
        os.environ.setdefault("PASST_BATCH_SIZE", "8")


def _enrich_inprocess(
    pending: list[Path],
    *,
    status: Callable[[str], None],
    on_progress: ProgressCallback | None,
    on_result: ResultCallback | None,
    cancel: threading.Event | None,
    cached_n: int,
    grand_total: int,
) -> tuple[int, str | None]:
    """Run PaSST in this process (reuse ORT session across Analyze runs)."""
    it = _import_instrument_tagger()
    _passt_env_defaults(status, pending)
    try:
        backend = _get_inproc_backend(status)
    except Exception as exc:
        return 0, f"in-process tagger failed to load: {exc}"

    batch_size = max(1, int(getattr(it, "_passt_batch_size", lambda: 8)()))
    audio_workers = max(1, int(getattr(it, "_passt_audio_workers", lambda: 2)()))
    status(f"  batch={batch_size} decode_workers={audio_workers} (in-process)")

    def _safe_load(path: Path):
        try:
            return path, it.load_mono_32k(path), None
        except Exception as exc:
            return path, None, str(exc)

    classified = 0
    done = cached_n
    predict_batch = getattr(backend, "predict_batch", None)

    def _emit_loaded(loaded) -> None:
        nonlocal classified, done
        ok_paths: list[Path] = []
        ok_audios: list = []
        for path, audio, err in loaded:
            if cancel is not None and cancel.is_set():
                return
            if err is not None or audio is None:
                done += 1
                _emit_result(
                    on_result,
                    path=path,
                    label="",
                    score=0.0,
                    second_score=0.0,
                    error=err or "load failed",
                    index=done,
                    total=grand_total,
                )
                if on_progress:
                    on_progress(done, grand_total)
                continue
            ok_paths.append(path)
            ok_audios.append(audio)
        if not ok_audios:
            return
        try:
            if callable(predict_batch):
                probs_batch = predict_batch(ok_audios)
            else:
                import numpy as np

                probs_batch = np.stack([backend.predict(a) for a in ok_audios], axis=0)
        except Exception as exc:
            for path, audio in zip(ok_paths, ok_audios):
                if cancel is not None and cancel.is_set():
                    return
                try:
                    probs = backend.predict(audio)
                    row = it.probs_to_result(probs, top_k=2, threshold=0.0)
                    row["path"] = str(path.resolve())
                    _store_result(path, row)
                    classified += 1
                    done += 1
                    _emit_result(
                        on_result,
                        path=path,
                        label=str(row.get("label") or ""),
                        score=float(row.get("score") or 0.0),
                        second_score=_second_from_row(row),
                        index=done,
                        total=grand_total,
                    )
                    if on_progress:
                        on_progress(done, grand_total)
                except Exception as exc2:
                    done += 1
                    _emit_result(
                        on_result,
                        path=path,
                        label="",
                        score=0.0,
                        second_score=0.0,
                        error=str(exc2),
                        index=done,
                        total=grand_total,
                    )
                    if on_progress:
                        on_progress(done, grand_total)
            status(f"  [warn] batch infer failed ({exc}); fell back per-file")
            return

        for path, probs in zip(ok_paths, probs_batch):
            if cancel is not None and cancel.is_set():
                return
            row = it.probs_to_result(probs, top_k=2, threshold=0.0)
            row["path"] = str(path.resolve())
            _store_result(path, row)
            classified += 1
            done += 1
            _emit_result(
                on_result,
                path=path,
                label=str(row.get("label") or ""),
                score=float(row.get("score") or 0.0),
                second_score=_second_from_row(row),
                index=done,
                total=grand_total,
            )
            if on_progress:
                on_progress(done, grand_total)

    with ThreadPoolExecutor(max_workers=audio_workers) as pool:
        starts = list(range(0, len(pending), batch_size))

        def _load_chunk(paths: list[Path]):
            return list(pool.map(_safe_load, paths))

        next_fut = None
        for i, start in enumerate(starts):
            if cancel is not None and cancel.is_set():
                break
            chunk = pending[start : start + batch_size]
            if next_fut is not None:
                loaded = next_fut.result()
                next_fut = None
            else:
                loaded = _load_chunk(chunk)
            if i + 1 < len(starts) and not (cancel is not None and cancel.is_set()):
                next_chunk = pending[starts[i + 1] : starts[i + 1] + batch_size]
                next_fut = pool.submit(_load_chunk, next_chunk)
            _emit_loaded(loaded)

    _flush_disk_cache()
    if cancel is not None and cancel.is_set():
        return classified, None
    return classified, None


def enrich_tracks(
    tracks: list[Track],
    *,
    status: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
    cancel: threading.Event | None = None,
    on_process: ProcessCallback | None = None,
) -> tuple[int, str | None]:
    """
    Classify tracks missing cache entries via instrument_tagger.

    Prefers in-process ORT (session reused across Analyze). Set
    ``STEM_PASST_SUBPROCESS=1`` to force the old exe spawn path.
    """
    _ensure_disk_loaded()
    apply_cached_labels(tracks)
    pending = _paths_needing_infer(tracks)
    pending_keys = {str(p.resolve()) for p in pending}

    cache_hits: list[tuple[Path, str, float, float]] = []
    for track in tracks:
        if cancel is not None and cancel.is_set():
            return 0, None
        path = track.file_path
        if path is None or not path.is_file():
            continue
        key = str(path.resolve())
        if key in pending_keys:
            continue
        cached = _CACHE.get(key)
        if not cached:
            continue
        unpacked = _unpack_cache(cached)
        if not unpacked:
            continue
        _mtime, label, score, second = unpacked
        cache_hits.append((path, label, score, second))

    cached_n = len(cache_hits)
    grand_total = cached_n + len(pending)

    for i, (path, label, score, second) in enumerate(cache_hits, start=1):
        if cancel is not None and cancel.is_set():
            return i - 1, None
        _emit_result(
            on_result,
            path=path,
            label=label,
            score=score,
            second_score=second,
            index=i,
            total=grand_total,
        )
        if on_progress:
            on_progress(i, grand_total)

    status = status or (lambda _msg: None)
    if cached_n and pending:
        status(f"Cache hit {cached_n:,} — inferring {len(pending):,}…")
    elif cached_n and not pending:
        status(f"All {cached_n:,} from cache.")
        return cached_n, None
    elif not pending:
        return cached_n, None

    if cancel is not None and cancel.is_set():
        return cached_n, None

    if not _force_subprocess() and _tagger_script().is_file():
        status(f"Starting tagger for {len(pending):,} file(s)…")
        try:
            classified, err = _enrich_inprocess(
                pending,
                status=status,
                on_progress=on_progress,
                on_result=on_result,
                cancel=cancel,
                cached_n=cached_n,
                grand_total=grand_total,
            )
            apply_cached_labels(tracks)
            _flush_disk_cache()
            if err:
                status(f"  [warn] in-process failed ({err}); trying subprocess…")
            else:
                return cached_n + classified, None
        except Exception as exc:
            status(f"  [warn] in-process failed ({exc}); trying subprocess…")

    py = resolve_tagger_python()
    tagger_script = _tagger_script()
    tagger_dir = _tagger_dir()
    if py is None or not tagger_script.is_file():
        return cached_n, (
            "Instrument tagger not installed.\n"
            f"{missing_tagger_python_hint()}"
        )

    total = grand_total
    status(f"Starting tagger subprocess for {len(pending):,} file(s)…")

    list_path: Path | None = None
    classified = 0
    done = cached_n
    proc: subprocess.Popen | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as handle:
            list_path = Path(handle.name)
            for path in pending:
                handle.write(f"{path.resolve()}\n")

        if cancel is not None and cancel.is_set():
            return cached_n, None

        spawn_env = tagger_subprocess_env()
        try:
            from ort_util import cuda_ep_usable, nvidia_gpu_present
            from stem_organizer.io_tune import ensure_tuned

            on_gpu = bool(
                spawn_env.get("STEM_ORT_CUDA", "1").strip() != "0"
                and cuda_ep_usable()
                and nvidia_gpu_present()
            )
            probe_dir = pending[0].parent if pending else Path(".")
            hint = ensure_tuned(
                probe_dir,
                workload="gender",
                log=lambda msg, _tag="info": status(msg),
                inference_on_gpu=on_gpu,
            )
            spawn_env.setdefault(
                "PASST_AUDIO_WORKERS",
                str(max(1, min(4, int(hint.audio_workers)))),
            )
            spawn_env.setdefault("PASST_BATCH_SIZE", "8" if on_gpu else "4")
            if not on_gpu:
                spawn_env.setdefault("OMP_NUM_THREADS", "1")
                spawn_env.setdefault("MKL_NUM_THREADS", "1")
                spawn_env.setdefault("OPENBLAS_NUM_THREADS", "1")
                spawn_env.setdefault("NUMEXPR_NUM_THREADS", "1")
                spawn_env.setdefault("STEM_ORT_INTRA_OP", "2")
        except Exception as tune_exc:
            status(f"  [warn] quick-tune skipped: {tune_exc}")
            spawn_env.setdefault("PASST_AUDIO_WORKERS", "2")
            spawn_env.setdefault("PASST_BATCH_SIZE", "8")

        cmd = build_tagger_command(
            tagger_script,
            "--files-from",
            str(list_path),
            "--top",
            "2",
        )
        from ffmpeg_bootstrap import subprocess_kwargs

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(tagger_dir),
            env=spawn_env,
            **subprocess_kwargs(),
        )
        if on_process is not None:
            on_process(proc)
        assert proc.stdout is not None

        log_tail: list[str] = []
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                terminate_tagger_process(proc)
                _flush_disk_cache()
                return cached_n + classified, None
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                log_tail.append(stripped)
                if len(log_tail) > 40:
                    log_tail = log_tail[-40:]
                if (
                    stripped.startswith("[")
                    and "/" in stripped[:12]
                    and "]" in stripped[:16]
                ):
                    continue
                low = stripped.lower()
                if (
                    "torch.size" in low
                    or "userwarning" in low
                    or "warnings.warn" in low
                    or "input image size" in low
                    or stripped.startswith(
                        (
                            "X flattened",
                            "forward_features",
                            "head ",
                            " self.",
                            "patch_embed",
                            "Loading PASST",
                            "Loading PaSST",
                            "(1): Linear",
                            "(head_dist):",
                            "Sequential(",
                            "  (",
                        )
                    )
                ):
                    continue
                status(stripped.lstrip())
                if stripped.lstrip().lower().startswith("backend:"):
                    status("")
                continue

            done += 1
            path = Path(str(row.get("path") or ""))
            if "error" in row or not path.name:
                _emit_result(
                    on_result,
                    path=path if path.name else Path("unknown"),
                    label="",
                    score=0.0,
                    second_score=0.0,
                    error=str(row.get("error") or "error"),
                    index=done,
                    total=total,
                )
                if on_progress:
                    on_progress(done, total)
                continue

            _store_result(path, row)
            classified += 1
            label = str(row.get("label") or "")
            try:
                score = float(row.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            second = _second_from_row(row)
            _emit_result(
                on_result,
                path=path,
                label=label,
                score=score,
                second_score=second,
                index=done,
                total=total,
            )
            if on_progress:
                on_progress(done, total)

        if cancel is not None and cancel.is_set():
            terminate_tagger_process(proc)
            _flush_disk_cache()
            return cached_n + classified, None

        returncode = proc.wait()
        terminate_tagger_process(proc)
        if cancel is not None and cancel.is_set():
            _flush_disk_cache()
            return cached_n + classified, None
        if returncode != 0 and classified == 0:
            useful = [
                ln
                for ln in log_tail
                if any(
                    k in ln
                    for k in (
                        "Error",
                        "ERROR",
                        "Traceback",
                        "Exception",
                        "UnicodeEncode",
                        "not installed",
                        "import failed",
                        "PYTHONPATH",
                        "STEM_SITE",
                        "detail:",
                        "python:",
                    )
                )
            ]
            err = "\n".join(useful or log_tail[-12:]).strip() or "tagger failed"
            diag = (
                f"\n  launch python: {py}"
                f"\n  launch PYTHONPATH: {spawn_env.get('PYTHONPATH', '') or '(empty)'}"
                f"\n  launch {STEM_SITE_PACKAGES_ENV}: "
                f"{spawn_env.get(STEM_SITE_PACKAGES_ENV, '') or '(empty)'}"
                f"\n  launch script: {tagger_script}"
            )
            if diag.strip() not in err:
                err = (err + diag).strip()
            return cached_n, err[:1200]
    except OSError as exc:
        return cached_n, str(exc)
    finally:
        if cancel is not None and cancel.is_set() and proc is not None:
            terminate_tagger_process(proc)
        if list_path is not None:
            try:
                list_path.unlink()
            except OSError:
                pass
        _flush_disk_cache()

    apply_cached_labels(tracks)
    return cached_n + classified, None


def _store_result(path: Path, row: dict) -> None:
    global _DISK_DIRTY
    label = str(row.get("label") or "")
    try:
        score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    second = _second_from_row(row)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    key = str(resolved)
    _CACHE[key] = (
        _mtime_ns(resolved),
        label,
        score,
        second,
        _CACHE_MODEL,
    )
    _DISK_DIRTY = True


def clear_instrument_cache() -> None:
    global _DISK_DIRTY
    _CACHE.clear()
    _DISK_DIRTY = True
    path = _cache_path()
    try:
        if path.is_file():
            path.unlink()
        _DISK_DIRTY = False
    except OSError:
        _flush_disk_cache()


def relocate_instrument_cache(old_path: Path, new_path: Path) -> None:
    """Move a cache entry when a file is renamed or moved on disk."""
    global _DISK_DIRTY
    _ensure_disk_loaded()
    try:
        old_key = str(Path(old_path).resolve())
    except OSError:
        old_key = str(old_path)
    try:
        new_key = str(Path(new_path).resolve())
    except OSError:
        new_key = str(new_path)
    if old_key == new_key:
        return
    entry = _CACHE.pop(old_key, None)
    if entry is None:
        return
    unpacked = _unpack_cache(entry)
    if unpacked is None:
        _CACHE[new_key] = entry
        _DISK_DIRTY = True
        _flush_disk_cache()
        return
    _mtime, label, score, second = unpacked
    try:
        mtime = _mtime_ns(Path(new_path))
    except OSError:
        mtime = _mtime
    _CACHE[new_key] = (mtime, label, score, second, _CACHE_MODEL)
    _DISK_DIRTY = True
    _flush_disk_cache()


def tagger_available() -> bool:
    return _tagger_script().is_file()

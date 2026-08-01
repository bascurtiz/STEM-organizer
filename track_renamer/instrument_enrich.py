"""Fill Track.instrument from PaSST OpenMIC worker (subprocess)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
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


# Resolve at call time (frozen exe dir can differ from import-time assumptions).
def _tagger_dir() -> Path:
    return instrument_tagger_dir()


def _tagger_script() -> Path:
    return instrument_tagger_script()

# Bump when model/label set / primary-pick policy changes so stale cache dies.
_CACHE_MODEL = "passt-openmic-nosynth-g35"

# path → (mtime_ns, label, score, second_score, model_id)
_CACHE: dict[str, tuple] = {}

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


def classify_decision(
    label: str,
    score: float = 0.0,
    *,
    second_score: float = 0.0,
) -> tuple[str, str]:
    """
    Return (action, category_name).

    action: 'apply' | 'skip_unmap'
    Always apply when label maps to a Category Macro row.
    score / second_score ignored (kept for call-site compatibility).
    """
    _ = (score, second_score)
    category = map_instrument_to_category(label)
    if not category:
        return "skip_unmap", category
    return "apply", category


def _unpack_cache(cached: tuple) -> tuple[int, str, float, float] | None:
    """Return (mtime, label, score, second) if entry matches current model."""
    if len(cached) < 5:
        return None  # legacy cache — force re-infer
    mtime, label, score, second, model_id = cached[:5]
    if model_id != _CACHE_MODEL:
        return None
    return int(mtime), str(label), float(score), float(second)


def apply_cached_labels(tracks: list[Track]) -> int:
    """Apply cache hits onto tracks. Returns number filled from cache."""
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

    on_result receives one dict per file (cache hit or fresh infer).
    Returns (classified_count, error_message_or_None).
    If *cancel* is set, the tagger subprocess is killed and the call returns
    early with error None (caller should treat it as abandoned).
    """
    apply_cached_labels(tracks)
    pending = _paths_needing_infer(tracks)
    pending_keys = {str(p.resolve()) for p in pending}

    # Collect cache hits first so we know grand total for === [i/n] headers.
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

    # Emit cache hits so the analyze log fills immediately.
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

    py = resolve_tagger_python()
    tagger_script = _tagger_script()
    tagger_dir = _tagger_dir()
    if py is None or not tagger_script.is_file():
        return cached_n, (
            "Instrument tagger not installed.\n"
            f"{missing_tagger_python_hint()}"
        )

    total = grand_total
    status(f"Starting tagger for {len(pending):,} file(s)…")

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
        cmd = build_tagger_command(
            tagger_script,
            "--files-from",
            str(list_path),
            "--top",
            "2",
        )
        # Merge stderr→stdout so status/TF logs cannot fill the stderr pipe
        # and deadlock the worker (classic hang ~dozens of files in).
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
            **subprocess_kwargs(),  # hide console window on Windows
        )
        if on_process is not None:
            on_process(proc)
        assert proc.stdout is not None

        log_tail: list[str] = []
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                terminate_tagger_process(proc)
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
                # Forward warmup lines; skip per-file "[n/total] name" spam
                # and hear21passt / torch noise (model dump, shapes, warnings).
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
                # Blank before === file headers (Rename analyze PREVIEW).
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
            return cached_n + classified, None

        returncode = proc.wait()
        # Ensure any tagger child workers are gone before the UI renames files.
        terminate_tagger_process(proc)
        if cancel is not None and cancel.is_set():
            return cached_n + classified, None
        if returncode != 0 and classified == 0:
            # Prefer real traceback / ERROR lines over model-dump noise.
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
            # Always append launch diagnostics when Auto-detect fails cold.
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

    apply_cached_labels(tracks)
    return cached_n + classified, None


def _store_result(path: Path, row: dict) -> None:
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


def clear_instrument_cache() -> None:
    _CACHE.clear()


def relocate_instrument_cache(old_path: Path, new_path: Path) -> None:
    """Move a cache entry when a file is renamed or moved on disk."""
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
        return
    _mtime, label, score, second = unpacked
    try:
        mtime = _mtime_ns(Path(new_path))
    except OSError:
        mtime = _mtime
    _CACHE[new_key] = (mtime, label, score, second, _CACHE_MODEL)


def tagger_available() -> bool:
    return resolve_tagger_python() is not None and _tagger_script().is_file()

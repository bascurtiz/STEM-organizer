"""Fill results from PANNs Cnn14 worker (subprocess) — UI-ready client.

Same shape as ``track_renamer.instrument_enrich``: spawn
``panns_tagger/panns_tagger.py``, parse JSON lines, optional path cache.
Does not import torch into the GUI process.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from tagger_launch import (
    missing_tagger_python_hint,
    panns_tagger_dir,
    panns_tagger_script,
    resolve_tagger_python,
    tagger_subprocess_env,
)

# Bump when model / focus set / primary-pick policy changes.
_CACHE_MODEL = "panns-cnn14-focus-v2"

# path → (mtime_ns, payload dict, model_id)
_CACHE: dict[str, tuple] = {}

ResultCallback = Callable[[dict[str, Any]], None]
ProgressCallback = Callable[[int, int], None]
ProcessCallback = Callable[[subprocess.Popen], None]


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def terminate_panns_process(proc: subprocess.Popen | None) -> None:
    """Kill the PANNs tagger and any children (Cancel / cleanup)."""
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


def classify_paths(
    paths: list[str | Path],
    *,
    segment_sec: float = 0.0,
    hop_sec: float = 0.0,
    top_k: int = 10,
    focus_only: bool = False,
    use_cache: bool = True,
    on_result: Optional[ResultCallback] = None,
    on_progress: Optional[ProgressCallback] = None,
    on_process: Optional[ProcessCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[dict[str, Any]]:
    """Run PANNs on ``paths``; return list of result dicts (one per file)."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)

    results: list[dict[str, Any]] = []
    if not files:
        return results

    pending: list[Path] = []
    for path in files:
        key = str(path.resolve())
        mtime = _mtime_ns(path)
        if use_cache and key in _CACHE:
            cached_mtime, payload, model_id = _CACHE[key]
            if cached_mtime == mtime and model_id == _CACHE_MODEL:
                results.append(dict(payload))
                if on_result is not None:
                    on_result(payload)
                continue
        pending.append(path)

    total = len(files)
    done = total - len(pending)
    if on_progress is not None:
        on_progress(done, total)

    if not pending:
        return results

    py = resolve_tagger_python()
    script = panns_tagger_script()
    if py is None or not script.is_file():
        err = {
            "error": missing_tagger_python_hint()
            if py is None
            else f"PANNs script missing: {script}",
        }
        for path in pending:
            row = {"path": str(path.resolve()), **err}
            results.append(row)
            if on_result is not None:
                on_result(row)
        return results

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as tmp:
        list_path = Path(tmp.name)
        for path in pending:
            tmp.write(str(path.resolve()) + "\n")

    cmd = [
        str(py),
        "-u",
        str(script),
        "--files-from",
        str(list_path),
        "--top",
        str(max(1, int(top_k))),
    ]
    if focus_only:
        cmd.append("--focus-only")
    if segment_sec and segment_sec > 0:
        cmd.extend(["--segment-sec", str(float(segment_sec))])
        if hop_sec and hop_sec > 0:
            cmd.extend(["--hop-sec", str(float(hop_sec))])

    env = tagger_subprocess_env()
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(panns_tagger_dir()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if on_process is not None:
            on_process(proc)

        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                terminate_panns_process(proc)
                break
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            path_str = str(payload.get("path", ""))
            if path_str and "error" not in payload:
                try:
                    _CACHE[path_str] = (
                        _mtime_ns(Path(path_str)),
                        dict(payload),
                        _CACHE_MODEL,
                    )
                except Exception:
                    pass
            results.append(payload)
            done += 1
            if on_result is not None:
                on_result(payload)
            if on_progress is not None:
                on_progress(min(done, total), total)

        proc.wait(timeout=30)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass
        if proc is not None and proc.poll() is None:
            terminate_panns_process(proc)

    return results


def clear_cache() -> None:
    _CACHE.clear()

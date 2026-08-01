"""Wrap flac-detective to write COMPRESSION=lossless|lossy tags.

The 11-rule spectral path is CPU-bound (scipy FFT) — no GPU API in
flac-detective. We speed up by running file analyses in a process pool
(same approach as flac-detective's own CLI).
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Callable, Optional

from ..integrity_pool import integrity_pool
from ..meta_tags import read_custom_tag, write_compression_tag
from ..run_summary import emit_run_summary

LogFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, str], None]

_LOSSY_EXTS = {".mp3", ".ogg", ".opus", ".aac"}
_DETECT_EXTS = {".flac", ".wav", ".m4a", ".mp4", ".ape", ".aiff", ".aif"}

_LOSSLESS_VERDICTS = frozenset({"AUTHENTIC"})
_LOSSY_VERDICTS = frozenset({"WARNING", "SUSPICIOUS", "FAKE_CERTAIN"})


def _is_native_lossy(path: Path) -> bool:
    return path.suffix.lower() in _LOSSY_EXTS


def verdict_to_compression(verdict: str) -> Optional[str]:
    v = (verdict or "").strip().upper()
    if v in _LOSSLESS_VERDICTS:
        return "lossless"
    if v in _LOSSY_VERDICTS:
        return "lossy"
    return None


def _worker_count() -> int:
    try:
        from flac_detective.config import analysis_config

        n = int(getattr(analysis_config, "MAX_WORKERS", 0) or 0)
        if n > 0:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 4))


def _analyze_path_worker(
    path_str: str,
) -> tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Process-pool entry: (path, compression, verdict, note, error)."""
    path = Path(path_str)
    try:
        if _is_native_lossy(path):
            return path_str, "lossy", None, None, None
        if path.suffix.lower() not in _DETECT_EXTS:
            return path_str, None, None, None, None
        from flac_detective import FLACAnalyzer

        result = FLACAnalyzer().analyze_file(path_str)
        if not isinstance(result, dict):
            return path_str, None, None, None, "bad result"
        verdict = str(result.get("verdict", "")).strip().upper() or None
        value = verdict_to_compression(verdict or "")
        note = _verdict_note(result, verdict)
        return path_str, value, verdict, note, None
    except Exception as exc:
        return path_str, None, None, None, str(exc)


def _verdict_note(result: dict, verdict: Optional[str]) -> Optional[str]:
    """Plain-language reason for WARNING / SUSPICIOUS (flac-detective easy mode)."""
    if verdict not in ("WARNING", "SUSPICIOUS"):
        return None
    try:
        from flac_detective.presentation import plain_explanation

        text = (plain_explanation(result) or "").strip()
        if text:
            if verdict == "WARNING":
                text = text.replace(
                    "Most likely genuine.",
                    "Most likely fake.",
                )
            return text
    except Exception:
        pass
    try:
        from flac_detective.presentation import verdict_plain

        _icon, _label, action = verdict_plain(verdict)
        if verdict == "WARNING":
            return "A couple of mild oddities, but nothing conclusive. Most likely fake."
        text = (action or "").strip()
        return text or None
    except Exception:
        if verdict == "WARNING":
            return "A couple of mild oddities, but nothing conclusive. Most likely fake."
        return None


def detect_file_compression(path: Path, *, analyzer=None) -> Optional[str]:
    """Return lossless/lossy for one file, or None on failure / unsupported."""
    if _is_native_lossy(path):
        return "lossy"
    if path.suffix.lower() not in _DETECT_EXTS:
        return None
    if analyzer is None:
        from flac_detective import FLACAnalyzer

        analyzer = FLACAnalyzer()
    try:
        result = analyzer.analyze_file(str(path))
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return verdict_to_compression(str(result.get("verdict", "")))


def _log_file_result(
    log: LogFn,
    path: Path,
    value: str,
    verdict: Optional[str] = None,
    note: Optional[str] = None,
    *,
    index: int | None = None,
    total: int | None = None,
) -> None:
    """Gender-style LOG: === [i/n] file === + lossless/lossy + verdict (+ note).

    Blank after the block only when 2+ badge chips (value + verdict).
    """
    from ..run_summary import file_progress_header

    log(file_progress_header(path.name, index, total), "info")
    log(f"  {value}", "info")
    if not verdict:
        return
    label = verdict.strip().lower()
    note_text = (note or "").strip()
    if note_text and label in ("warning", "suspicious"):
        log(f"  {label}  {note_text}", "info")
    else:
        log(f"  {label}", "info")
    # value + verdict → two chips → blank; single value stays tight
    log("", "info")


def _log_compression_summary(
    log: LogFn,
    *,
    files: int,
    written: int,
    skipped: int,
    errors: int,
    error_files: list[str],
    lossless_n: int,
    lossy_n: int,
    elapsed: float,
) -> None:
    stat_lines: list[tuple[str, str]] = [
        (f"Tagged: {written:,}", "info"),
    ]
    if lossless_n:
        stat_lines.append((f"Lossless: {lossless_n:,}", "ok"))
    if lossy_n:
        stat_lines.append((f"Lossy: {lossy_n:,}", "err"))
    if skipped:
        # Own line so log_panel paints "  Skipped…" yellow (same as Corruption).
        stat_lines.append(
            (f"Skipped (COMPRESSION=lossless or lossy): {skipped:,}", "info")
        )
    if errors:
        stat_lines.append((f"Errors: {errors:,}", "err"))
        for name in error_files:
            stat_lines.append((f"  {name}", "err"))
    emit_run_summary(
        log,
        "Compression",
        elapsed=elapsed,
        files=files,
        stat_lines=stat_lines,
    )


def run_compression_detect(
    paths: list[Path],
    *,
    skip_existing: bool = True,
    on_log: Optional[LogFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
    max_workers: Optional[int] = None,
) -> dict:
    """Analyze files (multiprocess) and write COMPRESSION tags."""

    def log(msg: str, tag: str = "info") -> None:
        if on_log:
            on_log(msg, tag)

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    try:
        import flac_detective  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "flac-detective is required. Install with:\n"
            "  pip install flac-detective==1.7.0"
        ) from exc

    t0 = time.monotonic()
    total = max(1, len(paths))
    written = skipped = errors = lossless_n = lossy_n = 0
    error_files: list[str] = []
    to_analyze: list[Path] = []

    def live(n: int, count: int, label: str) -> None:
        log(f"__live_progress__\t{int(n)}\t{int(count)}\t{label}", "")

    def record_error(path: Path) -> None:
        nonlocal errors
        errors += 1
        error_files.append(path.name)

    def finish() -> dict:
        _log_compression_summary(
            log,
            files=len(paths),
            written=written,
            skipped=skipped,
            errors=errors,
            error_files=error_files,
            lossless_n=lossless_n,
            lossy_n=lossy_n,
            elapsed=time.monotonic() - t0,
        )
        return _summary(written, skipped, errors, lossless_n, lossy_n)

    # Fast path on the calling thread: native lossy + skip-existing.
    live(0, total, "files checked")
    for i, path in enumerate(paths, start=1):
        if stopped():
            break
        if on_progress:
            on_progress(i, total, "files checked")
        if i == 1 or i == total or i % 25 == 0:
            live(i, total, "files checked")

        existing = read_custom_tag(path, "COMPRESSION").lower()
        if skip_existing and existing in ("lossless", "lossy"):
            skipped += 1
            if existing == "lossless":
                lossless_n += 1
            else:
                lossy_n += 1
            continue

        if _is_native_lossy(path):
            value = "lossy"
            if skip_existing and existing == value:
                skipped += 1
                lossy_n += 1
                continue
            if write_compression_tag(path, value):
                written += 1
                lossy_n += 1
                _log_file_result(log, path, value, index=i, total=total)
            else:
                record_error(path)
                log(f"Write failed · {path.name}", "err")
            continue

        if path.suffix.lower() not in _DETECT_EXTS:
            skipped += 1
            log(f"No verdict · {path.name}", "detail")
            continue

        to_analyze.append(path)

    log("__live_progress_end__", "")

    if not stopped() and to_analyze:
        workers = max(1, int(max_workers or _worker_count()))
        log(
            f"Spectral analyze {len(to_analyze):,} file(s) · {workers} CPU worker(s)",
            "info",
        )
        log("", "info")

        done_spectral = 0
        spectral_total = max(1, len(to_analyze))
        live(0, spectral_total, "files analyzed")

        # Process pool — flac-detective CLI uses the same pattern.
        # Log "stopped" only after the pool fully shuts down (workers released).
        with integrity_pool(max_workers=workers) as pool:
            futures = {
                pool.submit(_analyze_path_worker, str(p)): p for p in to_analyze
            }
            for future in as_completed(futures):
                if stopped():
                    for f in futures:
                        f.cancel()
                    break
                path = futures[future]
                done_spectral += 1
                if on_progress:
                    on_progress(
                        done_spectral,
                        spectral_total,
                        "files analyzed",
                    )
                if (
                    done_spectral == 1
                    or done_spectral == spectral_total
                    or done_spectral % 5 == 0
                ):
                    live(done_spectral, spectral_total, "files analyzed")
                try:
                    _path_str, value, verdict, note, err = future.result()
                except Exception as exc:
                    record_error(path)
                    log(f"Analyze failed · {path.name}: {exc}", "err")
                    continue
                if err:
                    record_error(path)
                    log(f"Analyze failed · {path.name}: {err}", "err")
                    continue
                if value is None:
                    skipped += 1
                    log(f"No verdict · {path.name}", "detail")
                    continue
                existing = read_custom_tag(path, "COMPRESSION").lower()
                if skip_existing and existing == value:
                    skipped += 1
                    if value == "lossless":
                        lossless_n += 1
                    else:
                        lossy_n += 1
                    continue
                if write_compression_tag(path, value):
                    written += 1
                    if value == "lossless":
                        lossless_n += 1
                    else:
                        lossy_n += 1
                    _log_file_result(
                        log, path, value, verdict, note,
                        index=done_spectral, total=spectral_total,
                    )
                else:
                    record_error(path)
                    log(f"Write failed · {path.name}", "err")

        log("__live_progress_end__", "")

    if stopped():
        log("Compression detect stopped.", "warn")
    return finish()


def _summary(
    written: int, skipped: int, errors: int, lossless_n: int, lossy_n: int
) -> dict:
    return {
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "lossless": lossless_n,
        "lossy": lossy_n,
    }

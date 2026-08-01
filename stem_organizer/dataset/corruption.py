"""Detect audio corruption — Fast / Deep / Both + optional Fix / Quarantine."""

from __future__ import annotations

import subprocess
import threading
import time
from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from ..integrity_pool import integrity_pool
from ..meta_tags import read_corruption_tag, write_corruption_tag
from ..run_summary import emit_run_summary
from ..corruption.deep import verify_deep
from ..corruption.fast_mp3 import MP3_EXTS, verify_mp3_fast
from ..corruption.fix import fix_file, quarantine_file, remove_fixed_sibling
from ..corruption.tools import find_ffmpeg, find_flac, find_mp3val

try:
    from ffmpeg_bootstrap import subprocess_kwargs
except Exception:  # pragma: no cover
    def subprocess_kwargs() -> dict:
        return {}

LogFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, str], None]

CorruptionMode = Literal["fast", "deep", "both"]
CORRUPTION_STATUSES = ("ok", "minor", "failed", "suspect")


@dataclass
class FileVerdict:
    path: Path
    status: str
    detail: str = ""
    fast_ok: Optional[bool] = None
    deep_status: Optional[str] = None


def _mp3val_check(path: Path, mp3val_bin: str) -> tuple[bool, str]:
    """Return (ok, detail). mp3val exit 0 = clean; warnings still often exit 0."""
    try:
        proc = subprocess.run(
            [mp3val_bin, "-si", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            **subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return True, f"mp3val skip: {exc}"  # don't fail Fast if tool errors
    out = ((proc.stdout or "") + (proc.stderr or "")).lower()
    if "error:" in out or "broken" in out or "garbage" in out:
        # Extract last ERROR line
        detail = "mp3val error"
        for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
            if "error" in line.lower() or "warning" in line.lower():
                detail = line.strip()[-160:]
        # Treat ERROR as fail; WARNING alone → still fail Fast (structural)
        if "error:" in out:
            return False, detail
        if "warning:" in out:
            return False, detail
    return True, ""


def _fast_check(path: Path, mp3val_bin: str | None) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    if suffix in MP3_EXTS:
        r = verify_mp3_fast(path)
        if not r.ok:
            return False, r.error or "structural fail"
        if mp3val_bin:
            ok, detail = _mp3val_check(path, mp3val_bin)
            if not ok:
                return False, detail or "mp3val"
        return True, ""
    if suffix in (".flac", ".fla"):
        # Catch illegal/unsupported STREAMINFO early (Deep may use ffmpeg only).
        try:
            from mutagen.flac import FLAC

            bits = getattr(FLAC(str(path)).info, "bits_per_sample", None)
            if bits is not None and int(bits) > 24:
                return False, f"FLAC bit depth {int(bits)} unsupported (max 24)"
        except Exception as exc:
            return False, str(exc)[:160]
        return True, ""
    # Other formats: Fast path has no structural walker — Deep handles
    return True, "n/a"


def _merge_status(
    *,
    mode: CorruptionMode,
    fast_ok: Optional[bool],
    deep_status: Optional[str],
) -> tuple[str, str]:
    """Return (status, note)."""
    if mode == "fast":
        if fast_ok is False:
            return "suspect", "fast structural fail"
        return "ok", ""
    if mode == "deep":
        return deep_status or "failed", ""
    # both
    deep = deep_status or "failed"
    if fast_ok is False and deep == "ok":
        return "suspect", "fast fail, deep ok"
    if fast_ok is False and deep == "minor":
        return "minor", "fast fail, deep minor"
    if fast_ok is False:
        return "failed", "fast+deep fail"
    return deep, ""


def _analyze_one(
    path_s: str,
    mode: str,
    mp3val_bin: str | None,
    flac_bin: str | None,
    ffmpeg_bin: str | None,
) -> tuple[str, str, str, Optional[bool], Optional[str]]:
    """Picklable worker: path, status, detail, fast_ok, deep_status."""
    path = Path(path_s)
    fast_ok: Optional[bool] = None
    deep_status: Optional[str] = None
    detail = ""

    if mode in ("fast", "both"):
        ok, d = _fast_check(path, mp3val_bin)
        fast_ok = ok
        if not ok:
            detail = d

    if mode in ("deep", "both"):
        dr = verify_deep(path, flac_bin=flac_bin, ffmpeg_bin=ffmpeg_bin)
        deep_status = dr.status
        if dr.detail and (not detail or deep_status == "failed"):
            detail = dr.detail

    status, note = _merge_status(
        mode=mode,  # type: ignore[arg-type]
        fast_ok=fast_ok,
        deep_status=deep_status,
    )
    if note and not detail:
        detail = note
    return path_s, status, detail, fast_ok, deep_status


def run_corruption_detect(
    paths: list[Path],
    *,
    mode: CorruptionMode = "both",
    skip_existing: bool = True,
    do_fix: bool = False,
    do_quarantine: bool = False,
    input_root: str | Path = "",
    mp3val_path: str = "",
    on_log: Optional[LogFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
    max_workers: Optional[int] = None,
) -> dict:
    """Scan files, write CORRUPTION tags, optional fix/quarantine.

    When ``skip_existing``, files already tagged ``CORRUPTION=ok`` are skipped
    (failed/suspect/minor are still re-checked).
    """

    def log(msg: str, tag: str = "info") -> None:
        if on_log:
            on_log(msg, tag)

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    t0 = time.monotonic()
    ffmpeg_bin = find_ffmpeg()
    flac_bin = find_flac(ensure=False)
    mp3val_bin = find_mp3val(mp3val_path, ensure=False)
    if mode in ("deep", "both") and not flac_bin:
        log("flac not found locally — downloading once into flac\\ …", "info")
        flac_bin = find_flac(ensure=True)
        if flac_bin:
            log(f"  flac: {flac_bin}", "info")
        else:
            log(
                "  flac download failed — Deep FLAC verify uses ffmpeg only "
                "(no STREAMINFO MD5 check).",
                "warn",
            )
    if not mp3val_bin:
        log("mp3val not found locally — downloading once into mp3val\\ …", "info")
        mp3val_bin = find_mp3val(mp3val_path, ensure=True)

    if mode in ("deep", "both") and not ffmpeg_bin and not flac_bin:
        log("ffmpeg not found — Deep verify limited (FLAC-only if flac present).", "warn")
    if mode in ("fast", "both") and not mp3val_bin:
        log("mp3val unavailable — Fast uses built-in MP3 frame walk only.", "info")
    if do_fix and not mp3val_bin:
        log("mp3val unavailable — MP3 Fix will use ffmpeg remux/re-encode.", "warn")
    if mp3val_bin:
        log(f"mp3val: {mp3val_bin}", "info")
    if ffmpeg_bin:
        log(f"ffmpeg: {ffmpeg_bin}", "info")

    counts = {k: 0 for k in CORRUPTION_STATUSES}
    skipped = errors = fixed = quarantined = 0
    to_scan: list[Path] = []

    def live(n: int, total: int, label: str) -> None:
        log(f"__live_progress__\t{int(n)}\t{int(total)}\t{label}", "")

    n_paths = len(paths)
    if skip_existing and n_paths:
        live(0, n_paths, "files read")
        for i, path in enumerate(paths, start=1):
            if stopped():
                break
            if on_progress and (i == 1 or i == n_paths or i % 25 == 0):
                on_progress(i, max(1, n_paths), "files read")
            if i == 1 or i == n_paths or i % 25 == 0:
                live(i, n_paths, "files read")
            existing = read_corruption_tag(path)
            if existing == "ok":
                skipped += 1
                counts["ok"] = counts.get("ok", 0) + 1
                continue
            to_scan.append(path)
        log("__live_progress_end__", "")
    else:
        for path in paths:
            if stopped():
                break
            to_scan.append(path)

    total = max(1, len(to_scan))
    workers = max(1, int(max_workers or min(8, (os_cpu_count() or 4))))
    log(
        f"Mode={mode} · {len(to_scan):,} to scan · {skipped:,} skipped (ok) · {workers} worker(s)",
        "info",
    )
    log("", "info")

    results: list[FileVerdict] = []
    root = Path(input_root) if str(input_root).strip() else None
    # Quarantine folder for checkbox + unfixable leftovers after a failed Fix
    quarantine_root = None
    if root is not None and (do_quarantine or do_fix):
        quarantine_root = root.parent / f"{root.name}_CORRUPT"

    if to_scan and not stopped():
        live(0, len(to_scan), "files scanned")
        # Process pool — picklable top-level worker
        with integrity_pool(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _analyze_one,
                    str(p),
                    mode,
                    mp3val_bin,
                    flac_bin,
                    ffmpeg_bin,
                ): p
                for p in to_scan
            }
            done = 0
            for fut in as_completed(futures):
                if stopped():
                    for f in futures:
                        f.cancel()
                    break
                path = futures[fut]
                done += 1
                if on_progress:
                    on_progress(done, total, "files scanned")
                if done == 1 or done == len(to_scan) or done % 5 == 0:
                    live(done, len(to_scan), "files scanned")
                try:
                    path_s, status, detail, fast_ok, deep_status = fut.result()
                except Exception as exc:
                    errors += 1
                    log(f"Analyze failed · {path.name}: {exc}", "err")
                    continue

                verdict = FileVerdict(
                    path=Path(path_s),
                    status=status,
                    detail=detail,
                    fast_ok=fast_ok,
                    deep_status=deep_status,
                )
                results.append(verdict)
                counts[status] = counts.get(status, 0) + 1

                _log_file_result(
                    log, verdict.path, status, detail,
                    index=done, total=len(to_scan),
                )

                tag_path = verdict.path
                tag_status = status
                replaced = False
                removed = False

                if do_fix and status in ("failed", "suspect", "minor"):
                    fr = fix_file(
                        verdict.path, mp3val_bin=mp3val_bin, ffmpeg_bin=ffmpeg_bin
                    )
                    if fr.ok and fr.out_path:
                        fixed_path = Path(fr.out_path)
                        how = fr.detail or "ok"
                        # Tag the clean rebuild first (corrupt originals often cannot
                        # accept tags), then replace the original so we never keep
                        # original + *_FIXED side by side.
                        if not write_corruption_tag(fixed_path, "ok"):
                            log(
                                "  [warn] could not write CORRUPTION tag on fixed file",
                                "warn",
                            )
                        try:
                            orig = verdict.path
                            if fixed_path.resolve() != orig.resolve():
                                # Wrong-extension fix may change suffix (.wav→.mp3);
                                # keep the corrected name instead of renaming back.
                                if fixed_path.suffix.lower() != orig.suffix.lower():
                                    final = orig.with_suffix(fixed_path.suffix)
                                    if (
                                        final.exists()
                                        and final.resolve() != fixed_path.resolve()
                                    ):
                                        final = fixed_path  # already *_FIXED.ext
                                    orig.unlink(missing_ok=True)
                                    if fixed_path.resolve() != final.resolve():
                                        if final.exists():
                                            final.unlink()
                                        fixed_path.replace(final)
                                    tag_path = final
                                    log(
                                        f"  fixed -> {final.name} ({how})",
                                        "detail",
                                    )
                                else:
                                    orig.unlink(missing_ok=True)
                                    fixed_path.replace(orig)
                                    tag_path = orig
                                    log(
                                        f"  fixed -> replaced original ({how})",
                                        "detail",
                                    )
                            else:
                                tag_path = orig
                                log(
                                    f"  fixed -> replaced original ({how})",
                                    "detail",
                                )
                            replaced = True
                            tag_status = "ok"
                            fixed += 1
                            counts[status] = max(0, counts.get(status, 0) - 1)
                            counts["ok"] = counts.get("ok", 0) + 1
                        except OSError as exc:
                            # Keep *_FIXED; original may still be present
                            tag_path = fixed_path
                            tag_status = "ok"
                            fixed += 1
                            counts[status] = max(0, counts.get(status, 0) - 1)
                            counts["ok"] = counts.get("ok", 0) + 1
                            replaced = True  # already tagged ok on fixed_path
                            log(
                                f"  fixed -> {fixed_path.name} ({how}); "
                                f"could not replace original: {exc}",
                                "warn",
                            )
                    else:
                        # Never leave a destroyed *_FIXED beside the original
                        remove_fixed_sibling(fr.out_path or verdict.path)
                        log(
                            f"  [warn] fix failed: {fr.detail or 'unknown'}",
                            "log_note",
                        )
                        # Unfixable (failed) originals → quarantine (not delete)
                        if (
                            status == "failed"
                            and verdict.path.exists()
                            and quarantine_root is not None
                        ):
                            ok_q, dest = quarantine_file(
                                verdict.path, quarantine_root
                            )
                            if ok_q:
                                removed = True
                                quarantined += 1
                                counts[status] = max(0, counts.get(status, 0) - 1)
                                log(
                                    f"  quarantined (unfixable) -> {dest}",
                                    "log_note_err",
                                )
                            else:
                                log(
                                    f"  [warn] quarantine failed: {dest}",
                                    "warn",
                                )

                # Tag after Fix attempt (skipped above when replace already tagged ok)
                if not replaced and not removed:
                    if not write_corruption_tag(tag_path, tag_status):
                        # Broken files often cannot hold tags — warn only
                        log("  [warn] could not write CORRUPTION tag", "warn")

                # Quarantine only if still bad and the original path still exists
                still_bad = tag_status in ("failed", "suspect") and not replaced and not removed
                if do_quarantine and quarantine_root and still_bad and verdict.path.exists():
                    ok_q, dest = quarantine_file(verdict.path, quarantine_root)
                    if ok_q:
                        quarantined += 1
                        log(f"  quarantined -> {dest}", "warn")
                    else:
                        log(f"  [warn] quarantine failed: {dest}", "warn")

    if stopped():
        log("Corruption detect stopped.", "warn")

    elapsed = time.monotonic() - t0
    stat_lines: list[tuple[str, str]] = [
        (f"Scanned: {len(results):,}", "info"),
    ]
    for key in CORRUPTION_STATUSES:
        if counts.get(key):
            stat_lines.append((f"{key}: {counts[key]:,}", "info"))
    if skipped:
        stat_lines.append((f"Skipped (CORRUPTION=ok): {skipped:,}", "info"))
    if fixed:
        stat_lines.append((f"Fixed: {fixed:,}", "ok"))
    if quarantined:
        stat_lines.append((f"Quarantined: {quarantined:,}", "warn"))
    if errors:
        stat_lines.append((f"Errors: {errors:,}", "err"))
    emit_run_summary(
        log,
        "Corruption",
        elapsed=elapsed,
        stat_lines=stat_lines,
    )

    return {
        "scanned": len(results),
        "skipped": skipped,
        "errors": errors,
        "fixed": fixed,
        "quarantined": quarantined,
        "counts": counts,
    }


def _log_file_result(
    log: LogFn,
    path: Path,
    status: str,
    detail: str = "",
    *,
    index: int | None = None,
    total: int | None = None,
) -> None:
    """Compression-style LOG: === [i/n] file === + badge (+ note for bad/warn).

    minor / suspect display as a single ``warning`` badge (yellow bg, dark text).
    """
    from ..run_summary import file_progress_header

    log(file_progress_header(path.name, index, total), "info")
    label = str(status).strip().lower() or "failed"
    display = "warning" if label in ("minor", "suspect") else label
    note = " ".join(str(detail or "").split()).strip()
    if note and label in ("minor", "failed", "suspect"):
        # Two spaces before note → log_panel badge + log_note
        log(f"  {display}  {note}", "info")
    else:
        log(f"  {display}", "info")
    # Single badge chip — no blank after (blank only for 2+ badges elsewhere)


def os_cpu_count() -> Optional[int]:
    import os

    return os.cpu_count()

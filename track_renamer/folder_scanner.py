"""Scan folders and rename files on disk."""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from track_renamer.engine.models import Track, TrackType


def _is_sharing_violation(exc: OSError) -> bool:
    """True for Windows WinError 32 (file in use by another process)."""
    return getattr(exc, "winerror", None) == 32 or getattr(exc, "errno", None) == 32


def _rename_with_retry(
    src: Path,
    dst: Path,
    *,
    attempts: int = 5,
    delays: tuple[float, ...] = (0.2, 0.4, 0.8, 1.2),
) -> None:
    """Rename with backoff retries on sharing violations (preview/tagger locks)."""
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            src.rename(dst)
            return
        except OSError as exc:
            last_exc = exc
            if attempt + 1 < attempts and _is_sharing_violation(exc):
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise
    if last_exc is not None:
        raise last_exc


def _move_with_retry(
    src: Path,
    dst: Path,
    *,
    attempts: int = 5,
    delays: tuple[float, ...] = (0.2, 0.4, 0.8, 1.2),
) -> None:
    """shutil.move with backoff retries on sharing violations."""
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.move(str(src), str(dst))
            return
        except OSError as exc:
            last_exc = exc
            if attempt + 1 < attempts and _is_sharing_violation(exc):
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise
    if last_exc is not None:
        raise last_exc


AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg", ".m4a", ".wma"}
MIDI_EXTENSIONS = {".mid", ".midi"}
DEFAULT_EXTENSIONS = AUDIO_EXTENSIONS | MIDI_EXTENSIONS

_BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.I)
_KEY_RE = re.compile(r"\b([A-G][#b]?(?:m|min|maj|major|minor)?)\b", re.I)
_PREFIX_RE = re.compile(r"^\s*(.+?)\s+-\s+")


def _track_type_for_extension(ext: str) -> TrackType:
    ext = ext.lower()
    if ext in MIDI_EXTENSIONS:
        return "midi"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "audio"


def _parse_filename_metadata(stem: str) -> tuple[str, str]:
    bpm = ""
    key = ""
    bpm_match = _BPM_RE.search(stem)
    if bpm_match:
        bpm = bpm_match.group(1)
    key_match = _KEY_RE.search(stem)
    if key_match:
        key = key_match.group(1)
    return bpm, key


def scan_folder(
    root: Path,
    *,
    recursive: bool = True,
    extensions: set[str] | None = None,
    progress: Callable[[int], None] | None = None,
) -> list[Track]:
    """Scan a folder and return one Track per matching file."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or DEFAULT_EXTENSIONS)}
    paths: list[Path] = []

    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        paths.append(path)
        if progress and len(paths) % 1000 == 0:
            progress(len(paths))

    paths.sort()
    tracks: list[Track] = []
    for path in paths:
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        parent_folder = path.parent.name if path.parent != root else ""
        stem = path.stem
        bpm, key = _parse_filename_metadata(stem)

        tracks.append(
            Track(
                id=str(path),
                name=stem,
                track_type=_track_type_for_extension(path.suffix.lower()),
                parent_id=str(path.parent) if depth > 0 else None,
                depth=depth,
                file_path=path,
                extension=path.suffix.lower(),
                relative_path=str(rel),
                group=parent_folder,
                bpm=bpm,
                key=key,
            )
        )
        if progress and len(tracks) % 1000 == 0:
            progress(len(tracks))

    if progress:
        progress(len(tracks))
    return tracks


def apply_file_renames_detailed(
    renames: dict[str, str],
) -> tuple[int, list[str], list[Path]]:
    """
    Rename files on disk.

    *renames* maps track id (absolute file path) → new stem (without extension).
    Returns (success_count, error_messages, successfully_renamed_paths).
    """
    errors: list[str] = []
    success = 0
    renamed_paths: list[Path] = []

    validated: list[tuple[Path, str]] = []
    for file_id, new_stem in renames.items():
        source = Path(file_id)
        if not source.exists():
            errors.append(f"Missing: {source.name}")
            continue
        if not new_stem or new_stem.strip() == "":
            errors.append(f"Empty name for: {source.name}")
            continue

        # Disallow path separators in new names.
        if any(sep in new_stem for sep in ("/", "\\", ":")):
            errors.append(f"Invalid characters in new name for: {source.name}")
            continue
        validated.append((source, new_stem.strip()))

    def path_key(path: Path) -> str:
        return str(path.resolve(strict=False)).casefold()

    validated = [
        (source, new_stem)
        for source, new_stem in validated
        if source.with_name(new_stem + source.suffix) != source
    ]
    source_keys = {path_key(source) for source, _stem in validated}
    reserved: set[str] = set()
    planned: list[tuple[Path, Path]] = []

    for source, new_stem in validated:
        target = source.with_name(new_stem + source.suffix)

        suffix = 0
        candidate = target
        while True:
            key = path_key(candidate)
            occupied_on_disk = candidate.exists() and key not in source_keys
            if key not in reserved and not occupied_on_disk:
                break
            suffix += 1
            candidate = source.with_name(
                f"{new_stem}_{suffix}{source.suffix}"
            )

        reserved.add(path_key(candidate))
        planned.append((source, candidate))

    # Stage every source first so swaps and targets occupied by another source
    # cannot fail midway through the operation.
    staged: list[tuple[Path, Path, Path]] = []
    for source, target in planned:
        temporary = source.with_name(
            f".__track_renamer_{uuid.uuid4().hex}{source.suffix}"
        )
        try:
            _rename_with_retry(source, temporary)
            staged.append((source, temporary, target))
        except OSError as exc:
            errors.append(f"{source.name}: {exc}")

    for source, temporary, target in staged:
        try:
            _rename_with_retry(temporary, target)
            success += 1
            renamed_paths.append(target)
            try:
                from track_renamer.instrument_enrich import relocate_instrument_cache

                relocate_instrument_cache(source, target)
            except Exception:
                pass
        except OSError as exc:
            errors.append(f"{source.name}: {exc}")
            try:
                _rename_with_retry(temporary, source)
            except OSError as restore_exc:
                errors.append(f"Could not restore {source.name}: {restore_exc}")

    return success, errors, renamed_paths


def apply_file_renames(renames: dict[str, str]) -> tuple[int, list[str]]:
    """Rename files while preserving the original public return shape."""
    success, errors, _renamed_paths = apply_file_renames_detailed(renames)
    return success, errors


def move_files_to_prefix_folders(
    files: list[Path],
    root: Path,
) -> tuple[int, int, list[str]]:
    """
    Move files into ``root / PREFIX`` using their ``PREFIX - name`` pattern.

    Returns (moved_count, skipped_without_prefix_count, error_messages).
    Existing destination names receive an incrementing numeric suffix.
    """
    moved = 0
    skipped = 0
    errors: list[str] = []
    root = root.resolve()

    for source_value in files:
        source = Path(source_value)
        if not source.is_file():
            errors.append(f"Missing: {source.name}")
            continue

        match = _PREFIX_RE.match(source.stem)
        if not match:
            skipped += 1
            continue

        prefix = match.group(1).strip()
        if prefix in {"", ".", ".."}:
            errors.append(f"{source.name}: invalid prefix folder")
            continue
        target_folder = root / prefix
        try:
            target_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"{source.name}: could not create {prefix}: {exc}")
            continue

        target = target_folder / source.name
        if source.resolve(strict=False) == target.resolve(strict=False):
            continue

        suffix = 0
        candidate = target
        while candidate.exists():
            suffix += 1
            candidate = target.with_name(
                f"{target.stem}_{suffix}{target.suffix}"
            )

        try:
            _move_with_retry(source, candidate)
            moved += 1
            try:
                from track_renamer.instrument_enrich import relocate_instrument_cache

                relocate_instrument_cache(source, candidate)
            except Exception:
                pass
        except OSError as exc:
            errors.append(f"{source.name}: {exc}")

    return moved, skipped, errors

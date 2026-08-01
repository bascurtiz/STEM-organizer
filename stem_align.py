from __future__ import annotations

import re
import shutil
from math import gcd
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from pair_matcher import (
    AUDIO_EXTS,
    IgnoreRules,
    LOG_INDENT,
    LogFn,
    ProgressFn,
    _progress_header,
    _report_log,
    _report_progress,
    normalize_tag,
)

ORIGINAL_SUFFIX = '(original song)'
ORIGINAL_MARKERS = ('original song', 'original')
SORT_SUBDIR_WITH = 'with_original'
SORT_SUBDIR_WITHOUT = 'without_original'
LEGACY_SORT_SUBDIR_WITH = '_with original'
LEGACY_SORT_SUBDIR_WITHOUT = '_without original'
SORT_BUCKET_NAMES = frozenset({
    SORT_SUBDIR_WITH,
    SORT_SUBDIR_WITHOUT,
    LEGACY_SORT_SUBDIR_WITH,
    LEGACY_SORT_SUBDIR_WITHOUT,
})
ACA_MARKERS = (
    'acapella', 'aca', 'vocals', 'vocal', 'lead', 'bgv', 'backing vocals', 'backing vocal',
)
INST_MARKERS = ('instrumental', 'inst', 'karaoke')
ROLE_RE = re.compile(r'\(([^)]+)\)\s*$')
BACKUP_DIR_NAME = '_backup_before_align'

DEFAULT_SR = 22050
DEFAULT_ANALYSIS_SEC = 30.0
MAX_SHIFT_SEC = 120.0
PROGRESS_EVERY = 25
DEFAULT_FOLDER_MATCH_RULES = IgnoreRules(
    ignore_parentheses=True,
    ignore_square_brackets=True,
    ignore_extra_spaces=True,
)
DEFAULT_MATCH_THRESHOLD = 0.60


@dataclass(frozen=True)
class SongFolder:
    path: Path
    name: str
    instrumental: Path | None
    acapella: Path | None
    original: Path | None


@dataclass(frozen=True)
class AlignResult:
    folder: Path
    instrumental_shift_sec: float
    acapella_shift_sec: float
    vocal_onset_original_sec: float
    vocal_onset_acapella_sec: float
    output_paths: tuple[Path, Path]


@dataclass(frozen=True)
class FolderMatch:
    folder: Path
    score: float


def _is_sort_bucket_dir(path: Path) -> bool:
    return path.name.lower() in {name.lower() for name in SORT_BUCKET_NAMES}


def list_song_subfolders(root: Path) -> list[Path]:
    """All song subfolders: direct children plus any inside sort subdirs."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if _is_sort_bucket_dir(path):
            out.extend(sorted(p for p in path.iterdir() if p.is_dir()))
        else:
            out.append(path)
    return out


def list_unsorted_song_subfolders(root: Path) -> list[Path]:
    """Direct song subfolders still sitting in the library root (not yet sorted)."""
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not _is_sort_bucket_dir(p)
    )


def default_with_original_dir(stems_root: Path) -> Path:
    return stems_root / SORT_SUBDIR_WITH


def default_without_original_dir(stems_root: Path) -> Path:
    return stems_root / SORT_SUBDIR_WITHOUT


def resolve_with_original_dir(stems_root: Path, with_original_dir: Path | None = None) -> Path:
    if with_original_dir is not None:
        return with_original_dir
    primary = default_with_original_dir(stems_root)
    if primary.is_dir():
        return primary
    legacy = stems_root / LEGACY_SORT_SUBDIR_WITH
    if legacy.is_dir():
        return legacy
    return primary


def resolve_without_original_dir(stems_root: Path, without_original_dir: Path | None = None) -> Path:
    if without_original_dir is not None:
        return without_original_dir
    primary = default_without_original_dir(stems_root)
    if primary.is_dir():
        return primary
    legacy = stems_root / LEGACY_SORT_SUBDIR_WITHOUT
    if legacy.is_dir():
        return legacy
    return primary


def list_with_original_subfolders(
    stems_root: Path,
    with_original_dir: Path | None = None,
) -> list[Path]:
    """Song subfolders inside with_original only (used for alignment)."""
    with_dir = resolve_with_original_dir(stems_root, with_original_dir)
    if not with_dir.is_dir():
        return []
    return sorted(p for p in with_dir.iterdir() if p.is_dir())


def resolve_export_list_path(output_file: Path) -> Path:
    """Export target must be a .txt path. If user picks/pastes a folder, write inside it."""
    path = Path(output_file)
    if path.is_dir():
        return path / 'songs_to_download.txt'
    return path


def export_song_list(root: Path, output_file: Path) -> int:
    folders = list_song_subfolders(root)
    lines = [p.name for p in folders]
    out = resolve_export_list_path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)


def _strip_role_suffix(stem: str) -> str:
    text = stem.strip()
    text = re.sub(r'\s*\(original song\)\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(original\)\s*$', '', text, flags=re.IGNORECASE)
    while True:
        match = ROLE_RE.search(text)
        if not match:
            break
        role = match.group(1).strip().lower()
        if not _role_kind(role):
            break
        text = text[: match.start()].strip()
    return text


def _role_kind(role: str) -> str | None:
    role = role.strip().lower()
    if role in ('original song', 'original'):
        return 'original'
    if any(marker == role or role.endswith(f' {marker}') for marker in ACA_MARKERS):
        return 'acapella'
    if any(marker == role or role.endswith(f' {marker}') for marker in INST_MARKERS):
        return 'instrumental'
    if any(marker in role for marker in ACA_MARKERS):
        return 'acapella'
    if any(marker in role for marker in INST_MARKERS):
        return 'instrumental'
    return None


def _strip_trailing_bracket_tags(stem: str) -> str:
    """Drop trailing download tags like [MP3-320] so (role) suffixes can be detected."""
    text = stem.strip()
    while True:
        match = re.search(r'\s*\[[^\]]+\]\s*$', text, flags=re.IGNORECASE)
        if not match:
            break
        text = text[: match.start()].strip()
    return text


def classify_audio_file(path: Path) -> str | None:
    if path.suffix.lower() not in AUDIO_EXTS:
        return None
    stem_raw = path.stem
    stem = stem_raw.lower()
    if ORIGINAL_SUFFIX.lower() in stem or any(f'({m})' in stem for m in ORIGINAL_MARKERS):
        return 'original'
    stem_for_role = _strip_trailing_bracket_tags(stem_raw)
    match = ROLE_RE.search(stem_for_role)
    if match:
        kind = _role_kind(match.group(1))
        if kind:
            return kind
    lower = stem_for_role.lower()
    if any(f'({m})' in lower or lower.endswith(m) for m in ACA_MARKERS):
        return 'acapella'
    if any(m in lower for m in ('acapella', 'acappella', 'a capella')):
        return 'acapella'
    if any(f'({m})' in lower or lower.endswith(m) for m in INST_MARKERS):
        return 'instrumental'
    if 'instrumental' in lower:
        return 'instrumental'
    return None


def scan_song_folder(folder: Path) -> SongFolder:
    instrumental = acapella = original = None
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        role = classify_audio_file(path)
        if role == 'instrumental' and instrumental is None:
            instrumental = path
        elif role == 'acapella' and acapella is None:
            acapella = path
        elif role == 'original' and original is None:
            original = path
    return SongFolder(folder, folder.name, instrumental, acapella, original)


def name_similarity(
    a: str,
    b: str,
    rules: IgnoreRules | None = DEFAULT_FOLDER_MATCH_RULES,
) -> float:
    na = normalize_tag(_strip_role_suffix(a), rules)
    nb = normalize_tag(_strip_role_suffix(b), rules)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        if len(shorter) >= 8 and len(shorter) / len(longer) >= 0.55:
            return max(SequenceMatcher(None, na, nb).ratio(), 0.92)
    return SequenceMatcher(None, na, nb).ratio()


def _best_folder_match(
    filename_stem: str,
    folders: list[Path],
    *,
    rules: IgnoreRules | None = DEFAULT_FOLDER_MATCH_RULES,
) -> FolderMatch | None:
    base = _strip_role_suffix(filename_stem)
    best_score = 0.0
    best_folder: Path | None = None
    for folder in folders:
        score = name_similarity(base, folder.name, rules)
        if score > best_score:
            best_score = score
            best_folder = folder
    if best_folder is None:
        return None
    return FolderMatch(best_folder, best_score)


def match_original_to_folder(
    filename_stem: str,
    folders: list[Path],
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    rules: IgnoreRules | None = DEFAULT_FOLDER_MATCH_RULES,
) -> FolderMatch | None:
    best = _best_folder_match(filename_stem, folders, rules=rules)
    if best is not None and best.score >= threshold:
        return best
    return None


def distribute_originals(
    inbox: Path,
    stems_root: Path,
    *,
    on_log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    sort_after: bool = False,
    with_original_dir: Path | None = None,
    without_original_dir: Path | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Move inbox originals into song folders. Returns moved, skipped, unmatched, rejected, sorted_with, sorted_without."""
    folders = list_song_subfolders(stems_root)
    originals = sorted(
        p for p in inbox.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    total = max(1, len(originals))
    moved = skipped = unmatched = rejected = 0

    _report_log(on_log, f'{LOG_INDENT}Distributing {len(originals):,} file(s) to {len(folders):,} subfolder(s)', 'info')
    if not folders:
        _report_log(
            on_log,
            f'{LOG_INDENT}No song subfolders found under stems root (check path, or run Export to verify folder names).',
            'err',
        )

    candidates: list[tuple[Path, Path, float]] = []
    for idx, src in enumerate(originals):
        best = _best_folder_match(src.stem, folders)
        match = match_original_to_folder(src.stem, folders, threshold=match_threshold)
        _report_log(on_log, _progress_header(src.name, idx + 1, total), 'info')
        if match is None:
            unmatched += 1
            if best is not None and folders:
                _report_log(on_log, '  no folder match', 'err')
                _report_log(
                    on_log,
                    f'  closest: {best.folder.name} · {best.score:.0%}',
                    'detail',
                )
            else:
                _report_log(on_log, '  no folder match', 'err')
        else:
            candidates.append((src, match.folder, match.score))
            _report_log(
                on_log,
                f'  matched: {match.folder.name} · {match.score:.0%}',
                'info',
            )
        _report_progress(
            on_progress, idx + 1, total,
            f'Matching originals ({idx + 1:,}/{total:,})',
            step=idx + 1,
            force=False,
        )

    by_folder: dict[Path, list[tuple[Path, float]]] = {}
    for src, folder, score in candidates:
        by_folder.setdefault(folder, []).append((src, score))

    move_plan: list[tuple[Path, Path, float]] = []
    for folder, group in by_folder.items():
        group.sort(key=lambda item: item[1], reverse=True)
        winner_src, winner_score = group[0]
        song = scan_song_folder(folder)
        if song.original is not None:
            skipped += len(group)
            for src, score in group:
                _report_log(
                    on_log,
                    f'  [skip] folder already has original ← {src.name} ({score:.0%})',
                    'warn',
                )
            continue
        move_plan.append((winner_src, folder, winner_score))
        for src, score in group[1:]:
            rejected += 1
            _report_log(
                on_log,
                f'· Rejected weaker match {src.name} ({score:.0%}) — '
                f'keeping {winner_src.name} ({winner_score:.0%}) for {folder.name}',
                'warn',
            )

    for idx, (src, dest_folder, score) in enumerate(move_plan):
        dest_name = src.name
        if ORIGINAL_SUFFIX.lower() not in src.stem.lower():
            dest_name = f'{src.stem} {ORIGINAL_SUFFIX}{src.suffix}'
        dest = dest_folder / dest_name
        plan_total = max(1, len(move_plan))
        if dest.exists():
            skipped += 1
            _report_log(
                on_log,
                f'  [skip] exists {dest_folder.name}/{dest_name}',
                'warn',
            )
        else:
            shutil.move(str(src), str(dest))
            moved += 1
        _report_progress(
            on_progress, idx + 1, plan_total,
            f'Moving originals ({idx + 1:,}/{plan_total:,})',
            step=idx + 1,
            force=idx + 1 == plan_total,
        )

    sorted_with = sorted_without = 0
    if sort_after:
        sorted_with, sorted_without, sort_skipped = sort_folders_by_original(
            stems_root,
            with_original_dir=with_original_dir,
            without_original_dir=without_original_dir,
            on_log=on_log,
            on_progress=on_progress,
        )
        skipped += sort_skipped

    return moved, skipped, unmatched, rejected, sorted_with, sorted_without


def sort_folders_by_original(
    stems_root: Path,
    *,
    with_original_dir: Path | None = None,
    without_original_dir: Path | None = None,
    on_log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> tuple[int, int, int]:
    """Move unsorted song subfolders into with/without-original buckets. Returns with, without, skipped."""
    with_dir = with_original_dir or resolve_with_original_dir(stems_root)
    without_dir = without_original_dir or resolve_without_original_dir(stems_root)
    with_dir.mkdir(parents=True, exist_ok=True)
    without_dir.mkdir(parents=True, exist_ok=True)

    folders = list_unsorted_song_subfolders(stems_root)
    total = max(1, len(folders))
    moved_with = moved_without = skipped = 0

    _report_log(
        on_log,
        f'{LOG_INDENT}Sorting {len(folders):,} folder(s) → {with_dir.name} / {without_dir.name}',
        'info',
    )

    for idx, folder in enumerate(folders):
        song = scan_song_folder(folder)
        if song.original is not None:
            dest_parent = with_dir
            label = with_dir.name
            moved_with += 1
        else:
            dest_parent = without_dir
            label = without_dir.name
            moved_without += 1

        _report_log(on_log, _progress_header(folder.name, idx + 1, total), 'info')
        dest = dest_parent / folder.name
        if dest.exists():
            skipped += 1
            _report_log(on_log, f'  [skip] exists in {label}', 'warn')
        else:
            shutil.move(str(folder), str(dest))
            tag = 'ok' if song.original is not None else 'err'
            _report_log(on_log, f'  → {label}/', tag)

        _report_progress(
            on_progress, idx + 1, total,
            f'Sorting folders ({idx + 1:,}/{total:,})',
            step=idx + 1,
            force=idx + 1 == total,
        )

    return moved_with, moved_without, skipped


SOUNDFILE_READ_EXTS = frozenset({'.flac', '.wav', '.ogg', '.aiff', '.aif'})
STREAM_BLOCK_FRAMES = 262_144
MAX_SAMPLES_IN_MEMORY = 30_000_000
MAX_REASONABLE_FRAMES = 48_000 * 7_200  # ~2 hours at 48 kHz


@dataclass(frozen=True)
class AudioFileInfo:
    samplerate: int
    channels: int
    frames: int | None
    can_stream: bool


def _as_samples_channels(y: np.ndarray) -> np.ndarray:
    """Shape (samples,) or (samples, channels)."""
    if y.ndim == 1:
        return y
    if y.shape[0] < y.shape[1]:
        return y.T
    return y


def _to_sample_channel_matrix(y: np.ndarray, channels: int | None = None) -> np.ndarray:
    """Return audio as (samples, channels). Optionally expand/trim to channel count."""
    y = _as_samples_channels(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if channels is None:
        return y
    current = y.shape[1]
    if current == channels:
        return y
    if current == 1 and channels > 1:
        return np.repeat(y, channels, axis=1)
    return y[:, :channels]


def _sanity_frame_count(frames: int | None) -> int | None:
    if frames is None or frames <= 0 or frames > MAX_REASONABLE_FRAMES:
        return None
    return frames


def _fits_in_memory(frames: int | None, channels: int) -> bool:
    if frames is None:
        return False
    return frames * max(channels, 1) <= MAX_SAMPLES_IN_MEMORY


def _probe_audio(path: Path) -> AudioFileInfo:
    ext = path.suffix.lower()
    if ext in SOUNDFILE_READ_EXTS:
        import soundfile as sf

        info = sf.info(str(path))
        return AudioFileInfo(
            samplerate=info.samplerate,
            channels=info.channels,
            frames=_sanity_frame_count(info.frames),
            can_stream=True,
        )

    try:
        from mutagen import File as MutagenFile

        meta = MutagenFile(path)
        if meta is not None and meta.info is not None:
            sample_rate = int(getattr(meta.info, 'sample_rate', 0) or 44_100)
            channels = int(getattr(meta.info, 'channels', 0) or 2)
            length = float(getattr(meta.info, 'length', 0) or 0)
            frames = _sanity_frame_count(int(length * sample_rate)) if length > 0 else None
            return AudioFileInfo(sample_rate, channels, frames, False)
    except Exception:
        pass
    return AudioFileInfo(44_100, 2, None, False)


def _output_soundfile_format(path: Path) -> str:
    return {'.flac': 'FLAC', '.wav': 'WAV', '.ogg': 'OGG'}.get(path.suffix.lower(), 'FLAC')


def _write_silence_block(
    outf,
    frames: int,
    channels: int,
    *,
    block: int = STREAM_BLOCK_FRAMES,
) -> None:
    if frames <= 0:
        return
    written = 0
    while written < frames:
        count = min(block, frames - written)
        outf.write(np.zeros((count, channels), dtype='float32'))
        written += count


def _resample_ratio(in_sr: int, out_sr: int) -> tuple[int, int]:
    g = gcd(in_sr, out_sr)
    return out_sr // g, in_sr // g


def _resample_block(block: np.ndarray, up: int, down: int) -> np.ndarray:
    from scipy.signal import resample_poly

    block = _to_sample_channel_matrix(block)
    if block.shape[1] == 1:
        return resample_poly(block[:, 0], up, down)[:, np.newaxis]
    channels = [resample_poly(block[:, ch], up, down) for ch in range(block.shape[1])]
    min_len = min(len(ch) for ch in channels)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def _align_stem_stream_resample(
    src: Path,
    dest: Path,
    *,
    in_sr: int,
    target_sr: int,
    target_frames: int,
    delay_sec: float,
    out_channels: int,
) -> tuple[float, float]:
    """Stream-read, resample, align, and write without loading the full file."""
    import soundfile as sf

    pad_sec = trim_sec = 0.0
    pad_frames = trim_frames_in = 0
    if delay_sec >= 0.0005:
        pad_sec = delay_sec
        pad_frames = int(round(pad_sec * target_sr))
    elif delay_sec <= -0.0005:
        trim_sec = -delay_sec
        trim_frames_in = int(round(trim_sec * in_sr))

    up, down = _resample_ratio(in_sr, target_sr)
    tmp = dest.with_name(dest.stem + '._align_tmp' + dest.suffix)
    if tmp.exists():
        tmp.unlink()

    with sf.SoundFile(str(src)) as inf:
        if inf.samplerate != in_sr:
            in_sr = inf.samplerate
            up, down = _resample_ratio(in_sr, target_sr)
        channels = out_channels
        with sf.SoundFile(
            str(tmp),
            'w',
            samplerate=target_sr,
            channels=channels,
            subtype='PCM_16',
            format=_output_soundfile_format(dest),
        ) as outf:
            _write_silence_block(outf, pad_frames, channels)

            if trim_frames_in > 0:
                inf.seek(min(trim_frames_in, len(inf)))

            out_frames = pad_frames
            while out_frames < target_frames:
                block = inf.read(STREAM_BLOCK_FRAMES, dtype='float32', always_2d=True)
                if block.size == 0:
                    break
                block = _to_sample_channel_matrix(block, channels)
                out_block = _resample_block(block, up, down)
                to_write = min(len(out_block), target_frames - out_frames)
                if to_write <= 0:
                    break
                outf.write(out_block[:to_write])
                out_frames += to_write

            if out_frames < target_frames:
                _write_silence_block(outf, target_frames - out_frames, channels)

    tmp.replace(dest)
    return pad_sec, trim_sec


def _align_stem_stream(
    src: Path,
    dest: Path,
    *,
    target_sr: int,
    target_frames: int,
    delay_sec: float,
    out_channels: int,
) -> tuple[float, float]:
    """Stream-align a stem without loading the full file into memory."""
    import soundfile as sf

    pad_sec = trim_sec = 0.0
    pad_frames = trim_frames = 0
    if delay_sec >= 0.0005:
        pad_sec = delay_sec
        pad_frames = int(round(pad_sec * target_sr))
    elif delay_sec <= -0.0005:
        trim_sec = -delay_sec
        trim_frames = int(round(trim_sec * target_sr))

    tmp = dest.with_name(dest.stem + '._align_tmp' + dest.suffix)
    if tmp.exists():
        tmp.unlink()

    with sf.SoundFile(str(src)) as inf:
        if inf.samplerate != target_sr:
            raise ValueError(f'sample rate mismatch ({inf.samplerate} vs {target_sr})')
        channels = out_channels
        with sf.SoundFile(
            str(tmp),
            'w',
            samplerate=target_sr,
            channels=channels,
            subtype='PCM_16',
            format=_output_soundfile_format(dest),
        ) as outf:
            _write_silence_block(outf, pad_frames, channels)

            if trim_frames > 0:
                inf.seek(min(trim_frames, len(inf)))

            out_frames = pad_frames
            while out_frames < target_frames:
                to_read = min(STREAM_BLOCK_FRAMES, target_frames - out_frames)
                block = inf.read(to_read, dtype='float32', always_2d=True)
                if block.size == 0:
                    break
                block = _to_sample_channel_matrix(block, channels)
                outf.write(block)
                out_frames += len(block)

            if out_frames < target_frames:
                _write_silence_block(outf, target_frames - out_frames, channels)

    tmp.replace(dest)
    return pad_sec, trim_sec


def _align_stem_to_path(
    src: Path,
    dest: Path,
    *,
    target_sr: int,
    target_frames: int,
    delay_sec: float,
    out_channels: int,
    on_log: LogFn | None = None,
) -> tuple[float, float]:
    info = _probe_audio(src)
    needs_resample = info.samplerate != target_sr

    if info.can_stream and not needs_resample:
        return _align_stem_stream(
            src,
            dest,
            target_sr=target_sr,
            target_frames=target_frames,
            delay_sec=delay_sec,
            out_channels=out_channels,
        )

    if needs_resample and info.can_stream:
        _report_log(
            on_log,
            f'· {src.name}: streaming resample {info.samplerate:,} → {target_sr:,} Hz',
            'info',
        )
        return _align_stem_stream_resample(
            src,
            dest,
            in_sr=info.samplerate,
            target_sr=target_sr,
            target_frames=target_frames,
            delay_sec=delay_sec,
            out_channels=out_channels,
        )

    if needs_resample and not _fits_in_memory(info.frames, info.channels):
        raise MemoryError(
            f'{src.name} is too large to resample in memory ({info.frames:,} frames)',
        )

    if needs_resample:
        _report_log(
            on_log,
            f'· {src.name}: resampling {info.samplerate:,} → {target_sr:,} Hz',
            'info',
        )

    y, file_sr = _read_audio_file(src, sr=target_sr if needs_resample else None, mono=False)
    if not needs_resample and file_sr != target_sr:
        y = _resample_audio(y, file_sr, target_sr)
    y = _to_sample_channel_matrix(y, out_channels)
    y, pad_sec, trim_sec = _shift_stem_to_original(y, target_sr, delay_sec)
    y = _fit_to_length(y, target_frames)
    _write_audio(dest, y, target_sr, channels=out_channels)
    return pad_sec, trim_sec


def _target_from_original(path: Path) -> tuple[int, int, int]:
    """Return (samplerate, frame_count, channels) for the original without loading audio."""
    info = _probe_audio(path)
    if info.frames is not None:
        return info.samplerate, info.frames, info.channels
    if not info.can_stream:
        y, sr = _read_audio_file(path, mono=False)
        y = _to_sample_channel_matrix(y)
        return sr, len(y), y.shape[1]

    import soundfile as sf

    with sf.SoundFile(str(path)) as handle:
        frames = _sanity_frame_count(len(handle))
        if frames is None:
            raise ValueError(f'could not determine a sane length for {path.name}')
        return handle.samplerate, frames, handle.channels


def _target_frames_from_original(path: Path) -> tuple[int, int]:
    sr, frames, _channels = _target_from_original(path)
    return sr, frames


def _read_audio_file(
    path: Path,
    *,
    sr: int | None = None,
    duration: float | None = None,
    mono: bool = False,
) -> tuple[np.ndarray, int]:
    """Load audio. Returns (samples[, channels], samplerate)."""
    import librosa

    ext = path.suffix.lower()
    if ext in SOUNDFILE_READ_EXTS:
        import soundfile as sf

        try:
            with sf.SoundFile(str(path)) as handle:
                file_sr = handle.samplerate
                frame_count = len(handle)
                if duration is not None:
                    frame_count = min(frame_count, int(duration * file_sr))
                elif not _fits_in_memory(frame_count, handle.channels):
                    raise MemoryError(
                        f'{path.name} is too large to load into memory ({frame_count:,} frames)',
                    )
                y = handle.read(frame_count, dtype='float32', always_2d=True)
            y = _as_samples_channels(y)
            if mono:
                y = y.mean(axis=1)
            target_sr = sr or file_sr
            if target_sr != file_sr:
                y = _resample_audio(y, file_sr, target_sr)
            return y, target_sr
        except MemoryError:
            raise
        except ValueError as exc:
            if 'too large' in str(exc).lower() or 'array is too big' in str(exc).lower():
                raise MemoryError(f'{path.name} is too large to load into memory') from exc
            pass
        except Exception:
            pass

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='PySoundFile failed.*')
        warnings.filterwarnings('ignore', category=FutureWarning, module=r'librosa\.core\.audio')
        if duration is None and ext not in SOUNDFILE_READ_EXTS:
            info = _probe_audio(path)
            if not _fits_in_memory(info.frames, info.channels):
                raise MemoryError(f'{path.name} is too large to load into memory')
        y, file_sr = librosa.load(
            str(path),
            sr=sr,
            mono=mono,
            duration=duration,
        )
    if not mono:
        y = _to_sample_channel_matrix(y)
    return y, file_sr if sr is None else sr


def _resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    import librosa

    if y.ndim == 1:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast')
    channels = [
        librosa.resample(y[:, ch], orig_sr=orig_sr, target_sr=target_sr, res_type='kaiser_fast')
        for ch in range(y.shape[1])
    ]
    min_len = min(len(ch) for ch in channels)
    return np.stack([ch[:min_len] for ch in channels], axis=1)


def _read_audio_mono(path: Path, sr: int, *, duration: float | None = None) -> np.ndarray:
    y, _ = _read_audio_file(path, sr=sr, duration=duration, mono=True)
    return y


def _load_mono(path: Path, sr: int, duration: float | None = None) -> np.ndarray:
    return _read_audio_mono(path, sr, duration=duration)


def _write_audio(path: Path, y: np.ndarray, sr: int, *, channels: int | None = None) -> None:
    import soundfile as sf

    y = _to_sample_channel_matrix(y, channels)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(
        str(path),
        y,
        sr,
        subtype='PCM_16',
        format=_output_soundfile_format(path),
    )


def _normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak < 1e-9:
        return y
    return y / peak


def _cross_corr_delay_sec(reference: np.ndarray, query: np.ndarray, sr: int, max_shift_sec: float) -> float:
    from scipy.signal import correlate

    n = int(min(len(reference), len(query), max_shift_sec * sr * 2 + sr * 10))
    if n < sr:
        return 0.0
    ref = _normalize_audio(reference[:n])
    qry = _normalize_audio(query[:n])
    corr = correlate(ref, qry, mode='full', method='fft')
    lags = np.arange(-len(qry) + 1, len(ref))
    max_shift = int(max_shift_sec * sr)
    valid = (lags >= -max_shift) & (lags <= max_shift)
    if not np.any(valid):
        return 0.0
    best = int(lags[valid][np.argmax(corr[valid])])
    return best / sr


def _vocal_onset_sec(y: np.ndarray, sr: int, *, analysis_sec: float) -> float:
    import librosa

    clip = y[: int(analysis_sec * sr)]
    if clip.size < sr // 4:
        return 0.0
    rms = librosa.feature.rms(y=clip, frame_length=2048, hop_length=512)[0]
    if rms.size == 0:
        return 0.0
    peak = float(np.max(rms))
    if peak < 1e-9:
        return 0.0
    threshold = max(peak * 0.18, float(np.percentile(rms, 85)) * 0.35)
    for idx, value in enumerate(rms):
        if float(value) >= threshold:
            return librosa.frames_to_time(idx, sr=sr, hop_length=512)
    return 0.0


def _prepend_silence(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    if seconds <= 0.0005:
        return y
    pad = int(round(seconds * sr))
    if y.ndim == 1:
        return np.concatenate([np.zeros(pad, dtype=y.dtype), y])
    return np.concatenate([np.zeros((pad, y.shape[1]), dtype=y.dtype), y], axis=0)


def _shift_stem_to_original(
    y: np.ndarray,
    sr: int,
    delay_sec: float,
) -> tuple[np.ndarray, float, float]:
    """Place stem on the original song timeline. Positive delay → prepend silence; negative → trim start."""
    pad_sec = trim_sec = 0.0
    if delay_sec >= 0.0005:
        pad_sec = delay_sec
        y = _prepend_silence(y, sr, pad_sec)
    elif delay_sec <= -0.0005:
        trim_sec = -delay_sec
        trim_samples = int(round(trim_sec * sr))
        if trim_samples >= len(y):
            y = np.zeros((0, y.shape[1]), dtype=y.dtype) if y.ndim > 1 else np.zeros(0, dtype=y.dtype)
        else:
            y = y[trim_samples:]
    return y, pad_sec, trim_sec


def _fit_to_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    if y.ndim == 1:
        return np.pad(y, (0, target_len - len(y)))
    return np.pad(y, ((0, target_len - len(y)), (0, 0)))


def _match_length(tracks: list[np.ndarray]) -> list[np.ndarray]:
    max_len = max(len(t) for t in tracks)
    return [np.pad(t, (0, max_len - len(t))) for t in tracks]


def has_align_backup(folder: Path) -> bool:
    """True when a prior align run left a _backup_before_align folder."""
    return (folder / BACKUP_DIR_NAME).is_dir()


def align_song_folder(
    folder: SongFolder,
    *,
    sr: int = DEFAULT_SR,
    analysis_sec: float = DEFAULT_ANALYSIS_SEC,
    backup: bool = True,
    on_log: LogFn | None = None,
    index: int | None = None,
    total: int | None = None,
) -> AlignResult | None:
    if not folder.instrumental or not folder.acapella or not folder.original:
        missing = []
        if not folder.instrumental:
            missing.append('instrumental')
        if not folder.acapella:
            missing.append('acapella')
        if not folder.original:
            missing.append('original')
        _report_log(on_log, _progress_header(folder.name, index, total), 'info')
        _report_log(on_log, f'  missing {", ".join(missing)}', 'warn')
        return None

    _report_log(on_log, _progress_header(folder.name, index, total), 'info')

    orig_clip = _load_mono(folder.original, sr, duration=analysis_sec + MAX_SHIFT_SEC)
    inst_clip = _load_mono(folder.instrumental, sr, duration=analysis_sec + MAX_SHIFT_SEC)
    acap_clip = _load_mono(folder.acapella, sr, duration=analysis_sec + MAX_SHIFT_SEC)

    delay_inst = _cross_corr_delay_sec(orig_clip, inst_clip, sr, MAX_SHIFT_SEC)
    delay_acap = _cross_corr_delay_sec(orig_clip, acap_clip, sr, MAX_SHIFT_SEC)

    vocal_orig = _vocal_onset_sec(orig_clip, sr, analysis_sec=analysis_sec)
    vocal_acap = _vocal_onset_sec(acap_clip, sr, analysis_sec=analysis_sec)

    out_sr, target_len, out_channels = _target_from_original(folder.original)

    if backup:
        backup_dir = folder.path / BACKUP_DIR_NAME
        backup_dir.mkdir(exist_ok=True)
        for src in (folder.instrumental, folder.acapella):
            dest = backup_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)

    pad_inst, trim_inst = _align_stem_to_path(
        folder.instrumental,
        folder.instrumental,
        target_sr=out_sr,
        target_frames=target_len,
        delay_sec=delay_inst,
        out_channels=out_channels,
        on_log=on_log,
    )
    pad_acap, trim_acap = _align_stem_to_path(
        folder.acapella,
        folder.acapella,
        target_sr=out_sr,
        target_frames=target_len,
        delay_sec=delay_acap,
        out_channels=out_channels,
        on_log=on_log,
    )

    _report_log(
        on_log,
        f'  ok · inst +{pad_inst:.3f}s / −{trim_inst:.3f}s · '
        f'acap +{pad_acap:.3f}s / −{trim_acap:.3f}s · vocal {vocal_orig:.2f}s · {out_channels}ch',
        'ok',
    )
    return AlignResult(
        folder=folder.path,
        instrumental_shift_sec=pad_inst - trim_inst,
        acapella_shift_sec=pad_acap - trim_acap,
        vocal_onset_original_sec=vocal_orig,
        vocal_onset_acapella_sec=vocal_acap,
        output_paths=(folder.instrumental, folder.acapella),
    )


def align_all_songs(
    stems_root: Path,
    *,
    with_original_dir: Path | None = None,
    sr: int = DEFAULT_SR,
    analysis_sec: float = DEFAULT_ANALYSIS_SEC,
    backup: bool = True,
    skip_existing: bool = False,
    on_log: LogFn | None = None,
    on_progress: ProgressFn | None = None,
) -> tuple[list[AlignResult], int]:
    align_root = resolve_with_original_dir(stems_root, with_original_dir)
    folders = [scan_song_folder(p) for p in list_with_original_subfolders(stems_root, with_original_dir)]
    ready = [f for f in folders if f.instrumental and f.acapella and f.original]
    total = max(1, len(folders))

    _report_log(
        on_log,
        f'{LOG_INDENT}Aligning {len(ready):,}/{len(folders):,} folder(s) in {align_root.name} with original + stems',
        'info',
    )
    if skip_existing:
        _report_log(
            on_log,
            f'{LOG_INDENT}Resume: skipping folders that already contain {BACKUP_DIR_NAME}',
            'info',
        )

    results: list[AlignResult] = []
    skipped = 0
    for idx, song in enumerate(folders):
        if song.instrumental and song.acapella and song.original:
            if skip_existing and has_align_backup(song.path):
                skipped += 1
                _report_log(
                    on_log,
                    _progress_header(song.name, idx + 1, total),
                    'info',
                )
                _report_log(on_log, '  [skip] already aligned', 'warn')
            else:
                try:
                    result = align_song_folder(
                        song,
                        sr=sr,
                        analysis_sec=analysis_sec,
                        backup=backup,
                        on_log=on_log,
                        index=idx + 1,
                        total=total,
                    )
                except Exception as exc:
                    _report_log(on_log, f'  align failed — {exc}', 'err')
                    result = None
                if result is not None:
                    results.append(result)
        _report_progress(
            on_progress,
            idx + 1,
            total,
            f'Aligning ({idx + 1:,}/{total:,})',
            step=idx + 1,
            force=idx + 1 == total,
        )
    return results, skipped

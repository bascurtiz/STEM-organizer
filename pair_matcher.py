from __future__ import annotations

import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from mutagen import File as MutagenFile

AUDIO_EXTS = ('.flac', '.mp3')
ProgressFn = Callable[[int, int, str], None]
LogFn = Callable[[str, str], None]

LOG_EVERY_READ = 500
LOG_EVERY_MATCH = 1000
LOG_EVERY_MOVE = 200
PROGRESS_EVERY = 250
# Classify-style startup/config indent; === Summary headers stay flush-left.
LOG_INDENT = '  '
TITLE_PREFIX_LEN = 6
TITLE_HEAD_LEN = 3
INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
BRACKET_PREFIX_RE = re.compile(r'^\[[^\]]+\]\s*-\s*')
STEM_SUFFIX_RE = re.compile(
    r'\s*\((?:acapella|aca(?:\s|$)|instrumental|inst(?:\s|$)|bgv|lead|vocals?|'
    r'backing(?:\s+vocals?)?)\)\s*$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IgnoreRules:
    ignore_parentheses: bool = True
    ignore_square_brackets: bool = True
    ignore_extra_spaces: bool = True
    custom_keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict | None) -> IgnoreRules:
        if not data:
            return cls()
        keywords = data.get('custom_keywords', [])
        if not isinstance(keywords, list):
            keywords = []
        cleaned = tuple(
            str(k).strip() for k in keywords if str(k).strip()
        )
        # Legacy: ignore_all_brackets meant both paren + square on.
        legacy_both = bool(data.get('ignore_all_brackets', False))
        return cls(
            ignore_parentheses=bool(data.get('ignore_parentheses', True)) or legacy_both,
            ignore_square_brackets=bool(data.get('ignore_square_brackets', True)) or legacy_both,
            ignore_extra_spaces=bool(data.get('ignore_extra_spaces', True)),
            custom_keywords=cleaned,
        )

    def to_dict(self) -> dict:
        return {
            'ignore_parentheses': self.ignore_parentheses,
            'ignore_square_brackets': self.ignore_square_brackets,
            'ignore_extra_spaces': self.ignore_extra_spaces,
            'custom_keywords': list(self.custom_keywords),
        }


@dataclass(frozen=True)
class TrackTags:
    path: Path
    artist: str
    title: str

    @property
    def key(self) -> tuple[str, str]:
        return normalize_tag(self.artist), normalize_tag(self.title)

    @property
    def display_name(self) -> str:
        artist = self.artist.strip() or 'Unknown Artist'
        title = self.title.strip() or self.path.stem
        return f'{artist} - {title}'


@dataclass(frozen=True)
class PairMatch:
    reference: TrackTags
    partner: TrackTags
    score: float


@dataclass(frozen=True)
class MatchResult:
    pairs: list[PairMatch]
    unmatched_reference: list[TrackTags]
    unmatched_partner: list[TrackTags]


@dataclass(frozen=True)
class OrganizeGroup:
    folder_name: str
    files: list[Path]


def apply_ignore_rules(value: str, rules: IgnoreRules | None) -> str:
    text = value or ''
    if not rules:
        return text

    if rules.ignore_parentheses:
        text = re.sub(r'\([^)]*\)', ' ', text)
    if rules.ignore_square_brackets:
        text = re.sub(r'\[[^\]]*\]', ' ', text)

    for keyword in rules.custom_keywords:
        keyword = keyword.strip()
        if not keyword:
            continue
        text = re.sub(re.escape(keyword), ' ', text, flags=re.IGNORECASE)

    if rules.ignore_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_tag(value: str, rules: IgnoreRules | None = None) -> str:
    text = apply_ignore_rules(value, rules)
    text = unicodedata.normalize('NFKD', text)
    text = text.casefold()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^\w\s]+', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def similarity(a: str, b: str, rules: IgnoreRules | None = None) -> float:
    na, nb = normalize_tag(a, rules), normalize_tag(b, rules)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def tag_match_score(
    left: TrackTags,
    right: TrackTags,
    rules: IgnoreRules | None = None,
) -> float:
    artist_score = similarity(left.artist, right.artist, rules)
    title_score = similarity(left.title, right.title, rules)
    return min(artist_score, title_score)


def strictness_to_threshold(strictness: float) -> float:
    """Map UI strictness 0..100 to similarity threshold 0.55..1.0."""
    clamped = max(0.0, min(100.0, strictness))
    return 0.55 + (clamped / 100.0) * 0.45


def read_audio_tags(path: Path, *, use_filename_fallback: bool = True) -> TrackTags:
    """Read artist/title for matching.

    use_filename_fallback=True  -> parse from filename only (ignore tags)
    use_filename_fallback=False -> metadata tags only
    """
    if use_filename_fallback:
        artist, title = parse_filename_tags(path)
        return TrackTags(path=path, artist=artist, title=title)
    artist, title = _read_tags_from_file(path)
    return TrackTags(path=path, artist=artist, title=title)


def _read_tags_from_file(path: Path) -> tuple[str, str]:
    artist = ''
    title = ''
    try:
        audio = MutagenFile(path, easy=True)
        if audio is not None and audio.tags is not None:
            artist = _pick_easy_tag(audio.tags, 'artist', 'albumartist')
            title = _pick_easy_tag(audio.tags, 'title')
            if artist and title:
                return artist, title
    except Exception:
        pass

    try:
        audio = MutagenFile(path)
        if audio is None or audio.tags is None:
            return artist, title
        tags = audio.tags
        artist = artist or _pick_raw_tag(
            tags, 'artist', 'albumartist', 'TPE1', 'TPE2', '©ART',
        )
        title = title or _pick_raw_tag(tags, 'title', 'TIT2', '©nam')
    except Exception:
        pass
    return artist, title


def _pick_easy_tag(tags, *keys: str) -> str:
    for key in keys:
        if key not in tags:
            continue
        value = tags[key]
        if isinstance(value, (list, tuple)):
            text = str(value[0]).strip() if value else ''
        else:
            text = str(value).strip()
        if text:
            return text
    return ''


def _pick_raw_tag(tags, *keys: str) -> str:
    for key in keys:
        if key not in tags:
            continue
        value = tags[key]
        if hasattr(value, 'text'):
            items = value.text
            text = str(items[0]).strip() if items else ''
        elif isinstance(value, (list, tuple)):
            text = str(value[0]).strip() if value else ''
        else:
            text = str(value).strip()
        if text:
            return text
    return ''


def parse_filename_tags(path: Path) -> tuple[str, str]:
    stem = path.stem
    stem = BRACKET_PREFIX_RE.sub('', stem)
    while True:
        cleaned = STEM_SUFFIX_RE.sub('', stem).strip()
        if cleaned == stem:
            break
        stem = cleaned
    if ' - ' in stem:
        artist, title = stem.split(' - ', 1)
        return artist.strip(), title.strip()
    return '', stem.strip()


def iter_audio_files(folder: Path, *, recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        return []
    if recursive:
        return sorted(
            p for p in folder.rglob('*')
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def _report_progress(
    on_progress: ProgressFn | None,
    done: int,
    total: int,
    message: str,
    *,
    step: int,
    force: bool = False,
) -> None:
    if on_progress is None:
        return
    if force or step <= 1 or step >= total or step % PROGRESS_EVERY == 0:
        on_progress(done, total, message)


def _report_log(on_log: LogFn | None, message: str, tag: str = 'info') -> None:
    if on_log is None:
        return
    msg = message
    # Match Classify / Genre indent; keep === headers and already-indented lines.
    if msg and not msg[0].isspace() and not msg.startswith('==='):
        msg = LOG_INDENT + msg
    on_log(msg, tag)


def _progress_header(name: str, index: int | None = None, total: int | None = None) -> str:
    try:
        from stem_organizer.run_summary import file_progress_header
        return file_progress_header(name, index, total)
    except Exception:
        label = (name or '').strip() or '?'
        try:
            i = int(index) if index is not None else 0
            n = int(total) if total is not None else 0
        except (TypeError, ValueError):
            i = n = 0
        if i > 0 and n > 0:
            return f'=== [{i}/{n}] {label} ==='
        return f'=== {label} ==='


def _norm_pair(track: TrackTags, rules: IgnoreRules | None) -> tuple[str, str]:
    return (
        normalize_tag(track.artist, rules),
        normalize_tag(track.title, rules),
    )


def _score_norms(na1: str, nt1: str, na2: str, nt2: str) -> float:
    if not na1 and not na2:
        artist_score = 1.0
    elif not na1 or not na2:
        artist_score = 0.0
    elif na1 == na2:
        artist_score = 1.0
    else:
        artist_score = SequenceMatcher(None, na1, na2).ratio()

    if not nt1 and not nt2:
        title_score = 1.0
    elif not nt1 or not nt2:
        title_score = 0.0
    elif nt1 == nt2:
        title_score = 1.0
    else:
        title_score = SequenceMatcher(None, nt1, nt2).ratio()

    return min(artist_score, title_score)


def _title_prefix(title: str) -> str:
    return title[:TITLE_PREFIX_LEN] if title else ''


@dataclass
class _PartnerIndex:
    by_exact: dict[tuple[str, str], list[int]]
    by_title: dict[str, list[int]]
    by_artist: dict[str, list[int]]
    by_title_prefix: dict[str, list[int]]
    by_title_head: dict[str, list[int]]
    norms: list[tuple[str, str]]


def _build_partner_index(
    partner_tracks: list[TrackTags],
    rules: IgnoreRules | None,
) -> _PartnerIndex:
    by_exact: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_title: dict[str, list[int]] = defaultdict(list)
    by_artist: dict[str, list[int]] = defaultdict(list)
    by_title_prefix: dict[str, list[int]] = defaultdict(list)
    by_title_head: dict[str, list[int]] = defaultdict(list)
    norms: list[tuple[str, str]] = []

    for idx, track in enumerate(partner_tracks):
        na, nt = _norm_pair(track, rules)
        norms.append((na, nt))
        by_exact[(na, nt)].append(idx)
        by_title[nt].append(idx)
        if na:
            by_artist[na].append(idx)
        by_title_prefix[_title_prefix(nt)].append(idx)
        if nt:
            by_title_head[nt[:TITLE_HEAD_LEN]].append(idx)

    return _PartnerIndex(
        by_exact, by_title, by_artist, by_title_prefix, by_title_head, norms,
    )


def _partner_candidates(
    na: str,
    nt: str,
    index: _PartnerIndex,
    *,
    fuzzy: bool,
) -> set[int]:
    found: set[int] = set()
    found.update(index.by_exact.get((na, nt), ()))
    found.update(index.by_title.get(nt, ()))
    if na:
        found.update(index.by_artist.get(na, ()))
    if fuzzy and nt:
        head = nt[:TITLE_HEAD_LEN]
        found.update(index.by_title_head.get(head, ()))
        prefix = _title_prefix(nt)
        found.update(index.by_title_prefix.get(prefix, ()))
    return found


def _match_pairs_indexed(
    reference_tracks: list[TrackTags],
    partner_tracks: list[TrackTags],
    *,
    threshold: float,
    ignore_rules: IgnoreRules | None,
    on_progress: ProgressFn | None,
    on_log: LogFn | None,
    progress_done: int,
    progress_total: int,
) -> tuple[list[PairMatch], set[int], set[int]]:
    ref_norms = [_norm_pair(track, ignore_rules) for track in reference_tracks]
    index = _build_partner_index(partner_tracks, ignore_rules)
    fuzzy = threshold < 0.995

    _report_log(
        on_log,
        f'{LOG_INDENT}Built partner index · {len(partner_tracks):,} tracks · '
        f'{len(index.by_title):,} unique titles',
        'info',
    )

    candidates: list[tuple[float, int, int]] = []
    ref_count = len(reference_tracks)
    for ref_idx, (na, nt) in enumerate(ref_norms):
        for partner_idx in _partner_candidates(na, nt, index, fuzzy=fuzzy):
            pna, pnt = index.norms[partner_idx]
            score = _score_norms(na, nt, pna, pnt)
            if score >= threshold:
                candidates.append((score, ref_idx, partner_idx))

        done = progress_done + ref_idx + 1
        if ref_idx == 0 or (ref_idx + 1) % LOG_EVERY_MATCH == 0 or ref_idx + 1 == ref_count:
            _report_log(
                on_log,
                f'Matching {ref_idx + 1:,}/{ref_count:,} · {len(candidates):,} candidate pair(s) so far',
                'info',
            )
        _report_progress(
            on_progress,
            done,
            progress_total,
            f'Matching tags ({ref_idx + 1:,}/{ref_count:,})',
            step=ref_idx + 1,
            force=ref_idx + 1 == ref_count,
        )

    _report_log(on_log, f'Sorting {len(candidates):,} candidate pair(s)…', 'info')
    candidates.sort(key=lambda item: item[0], reverse=True)

    used_reference: set[int] = set()
    used_partner: set[int] = set()
    pairs: list[PairMatch] = []

    for score, ref_idx, partner_idx in candidates:
        if ref_idx in used_reference or partner_idx in used_partner:
            continue
        used_reference.add(ref_idx)
        used_partner.add(partner_idx)
        pairs.append(
            PairMatch(
                reference=reference_tracks[ref_idx],
                partner=partner_tracks[partner_idx],
                score=score,
            )
        )

    _report_log(on_log, f'Matched {len(pairs):,} pair(s) from candidates', 'ok')
    return pairs, used_reference, used_partner


def find_pairs(
    reference_dir: Path,
    partner_dir: Path,
    *,
    reference_is_acapella: bool,
    strictness: float,
    use_filename_fallback: bool = True,
    ignore_rules: IgnoreRules | None = None,
    include_subfolders: bool = False,
    move_to: Path | None = None,
    on_progress: ProgressFn | None = None,
    on_log: LogFn | None = None,
) -> MatchResult:
    threshold = strictness_to_threshold(strictness)
    ref_paths = iter_audio_files(reference_dir, recursive=include_subfolders)
    partner_paths = iter_audio_files(partner_dir, recursive=include_subfolders)
    ref_count = len(ref_paths)
    partner_count = len(partner_paths)
    move_budget = min(ref_count, partner_count) if move_to is not None else 0
    total = max(1, ref_count + partner_count + ref_count + move_budget)
    scan_note = 'including subfolders' if include_subfolders else 'top level only'

    _report_log(
        on_log,
        f'{LOG_INDENT}Scanning folders ({scan_note}) · reference: {ref_count:,} · partner: {partner_count:,} · '
        f'threshold {threshold:.0%}',
        'info',
    )

    done = 0
    reference_tracks: list[TrackTags] = []
    for idx, path in enumerate(ref_paths):
        reference_tracks.append(
            read_audio_tags(path, use_filename_fallback=use_filename_fallback)
        )
        done += 1
        if idx == 0 or (idx + 1) % LOG_EVERY_READ == 0 or idx + 1 == ref_count:
            _report_log(
                on_log,
                f'Reading reference tags {idx + 1:,}/{ref_count:,}',
                'info',
            )
        _report_progress(
            on_progress, done, total,
            f'Reading reference tags ({idx + 1:,}/{ref_count:,})',
            step=idx + 1,
            force=idx + 1 == ref_count,
        )

    partner_tracks: list[TrackTags] = []
    for idx, path in enumerate(partner_paths):
        partner_tracks.append(
            read_audio_tags(path, use_filename_fallback=use_filename_fallback)
        )
        done += 1
        if idx == 0 or (idx + 1) % LOG_EVERY_READ == 0 or idx + 1 == partner_count:
            _report_log(
                on_log,
                f'Reading partner tags {idx + 1:,}/{partner_count:,}',
                'info',
            )
        _report_progress(
            on_progress, done, total,
            f'Reading partner tags ({idx + 1:,}/{partner_count:,})',
            step=idx + 1,
            force=idx + 1 == partner_count,
        )

    pairs, used_reference, used_partner = _match_pairs_indexed(
        reference_tracks,
        partner_tracks,
        threshold=threshold,
        ignore_rules=ignore_rules,
        on_progress=on_progress,
        on_log=on_log,
        progress_done=done,
        progress_total=total,
    )
    done += ref_count

    if move_to is not None and pairs:
        _report_log(on_log, f'Moving {len(pairs):,} pair(s) to {move_to}…', 'info')
        _report_log(on_log, '', 'info')
        moved = 0
        pair_total = len(pairs)
        for idx, match in enumerate(pairs, start=1):
            name = match.reference.path.name
            try:
                move_pair(match, move_to)
                moved += 1
                done += 1
                _report_log(on_log, _progress_header(name, idx, pair_total), 'info')
                _report_progress(
                    on_progress,
                    done,
                    total,
                    f'Moving pairs ({moved:,}/{pair_total:,})',
                    step=moved,
                    force=moved == pair_total,
                )
            except OSError as exc:
                _report_log(on_log, _progress_header(name, idx, pair_total), 'info')
                _report_log(on_log, f'  failed: {exc}', 'err')

    unmatched_reference = [
        track for idx, track in enumerate(reference_tracks) if idx not in used_reference
    ]
    unmatched_partner = [
        track for idx, track in enumerate(partner_tracks) if idx not in used_partner
    ]
    _ = reference_is_acapella  # reserved for future role-specific rules
    return MatchResult(
        pairs=pairs,
        unmatched_reference=unmatched_reference,
        unmatched_partner=unmatched_partner,
    )


def safe_folder_name(name: str) -> str:
    cleaned = INVALID_NAME_CHARS.sub('', name).strip().rstrip('.')
    return cleaned or 'Untitled'


def unique_destination(folder: Path) -> Path:
    if not folder.exists():
        return folder
    stem = folder.name
    parent = folder.parent
    for index in range(2, 10_000):
        candidate = parent / f'{stem} ({index})'
        if not candidate.exists():
            return candidate
    raise OSError(f'Could not allocate unique folder for {folder}')


def move_pair(match: PairMatch, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_dest = _unique_file_path(output_dir / match.reference.path.name)
    partner_dest = _unique_file_path(output_dir / match.partner.path.name)
    shutil.move(str(match.reference.path), str(ref_dest))
    shutil.move(str(match.partner.path), str(partner_dest))
    return ref_dest, partner_dest


def _unique_file_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    parent = path.parent
    for index in range(2, 10_000):
        candidate = parent / f'{stem} ({index}){suffix}'
        if not candidate.exists():
            return candidate
    raise OSError(f'Could not allocate unique path for {path}')


def group_for_organize(
    matched_dir: Path,
    *,
    strictness: float,
    use_filename_fallback: bool = True,
    ignore_rules: IgnoreRules | None = None,
    include_subfolders: bool = False,
    tracks: list[TrackTags] | None = None,
    on_log: LogFn | None = None,
) -> list[OrganizeGroup]:
    if tracks is None:
        tracks = [
            read_audio_tags(p, use_filename_fallback=use_filename_fallback)
            for p in iter_audio_files(matched_dir, recursive=include_subfolders)
        ]
    if not tracks:
        return []

    threshold = strictness_to_threshold(strictness)
    groups: list[list[TrackTags]] = []
    group_keys: list[tuple[str, str]] = []
    by_title: dict[str, list[int]] = defaultdict(list)
    track_count = len(tracks)

    for idx, track in enumerate(tracks):
        na, nt = _norm_pair(track, ignore_rules)
        placed = False
        for group_idx in by_title.get(nt, ()):
            anchor_na, anchor_nt = group_keys[group_idx]
            if _score_norms(na, nt, anchor_na, anchor_nt) >= threshold:
                groups[group_idx].append(track)
                placed = True
                break
        if not placed and na and nt:
            head = nt[:TITLE_HEAD_LEN]
            for group_idx, (anchor_na, anchor_nt) in enumerate(group_keys):
                if anchor_nt[:TITLE_HEAD_LEN] != head:
                    continue
                if _score_norms(na, nt, anchor_na, anchor_nt) >= threshold:
                    groups[group_idx].append(track)
                    by_title[anchor_nt].append(group_idx)
                    placed = True
                    break
        if not placed:
            group_keys.append((na, nt))
            by_title[nt].append(len(groups))
            groups.append([track])

        if idx == 0 or (idx + 1) % LOG_EVERY_MATCH == 0 or idx + 1 == track_count:
            _report_log(
                on_log,
                f'Grouping {idx + 1:,}/{track_count:,} · {len(groups):,} folder(s) so far',
                'info',
            )

    result: list[OrganizeGroup] = []
    for group in groups:
        anchor = group[0]
        base_name = safe_folder_name(anchor.display_name)
        if len(group) > 2:
            folder_name = f'{base_name} [{len(group)}]'
        else:
            folder_name = base_name
        result.append(
            OrganizeGroup(
                folder_name=folder_name,
                files=[track.path for track in group],
            )
        )
    return result


def organize_matched_folder(
    matched_dir: Path,
    *,
    strictness: float,
    use_filename_fallback: bool = True,
    ignore_rules: IgnoreRules | None = None,
    include_subfolders: bool = False,
    on_progress: ProgressFn | None = None,
    on_log: LogFn | None = None,
) -> list[tuple[Path, list[Path]]]:
    paths = iter_audio_files(matched_dir, recursive=include_subfolders)
    file_count = len(paths)
    total = max(1, file_count * 2 + 1)
    done = 0
    scan_note = 'including subfolders' if include_subfolders else 'top level only'

    _report_log(on_log, f'{LOG_INDENT}Organizing {file_count:,} file(s) in {matched_dir} ({scan_note})', 'info')

    tracks: list[TrackTags] = []
    for idx, path in enumerate(paths):
        tracks.append(read_audio_tags(path, use_filename_fallback=use_filename_fallback))
        done += 1
        if idx == 0 or (idx + 1) % LOG_EVERY_READ == 0 or idx + 1 == file_count:
            _report_log(on_log, f'Reading tags {idx + 1:,}/{file_count:,}', 'info')
        _report_progress(
            on_progress, done, total,
            f'Reading tags ({idx + 1:,}/{file_count:,})',
            step=idx + 1,
            force=idx + 1 == file_count,
        )

    groups = group_for_organize(
        matched_dir,
        strictness=strictness,
        use_filename_fallback=use_filename_fallback,
        ignore_rules=ignore_rules,
        tracks=tracks,
        on_log=on_log,
        include_subfolders=include_subfolders,
    )
    done += 1
    _report_log(on_log, f'Created {len(groups):,} group(s)', 'info')
    _report_log(on_log, '', 'info')
    _report_progress(
        on_progress, done, total, 'Grouping tracks',
        step=done, force=True,
    )

    moved: list[tuple[Path, list[Path]]] = []
    files_moved = 0
    total_files = sum(len(group.files) for group in groups)
    group_total = len(groups)
    for g_idx, group in enumerate(groups, start=1):
        _report_log(
            on_log,
            _progress_header(group.folder_name, g_idx, group_total),
            'info',
        )
        dest_dir = unique_destination(matched_dir / group.folder_name)
        dest_dir.mkdir(parents=True, exist_ok=False)
        new_paths: list[Path] = []
        for src in group.files:
            target = _unique_file_path(dest_dir / src.name)
            shutil.move(str(src), str(target))
            new_paths.append(target)
            files_moved += 1
            done += 1
            _report_progress(
                on_progress, done, total,
                f'Moving files ({files_moved:,}/{total_files:,})',
                step=files_moved,
                force=files_moved == total_files,
            )
        moved.append((dest_dir, new_paths))
    return moved

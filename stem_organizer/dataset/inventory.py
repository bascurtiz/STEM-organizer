"""Walk Instrumental / Vocal / Pairs / Samples roots and build OverviewStats."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..meta_tags import read_custom_tag, read_key_tag
from ..musical_keys import normalize_key_short, short_to_display
from ..run_summary import emit_run_summary
from .models import (
    VOCAL_TYPES,
    ClassBucket,
    ContinuousStats,
    OverviewStats,
    RoleCounts,
)

BACKUP_DIR_NAME = "_backup_before_align"
AUDIO_EXTS = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aiff",
    ".aif",
    ".opus",
    ".aac",
    ".ape",
)

LogFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, str], None]

_VOCAL_TYPE_LOOKUP = {v.lower(): v for v in VOCAL_TYPES}
_VOCAL_SAMPLE_RE = re.compile(
    r"(?:vocal|acapella|acapellas|vox|vocoder)",
    re.IGNORECASE,
)


def _sample_treated_as_vocal(path: Path) -> bool:
    """Samples with vocal-ish names count toward vocal facets, not genre/style."""
    return bool(_VOCAL_SAMPLE_RE.search(path.stem))


@dataclass
class ScanUnit:
    """One countable inventory unit (file or pair folder)."""

    role: str  # "instrumental" | "vocal" | "sample" | "pair"
    path: Path
    root: Path
    relative: str
    bytes: int = 0
    duration_sec: float = 0.0
    genre: str = ""
    style: str = ""
    gender: str = ""
    reverb: str = ""
    vocal_type: str = ""
    sdr: Optional[float] = None
    compression: str = ""
    key: str = ""  # canonical short key (Db, Am, …)
    # Pair folders also expose representative file paths for tag/copy ops.
    pair_inst: Optional[Path] = None
    pair_voc: Optional[Path] = None


@dataclass
class ScanResult:
    stats: OverviewStats
    units: list[ScanUnit] = field(default_factory=list)
    inst_root: str = ""
    vocal_root: str = ""
    pairs_root: str = ""
    samples_root: str = ""


def _path_has_backup_segment(path: Path) -> bool:
    return any(part == BACKUP_DIR_NAME for part in path.parts)


def iter_audio_files(
    root: Path,
    *,
    recursive: bool,
    stop_event: Optional[threading.Event] = None,
) -> list[Path]:
    """List audio files under root, skipping ``_backup_before_align`` trees."""
    if not root.is_dir():
        return []
    out: list[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if stop_event is not None and stop_event.is_set():
                break
            if not p.is_file():
                continue
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            if _path_has_backup_segment(p.relative_to(root)):
                continue
            out.append(p)
    else:
        for p in root.iterdir():
            if stop_event is not None and stop_event.is_set():
                break
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                out.append(p)
    return sorted(out)


def list_pair_song_folders(
    root: Path,
    *,
    stop_event: Optional[threading.Event] = None,
) -> list[Path]:
    """Song folders under the pairs root (any nesting depth).

    A song folder is a directory that contains both an instrumental and a
    vocal/acapella stem. Grouping folders (e.g. ``EM Pairs/``) are walked
    through; matched song folders are not descended into.
    """
    if not root.is_dir():
        return []

    folders: list[Path] = []

    def _walk(dir_path: Path) -> None:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            children = sorted(dir_path.iterdir())
        except OSError:
            return
        for p in children:
            if stop_event is not None and stop_event.is_set():
                return
            if not p.is_dir():
                continue
            if p.name == BACKUP_DIR_NAME:
                continue
            inst, voc = _classify_pair_files(p)
            if inst is not None and voc is not None:
                folders.append(p)
                continue
            _walk(p)

    _walk(root)
    return folders


def _audio_duration_and_size(path: Path) -> tuple[float, int]:
    size = 0
    try:
        size = path.stat().st_size
    except OSError:
        pass
    dur = 0.0
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path))
        if audio is not None and getattr(audio, "info", None) is not None:
            length = getattr(audio.info, "length", None)
            if length is not None:
                dur = float(length)
    except Exception:
        pass
    return dur, size


def _split_slash(value: str) -> tuple[str, str]:
    text = (value or "").strip()
    if not text:
        return "", ""
    if "/" in text:
        a, b = text.split("/", 1)
        return a.strip(), b.strip()
    return text, ""


def _read_file_tags(path: Path) -> dict:
    """Pull overview-relevant tags from one audio file."""
    genre = read_custom_tag(path, "genre")
    style = read_custom_tag(path, "style")
    # Combined Genre/Style in genre field (GG combined mode).
    if genre and "/" in genre and not style:
        genre, style = _split_slash(genre)

    gender = read_custom_tag(path, "gender")
    reverb = read_custom_tag(path, "reverb")
    comment = read_custom_tag(path, "comment")
    if gender and "/" in gender and not reverb:
        gender, reverb = _split_slash(gender)
    elif comment and not gender:
        head, tail = _split_slash(comment)
        if head.lower() in ("female", "male"):
            gender = head
            if tail.lower() in ("dry", "wet") and not reverb:
                reverb = tail

    vocal_type = read_custom_tag(path, "VOCAL_TYPE")
    if not vocal_type and comment:
        head = comment.split("/", 1)[0].strip()
        mapped = _VOCAL_TYPE_LOOKUP.get(head.lower())
        if mapped:
            vocal_type = mapped
    else:
        mapped = _VOCAL_TYPE_LOOKUP.get(vocal_type.lower()) if vocal_type else None
        vocal_type = mapped or vocal_type

    sdr_raw = read_custom_tag(path, "SDR")
    sdr: Optional[float] = None
    if sdr_raw:
        try:
            sdr = float(sdr_raw.replace("dB", "").strip())
        except ValueError:
            sdr = None

    compression = read_custom_tag(path, "COMPRESSION").lower()
    if compression not in ("lossless", "lossy"):
        compression = ""

    key_raw = read_key_tag(path)
    key = normalize_key_short(key_raw)
    if not key and comment:
        key = normalize_key_short(comment)

    gender_l = gender.lower() if gender else ""
    if gender_l not in ("female", "male"):
        gender_l = ""
    reverb_l = reverb.lower() if reverb else ""
    if reverb_l not in ("dry", "wet"):
        reverb_l = ""

    return {
        "genre": genre,
        "style": style,
        "gender": gender_l,
        "reverb": reverb_l,
        "vocal_type": vocal_type,
        "sdr": sdr,
        "compression": compression,
        "key": key,
    }


def _classify_pair_files(folder: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Return (instrumental, vocal/acapella) paths inside a song folder.

    Prefer role/keyword names. If the folder has exactly two audio files and only
    one side is recognized, the other file is treated as the complement. With
    two unmarked files, larger → instrumental (same rule as SI-SDR).
    """
    try:
        from stem_align import classify_audio_file
    except Exception:
        classify_audio_file = None  # type: ignore

    audio: list[Path] = []
    try:
        children = sorted(folder.iterdir())
    except OSError:
        return None, None
    for path in children:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            audio.append(path)

    inst = voc = None
    for path in audio:
        role = None
        if classify_audio_file is not None:
            try:
                role = classify_audio_file(path)
            except Exception:
                role = None
        if role == "instrumental" and inst is None:
            inst = path
            continue
        if role == "acapella" and voc is None:
            voc = path
            continue
        lower = path.stem.lower()
        if inst is None and any(
            m in lower
            for m in (
                "instrumental",
                "instrumenta",  # typo "(Instrumenta)l"
                "instrum",
                "(inst)",
                "_inst",
                "-inst",
                "inst.",
            )
        ):
            inst = path
        elif inst is None and (
            " beat" in f" {lower}" or lower.endswith("beat") or "-beat" in lower
            or lower.startswith("beat ")
        ):
            inst = path
        if voc is None and any(
            m in lower
            for m in (
                "acapella",
                "acappella",
                "a cappella",
                "a capella",
                "vocal",
                "vocals",
                "(aca)",
            )
        ):
            voc = path

    if inst is not None and voc is not None and inst != voc:
        return inst, voc

    # Exactly two stems: assume a pair when keywords only hit one side (or neither).
    if len(audio) == 2:
        a, b = audio[0], audio[1]
        if inst is not None and voc is None:
            other = b if inst == a else a
            return inst, other
        if voc is not None and inst is None:
            other = b if voc == a else a
            return other, voc
        if inst is None and voc is None:
            try:
                bigger, smaller = sorted(
                    audio, key=lambda p: (-p.stat().st_size, p.name.lower())
                )
            except OSError:
                bigger, smaller = a, b
            return bigger, smaller

    return inst, voc


def _bump(
    store: dict[str, ClassBucket],
    name: str,
    *,
    nbytes: int,
    duration: float,
) -> None:
    if not name:
        return
    b = store.get(name)
    if b is None:
        store[name] = ClassBucket(name=name, count=1, bytes=nbytes, duration_sec=duration)
    else:
        b.count += 1
        b.bytes += nbytes
        b.duration_sec += duration


def scan_library(
    *,
    instrumental: str | Path,
    vocal: str | Path,
    pairs: str | Path,
    samples: str | Path = "",
    include_subfolders: bool = True,
    on_log: Optional[LogFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
) -> ScanResult:
    """Scan roots and return aggregated stats + unit list for Balance.

    Samples are counted and facet-tagged like Instrumental (genre / style).
    """

    def log(msg: str, tag: str = "info") -> None:
        if on_log:
            on_log(msg, tag)

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    inst_root = Path(instrumental) if str(instrumental).strip() else None
    voc_root = Path(vocal) if str(vocal).strip() else None
    pairs_root = Path(pairs) if str(pairs).strip() else None
    samples_root = Path(samples) if str(samples).strip() else None

    roots_ok = [
        (label, p)
        for label, p in (
            ("Instrumental", inst_root),
            ("Vocal", voc_root),
            ("Pairs", pairs_root),
            ("Samples", samples_root),
        )
        if p is not None and p.is_dir()
    ]
    if not roots_ok:
        raise ValueError(
            "Set at least one existing Instrumental, Vocal, Pairs, or Samples folder."
        )

    units: list[ScanUnit] = []
    stats = OverviewStats(demo=False)
    stats.roles = RoleCounts()
    stats.duration = ContinuousStats()
    stats.sdr = ContinuousStats()
    t0 = time.perf_counter()

    # ---- collect work items ----
    work: list[tuple[str, Path, Path]] = []  # (kind, path, root)
    # kind: "instrumental" | "vocal" | "sample" | "pair"

    if inst_root and inst_root.is_dir():
        files = iter_audio_files(
            inst_root, recursive=include_subfolders, stop_event=stop_event
        )
        log(f"Instrumental: {len(files):,} file(s) under {inst_root}", "info")
        for f in files:
            work.append(("instrumental", f, inst_root))

    if samples_root and samples_root.is_dir():
        files = iter_audio_files(
            samples_root, recursive=include_subfolders, stop_event=stop_event
        )
        log(f"Samples: {len(files):,} file(s) under {samples_root}", "info")
        for f in files:
            work.append(("sample", f, samples_root))

    if voc_root and voc_root.is_dir():
        files = iter_audio_files(
            voc_root, recursive=include_subfolders, stop_event=stop_event
        )
        log(f"Vocal: {len(files):,} file(s) under {voc_root}", "info")
        for f in files:
            work.append(("vocal", f, voc_root))

    pair_folders: list[Path] = []
    if pairs_root and pairs_root.is_dir():
        pair_folders = list_pair_song_folders(pairs_root, stop_event=stop_event)
        log(f"Pairs: {len(pair_folders):,} song folder(s) under {pairs_root}", "info")
        for folder in pair_folders:
            work.append(("pair", folder, pairs_root))

    total = max(1, len(work))
    done = 0

    def live(n: int, count: int) -> None:
        # Host LOG replaces one line via __live_progress__ (same as Compression).
        log(f"__live_progress__\t{int(n)}\t{int(count)}\tunits scanned", "")

    live(0, total)
    for kind, path, root in work:
        if stopped():
            log("__live_progress_end__", "")
            log("Scan stopped.", "warn")
            break
        done += 1
        if on_progress:
            on_progress(done, total, "Scanning")
        if done == 1 or done == total or done % 25 == 0:
            live(done, total)

        if kind == "pair":
            inst_f, voc_f = _classify_pair_files(path)
            if inst_f is None or voc_f is None:
                from ..run_summary import file_progress_header

                log(file_progress_header(path.name, done, total), "info")
                log("  [skip] need instrumental + vocal", "warn")
                continue
            tags_i = _read_file_tags(inst_f)
            tags_v = _read_file_tags(voc_f)
            # Genre/Style ← instrumental only; Gender/Reverb/Vocal type ← vocal only.
            # Unit SDR/COMPRESSION keep a representative (prefer instrumental).
            # Chart stats count each stem's SDR / COMPRESSION tags separately.
            sdr = tags_i["sdr"] if tags_i["sdr"] is not None else tags_v["sdr"]
            compression = tags_i["compression"] or tags_v["compression"]
            dur_i, sz_i = _audio_duration_and_size(inst_f)
            dur_v, sz_v = _audio_duration_and_size(voc_f)
            nbytes = sz_i + sz_v
            duration = dur_i + dur_v
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
            unit = ScanUnit(
                role="pair",
                path=path,
                root=root,
                relative=rel.replace("\\", "/"),
                bytes=nbytes,
                duration_sec=duration,
                genre=tags_i["genre"],
                style=tags_i["style"],
                gender=tags_v["gender"],
                reverb=tags_v["reverb"],
                vocal_type=tags_v["vocal_type"],
                sdr=sdr,
                compression=compression,
                key=tags_i["key"] or tags_v["key"],
                pair_inst=inst_f,
                pair_voc=voc_f,
            )
            units.append(unit)
            stats.roles.pair_folders += 1
            stats.roles.instrumental += 1
            stats.roles.vocal += 1
            stats.total_bytes += nbytes
            if duration > 0:
                stats.duration.values.append(duration)
            # Facets once per side (genre←inst, gender/reverb/vt←voc).
            # SDR / COMPRESSION / Initial key: one chart sample per tagged audio file.
            if tags_i["sdr"] is not None:
                stats.sdr.values.append(float(tags_i["sdr"]))
            if tags_v["sdr"] is not None:
                stats.sdr.values.append(float(tags_v["sdr"]))
            if tags_i["compression"]:
                _bump(
                    stats.compression,
                    tags_i["compression"],
                    nbytes=sz_i,
                    duration=dur_i,
                )
            if tags_v["compression"]:
                _bump(
                    stats.compression,
                    tags_v["compression"],
                    nbytes=sz_v,
                    duration=dur_v,
                )
            if tags_i["key"]:
                _bump(
                    stats.key,
                    short_to_display(tags_i["key"]),
                    nbytes=sz_i,
                    duration=dur_i,
                )
            if tags_v["key"]:
                _bump(
                    stats.key,
                    short_to_display(tags_v["key"]),
                    nbytes=sz_v,
                    duration=dur_v,
                )
            _apply_unit_facets(
                stats,
                unit,
                nbytes=sz_i,
                duration=dur_i,
                facets="instrumental",
                include_shared=False,
            )
            _apply_unit_facets(
                stats,
                unit,
                nbytes=sz_v,
                duration=dur_v,
                facets="vocal",
                include_shared=False,
            )
            continue

        # Single-file instrumental / sample / vocal
        tags = _read_file_tags(path)
        duration, nbytes = _audio_duration_and_size(path)
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name
        if kind in ("instrumental", "sample"):
            if kind == "sample" and _sample_treated_as_vocal(path):
                unit = ScanUnit(
                    role=kind,
                    path=path,
                    root=root,
                    relative=rel.replace("\\", "/"),
                    bytes=nbytes,
                    duration_sec=duration,
                    genre="",
                    style="",
                    gender=tags["gender"],
                    reverb=tags["reverb"],
                    vocal_type=tags["vocal_type"],
                    sdr=tags["sdr"],
                    compression=tags["compression"],
                    key=tags["key"],
                )
                stats.roles.samples += 1
                facet = "vocal"
            else:
                # Samples are a separate role but use instrumental facets (genre/style).
                unit = ScanUnit(
                    role=kind,
                    path=path,
                    root=root,
                    relative=rel.replace("\\", "/"),
                    bytes=nbytes,
                    duration_sec=duration,
                    genre=tags["genre"],
                    style=tags["style"],
                    gender="",
                    reverb="",
                    vocal_type="",
                    sdr=tags["sdr"],
                    compression=tags["compression"],
                    key=tags["key"],
                )
                if kind == "sample":
                    stats.roles.samples += 1
                else:
                    stats.roles.instrumental += 1
                facet = "instrumental"
        else:
            unit = ScanUnit(
                role=kind,
                path=path,
                root=root,
                relative=rel.replace("\\", "/"),
                bytes=nbytes,
                duration_sec=duration,
                genre="",
                style="",
                gender=tags["gender"],
                reverb=tags["reverb"],
                vocal_type=tags["vocal_type"],
                sdr=tags["sdr"],
                compression=tags["compression"],
                key=tags["key"],
            )
            stats.roles.vocal += 1
            facet = "vocal"
        units.append(unit)
        stats.total_bytes += nbytes
        if duration > 0:
            stats.duration.values.append(duration)
        if tags["compression"]:
            _bump(
                stats.compression,
                tags["compression"],
                nbytes=nbytes,
                duration=duration,
            )
        if tags["key"]:
            _bump(
                stats.key,
                short_to_display(tags["key"]),
                nbytes=nbytes,
                duration=duration,
            )
        _apply_unit_facets(
            stats,
            unit,
            nbytes=nbytes,
            duration=duration,
            facets=facet,
            include_compression=False,
        )

    stats.total_files = stats.roles.total_units
    _finalize_style_top(stats)

    log("__live_progress_end__", "")
    elapsed = time.perf_counter() - t0
    stat_lines: list[tuple[str, str]] = [
        (f"Instrumental: {stats.roles.instrumental:,}", "info"),
        (f"Vocal: {stats.roles.vocal:,}", "info"),
        (f"Samples: {stats.roles.samples:,}", "info"),
        (f"Pairs: {stats.roles.pair_folders:,}", "info"),
    ]
    if stopped():
        stat_lines.append(("Stopped early", "warn"))
    emit_run_summary(
        log,
        "Charts",
        elapsed=elapsed,
        files=stats.total_files,
        stat_lines=stat_lines,
    )
    return ScanResult(
        stats=stats,
        units=units,
        inst_root=str(inst_root) if inst_root else "",
        vocal_root=str(voc_root) if voc_root else "",
        pairs_root=str(pairs_root) if pairs_root else "",
        samples_root=str(samples_root) if samples_root else "",
    )


def _apply_unit_facets(
    stats: OverviewStats,
    unit: ScanUnit,
    *,
    nbytes: int,
    duration: float,
    facets: str,
    include_shared: bool = True,
    include_compression: bool = True,
) -> None:
    """Apply chart facets for one unit.

    ``facets``:
      - ``instrumental`` — genre / style
      - ``vocal`` — gender / reverb / vocal_type
      - ``all`` — both role-specific sets

    Shared SDR / COMPRESSION apply when ``include_shared`` is True.
    Pair scans push both stems' SDR / COMPRESSION directly (include_shared=False).
    """
    if include_shared:
        if unit.sdr is not None:
            stats.sdr.values.append(float(unit.sdr))
        if include_compression and unit.compression:
            _bump(stats.compression, unit.compression, nbytes=nbytes, duration=duration)

    if facets in ("instrumental", "all"):
        if unit.genre:
            _bump(stats.genre, unit.genre, nbytes=nbytes, duration=duration)
            if unit.style:
                nested = stats.styles_by_genre.setdefault(unit.genre, {})
                _bump(nested, unit.style, nbytes=nbytes, duration=duration)

    if facets in ("vocal", "all"):
        if unit.vocal_type:
            _bump(stats.vocal_type, unit.vocal_type, nbytes=nbytes, duration=duration)
        if unit.gender:
            _bump(stats.gender, unit.gender, nbytes=nbytes, duration=duration)
        if unit.reverb:
            _bump(stats.reverb, unit.reverb, nbytes=nbytes, duration=duration)


def _finalize_style_top(stats: OverviewStats) -> None:
    flat: dict[str, ClassBucket] = {}
    for nested in stats.styles_by_genre.values():
        for name, b in nested.items():
            prev = flat.get(name)
            if prev is None:
                flat[name] = ClassBucket(
                    name=name,
                    count=b.count,
                    bytes=b.bytes,
                    duration_sec=b.duration_sec,
                )
            else:
                prev.count += b.count
                prev.bytes += b.bytes
                prev.duration_sec += b.duration_sec
    ranked = sorted(flat.values(), key=lambda b: -b.count)
    stats.style_top = {b.name: b for b in ranked[:20]}
    tagged = sum(b.count for b in flat.values())
    stats.style_other_count = max(0, stats.total_files - tagged)


def collect_audio_paths_for_roots(
    *,
    instrumental: str | Path,
    vocal: str | Path,
    pairs: str | Path,
    samples: str | Path = "",
    include_subfolders: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> list[Path]:
    """Flat list of audio files across all roots (for compression detect)."""
    paths: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        paths.append(p)

    for root_s in (instrumental, vocal, samples):
        root = Path(root_s) if str(root_s).strip() else None
        if root is None or not root.is_dir():
            continue
        for f in iter_audio_files(
            root, recursive=include_subfolders, stop_event=stop_event
        ):
            add(f)

    pairs_root = Path(pairs) if str(pairs).strip() else None
    if pairs_root and pairs_root.is_dir():
        for folder in list_pair_song_folders(pairs_root, stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                break
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    add(f)
    return paths

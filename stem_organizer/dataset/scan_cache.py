"""Persist last successful Scan library result across app restarts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..settings_store import app_dir
from .inventory import ScanResult, ScanUnit
from .models import ClassBucket, ContinuousStats, OverviewStats, RoleCounts

CACHE_FILENAME = "overview_scan_cache.json"


def cache_path() -> Path:
    return app_dir() / CACHE_FILENAME


def _path_str(p: Optional[Path | str]) -> str:
    if p is None:
        return ""
    return str(p)


def _opt_path(s: Any) -> Optional[Path]:
    text = str(s or "").strip()
    return Path(text) if text else None


def _bucket_to_dict(b: ClassBucket) -> dict[str, Any]:
    return {
        "name": b.name,
        "count": int(b.count),
        "bytes": int(b.bytes),
        "duration_sec": float(b.duration_sec),
    }


def _bucket_from_dict(d: dict[str, Any]) -> ClassBucket:
    return ClassBucket(
        name=str(d.get("name") or ""),
        count=int(d.get("count") or 0),
        bytes=int(d.get("bytes") or 0),
        duration_sec=float(d.get("duration_sec") or 0.0),
    )


def _buckets_map(store: dict[str, ClassBucket]) -> dict[str, Any]:
    return {k: _bucket_to_dict(v) for k, v in store.items()}


def _buckets_from_map(raw: Any) -> dict[str, ClassBucket]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, ClassBucket] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = _bucket_from_dict(v)
    return out


def stats_to_dict(stats: OverviewStats) -> dict[str, Any]:
    styles_by_genre = {
        g: _buckets_map(styles) for g, styles in stats.styles_by_genre.items()
    }
    return {
        "demo": False,
        "roles": {
            "instrumental": int(stats.roles.instrumental),
            "vocal": int(stats.roles.vocal),
            "samples": int(stats.roles.samples),
            "pair_folders": int(stats.roles.pair_folders),
        },
        "total_files": int(stats.total_files),
        "total_bytes": int(stats.total_bytes),
        "duration": list(map(float, stats.duration.values)),
        "sdr": list(map(float, stats.sdr.values)),
        "vocal_type": _buckets_map(stats.vocal_type),
        "gender": _buckets_map(stats.gender),
        "reverb": _buckets_map(stats.reverb),
        "genre": _buckets_map(stats.genre),
        "styles_by_genre": styles_by_genre,
        "style_top": _buckets_map(stats.style_top),
        "style_other_count": int(stats.style_other_count),
        "compression": _buckets_map(stats.compression),
        "key": _buckets_map(stats.key),
    }


def stats_from_dict(raw: dict[str, Any]) -> OverviewStats:
    roles_raw = raw.get("roles") if isinstance(raw.get("roles"), dict) else {}
    stats = OverviewStats(demo=False)
    stats.roles = RoleCounts(
        instrumental=int(roles_raw.get("instrumental") or 0),
        vocal=int(roles_raw.get("vocal") or 0),
        samples=int(roles_raw.get("samples") or 0),
        pair_folders=int(roles_raw.get("pair_folders") or 0),
    )
    stats.total_files = int(raw.get("total_files") or 0)
    stats.total_bytes = int(raw.get("total_bytes") or 0)
    stats.duration = ContinuousStats()
    stats.duration.values = [
        float(x) for x in (raw.get("duration") or []) if x is not None
    ]
    stats.sdr = ContinuousStats()
    stats.sdr.values = [float(x) for x in (raw.get("sdr") or []) if x is not None]
    stats.vocal_type = _buckets_from_map(raw.get("vocal_type"))
    stats.gender = _buckets_from_map(raw.get("gender"))
    stats.reverb = _buckets_from_map(raw.get("reverb"))
    stats.genre = _buckets_from_map(raw.get("genre"))
    stats.compression = _buckets_from_map(raw.get("compression"))
    stats.key = _buckets_from_map(raw.get("key"))
    # Migrate compact "Eb/D#" bucket names → spaced "Eb / D#".
    if stats.key:
        from ..musical_keys import short_to_display, normalize_key_short

        migrated: dict[str, ClassBucket] = {}
        for name, bucket in stats.key.items():
            short = normalize_key_short(name) or name
            display = short_to_display(short) if short else name
            prev = migrated.get(display)
            if prev is None:
                migrated[display] = ClassBucket(
                    name=display,
                    count=bucket.count,
                    bytes=bucket.bytes,
                    duration_sec=bucket.duration_sec,
                )
            else:
                prev.count += bucket.count
                prev.bytes += bucket.bytes
                prev.duration_sec += bucket.duration_sec
        stats.key = migrated
    stats.style_top = _buckets_from_map(raw.get("style_top"))
    stats.style_other_count = int(raw.get("style_other_count") or 0)
    nested: dict[str, dict[str, ClassBucket]] = {}
    raw_styles = raw.get("styles_by_genre")
    if isinstance(raw_styles, dict):
        for g, styles in raw_styles.items():
            nested[str(g)] = _buckets_from_map(styles)
    stats.styles_by_genre = nested
    return stats


def unit_to_dict(u: ScanUnit) -> dict[str, Any]:
    return {
        "role": u.role,
        "path": _path_str(u.path),
        "root": _path_str(u.root),
        "relative": u.relative,
        "bytes": int(u.bytes),
        "duration_sec": float(u.duration_sec),
        "genre": u.genre,
        "style": u.style,
        "gender": u.gender,
        "reverb": u.reverb,
        "vocal_type": u.vocal_type,
        "sdr": u.sdr,
        "compression": u.compression,
        "key": u.key,
        "pair_inst": _path_str(u.pair_inst),
        "pair_voc": _path_str(u.pair_voc),
    }


def unit_from_dict(d: dict[str, Any]) -> ScanUnit:
    sdr = d.get("sdr")
    try:
        sdr_f = float(sdr) if sdr is not None and str(sdr).strip() != "" else None
    except (TypeError, ValueError):
        sdr_f = None
    return ScanUnit(
        role=str(d.get("role") or ""),
        path=Path(str(d.get("path") or "")),
        root=Path(str(d.get("root") or "")),
        relative=str(d.get("relative") or ""),
        bytes=int(d.get("bytes") or 0),
        duration_sec=float(d.get("duration_sec") or 0.0),
        genre=str(d.get("genre") or ""),
        style=str(d.get("style") or ""),
        gender=str(d.get("gender") or ""),
        reverb=str(d.get("reverb") or ""),
        vocal_type=str(d.get("vocal_type") or ""),
        sdr=sdr_f,
        compression=str(d.get("compression") or ""),
        key=str(d.get("key") or ""),
        pair_inst=_opt_path(d.get("pair_inst")),
        pair_voc=_opt_path(d.get("pair_voc")),
    )


def scan_result_to_dict(result: ScanResult) -> dict[str, Any]:
    return {
        "version": 1,
        "stats": stats_to_dict(result.stats),
        "units": [unit_to_dict(u) for u in result.units],
        "inst_root": result.inst_root,
        "vocal_root": result.vocal_root,
        "pairs_root": result.pairs_root,
        "samples_root": result.samples_root,
    }


def scan_result_from_dict(raw: dict[str, Any]) -> Optional[ScanResult]:
    stats_raw = raw.get("stats")
    if not isinstance(stats_raw, dict):
        return None
    stats = stats_from_dict(stats_raw)
    if stats.demo or stats.total_files <= 0:
        return None
    units_raw = raw.get("units")
    units: list[ScanUnit] = []
    if isinstance(units_raw, list):
        for item in units_raw:
            if isinstance(item, dict):
                try:
                    units.append(unit_from_dict(item))
                except Exception:
                    continue
    return ScanResult(
        stats=stats,
        units=units,
        inst_root=str(raw.get("inst_root") or ""),
        vocal_root=str(raw.get("vocal_root") or ""),
        pairs_root=str(raw.get("pairs_root") or ""),
        samples_root=str(raw.get("samples_root") or ""),
    )


def save_scan_result(result: ScanResult) -> None:
    """Write last real scan beside settings.json (atomic replace)."""
    if result.stats.demo:
        return
    path = cache_path()
    payload = scan_result_to_dict(result)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        pass


def load_scan_result() -> Optional[ScanResult]:
    """Return last saved real scan, or None to keep demo charts."""
    path = cache_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return scan_result_from_dict(raw)
    except Exception:
        return None

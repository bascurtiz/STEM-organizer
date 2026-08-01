"""Build a balanced subset (copy / move / CSV) by equalizing enabled features."""

from __future__ import annotations

import csv
import random
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

from .inventory import ScanUnit

LogFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, str], None]

BalanceMode = Literal["copy", "move", "csv"]

# Chart-aligned facets the user can enable for min-count undersampling.
BALANCE_FEATURES: tuple[tuple[str, str], ...] = (
    ("sdr", "SI-SDR"),
    ("roles", "Roles"),
    ("genre", "Genre"),
    ("style", "Style"),
    ("gender", "Gender"),
    ("reverb", "Reverb"),
    ("vocal_type", "Vocal type"),
    ("key", "Keys"),
    ("compression", "Compression"),
)
BALANCE_FEATURE_KEYS = frozenset(k for k, _ in BALANCE_FEATURES)
# Roles stays out of the dialog checkboxes — always applied (see _normalize_features).
BALANCE_OPTIONAL_FEATURES: tuple[tuple[str, str], ...] = tuple(
    (k, lbl) for k, lbl in BALANCE_FEATURES if k != "roles"
)
BALANCE_ALWAYS_ON: tuple[str, ...] = ("roles",)
# Gender / reverb / vocal type come from the vocal side only — never stratify
# standalone instrumentals (or samples) on these, even when tags exist.
VOCAL_SIDE_FEATURES = frozenset({"gender", "reverb", "vocal_type"})
VOCAL_SIDE_ROLES = frozenset({"pair", "vocal"})
# Genre / style come from the instrumental side — standalone vocals are filled
# randomly to match the instrumental pick count (see select_balanced_units).
INST_SIDE_FEATURES = frozenset({"genre", "style"})
_UNTAGGED = "_untagged"

_ROLE_LABEL = {
    "instrumental": "Instrumental",
    "vocal": "Vocal",
    "sample": "Samples",
    "pair": "Pair",
}


@dataclass
class BalancePlan:
    mode: BalanceMode
    dest: Path
    target_n: int
    selected: list[ScanUnit]
    inst_root_name: str
    vocal_root_name: str
    pairs_root_name: str
    samples_root_name: str = "Samples_BALANCED"
    samples_root: str = ""
    features: tuple[str, ...] = ()


def _balanced_root_name(source_root: Path | str, fallback: str) -> str:
    name = Path(source_root).name if str(source_root).strip() else fallback
    if not name:
        name = fallback
    if name.upper().endswith("_BALANCED"):
        return name
    return f"{name}_BALANCED"


def _sdr_bin(value: Optional[float]) -> str:
    if value is None:
        return _UNTAGGED
    lo = int(value // 5 * 5)
    return f"{lo}–{lo + 5}"


def _feature_value(unit: ScanUnit, feature: str) -> str:
    if feature == "roles":
        return _ROLE_LABEL.get(unit.role, unit.role or _UNTAGGED)
    if feature == "genre":
        return unit.genre.strip() or _UNTAGGED
    if feature == "style":
        return unit.style.strip() or _UNTAGGED
    if feature == "gender":
        return unit.gender.strip() or _UNTAGGED
    if feature == "reverb":
        return unit.reverb.strip() or _UNTAGGED
    if feature == "vocal_type":
        return unit.vocal_type.strip() or _UNTAGGED
    if feature == "key":
        return unit.key.strip() or _UNTAGGED
    if feature == "compression":
        return unit.compression.strip() or _UNTAGGED
    if feature == "sdr":
        return _sdr_bin(unit.sdr)
    return _UNTAGGED


def _normalize_features(features: Sequence[str] | None) -> tuple[str, ...]:
    if not features:
        wanted: set[str] = set()
    else:
        wanted = {f for f in features if f in BALANCE_FEATURE_KEYS}
    wanted.update(BALANCE_ALWAYS_ON)
    order = [k for k, _ in BALANCE_FEATURES]
    # Roles first so strata read as Role|Genre|… and the dataset always keeps
    # Instrumental / Vocal / Pair / Samples as a required axis.
    rest = [k for k in order if k in wanted and k not in BALANCE_ALWAYS_ON]
    return tuple(BALANCE_ALWAYS_ON) + tuple(rest)


def _balance_key(unit: ScanUnit, feats: tuple[str, ...]) -> tuple[str, ...]:
    """Stratum key — vocal-side facets skip inst/sample; inst-side skips loose vocals."""
    parts: list[str] = []
    for f in feats:
        if f in VOCAL_SIDE_FEATURES and unit.role not in VOCAL_SIDE_ROLES:
            continue
        if f in INST_SIDE_FEATURES and unit.role == "vocal":
            continue
        parts.append(_feature_value(unit, f))
    return tuple(parts)


def _vocal_side_suffix(unit: ScanUnit, feats: tuple[str, ...]) -> tuple[str, ...]:
    vocal_feats = [f for f in feats if f in VOCAL_SIDE_FEATURES]
    if not vocal_feats:
        return ()
    return tuple(_feature_value(unit, f) for f in vocal_feats)


def _vocal_side_multiplier(units: list[ScanUnit], feats: tuple[str, ...]) -> int:
    """How many vocal-side buckets (e.g. female + male → 2) drive inst/sample counts."""
    vocal_feats = [f for f in feats if f in VOCAL_SIDE_FEATURES]
    if not vocal_feats:
        return 1
    suffixes: set[tuple[str, ...]] = set()
    for unit in units:
        if unit.role in VOCAL_SIDE_ROLES:
            suffixes.add(_vocal_side_suffix(unit, feats))
    return max(1, len(suffixes))


def select_balanced_units(
    units: list[ScanUnit],
    *,
    features: Sequence[str] | None = None,
    seed: int = 42,
) -> tuple[list[ScanUnit], int]:
    """Min-count undersample across joint strata of enabled features.

    Vocal-side facets (gender, reverb, vocal type) apply to pairs and vocals
    only. Instrumental-side facets (genre, style) apply to pairs, instrumentals,
    and samples — not standalone vocals.

    When only vocal-side facets are on, instrumentals/samples keep
    ``target × G`` (G = vocal-side buckets, e.g. female + male → 2).

    When only instrumental-side facets are on, pairs/instrumentals/samples keep
    ``target`` per genre/style bucket and loose vocals are chosen at random until
    their count matches standalone instrumentals selected.

    Returns ``(selected_units, per_stratum_target)``.
    """
    feats = _normalize_features(features)
    if not feats or not units:
        return [], 0

    vocal_feats = [f for f in feats if f in VOCAL_SIDE_FEATURES]
    inst_feats = [f for f in feats if f in INST_SIDE_FEATURES]
    use_vocal_side = bool(vocal_feats)
    use_inst_side = bool(inst_feats)
    use_vocal_fill = use_inst_side and not use_vocal_side

    by_key: dict[tuple[str, ...], list[ScanUnit]] = defaultdict(list)
    for unit in units:
        by_key[_balance_key(unit, feats)].append(unit)

    if not by_key:
        return [], 0

    def _vocal_side_keys() -> list[tuple[str, ...]]:
        return [
            k
            for k, pool in by_key.items()
            if any(u.role in VOCAL_SIDE_ROLES for u in pool)
        ]

    def _non_vocal_only_keys() -> list[tuple[str, ...]]:
        return [
            k
            for k, pool in by_key.items()
            if not all(u.role == "vocal" for u in pool)
        ]

    if use_vocal_side:
        keys_for_target = _vocal_side_keys()
        if not keys_for_target:
            return [], 0
        target = min(len(by_key[k]) for k in keys_for_target)
    elif use_vocal_fill:
        keys_for_target = _non_vocal_only_keys()
        if not keys_for_target:
            return [], 0
        target = min(len(by_key[k]) for k in keys_for_target)
    else:
        target = min(len(pool) for pool in by_key.values())

    if target <= 0:
        return [], 0

    # Gender-only (etc.): scale inst/sample pools; genre on inst blocks multiplier.
    inst_multiplier = (
        _vocal_side_multiplier(units, feats)
        if use_vocal_side and not use_inst_side
        else 1
    )
    rng = random.Random(seed)
    selected: list[ScanUnit] = []

    for key in sorted(by_key.keys()):
        pool = list(by_key[key])
        if use_vocal_fill and all(u.role == "vocal" for u in pool):
            continue
        rng.shuffle(pool)
        if any(u.role in VOCAL_SIDE_ROLES for u in pool):
            take = target
        elif use_vocal_side and not use_inst_side:
            take = target * inst_multiplier
        else:
            take = target
        selected.extend(pool[:take])

    if use_vocal_fill:
        inst_n = sum(1 for u in selected if u.role == "instrumental")
        vocal_pool = [u for u in units if u.role == "vocal"]
        rng.shuffle(vocal_pool)
        selected.extend(vocal_pool[:inst_n])

    return selected, target


def build_balance_plan(
    units: list[ScanUnit],
    *,
    mode: BalanceMode,
    dest: Path | str,
    features: Sequence[str] | None = None,
    inst_root: str = "",
    vocal_root: str = "",
    pairs_root: str = "",
    samples_root: str = "",
    seed: int = 42,
) -> BalancePlan:
    feats = _normalize_features(features)
    selected, target = select_balanced_units(units, features=feats, seed=seed)
    return BalancePlan(
        mode=mode,
        dest=Path(dest),
        target_n=target,
        selected=selected,
        inst_root_name=_balanced_root_name(inst_root, "Instrumental"),
        vocal_root_name=_balanced_root_name(vocal_root, "Vocal"),
        pairs_root_name=_balanced_root_name(pairs_root, "Pairs"),
        samples_root_name=_balanced_root_name(samples_root, "Samples"),
        samples_root=str(samples_root).strip(),
        features=feats,
    )


def _dest_for_unit(plan: BalancePlan, unit: ScanUnit) -> Path:
    if unit.role == "sample":
        root_name = plan.samples_root_name
    elif unit.role == "instrumental":
        root_name = plan.inst_root_name
    elif unit.role == "vocal":
        root_name = plan.vocal_root_name
    else:
        root_name = plan.pairs_root_name
    return plan.dest / root_name / Path(unit.relative)


BALANCE_OVERVIEW_LOG = "balance_overview.txt"

_MODE_LABELS = {
    "copy": "Copy",
    "move": "Move",
    "csv": "CSV list",
}


def write_balance_overview_log(plan: BalancePlan) -> Path:
    """Write the pre-run overview (plan + selection + destination) beside output roots."""
    labels = dict(BALANCE_FEATURES)
    feat_txt = ", ".join(labels.get(f, f) for f in plan.features)
    mode_label = _MODE_LABELS.get(plan.mode, plan.mode)
    target = int(plan.target_n)
    total = len(plan.selected)
    categories = total // target if target else 0
    n_pairs = sum(1 for u in plan.selected if u.role == "pair")
    n_inst = sum(1 for u in plan.selected if u.role == "instrumental")
    n_voc = sum(1 for u in plan.selected if u.role == "vocal")
    n_samples = sum(1 for u in plan.selected if u.role == "sample")

    dest = plan.dest.resolve()
    lines = [
        "Balance",
        "=======",
        "",
        "PLAN",
        f"Balance on:  {feat_txt}",
        f"Mode:        {mode_label}",
        f"Scale:       {categories:,} categories · {target:,} each · {total:,} total",
        "",
        "SELECTION",
        f"Pairs          {n_pairs:,}",
        f"Instrumental   {n_inst:,}",
        f"Vocal          {n_voc:,}",
        f"Samples        {n_samples:,}",
        "",
        "DESTINATION",
        str(dest),
        "",
        "OUTPUT FOLDERS",
    ]
    roles_present = {u.role for u in plan.selected}
    if "instrumental" in roles_present or "pair" in roles_present:
        lines.append(f"  {plan.inst_root_name}/")
    if "vocal" in roles_present or "pair" in roles_present:
        lines.append(f"  {plan.vocal_root_name}/")
    if "pair" in roles_present:
        lines.append(f"  {plan.pairs_root_name}/")
    if "sample" in roles_present:
        lines.append(f"  {plan.samples_root_name}/")
    lines.extend(
        [
            "",
            f"Written: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]
    )

    path = dest / BALANCE_OVERVIEW_LOG
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def execute_balance(
    plan: BalancePlan,
    *,
    on_log: Optional[LogFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Run copy / move / CSV. Returns summary counts."""

    def log(msg: str, tag: str = "info") -> None:
        if on_log:
            on_log(msg, tag)

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    plan.dest.mkdir(parents=True, exist_ok=True)
    try:
        overview_path = write_balance_overview_log(plan)
        log(f"Overview → {overview_path.name}", "info")
    except OSError as exc:
        log(f"Could not write {BALANCE_OVERVIEW_LOG}: {exc}", "warn")
    selected = plan.selected
    total = max(1, len(selected))
    copied = moved = written = skipped = errors = 0

    def live(n: int, count: int, label: str) -> None:
        # Host LOG replaces one line via __live_progress__ (same as Scan / Compression).
        log(f"__live_progress__\t{int(n)}\t{int(count)}\t{label}", "")

    if plan.features:
        labels = dict(BALANCE_FEATURES)
        feat_txt = ", ".join(labels.get(f, f) for f in plan.features)
        log(
            f"Balance on {feat_txt} · {plan.target_n:,} per stratum · "
            f"{len(selected):,} unit(s)",
            "info",
        )

    if plan.mode == "csv":
        csv_path = plan.dest / "balanced_selection.csv"
        live(0, total, "rows written")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "role",
                    "source_path",
                    "relative_path",
                    "genre",
                    "style",
                    "gender",
                    "reverb",
                    "vocal_type",
                    "key",
                    "compression",
                    "sdr",
                    "bytes",
                    "duration_sec",
                ]
            )
            for i, unit in enumerate(selected, start=1):
                if stopped():
                    log("__live_progress_end__", "")
                    log("Balance stopped.", "warn")
                    break
                writer.writerow(
                    [
                        unit.role,
                        str(unit.path),
                        unit.relative,
                        unit.genre,
                        unit.style,
                        unit.gender,
                        unit.reverb,
                        unit.vocal_type,
                        unit.key,
                        unit.compression,
                        "" if unit.sdr is None else f"{unit.sdr:.3f}",
                        unit.bytes,
                        f"{unit.duration_sec:.3f}",
                    ]
                )
                written += 1
                if on_progress:
                    on_progress(i, total, "CSV")
                if i == 1 or i == total or i % 25 == 0:
                    live(i, total, "rows written")
        log("__live_progress_end__", "")
        log(f"Wrote {written:,} row(s) → {csv_path}", "ok")
        return {
            "copied": 0,
            "moved": 0,
            "written": written,
            "skipped": 0,
            "errors": 0,
            "csv": str(csv_path),
        }

    op = shutil.copy2 if plan.mode == "copy" else shutil.move
    verb = "Copy" if plan.mode == "copy" else "Move"
    live_label = "units copied" if plan.mode == "copy" else "units moved"

    live(0, total, live_label)
    for i, unit in enumerate(selected, start=1):
        if stopped():
            log("__live_progress_end__", "")
            log("Balance stopped.", "warn")
            break
        if on_progress:
            on_progress(i, total, verb)
        if i == 1 or i == total or i % 25 == 0:
            live(i, total, live_label)
        dest = _dest_for_unit(plan, unit)
        try:
            if unit.role == "pair":
                if dest.exists() and plan.mode == "copy":
                    skipped += 1
                    continue
                if plan.mode == "move":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        skipped += 1
                        continue
                    op(str(unit.path), str(dest))
                    moved += 1
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                    for src_file in unit.path.iterdir():
                        if not src_file.is_file():
                            continue
                        target = dest / src_file.name
                        if target.exists():
                            skipped += 1
                            continue
                        shutil.copy2(str(src_file), str(target))
                    copied += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and plan.mode == "copy":
                    skipped += 1
                    continue
                op(str(unit.path), str(dest))
                if plan.mode == "copy":
                    copied += 1
                else:
                    moved += 1
        except Exception as exc:
            errors += 1
            from ..run_summary import file_progress_header

            log(file_progress_header(unit.path.name, i, total), "info")
            log(f"  failed: {exc}", "err")

    log("__live_progress_end__", "")
    log(
        f"Done · {verb.lower()}ed {copied + moved:,} · skipped {skipped:,} · errors {errors:,}",
        "ok",
    )
    return {
        "copied": copied,
        "moved": moved,
        "written": 0,
        "skipped": skipped,
        "errors": errors,
        "csv": "",
    }

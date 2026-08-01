"""Shared input layout detection for Genre / Gender / Vocal type tagging.

Uses the same rules as Classify → SI-SDR (pair folders, vocals/instrumental
keywords, process-all prompts). Genre tags instrumental-side files only;
Gender and Vocal type tag vocals only.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import classify_backend as cb

from .widgets.dialogs import ask_yes_no, show_info

TagPanel = Literal["genre", "gender", "vocal"]

_PREFERRED_CATEGORIES = cb.STEM_MODES["2 (instrumental/vocals)"]["categories"]


@dataclass
class TagInputOptions:
    categories: tuple[str, ...]
    layout: str
    flat_process_all: bool = False
    pair_process_all: bool = False
    user_picked_category: bool = False


def scan_mode_from_recursive(include_subfolders: bool) -> str:
    return "recursive" if include_subfolders else "subfolders"


def maybe_prompt_process_all(
    parent,
    root: Path,
    scan_mode: str,
    options: TagInputOptions,
    *,
    title_prefix: str = "Tagging",
) -> bool:
    """Same pair / flat process-all prompts as Classify SI-SDR."""
    preferred = _PREFERRED_CATEGORIES

    if set(preferred) == {"instrumental", "vocals"}:
        pair_hint = cb.build_pair_folder_process_all_hint(root, scan_mode)
        if pair_hint and pair_hint.get("should_ask_process_all"):
            if ask_yes_no(
                parent,
                f"{title_prefix} · all as pairs?",
                cb.pair_folder_process_all_message(pair_hint),
                yes_text="Yes, all pairs",
                no_text="Keywords only",
            ):
                options.categories = ("instrumental", "vocals")
                options.layout = cb.SDR_LAYOUT_MUSDB
                options.pair_process_all = True
                return True

    if options.layout not in (
        cb.SDR_LAYOUT_SINGLE_FLAT,
        cb.SDR_LAYOUT_MIXED_FLAT,
        None,
    ):
        return True

    candidates: list[dict] = []
    for kind in ("instrumental", "vocals"):
        hint = cb.build_single_stem_folder_hint(root, scan_mode, kind)
        if hint and hint.get("should_ask_process_all"):
            candidates.append(hint)
    if not candidates:
        return True

    hint = max(candidates, key=lambda h: int(h.get("keyword_matches") or 0))
    kind = str(hint["kind"])
    title = (
        f"{title_prefix} · all as vocals?"
        if kind == "vocals"
        else f"{title_prefix} · all as instrumental?"
    )
    if ask_yes_no(
        parent,
        title,
        cb.single_stem_process_all_message(hint),
        yes_text="Yes, all files",
        no_text="Keywords only",
    ):
        options.categories = (kind,)
        options.layout = cb.SDR_LAYOUT_SINGLE_FLAT
        options.flat_process_all = True
        options.user_picked_category = True
    return True


def resolve_tag_input(
    parent,
    root: Path,
    scan_mode: str,
    *,
    title_prefix: str = "Tagging",
) -> Optional[TagInputOptions]:
    """Detect folder layout; prompt when ambiguous (Classify parity)."""
    if not root.is_dir():
        return None

    categories, layout = cb.resolve_sdr_layout_and_categories(
        root, scan_mode, _PREFERRED_CATEGORIES
    )
    if layout is None or categories is None:
        show_info(
            parent,
            "Input folder",
            cb.describe_sdr_scan_failure(root, scan_mode, _PREFERRED_CATEGORIES),
        )
        return None

    options = TagInputOptions(categories=categories, layout=layout)
    maybe_prompt_process_all(
        parent, root, scan_mode, options, title_prefix=title_prefix
    )
    return options


def collect_paths_for_panel(
    root: Path,
    scan_mode: str,
    options: TagInputOptions,
    panel: TagPanel,
) -> list[Path]:
    if panel == "genre":
        return cb.collect_instrumental_tag_paths(
            root,
            options.categories,
            scan_mode,
            options.layout,
            flat_process_all=options.flat_process_all,
            pair_process_all=options.pair_process_all,
        )
    return cb.collect_vocal_tag_paths(
        root,
        options.categories,
        scan_mode,
        options.layout,
        flat_process_all=options.flat_process_all,
        pair_process_all=options.pair_process_all,
    )


def layout_log_line(options: TagInputOptions) -> str:
    return cb.describe_tag_layout_label(
        options.categories,
        options.layout,
        user_picked_category=options.user_picked_category,
    )


def write_files_list(paths: list[Path]) -> Path:
    fd, name = tempfile.mkstemp(suffix=".tag_files.txt", prefix="stem_org_")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(str(path.expanduser().resolve(strict=False)) + "\n")
    return Path(name)


def remove_files_list(path: Path | str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass

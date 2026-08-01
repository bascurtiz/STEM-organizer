"""Tab registry. Each tab module exposes a ``register(window, settings)`` function.

Stage 1 (skeleton): no tabs registered. As each tab lands it is added here.
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app import MainWindow
    from ..settings_store import SettingsStore

# Filled by register_all_tabs — frozen builds use console=False so prints are invisible.
LAST_TAB_ERRORS: list[str] = []


def _tab_errors_log_path() -> Path:
    try:
        from deps_bootstrap import app_dir

        return app_dir() / "tab_errors.log"
    except Exception:
        return Path.cwd() / "tab_errors.log"


def _record_tab_error(label: str, exc: BaseException) -> None:
    detail = f"{label} tab unavailable: {exc}"
    print(f"[stem_organizer] {detail}")
    tb = traceback.format_exc()
    LAST_TAB_ERRORS.append(f"{detail}\n{tb}")
    try:
        path = _tab_errors_log_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{detail}\n{tb}\n")
    except OSError:
        pass


def register_all_tabs(window: "MainWindow", settings: "SettingsStore") -> list[str]:
    """Register every tab that is currently implemented.

    Returns human-readable error lines for any tabs that failed (also written
    to ``tab_errors.log`` beside the exe when frozen).
    """
    LAST_TAB_ERRORS.clear()
    try:
        path = _tab_errors_log_path()
        if path.is_file():
            path.unlink()
    except OSError:
        pass

    try:
        from .classify_tab import register as register_classify

        register_classify(window, settings)
    except Exception as exc:
        _record_tab_error("classify", exc)

    try:
        from .genre_gender_tab import register as register_gg

        register_gg(window, settings)
    except Exception as exc:
        _record_tab_error("genre/gender", exc)

    try:
        from .key_detect_tab import register as register_key

        register_key(window, settings)
    except Exception as exc:
        _record_tab_error("key detect", exc)

    try:
        from .pair_finder_tab import register as register_pair

        register_pair(window, settings)
    except Exception as exc:
        _record_tab_error("pair-finder", exc)

    try:
        from .rename_tab import register as register_rename

        register_rename(window, settings)
    except Exception as exc:
        _record_tab_error("rename", exc)

    try:
        from .integrity_tab import register as register_integrity

        register_integrity(window, settings)
    except Exception as exc:
        _record_tab_error("integrity", exc)

    try:
        from .dataset_overview_tab import register as register_dataset

        register_dataset(window, settings)
    except Exception as exc:
        _record_tab_error("dataset overview", exc)

    return list(LAST_TAB_ERRORS)

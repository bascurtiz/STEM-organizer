"""Settings JSON store with atomic save.

Port of ui_theme.load_settings/save_settings + the App settings-snapshot/
load/save/autosave pattern. GUI-agnostic — just JSON I/O.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def app_dir() -> Path:
    """Writable app directory: next to exe when frozen, else script dir."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


SETTINGS_FILENAME = "settings.json"
SETTINGS_PATH = app_dir() / SETTINGS_FILENAME


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_settings(data: dict[str, Any]) -> None:
    try:
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(SETTINGS_PATH)
    except OSError:
        pass


def display_path(path: str | Path | None) -> str:
    """Normalize path slashes for display on Windows (port of ui_theme.display_path)."""
    if path is None:
        return ""
    s = str(path)
    return s.replace("/", "\\") if sys.platform == "win32" else s


class SettingsStore:
    """Thin convenience wrapper around load/save + an in-memory dict.

    Tabs merge their own snapshots into the shared dict via :meth:`merge`.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = load_settings()

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def merge(self, snapshot: dict[str, Any]) -> None:
        self._data.update(snapshot)

    def flush(self) -> None:
        save_settings(self._data)

    def reload(self) -> None:
        self._data = load_settings()

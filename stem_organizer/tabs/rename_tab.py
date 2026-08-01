"""Rename tab — embeds TrackRenamerApp and wires it to the main window's
status bar + log panel.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..renamer.app import TrackRenamerApp
from ..settings_store import SettingsStore

if TYPE_CHECKING:
    from ..app import MainWindow


class RenameTab(QWidget):
    """Thin wrapper around TrackRenamerApp."""

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.app = TrackRenamerApp(self, settings=settings)
        layout.addWidget(self.app)

    def on_tab_shown(self) -> None:
        self.app.on_tab_shown()

    def on_tab_hidden(self) -> None:
        self.app.on_tab_hidden()

    def flush_settings(self) -> None:
        self.app.flush_settings()

    def shutdown(self) -> None:
        self.app.shutdown()

    @property
    def destructive_busy(self) -> bool:
        return self.app.destructive_busy


def register(window: "MainWindow", settings: SettingsStore) -> None:
    tab = RenameTab(settings)
    window.register_tab("Rename", tab)
    # Renamer has its own bottom footer; no shared action-bar page needed.
    window.action_bar.make_page("Rename")
    tab.app.status_running.connect(lambda: window.status_bar.show_running())
    tab.app.status_idle.connect(window.status_bar.show_idle)
    tab.app.status_text.connect(window.status_bar.set_status)
    tab.app.log_line.connect(window.append_log)

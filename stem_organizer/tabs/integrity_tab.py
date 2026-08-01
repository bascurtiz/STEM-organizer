"""Integrity tab — Compression + Corruption + Convert sub-tabs."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import TabWidget

from .. import theme
from ..settings_store import SettingsStore
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from .compression_tab import CompressionTab
from .convert_tab import ConvertTab
from .corruption_tab import CorruptionTab

PANEL_TITLE = "Integrity"


class IntegrityTab(QWidget):
    """Compression + Corruption + Convert under one main tab."""

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._window = None
        self.compression = CompressionTab(settings)
        self.corruption = CorruptionTab(settings)
        self.convert = ConvertTab(settings)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._tabview = TabWidget()
        theme.configure_tab_widget(self._tabview)
        self._tabview.addTab(self.compression, "Compression")
        self._tabview.addTab(self.corruption, "Corruption")
        self._tabview.addTab(self.convert, "Convert")
        theme.configure_tab_widget(self._tabview)
        theme.inset_tab_bar(self._tabview)
        self._tabview.currentChanged.connect(self._on_subtab_changed)
        layout.addWidget(self._tabview)

    def _on_subtab_changed(self, _idx: int = 0) -> None:
        if not hasattr(self, "cmp_btn"):
            return
        idx = self._tabview.currentIndex()
        self.cmp_btn.setVisible(idx == 0)
        self.cor_btn.setVisible(idx == 1)
        self.cvt_btn.setVisible(idx == 2)
        # Corruption / Convert input follows Compression.
        if idx in (1, 2):
            self._sync_input_from_compression()

    def _sync_input_from_compression(self) -> None:
        """Corruption + Convert take over Compression's input folder."""
        try:
            self.compression.flush_settings()
        except Exception:
            pass
        path = self.compression.input_row.text().strip()
        if not path:
            return
        idx = self._tabview.currentIndex()
        if idx == 1:
            if self.corruption.input_row.text().strip() != path:
                self.corruption.input_row.set_text(path)
        elif idx == 2:
            if self.convert.input_row.text().strip() != path:
                self.convert.input_row.set_text(path)

    def on_tab_shown(self) -> None:
        """Pull Compression → Corruption/Convert when visiting Integrity."""
        if self._tabview.currentIndex() in (1, 2):
            self._sync_input_from_compression()

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._window = window
        self.compression.bind_window(window)
        self.corruption.bind_window(window)
        self.convert.bind_window(window)

        self.cmp_btn = action_button(
            "▶ Detect compression",
            on_click=self.compression.start_detect,
            accent=True,
            tip=self.compression.tip("detect"),
        )
        self.cor_btn = action_button(
            "▶ Detect corruption",
            on_click=self.corruption.start_detect,
            accent=True,
            tip=self.corruption.tip("detect"),
        )
        self.cvt_btn = action_button(
            "▶ Convert to FLAC",
            on_click=self.convert.start_convert,
            accent=True,
            tip=self.convert.tip("convert"),
        )
        self.stop_btn = action_button(
            "■ Stop",
            on_click=self._on_stop,
            tip="Stop the running integrity job.",
        )
        self.stop_btn.setEnabled(False)

        # Child busy helpers toggle these shared buttons.
        self.compression.bind_action_buttons(self.cmp_btn, self.stop_btn)
        self.corruption.bind_action_buttons(self.cor_btn, self.stop_btn)
        self.convert.bind_action_buttons(self.cvt_btn, self.stop_btn)
        self.compression.set_sibling_busy_hook(self._sync_action_busy)
        self.corruption.set_sibling_busy_hook(self._sync_action_busy)
        self.convert.set_sibling_busy_hook(self._sync_action_busy)

        page.add_button(self.cmp_btn)
        page.add_button(self.cor_btn)
        page.add_button(self.cvt_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()
        self._on_subtab_changed(0)

    def _sync_action_busy(self, busy: bool) -> None:
        """Disable primary buttons while any sub-tab is running."""
        if hasattr(self, "cmp_btn"):
            self.cmp_btn.setEnabled(not busy)
            self.cor_btn.setEnabled(not busy)
            self.cvt_btn.setEnabled(not busy)
            self.stop_btn.setEnabled(busy)

    def _on_stop(self) -> None:
        if self.compression.is_busy():
            self.compression.stop_detect()
        elif self.corruption.is_busy():
            self.corruption.stop_detect()
        elif self.convert.is_busy():
            self.convert.stop_convert()

    def flush_settings(self) -> None:
        self.compression.flush_settings()
        self.corruption.flush_settings()
        self.convert.flush_settings()


def register(window, settings: SettingsStore) -> None:
    tab = IntegrityTab(settings)
    window.register_tab(PANEL_TITLE, tab)
    page = window.action_bar.make_page(PANEL_TITLE)
    tab.attach_action_bar(page, window)

    for child in (tab.compression, tab.corruption, tab.convert):
        child.request_status_running.connect(lambda: window.status_bar.show_running())
        child.request_status_idle.connect(window.status_bar.show_idle)
        child.request_status_text.connect(window.status_bar.set_status)
        child.request_progress.connect(window.status_bar.update_progress)
        child.request_log.connect(window.append_log)
        child.request_clear_log.connect(window.clear_log)

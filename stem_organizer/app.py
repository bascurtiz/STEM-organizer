"""Main window for STEM organizer (PySide6 port).

Frameless QMainWindow with a custom title bar, a left column holding a tab
widget (Classify / Genre & Gender / Match & Align / Rename / Integrity / Charts) plus a shared
action bar, a right column with the log panel, and a bottom status bar.

This file currently wires up the skeleton; tab contents are added in their
own modules. As tabs land they are imported and registered here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import TabWidget

from . import theme
from .settings_store import SettingsStore
from .widgets.action_bar import ActionBarStack
from .widgets.log_panel import LogPanel
from .widgets.status_bar import StatusBar
from .widgets.titlebar import (
    CustomTitleBar,
    center_window_default_size,
    enable_win32_thick_frame,
    handle_native_frame_message,
    note_activation_chrome_refresh,
    note_minimize_restore_to_default,
    prepare_dark_frameless_chrome,
    toggle_work_area_maximize,
)

_ICON_PATH = Path(__file__).resolve().parent.parent / "logo.ico"


class MainWindow(QMainWindow):
    def __init__(self, settings_store: SettingsStore) -> None:
        super().__init__()
        self._settings = settings_store
        self._player_window: Optional[object] = None
        self._custom_maximized = False
        self._restore_geometry: Optional[QRect] = None
        self._was_minimized = False

        self._apply_window_flags()
        self.setWindowTitle("STEM organizer")
        self.setMinimumSize(theme.WIN_MIN_W, theme.WIN_MIN_H)
        self._center_on_screen(theme.WIN_DEFAULT_W, theme.WIN_DEFAULT_H)
        if _ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_ICON_PATH)))

        # Root container: title bar (top) + content (middle) + status bar (bottom)
        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        # Dark fill on HWND + client before first paint (avoids default gray seams).
        prepare_dark_frameless_chrome(self)
        self._root_layout = QVBoxLayout(root)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        # Title bar — full width across the top
        self.title_bar = CustomTitleBar(self, height=theme.TITLE_BAR_HEIGHT)
        self.title_bar.close_requested = self.close
        self.title_bar.minimize_requested = self.showMinimized
        self.title_bar.maximize_requested = self._toggle_maximize
        self._root_layout.addWidget(self.title_bar)

        # Content: upper (tabs | log) + shared action row (Start/Stop/Play | Clear/Save)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(0)

        upper = QWidget()
        self._upper_layout = QHBoxLayout(upper)
        self._upper_layout.setContentsMargins(0, 0, 0, 0)
        self._upper_layout.setSpacing(6)

        self.left_col = QWidget()
        self.left_col.setFixedWidth(theme.LEFT_PANEL_WIDTH)
        self.left_col.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.left_col_layout = QVBoxLayout(self.left_col)
        self.left_col_layout.setContentsMargins(0, 0, 0, 0)
        self.left_col_layout.setSpacing(0)

        # Tabs + optional Rename side panel (PATH+PREVIEW) beside the tab bar
        self._tabs_row = QWidget()
        tabs_row_lay = QHBoxLayout(self._tabs_row)
        tabs_row_lay.setContentsMargins(0, 0, 0, 0)
        tabs_row_lay.setSpacing(0)

        self.tabs = TabWidget()
        theme.configure_tab_widget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        tabs_row_lay.addWidget(self.tabs, stretch=1)

        self._rename_side_host = QWidget()
        self._rename_side_host.hide()
        rename_side_lay = QVBoxLayout(self._rename_side_host)
        rename_side_lay.setContentsMargins(0, 0, 0, 0)
        rename_side_lay.setSpacing(0)
        tabs_row_lay.addWidget(self._rename_side_host, stretch=1)

        self.left_col_layout.addWidget(self._tabs_row, stretch=1)

        self._rename_footer_host = QWidget()
        self._rename_footer_host.hide()
        rename_footer_lay = QVBoxLayout(self._rename_footer_host)
        rename_footer_lay.setContentsMargins(0, 0, 0, 0)
        rename_footer_lay.setSpacing(0)
        self.left_col_layout.addWidget(self._rename_footer_host)

        self.log_panel = LogPanel()
        self.log_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_panel.setMinimumWidth(280)

        self._upper_layout.addWidget(self.left_col, 0)
        self._upper_layout.addWidget(self.log_panel, 1)

        # Shared bottom row — same height/place the action bar had before Clear/Save moved
        self.action_bar = ActionBarStack()
        self._action_row = QWidget()
        self._action_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        action_row_layout = QHBoxLayout(self._action_row)
        action_row_layout.setContentsMargins(0, 0, 0, 0)
        action_row_layout.setSpacing(6)

        self.action_left = QWidget()
        self.action_left.setFixedWidth(theme.LEFT_PANEL_WIDTH)
        self.action_left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        action_left_layout = QHBoxLayout(self.action_left)
        action_left_layout.setContentsMargins(0, 0, 0, 0)
        action_left_layout.addWidget(self.action_bar)
        action_row_layout.addWidget(self.action_left, 0)

        self._log_actions = QWidget()
        log_actions_layout = QHBoxLayout(self._log_actions)
        # Right inset matches LogView; left 0 matches log panel left edge
        log_actions_layout.setContentsMargins(0, 0, theme.PAGE_CONTENT_INSET, 0)
        log_actions_layout.setSpacing(theme.ACTION_BTN_GAP)
        log_actions_layout.addStretch(1)
        log_actions_layout.addWidget(self.log_panel.clear_btn)
        log_actions_layout.addWidget(self.log_panel.save_btn)
        action_row_layout.addWidget(self._log_actions, 1)

        self.content_layout.addWidget(upper, stretch=1)
        # Explicit QSpacerItem (not addSpacing) so rename mode can resize it to
        # drop the in-tab footer to the same button height as other tabs.
        self._action_row_spacer = QSpacerItem(
            0, theme.ACTION_ROW_TOP_GAP, QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.content_layout.addItem(self._action_row_spacer)
        self.content_layout.addWidget(self._action_row)

        self._root_layout.addWidget(self.content, stretch=1)

        # Breathing room between action buttons / Clear-Save and the Idle status line
        self._root_layout.addSpacing(theme.STATUS_TOP_GAP)

        # Status bar — full width across the bottom
        self.status_bar = StatusBar()
        self.status_bar_widget = self.status_bar  # alias for legacy naming
        self._root_layout.addWidget(self.status_bar)

        # Placeholder tab registration — replaced when real tabs are added
        self._tabs_registered: dict[str, object] = {}
        self._prev_tab_widget: Optional[QWidget] = None
        # Layout modes: standard (tabs|log), rename (full-width rename chrome),
        # wide (Dataset overview idle — hide log so charts can breathe).
        self._layout_mode: str = "standard"
        self._dataset_log_busy = False

        # Keyboard shortcut: Esc does nothing special here; keep window-wide
        # shortcuts in the relevant panels.
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)
        # Focus the active tab content on Ctrl+1..4
        for i in range(4):
            QShortcut(QKeySequence(f"Ctrl+{i+1}"), self,
                      activated=lambda idx=i: self._jump_to_tab(idx))

        # Initialize device status now (lazy import to avoid torch at module load)
        try:
            self.status_bar.set_device_text(self._device_status_text())
        except Exception:
            self.status_bar.set_device_text("Device: CPU")

        # Edge grips need the central widget tree first (see install_frame_resize).
        self._install_edge_resize()

    # ----- public API used by tabs -------------------------------------

    def register_tab(self, name: str, widget: QWidget) -> None:
        """Add a tab widget and remember it for action-bar stacking."""
        self.tabs.addTab(widget, name)
        self._tabs_registered[name] = widget
        theme.configure_tab_widget(self.tabs)

    def attach_action_bar(self, name: str, page: QWidget) -> None:
        self.action_bar.add_page(name, page)

    def show_action_bar(self, name: str) -> None:
        self.action_bar.show_page(name)

    def log_queue(self):
        return self.log_panel.queue

    def append_log(self, text: str, tag: str = "") -> None:
        bare = (text or "").strip()
        if bare == "__live_progress_end__":
            self.log_panel.end_live_progress()
            return
        if bare.startswith("__live_progress__\t"):
            parts = bare.split("\t")
            try:
                n = int(parts[1])
                total = int(parts[2])
            except (IndexError, ValueError):
                return
            label = parts[3].strip() if len(parts) > 3 else "files"
            self.log_panel.update_live_progress(f"{n:,}/{total:,} {label}")
            return
        self.log_panel.append_line(text, tag)

    def append_sdr_log(self, filename: str, score: float, threshold: float) -> None:
        self.log_panel.append_sdr_line(filename, score, threshold)

    def update_gg_processed(self, n: int, total: int) -> None:
        self.log_panel.update_gg_processed(n, total)

    def update_live_progress(self, text: str) -> None:
        self.log_panel.update_live_progress(text)

    def clear_log(self) -> None:
        self.log_panel.clear()

    def set_log_export_prefix(self, prefix: str) -> None:
        """Suggested Save filename, e.g. compression → compression_stem_organizer.log."""
        self.log_panel.set_export_prefix(prefix)

    def save_log(self) -> None:
        self.log_panel.save_to_file(self)

    def set_dataset_log_busy(self, busy: bool) -> None:
        """Show host LOG while Dataset overview is scanning / balancing."""
        busy = bool(busy)
        if self._dataset_log_busy == busy:
            return
        self._dataset_log_busy = busy
        # Only affect layout when that tab is active.
        current = self.tabs.currentWidget()
        if current is self._tabs_registered.get("Charts"):
            self._apply_content_layout("standard" if busy else "wide")

    # ----- window chrome -----------------------------------------------

    def _center_on_screen(self, width: int, height: int) -> None:
        """Size the window to fit the available screen area and center it."""
        center_window_default_size(self, width, height)

    def _apply_window_flags(self) -> None:
        # Single atomic flags call — avoid sequential setWindowFlag recreates.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        from .widgets.titlebar import install_rounded_corner_watcher

        install_rounded_corner_watcher(self, radius=theme.WINDOW_CORNER_RADIUS)
        # Edge grips after content tree; Win32 thick-frame only after show().

    def _install_edge_resize(self) -> None:
        """Edge grips after central widget. Thick-frame waits for showEvent."""
        from .widgets.titlebar import install_frame_resize

        install_frame_resize(self)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)

        # Defer Win32 style tweak until the platform window exists and is shown.
        # Touching styles via winId() during __init__ caused CreateWindowEx failures.
        def _after_show() -> None:
            # Thick-frame FRAMECHANGED + NCCALCSIZE + re-apply rounded region.
            enable_win32_thick_frame(self)
            handler = getattr(self, "_frame_resize_handler", None)
            if handler is not None:
                handler._layout_grips()
                handler._raise_grips()

        QTimer.singleShot(0, _after_show)

    def changeEvent(self, event) -> None:  # noqa: N802
        # Minimize → taskbar restore: reset to default opening size so a prior
        # custom work-area fill does not come back as an awkward "max" window.
        note_minimize_restore_to_default(
            self, event, width=theme.WIN_DEFAULT_W, height=theme.WIN_DEFAULT_H
        )
        # Focus return (Explorer/Alt-Tab) can flash light thick-frame edges.
        note_activation_chrome_refresh(self, event)
        super().changeEvent(event)

    def _toggle_maximize(self) -> None:
        """Fill the monitor work area (CTk-style) — avoid OS showMaximized."""
        toggle_work_area_maximize(self)

    def nativeEvent(self, eventType, message):  # noqa: N802
        """Win32 edge resize via WM_NCHITTEST (+ WM_NCCALCSIZE for thick frame).

        Requires ``enable_win32_thick_frame`` so Windows actually delivers edge
        hit-tests. Returning HTLEFT/HTRIGHT/… lets the OS run the resize drag
        (cursors + rubber-band). Left panel stays Fixed; log stretches.
        """
        if sys.platform == "win32" and eventType in (b"windows_generic_MSG", "windows_generic_MSG"):
            try:
                from ctypes import wintypes

                msg = wintypes.MSG.from_address(int(message))
                handled = handle_native_frame_message(self, msg)
                if handled is not None:
                    return handled
            except Exception:
                pass
        return super().nativeEvent(eventType, message)

    def _jump_to_tab(self, idx: int) -> None:
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)
            # Fluent TabWidget.setCurrentIndex does not emit currentChanged (only
            # tab clicks do) — drive layout/dock hooks explicitly.
            self._on_tab_changed(idx)
            w = self.tabs.widget(idx)
            if w is not None:
                w.setFocus()

    def _undock_rename_if_needed(self) -> None:
        rename = self._tabs_registered.get("Rename")
        app = getattr(rename, "app", None) if rename is not None else None
        if app is not None and hasattr(app, "undock_from_host"):
            app.undock_from_host()
        self._rename_side_host.hide()
        self._rename_footer_host.hide()

    def _pin_left_column_fixed(self) -> None:
        """Classify / Genre / Match / busy Dataset — fixed left + stretching LOG."""
        self.left_col.setMinimumWidth(theme.LEFT_PANEL_WIDTH)
        self.left_col.setMaximumWidth(theme.LEFT_PANEL_WIDTH)
        self.left_col.setFixedWidth(theme.LEFT_PANEL_WIDTH)
        self.left_col.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.action_left.setMinimumWidth(theme.LEFT_PANEL_WIDTH)
        self.action_left.setMaximumWidth(theme.LEFT_PANEL_WIDTH)
        self.action_left.setFixedWidth(theme.LEFT_PANEL_WIDTH)
        self.action_left.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._upper_layout.setStretch(0, 0)
        self._upper_layout.setStretch(1, 1)

    def _expand_left_column(self) -> None:
        """Rename / Dataset overview idle — content uses full upper width."""
        self.left_col.setMinimumWidth(0)
        self.left_col.setMaximumWidth(16777215)
        self.left_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.action_left.setMinimumWidth(0)
        self.action_left.setMaximumWidth(16777215)
        self.action_left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._upper_layout.setStretch(0, 1)
        self._upper_layout.setStretch(1, 0)

    def _apply_content_layout(self, mode: str) -> None:
        """Switch between standard / rename / wide (charts without LOG)."""
        if mode not in ("standard", "rename", "wide"):
            mode = "standard"
        if mode == self._layout_mode:
            return
        prev = self._layout_mode
        self._layout_mode = mode

        if prev == "rename":
            self._undock_rename_if_needed()

        if mode == "rename":
            self.log_panel.hide()
            self._log_actions.hide()
            self._action_row.hide()
            # Shrink the gap reserved for the action row so the rename footer
            # (inside the tab) lines up with other tabs' action buttons.
            self._action_row_spacer.changeSize(
                0, 6, QSizePolicy.Minimum, QSizePolicy.Fixed
            )
            self.content_layout.invalidate()
            self._expand_left_column()

            rename = self._tabs_registered.get("Rename")
            app = getattr(rename, "app", None) if rename is not None else None
            if app is not None and hasattr(app, "dock_to_host"):
                app.dock_to_host(self._rename_side_host, self._rename_footer_host)
                self._rename_side_host.show()
                self._rename_footer_host.show()
            theme.configure_tab_widget(self.tabs)
            return

        # standard + wide share the host action row
        self._action_row.show()
        self._action_row_spacer.changeSize(
            0, theme.ACTION_ROW_TOP_GAP, QSizePolicy.Minimum, QSizePolicy.Fixed
        )
        self.content_layout.invalidate()

        if mode == "wide":
            self.log_panel.hide()
            self._log_actions.hide()
            self._expand_left_column()
            theme.configure_tab_widget(self.tabs)
            return

        # standard
        self.log_panel.show()
        self._log_actions.show()
        self._pin_left_column_fixed()
        theme.configure_tab_widget(self.tabs)

    def _show_rename_mode_layout(self) -> None:
        self._apply_content_layout("rename")

    def _show_standard_mode_layout(self) -> None:
        self._apply_content_layout("standard")

    def _show_wide_charts_layout(self) -> None:
        self._apply_content_layout("wide")

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        prev = self._prev_tab_widget
        if prev is not None and prev is not widget:
            # Persist paths before the next tab reads cross-tab takeovers.
            flush = getattr(prev, "flush_settings", None)
            if callable(flush):
                try:
                    flush()
                except Exception:
                    pass
            if hasattr(prev, "on_tab_hidden"):
                try:
                    prev.on_tab_hidden()
                except Exception:
                    pass
        self._prev_tab_widget = widget

        for name, w in self._tabs_registered.items():
            if w is widget:
                if name == "Rename":
                    self.action_bar.setVisible(False)
                    self._show_rename_mode_layout()
                elif name == "Charts":
                    self.action_bar.setVisible(True)
                    if self.action_bar.has_page(name):
                        self.show_action_bar(name)
                    if self._dataset_log_busy:
                        self._show_standard_mode_layout()
                    else:
                        self._show_wide_charts_layout()
                else:
                    self._show_standard_mode_layout()
                    self.action_bar.setVisible(True)
                    if self.action_bar.has_page(name):
                        self.show_action_bar(name)
                if hasattr(w, "on_tab_shown"):
                    try:
                        w.on_tab_shown()
                    except Exception:
                        pass
                return

    # ----- stem player launcher (lazy) ---------------------------------

    def _open_player(self) -> None:
        """Open the Stem Player window. Lazy import to keep startup lean."""
        try:
            from .player.stem_player_window import open_stem_player
        except Exception as exc:
            self.append_log(f"[player] not available: {exc}", "warn")
            try:
                from .widgets.dialogs import show_info

                show_info(self, "STEM Player", f"Player is not available:\n{exc}")
            except Exception:
                pass
            return
        library_root = ""
        # Active tab supplies the library whose children are song folders
        # (Classify: output_row; Align: with_original; Match: pairs output).
        widget = self.tabs.currentWidget()
        if widget is not None:
            resolver = getattr(widget, "player_library_root", None)
            if callable(resolver):
                try:
                    library_root = (resolver() or "").strip()
                except Exception:
                    library_root = ""
            if not library_root:
                for attr in ("output_row", "pairs_output_row", "with_original_row"):
                    row = getattr(widget, attr, None)
                    if row is not None and hasattr(row, "text"):
                        library_root = row.text().strip()
                        if library_root:
                            break
        try:
            open_stem_player(self, library_root=library_root or None)
        except Exception as exc:
            self.append_log(f"[player] failed to open: {exc}", "warn")
            try:
                from .widgets.dialogs import show_info

                show_info(self, "STEM Player", f"Could not open the player:\n{exc}")
            except Exception:
                pass

    # ----- device label ------------------------------------------------

    def _device_status_text(self) -> str:
        try:
            import torch  # noqa: WPS433  lazy
            if torch.cuda.is_available():
                try:
                    name = torch.cuda.get_device_name(0)
                    return f"Device: GPU ({name})"
                except Exception:
                    return "Device: GPU"
            return "Device: CPU"
        except Exception:
            return "Device: CPU"

    # ----- close -------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 Qt naming
        from .widgets.titlebar import disarm_win32_thick_frame

        disarm_win32_thick_frame(self)
        # Flush pending debounced settings from every tab before exit.
        for widget in self._tabs_registered.values():
            flush = getattr(widget, "flush_settings", None)
            if callable(flush):
                try:
                    flush()
                except Exception:
                    pass
        for widget in self._tabs_registered.values():
            shutdown = getattr(widget, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        if self._player_window is not None and hasattr(self._player_window, "close"):
            try:
                self._player_window.close()
            except Exception:
                pass
        try:
            from .player.stem_player_window import close_stem_player

            close_stem_player()
        except Exception:
            pass
        event.accept()

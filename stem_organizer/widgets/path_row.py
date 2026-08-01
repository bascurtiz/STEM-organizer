"""Path row widget — label + Fluent LineEdit + Browse + Open."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QSizePolicy, QWidget
from qfluentwidgets import BodyLabel, LineEdit

from .. import theme
from .action_button import action_button


class PathRow(QWidget):
    """One row: label + entry + Browse + Open.

    If ``parent`` already has a layout, this row is added to it automatically
    (matches the old CTk “create under parent” packing pattern).
    """

    def __init__(
        self,
        parent: QWidget,
        label: str,
        *,
        directory: bool = True,
        save_dialog: bool = False,
        filter_pattern: str = "",
        caption: str = "Select folder",
        tip_text: str = "",
        label_width: int = 56,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        self._lbl = BodyLabel(label)
        self._lbl.setFixedWidth(label_width)
        self._lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self._lbl)

        field_h = theme.PATH_FIELD_HEIGHT
        self.entry = LineEdit()
        self.entry.setFixedHeight(field_h)
        self.entry.setClearButtonEnabled(False)
        theme.style_line_edit(self.entry)
        layout.addWidget(self.entry, stretch=1)

        browse_tip = tip_text or (
            "Choose a file path."
            if (save_dialog or filter_pattern)
            else "Choose a folder."
        )
        open_tip = (
            "Open this file's folder in Explorer."
            if (save_dialog or filter_pattern)
            else "Open this folder in Explorer."
        )

        self.browse_btn = action_button(
            "Browse",
            on_click=self._browse,
            width=theme.PATH_BTN_WIDTH_BROWSE,
            height=field_h,
            tip=browse_tip,
        )
        layout.addWidget(self.browse_btn)

        self.open_btn = action_button(
            "Open",
            on_click=self._open,
            width=theme.PATH_BTN_WIDTH_OPEN,
            height=field_h,
            tip=open_tip,
        )
        layout.addWidget(self.open_btn)

        self._save_dialog = save_dialog
        self._filter = filter_pattern
        self._caption = caption

        if tip_text:
            tip = theme.format_tooltip(tip_text)
            self._lbl.setToolTip(tip)
            self.entry.setToolTip(tip)

        # Auto-pack into parent layout (fixes overlapping PathRows).
        if parent is not None:
            parent_layout = parent.layout()
            if parent_layout is not None:
                parent_layout.addWidget(self)

    # ---- public API ----

    def text(self) -> str:
        return self.entry.text().strip()

    def set_text(self, value: str) -> None:
        self.entry.setText(value or "")

    def changed_connect(self, callback: Callable[[str], None]) -> None:
        self.entry.textChanged.connect(callback)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 Qt name
        """Busy-lock greys the path field only — Browse / Open stay usable."""
        # Keep the row widget itself enabled so child buttons can stay active
        # (Qt forces children disabled when the parent is disabled).
        super().setEnabled(True)
        self._lbl.setEnabled(enabled)
        self.entry.setEnabled(enabled)
        self.browse_btn.setEnabled(True)
        self.open_btn.setEnabled(True)

    # ---- internals ----

    def _browse(self) -> None:
        cur = self.entry.text().strip()
        start = cur if (cur and os.path.isdir(cur)) else os.path.expanduser("~")
        if self._save_dialog:
            path, _ = QFileDialog.getSaveFileName(self, self._caption, start, self._filter)
        elif self._filter:
            path, _ = QFileDialog.getOpenFileName(self, self._caption, start, self._filter)
        else:
            path = QFileDialog.getExistingDirectory(self, self._caption, start)
        if path:
            # Normalize slashes on Windows for consistency with display_path.
            if sys.platform == "win32":
                path = path.replace("/", "\\")
            self.entry.setText(path)

    def _open(self) -> None:
        target = self.entry.text().strip()
        if not target or not os.path.exists(target):
            return
        try:
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", target], check=False)
            else:
                subprocess.run(["xdg-open", target], check=False)
        except Exception:
            pass

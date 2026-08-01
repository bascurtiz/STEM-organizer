"""Compression sub-tab (Integrity) — flac-detective → COMPRESSION tags."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CheckBox

from .. import theme
from ..dataset.compression import run_compression_detect
from ..dataset.inventory import iter_audio_files
from ..io_tune import ensure_tuned
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.option_hint import style_how_it_works
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.overview_worker import OverviewWorker

PANEL_TITLE = "Compression"

TIPS = {
    "input": (
        "Folder of audio to analyze. Nested folders included when checked. "
        "_backup_before_align folders are skipped."
    ),
    "include_subfolders": "Scan nested folders under the input root.",
    "skip_existing": (
        "On (default): skip files already tagged COMPRESSION=lossless or lossy. "
        "Compression tag is read by Charts."
    ),
    "detect": (
        "Run flac-detective on lossless containers and write COMPRESSION=lossless|lossy. "
        "Native MP3/OGG/… are tagged lossy without analysis."
    ),
    "stop": "Stop the running compression job.",
}
TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


class CompressionTab(QWidget):
    request_status_running = Signal()
    request_status_idle = Signal(str)
    request_status_text = Signal(str)
    request_progress = Signal(float, object, int, int, str)
    request_log = Signal(str, str)
    request_clear_log = Signal()

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._loading = False
        self._worker: Optional[OverviewWorker] = None
        self._busy = False
        self._build_ui()
        self.load_settings()
        self._bind_autosave()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Nested under Integrity TabWidget — no extra outer chrome.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {theme.COLORS['bg']}; border: none; }}"
        )
        inner = QWidget()
        inner.setObjectName("CompressionPage")
        # Scope to this widget only — bare `background:` would cascade onto #Card.
        inner.setStyleSheet(
            f"QWidget#CompressionPage {{ background: {theme.COLORS['bg']}; }}"
        )
        self._inner = QVBoxLayout(inner)
        self._inner.setAlignment(Qt.AlignTop)
        self._inner.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        self._inner.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel(
            "Detect (fake) lossless audio and write lossless / lossy tags"
        )
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=self._show_help))
        header.addStretch(1)
        self._inner.addLayout(header)

        paths = Section(inner, "Path")
        # Same Path top padding as Genre (Section default 8)
        paths.body.layout().setSpacing(12)
        self.input_row = PathRow(
            paths.body, "Input", tip_text=TIPS["input"], label_width=80
        )
        self.include_subfolders = CheckBox("Include subfolders")
        self.include_subfolders.setChecked(True)
        self.include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths.body.layout().addWidget(self.include_subfolders)
        self.skip_existing = CheckBox("Skip if already tagged")
        self.skip_existing.setChecked(True)
        self.skip_existing.setToolTip(TIPS["skip_existing"])
        paths.body.layout().addWidget(self.skip_existing)
        self._inner.addWidget(paths)

        note = Section(inner, "How it works")
        note_lay = note.body.layout()
        note_lay.setSpacing(10)
        for paragraph in (
            "Lossless containers (WAV/FLAC/AIFF) are scored with FLAC Detective’s "
            "11-rule spectral analysis.",
            "AUTHENTIC → lossless.",
            "WARNING / SUSPICIOUS / FAKE_CERTAIN → lossy.",
            "Native MP3 / OGG / Opus / AAC are tagged lossy without analysis.",
        ):
            lbl = BodyLabel(paragraph)
            lbl.setWordWrap(True)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            style_how_it_works(lbl)
            note_lay.addWidget(lbl)
        self._inner.addWidget(note)

        self._inner.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

    def bind_window(self, window) -> None:
        self._window = window

    def tip(self, key: str) -> str:
        return TIPS.get(key, "")

    def bind_action_buttons(self, detect_btn, stop_btn) -> None:
        self.detect_btn = detect_btn
        self.stop_btn = stop_btn

    def set_sibling_busy_hook(self, hook) -> None:
        self._sibling_busy_hook = hook

    def is_busy(self) -> bool:
        return bool(self._busy)

    def start_detect(self) -> None:
        self._on_detect()

    def stop_detect(self) -> None:
        self._on_stop()

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        """Standalone registration (unused when hosted under Integrity)."""
        self.bind_window(window)
        self.detect_btn = action_button(
            "▶ Detect compression",
            on_click=self.start_detect,
            accent=True,
            tip=TIPS["detect"],
        )
        self.stop_btn = action_button(
            "■ Stop", on_click=self.stop_detect, tip=TIPS["stop"]
        )
        self.stop_btn.setEnabled(False)
        page.add_button(self.detect_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()

    @staticmethod
    def _enable_widgets(enabled: bool, *widgets) -> None:
        for w in widgets:
            if w is None:
                continue
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def _set_settings_enabled(self, enabled: bool) -> None:
        self._enable_widgets(
            enabled,
            self.input_row,
            self.include_subfolders,
            self.skip_existing,
        )

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self._set_settings_enabled(not busy)
        hook = getattr(self, "_sibling_busy_hook", None)
        if hook is not None:
            hook(busy)
        elif hasattr(self, "detect_btn"):
            self.detect_btn.setEnabled(not busy)
            self.stop_btn.setEnabled(busy)
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def _on_detect(self) -> None:
        root = self.input_row.text().strip()
        if not root or not Path(root).is_dir():
            show_info(
                self,
                PANEL_TITLE,
                "Browse & select an input folder first.",
            )
            return
        recursive = bool(self.include_subfolders.isChecked())
        skip_existing = bool(self.skip_existing.isChecked())

        def action(on_log, on_progress, stop_event):
            hint = ensure_tuned(
                root,
                self._settings,
                workload="compression",
                log=on_log,
            )
            paths = iter_audio_files(
                Path(root),
                recursive=recursive,
                stop_event=stop_event,
            )
            on_log(f"{len(paths):,} file(s) under {root}", "info")
            return run_compression_detect(
                paths,
                skip_existing=skip_existing,
                max_workers=hint.process_workers,
                on_log=on_log,
                on_progress=on_progress,
                stop_event=stop_event,
            )

        if self._worker is not None:
            return
        self.request_clear_log.emit()
        if self._window is not None:
            self._window.set_log_export_prefix("compression")
        self.request_log.emit("=== Detect compression ===", "info")
        self._set_busy(True, "Detecting compression…")
        self._worker = OverviewWorker(action, parent=self)
        self._worker.log_line.connect(self.request_log)
        self._worker.progress.connect(self.request_progress)
        self._worker.status.connect(self.request_status_text)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, status: str) -> None:
        self._worker = None
        self._set_busy(False, status)

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.request_status_text.emit("Stopping…")
            self.request_log.emit("Compression detect stopping…", "warn")

    def _show_help(self) -> None:
        help_dialog(
            self,
            title="Compression help",
            heading="Fake-lossless detection",
            intro=(
                "Spot lossy audio re-saved as FLAC/WAV, then write a COMPRESSION tag."
            ),
            sections=[
                ("Path", [
                    "One input folder. Include subfolders when nested libraries should be scanned.",
                    "_backup_before_align folders are skipped.",
                ]),
                ("Tags", [
                    "COMPRESSION=lossless or lossy (Vorbis / ID3 TXXX / iTunes freeform).",
                    "Skip already-tagged files when the checkbox is on (default). "
                    "Turn off to re-tag everything.",
                ]),
                ("Verdicts", [
                    "AUTHENTIC → lossless.",
                    "WARNING / SUSPICIOUS / FAKE_CERTAIN → lossy.",
                    "Native MP3 / OGG / Opus / AAC → lossy (no spectral pass).",
                ]),
                ("Sources", [
                    '<a href="https://pypi.org/project/flac-detective/">FLAC Detective</a> '
                    "— 11-rule spectral analysis for lossless containers.",
                ]),
            ],
        )

    def settings_snapshot(self) -> dict:
        return {
            "cmp_input_dir": display_path(self.input_row.text()),
            "cmp_include_subfolders": bool(self.include_subfolders.isChecked()),
            "cmp_skip_existing": bool(self.skip_existing.isChecked()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("cmp_input_dir"):
                self.input_row.set_text(d["cmp_input_dir"])
            elif d.get("output_dir"):
                self.input_row.set_text(d["output_dir"])
            self.include_subfolders.setChecked(
                bool(d.get("cmp_include_subfolders", True))
            )
            if "cmp_skip_existing" in d:
                self.skip_existing.setChecked(bool(d["cmp_skip_existing"]))
            else:
                # Legacy: cmp_overwrite on meant do not skip.
                self.skip_existing.setChecked(not bool(d.get("cmp_overwrite", False)))
        finally:
            self._loading = False

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        for sig in (
            self.input_row.entry.textChanged,
            self.include_subfolders.toggled,
            self.skip_existing.toggled,
        ):
            sig.connect(self._schedule_save)

    def _schedule_save(self, *_) -> None:
        if self._loading:
            return
        self._autosave_timer.start()

    def _flush_settings(self) -> None:
        self._settings.merge(self.settings_snapshot())
        self._settings.flush()

    def flush_settings(self) -> None:
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()


def register(window, settings: SettingsStore) -> None:
    """Deprecated: use integrity_tab.register (kept for direct testing)."""
    from .integrity_tab import register as register_integrity

    register_integrity(window, settings)

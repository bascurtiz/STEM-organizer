"""Key Detect tab — MusicalKeyCNN → COMMENT or Initial key (TKEY)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CheckBox, RadioButton

from .. import theme
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.option_hint import split_option_label, style_option_hint
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.key_worker import KeyWorker

PANEL_TITLE = "Key"

TIPS = {
    "input": (
        "Folder of audio to analyze (instrumentals, vocals, pairs, samples). "
        "Nested folders included when checked. "
        "_backup_before_align folders are skipped."
    ),
    "include_subfolders": "Scan nested folders under the input root.",
    "run_mode_batch": (
        "Batch decodes several files in parallel, then runs the model in GPU "
        "batches. Fastest overall; LOG shows a single progress counter."
    ),
    "run_mode_per_file": (
        "Per-file processes one track at a time and prints each prediction "
        "live in the LOG. Still uses the GPU when available; slower than Batch."
    ),
    "tag_field_comment": "Write the short key (e.g. Db) into the standard Comment field.",
    "tag_field_key": (
        "Write the short key into the DJ-standard Initial key field "
        "(ID3 TKEY / Vorbis INITIALKEY / iTunes initialkey)."
    ),
    "write_meta": "Write detected keys to metadata. Turn off to preview in the LOG only.",
    "skip_existing": (
        "On (default): skip files that already have a key tag (resume-friendly). "
        "Off: re-tag every file."
    ),
    "detect": "Detect musical key with the in-house KeyNet model and write tags.",
    "stop": "Stop the running key-detect job.",
}
TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


def _load_skip_existing(d: dict, skip_key: str, overwrite_key: str) -> bool:
    """Prefer skip_existing; migrate legacy overwrite (inverted). Default on."""
    if skip_key in d:
        return bool(d[skip_key])
    if overwrite_key in d:
        return not bool(d[overwrite_key])
    return True


class _RadioRow(QWidget):
    """Two radio options split evenly across the card width."""

    def __init__(
        self,
        parent: QWidget,
        options,
        value: str,
        *,
        tooltips: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._buttons: dict[str, RadioButton] = {}
        self._extra: list[QWidget] = []

        for label, val in options:
            main, hint = split_option_label(label)
            key = str(val)
            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cell_lay = QHBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(6)
            rb = RadioButton(main)
            rb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            rb.setCursor(Qt.PointingHandCursor)
            tip_text = (tooltips or {}).get(key) or ""
            if tip_text:
                rb.setToolTip(tip_text)
            self._group.addButton(rb)
            self._buttons[key] = rb
            cell_lay.addWidget(rb, 0, Qt.AlignVCenter)
            if hint:
                hint_lbl = QLabel(hint)
                hint_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                style_option_hint(hint_lbl)
                hint_lbl.setCursor(Qt.PointingHandCursor)
                if tip_text:
                    hint_lbl.setToolTip(tip_text)
                hint_lbl.mousePressEvent = (  # type: ignore[method-assign]
                    lambda _e, v=key: self.set_value(v)
                )
                cell_lay.addWidget(hint_lbl, 0, Qt.AlignVCenter)
                self._extra.append(hint_lbl)
            cell_lay.addStretch(1)
            layout.addWidget(cell, 1)
            self._extra.append(cell)

        self.set_value(value)
        self.valueChanged = self._group.buttonClicked

    def set_value(self, value: str) -> None:
        rb = self._buttons.get(str(value))
        if rb is not None:
            rb.setChecked(True)

    def value(self) -> str:
        for val, rb in self._buttons.items():
            if rb.isChecked():
                return val
        return ""

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        for rb in self._buttons.values():
            rb.setEnabled(enabled)
        for w in self._extra:
            w.setEnabled(enabled)


class KeyDetectTab(QWidget):
    request_status_running = Signal()
    request_status_idle = Signal(str)
    request_progress = Signal(float, object, int, int, str)
    request_status_text = Signal(str)
    request_log = Signal(str, str)
    request_clear_log = Signal()
    request_gg_processed = Signal(int, int)

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._loading = False
        self._worker: Optional[KeyWorker] = None
        self._busy = False
        self._build_ui()
        self.load_settings()
        self._bind_autosave()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {theme.COLORS['bg']}; border: none; }}"
        )
        inner = QWidget()
        inner.setObjectName("KeyDetectPage")
        inner.setStyleSheet(
            f"QWidget#KeyDetectPage {{ background: {theme.COLORS['bg']}; }}"
        )
        v = QVBoxLayout(inner)
        v.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        v.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel(
            "Detect musical key (24 majors / minors) and write Comment or Initial key tags"
        )
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=self._show_help))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.input_row = PathRow(
            paths.body, "Input folder", tip_text=TIPS["input"], label_width=80
        )
        self.include_subfolders = CheckBox("Include subfolders")
        self.include_subfolders.setChecked(True)
        self.include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths.body.layout().addWidget(self.include_subfolders)
        v.addWidget(paths)

        run_card = Section(inner, "Run mode")
        self.run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
            tooltips={
                "batch": TIPS["run_mode_batch"],
                "per_file": TIPS["run_mode_per_file"],
            },
        )
        run_card.body.layout().addWidget(self.run_mode)
        v.addWidget(run_card)

        field_card = Section(inner, "Write label to")
        self.tag_field = _RadioRow(
            field_card.body,
            [("Comment", "comment"), ("Initial key", "key")],
            "key",
            tooltips={
                "comment": TIPS["tag_field_comment"],
                "key": TIPS["tag_field_key"],
            },
        )
        field_card.body.layout().addWidget(self.tag_field)
        v.addWidget(field_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.write_meta = CheckBox("Write metadata tags")
        self.write_meta.setChecked(True)
        self.write_meta.setToolTip(TIPS["write_meta"])
        self.skip_existing = CheckBox("Skip if already tagged")
        self.skip_existing.setChecked(True)
        self.skip_existing.setToolTip(TIPS["skip_existing"])
        opts_lay.addWidget(self.write_meta)
        opts_lay.addWidget(self.skip_existing)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._window = window
        self.detect_btn = action_button(
            "▶ Detect key",
            on_click=self._start,
            accent=True,
            tip=TIPS["detect"],
        )
        self.stop_btn = action_button("■ Stop", on_click=self._stop, tip=TIPS["stop"])
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

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._enable_widgets(
            not busy,
            self.input_row,
            self.include_subfolders,
            self.run_mode,
            self.tag_field,
            self.write_meta,
            self.skip_existing,
            self.detect_btn,
        )
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(busy)

    def _start(self) -> None:
        if self._busy:
            return
        folder = (self.input_row.text() or "").strip()
        if not folder:
            show_info(self, "Key Detect", "Choose an input folder first.")
            return
        from pathlib import Path

        if not Path(folder).is_dir():
            show_info(self, "Key Detect", f"Folder not found:\n{folder}")
            return

        self.request_clear_log.emit()
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("key")
        mode = "Batch" if self.run_mode.value() == "batch" else "Per-file"
        field = "Initial key" if (self.tag_field.value() or "key") == "key" else "Comment"
        self.request_log.emit("  Starting Key Detect:", "info")
        self.request_log.emit(f"  {folder}", "info")
        self.request_log.emit(
            f"  Mode: {mode}  ·  Write to: {field}  ·  "
            f"Skip if already tagged: {'on' if self.skip_existing.isChecked() else 'off'}",
            "info",
        )
        self.request_log.emit("  Loading model (first run can take a moment)…", "info")
        self.request_status_running.emit()
        self._set_busy(True)
        worker = KeyWorker(
            folder,
            include_subfolders=self.include_subfolders.isChecked(),
            write_meta=self.write_meta.isChecked(),
            overwrite_tags=not self.skip_existing.isChecked(),
            tag_field=self.tag_field.value() or "key",
            batch_mode=self.run_mode.value() == "batch",
            parent=self,
        )
        self._worker = worker
        worker.log_line.connect(self.request_log.emit)
        worker.progress.connect(self.request_progress.emit)
        worker.processed.connect(self.request_gg_processed.emit)
        worker.finished_ok.connect(self._on_finished)
        worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.request_status_text.emit("Stopping…")
            self.request_log.emit("Key Detect stopping…", "warn")

    def _on_finished(self, status: str) -> None:
        self._worker = None
        self._set_busy(False)
        self.request_status_idle.emit(status or "Done")

    def _show_help(self) -> None:
        help_dialog(
            self,
            title="Key Detect help",
            heading="Musical key detection",
            intro=(
                "Scores each audio file with an in-house KeyNet CNN, "
                "then writes a short key label."
            ),
            sections=[
                (
                    "Path",
                    [
                        "One input folder. Include subfolders when nested libraries "
                        "should be scanned. Processes all audio types STEM Organizer "
                        "supports (FLAC, WAV, MP3, OGG, Opus, M4A, AIFF, …). "
                        "_backup_before_align folders are skipped.",
                    ],
                ),
                (
                    "Run mode",
                    [
                        "Batch (fast) parallelizes CQT on CPU workers, then "
                        "GPU-batches 8 s chunks. LOG shows a progress counter only.",
                        "Per-file prints each prediction live in the LOG.",
                    ],
                ),
                (
                    "Tags",
                    [
                        "Short keys only (C, Am, Db, Gbm, …) — Charts show enharmonic "
                        "labels (Db/C#, Abm/G#m).",
                        "Comment or Initial key (TKEY). Skip if already tagged "
                        "(default on) is resume-friendly.",
                    ],
                ),
                (
                    "Sources",
                    [
                        "KeyNet CNN (checkpoint nf50-q05-221125.pt) — in-house model "
                        "that outperforms the original MusicalKeyCNN release and "
                        "Mixed In Key on "
                        '<a href="https://docs.google.com/spreadsheets/d/'
                        '1asmBVlIjimZ9XAmK5JE42SX4vAvjGqjLflukYBgFSuE/edit?usp=sharing">'
                        "Consensus Balanced dataset</a>.",
                    ],
                ),
            ],
        )

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            self.input_row.set_text(display_path(d.get("key_detect_input", "")))
            self.include_subfolders.setChecked(
                bool(d.get("key_detect_include_subfolders", True))
            )
            self.run_mode.set_value(
                "batch" if d.get("key_detect_batch_mode", True) else "per_file"
            )
            field = str(d.get("key_detect_tag_field", "key") or "key")
            self.tag_field.set_value(field if field in ("comment", "key") else "key")
            self.write_meta.setChecked(bool(d.get("key_detect_write_meta", True)))
            self.skip_existing.setChecked(
                _load_skip_existing(d, "key_detect_skip_existing", "key_detect_overwrite")
            )
            # Take over Classify output when Key input is empty.
            if not self.input_row.text().strip():
                classify_out = (d.get("output_dir") or "").strip()
                if classify_out:
                    self.input_row.set_text(display_path(classify_out))
        finally:
            self._loading = False

    def on_tab_shown(self) -> None:
        """Key input takes over Classify output path."""
        classify_out = (self._settings.data.get("output_dir") or "").strip()
        if classify_out:
            self._loading = True
            try:
                self.input_row.set_text(display_path(classify_out))
            finally:
                self._loading = False
            self._flush_settings()

    def _flush_settings(self) -> None:
        if self._loading:
            return
        self._settings.merge(
            {
                "key_detect_input": display_path(self.input_row.text()),
                "key_detect_include_subfolders": self.include_subfolders.isChecked(),
                "key_detect_batch_mode": self.run_mode.value() == "batch",
                "key_detect_tag_field": self.tag_field.value() or "key",
                "key_detect_write_meta": self.write_meta.isChecked(),
                "key_detect_skip_existing": self.skip_existing.isChecked(),
            }
        )
        self._settings.flush()

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(400)
        self._autosave_timer.timeout.connect(self._flush_settings)

        def bump(*_a) -> None:
            if not self._loading:
                self._autosave_timer.start()

        self.input_row.entry.textChanged.connect(bump)
        self.include_subfolders.toggled.connect(bump)
        self.run_mode.valueChanged.connect(bump)
        self.tag_field.valueChanged.connect(bump)
        self.write_meta.toggled.connect(bump)
        self.skip_existing.toggled.connect(bump)

    def flush_settings(self) -> None:
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()


def register(window, settings: SettingsStore) -> None:
    tab = KeyDetectTab(settings)
    window.register_tab(PANEL_TITLE, tab)
    page = window.action_bar.make_page(PANEL_TITLE)
    tab.attach_action_bar(page, window)
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_status_text.connect(window.status_bar.set_status)
    tab.request_log.connect(window.append_log)
    tab.request_clear_log.connect(window.clear_log)
    tab.request_gg_processed.connect(window.update_gg_processed)

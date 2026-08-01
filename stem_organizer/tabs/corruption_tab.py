"""Corruption sub-tab (Integrity) — Fast / Deep verify + optional Fix / Quarantine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CheckBox, RadioButton

from .. import theme
from ..corruption.tools import find_mp3val
from ..dataset.corruption import run_corruption_detect
from ..dataset.inventory import iter_audio_files
from ..io_tune import ensure_tuned
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.option_hint import option_with_hint, style_how_it_works
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.overview_worker import OverviewWorker

PANEL_TITLE = "Corruption"

TIPS = {
    "input": (
        "Folder of audio to verify. Nested folders included when checked. "
        "_backup_before_align folders are skipped."
    ),
    "include_subfolders": "Scan nested folders under the input root.",
    "skip_existing": (
        "On (default): skip files already tagged CORRUPTION=ok "
        "(WARNING and FAILED are always re-checked). "
        "Corruption tag is read by Charts."
    ),
    "fix": (
        "Rebuild flagged files, write CORRUPTION=ok on the rebuild, then replace "
        "the original in place (no *_FIXED left). MP3 re-encodes audio only "
        "(cover art ignored) + MP3val polish. Failed rebuilds delete *_FIXED; "
        "unfixable failed originals are moved to {input}_CORRUPT. Only reported "
        "fixed when post-verify is clean."
    ),
    "quarantine": (
        "Move failed/suspect files into {input}_CORRUPT beside the input folder. "
        "Unfixable leftovers after a failed Fix are quarantined even when this "
        "is off."
    ),
    "detect": "Run Fast / Deep integrity checks and write CORRUPTION tags.",
    "stop": "Stop the running corruption job.",
    "mode_fast": "Structural frame walk (AudioTester-style) — very fast for MP3.",
    "mode_deep": "Full decode via ffmpeg (foobar Verify–like) + flac -t when available.",
    "mode_both": "Run Fast and Deep; Fast-only fails become suspect when Deep is OK.",
}
TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


class CorruptionTab(QWidget):
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
        inner.setObjectName("CorruptionPage")
        # Scope to this widget only — bare `background:` would cascade onto #Card.
        inner.setStyleSheet(
            f"QWidget#CorruptionPage {{ background: {theme.COLORS['bg']}; }}"
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
        title = BodyLabel("Find and fix corrupt audio files")
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

        mode_sec = Section(inner, "Mode")
        mode_lay = mode_sec.body.layout()
        mm = mode_lay.contentsMargins()
        mode_lay.setContentsMargins(mm.left(), 14, mm.right(), mm.bottom())
        mode_lay.setSpacing(12)
        self._mode_group = QButtonGroup(self)
        self.mode_radios: dict[str, RadioButton] = {}
        self._mode_option_rows: list[QWidget] = []
        for key, label, tip in (
            ("fast", "Fast — structural (AudioTester-style)", TIPS["mode_fast"]),
            ("deep", "Deep — full decode (ffmpeg / flac -t)", TIPS["mode_deep"]),
            ("both", "Both — Fast + Deep (recommended)", TIPS["mode_both"]),
        ):
            rb = RadioButton("")
            self._mode_group.addButton(rb)
            self.mode_radios[key] = rb

            def _pick(v=key) -> None:
                self.mode_radios[v].setChecked(True)

            row = option_with_hint(rb, label, tip=tip, on_hint_click=_pick)
            self._mode_option_rows.append(row)
            mode_lay.addWidget(row)
        self.mode_radios["both"].setChecked(True)
        self._inner.addWidget(mode_sec)

        actions = Section(inner, "After detect")
        act_lay = actions.body.layout()
        am = act_lay.contentsMargins()
        act_lay.setContentsMargins(am.left(), 14, am.right(), am.bottom())
        act_lay.setSpacing(12)
        self.do_fix = CheckBox("")
        self.do_fix.setChecked(True)
        self._fix_option_row = option_with_hint(
            self.do_fix,
            "Fix flagged files (re-encode → replace original)",
            tip=TIPS["fix"],
            on_hint_click=lambda: self.do_fix.toggle(),
        )
        act_lay.addWidget(self._fix_option_row)
        self.do_quarantine = CheckBox("")
        self.do_quarantine.setChecked(False)
        self._quarantine_option_row = option_with_hint(
            self.do_quarantine,
            "Quarantine failed / suspect ({input}_CORRUPT)",
            tip=TIPS["quarantine"],
            on_hint_click=lambda: self.do_quarantine.toggle(),
        )
        act_lay.addWidget(self._quarantine_option_row)
        self._inner.addWidget(actions)

        note = Section(inner, "How it works")
        note_lay = note.body.layout()
        note_lay.setSpacing(10)
        for paragraph in (
            "Fast - checks MP3 structure. Deep - Fully decodes the audio file.",
            "Results show as OK, WARNING, or FAILED.",
            "Fix rebuilds WARNING and FAILED files and replaces the original.",
            "Unfixable files go to {input}_CORRUPT.",
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

    def _selected_mode(self) -> str:
        for key, rb in self.mode_radios.items():
            if rb.isChecked():
                return key
        return "both"

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
            "▶ Detect corruption",
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
            *self._mode_option_rows,
            self._fix_option_row,
            self._quarantine_option_row,
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
        do_fix = bool(self.do_fix.isChecked())
        do_quarantine = bool(self.do_quarantine.isChecked())
        mode = self._selected_mode()
        mp3val_override = str(self._settings.data.get("cor_mp3val_path") or "")

        if do_fix and not find_mp3val(mp3val_override, ensure=False):
            # Still allowed — first Detect run / install-deps fetches mp3val; ffmpeg fallback otherwise
            pass

        def action(on_log, on_progress, stop_event):
            hint = ensure_tuned(
                root,
                self._settings,
                workload="corruption",
                log=on_log,
            )
            paths = iter_audio_files(
                Path(root),
                recursive=recursive,
                stop_event=stop_event,
            )
            on_log(f"{len(paths):,} file(s) under {root}", "info")
            return run_corruption_detect(
                paths,
                mode=mode,  # type: ignore[arg-type]
                skip_existing=skip_existing,
                do_fix=do_fix,
                do_quarantine=do_quarantine,
                input_root=root,
                mp3val_path=mp3val_override,
                max_workers=hint.process_workers,
                on_log=on_log,
                on_progress=on_progress,
                stop_event=stop_event,
            )

        if self._worker is not None:
            return
        self.request_clear_log.emit()
        if self._window is not None:
            self._window.set_log_export_prefix("corruption")
        self.request_log.emit("=== Detect corruption ===", "info")
        self._set_busy(True, "Detecting corruption…")
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
            self.request_log.emit("Corruption detect stopping…", "warn")

    def _show_help(self) -> None:
        help_dialog(
            self,
            title="Corruption help",
            heading="File integrity verify + repair",
            intro=(
                "Fast structural scan plus deep decode verification. "
                "Optional Fix rebuilds damaged files."
            ),
            sections=[
                ("Modes", [
                    "Fast: quick MP3 structure check (no full decode).",
                    "Deep: full decode with ffmpeg (and flac -t for FLAC).",
                    "Both: runs Fast and Deep (recommended).",
                ]),
                ("Results", [
                    "OK — file looks clean.",
                    "WARNING — structure or decode issues that are still often playable.",
                    "FAILED — decode / integrity failure.",
                    "Skip already-OK when the checkbox is on; WARNING and FAILED are "
                    "always re-checked.",
                ]),
                ("Fix", [
                    "Rebuilds WARNING and FAILED files, then replaces the original.",
                    "MP3: re-encode audio + MP3val polish. Other formats: remux, then re-encode.",
                    "Unfixable FAILED files go to {input}_CORRUPT.",
                ]),
                ("Quarantine", [
                    "Optionally moves FAILED files into {InputFolderName}_CORRUPT "
                    "next to the input folder.",
                ]),
                ("Sources", [
                    '<a href="http://www.vuplayer.com/other.php">AudioTester</a> '
                    "— fast MP3 structural frame walk.",
                    "ffmpeg / flac -t — deep decode verification "
                    '(<a href="https://www.foobar2000.org/">foobar2000</a> Verify–like).',
                    '<a href="https://mp3val.sourceforge.net/">MP3val</a> '
                    "— MP3 rebuild polish (bundled via install-deps.bat).",
                ]),
            ],
        )

    def settings_snapshot(self) -> dict:
        return {
            "cor_input_dir": display_path(self.input_row.text()),
            "cor_include_subfolders": bool(self.include_subfolders.isChecked()),
            "cor_skip_existing": bool(self.skip_existing.isChecked()),
            "cor_mode": self._selected_mode(),
            "cor_fix": bool(self.do_fix.isChecked()),
            "cor_quarantine": bool(self.do_quarantine.isChecked()),
            "cor_mp3val_path": str(self._settings.data.get("cor_mp3val_path") or ""),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("cor_input_dir"):
                self.input_row.set_text(d["cor_input_dir"])
            elif d.get("output_dir"):
                self.input_row.set_text(d["output_dir"])
            self.include_subfolders.setChecked(
                bool(d.get("cor_include_subfolders", True))
            )
            self.skip_existing.setChecked(bool(d.get("cor_skip_existing", True)))
            self.do_fix.setChecked(bool(d.get("cor_fix", True)))
            self.do_quarantine.setChecked(bool(d.get("cor_quarantine", False)))
            mode = str(d.get("cor_mode") or "both")
            if mode in self.mode_radios:
                self.mode_radios[mode].setChecked(True)
            else:
                self.mode_radios["both"].setChecked(True)
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
            self.do_fix.toggled,
            self.do_quarantine.toggled,
        ):
            sig.connect(self._schedule_save)
        for rb in self.mode_radios.values():
            rb.toggled.connect(self._schedule_save)

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
    """Deprecated: use integrity_tab.register."""
    from .integrity_tab import register as register_integrity

    register_integrity(window, settings)

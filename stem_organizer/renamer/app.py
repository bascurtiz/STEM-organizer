"""Renamer app — port of track_renamer.gui.app.TrackRenamerApp.

A QWidget (not QMainWindow) that hosts the full Rename workflow: header
(presets / help), body (RulesPanel | Path + PreviewPanel), footer
(file count + AudioPlayerBar + Cancel / Rename).

Threading: daemon threads emit Signals (queued to the UI thread). Generation
guards prevent stale callbacks from overwriting newer state.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    PrimaryPushButton,
    PushButton,
)

from track_renamer.category_palette import (
    applied_category_colors,
    list_category_rules,
    normalize_rules_category_colors,
    sort_rule_category_keywords,
    sync_category_names_from_affix,
)
from track_renamer.engine.defaults import (
    make_default_rules,
    make_demo_tracks,
)
from track_renamer.engine.models import Rule, rule_from_dict, rule_to_dict
from track_renamer.engine.processor import compute_preview_row, prepare_rules
from track_renamer.folder_scanner import (
    apply_file_renames_detailed,
    move_files_to_prefix_folders,
    scan_folder,
)
from track_renamer.instrument_enrich import (
    classify_decision,
    enrich_tracks,
    rules_need_instrument_ml,
    terminate_tagger_process,
)

from .. import theme
from ..settings_store import SettingsStore, display_path
from ..widgets.dialogs import ask_yes_no, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from .audio_player_bar import AudioPlayerBar
from .help_dialog import show_rename_help_dialog
from .preview_panel import PreviewPanel
from .rules_panel import RulesPanel
from .theme import TIPS


PRESETS_DIR = Path.home() / ".track_renamer" / "presets"


def _ask_preset_name(parent: QWidget) -> str:
    """Themed Save Preset dialog (replaces unstyled QInputDialog)."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QDialog, QHBoxLayout as _QH, QVBoxLayout as _QV
    from qfluentwidgets import LineEdit

    from ..widgets.action_button import action_button

    t = theme.DARK
    host = parent.window() if parent is not None else parent
    dlg = QDialog(host)
    dlg.setWindowTitle("Save preset")
    dlg.setModal(True)
    dlg.setMinimumWidth(360)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = _QV(dlg)
    outer.setContentsMargins(12, 12, 12, 12)
    card = QFrame()
    card.setObjectName("HelpCard")
    r = theme.DIALOG_CORNER_RADIUS
    card.setStyleSheet(
        f"""
        QFrame#HelpCard {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = _QV(card)
    lay.setContentsMargins(18, 16, 18, 14)
    lay.setSpacing(10)

    lbl = BodyLabel("Preset name:")
    if hasattr(lbl, "setTextColor"):
        lbl.setTextColor(t["text"], t["text"])
    lay.addWidget(lbl)

    entry = LineEdit()
    entry.setPlaceholderText("My preset")
    entry.setClearButtonEnabled(False)
    entry.setToolTip("Name for this rule preset.")
    theme.style_line_edit(entry)
    lay.addWidget(entry)

    btns = _QH()
    btns.addStretch(1)
    cancel = action_button(
        "Cancel", on_click=dlg.reject, parent=card, tip=TIPS.get("cancel_dialog", "Dismiss without saving.")
    )
    ok = action_button(
        "OK", on_click=dlg.accept, accent=True, parent=card, tip=TIPS.get("ok_dialog", "Confirm and continue.")
    )
    ok.setMinimumWidth(72)
    btns.addWidget(cancel)
    btns.addWidget(ok)
    lay.addLayout(btns)
    outer.addWidget(card)

    entry.returnPressed.connect(dlg.accept)
    entry.setFocus()

    from ..widgets.dialogs import dim_behind

    dlg.adjustSize()
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(host):
        if dlg.exec() != QDialog.Accepted:
            return ""
    return entry.text().strip()


class TrackRenamerApp(QWidget):
    """Full rename workflow as an embeddable QWidget."""

    status_text = Signal(str)
    status_running = Signal()
    status_idle = Signal(str)
    log_line = Signal(str, str)
    # Worker → UI (auto queued across threads)
    _scan_progress = Signal(int, int)  # generation, count
    _scan_finished = Signal(object, object, object, int)  # path, tracks, error, generation
    _enrich_status = Signal(int, str)  # generation, message
    _enrich_progress = Signal(int, int, int)  # generation, done, total
    _enrich_result = Signal(int, object)  # generation, row dict
    _enrich_finished = Signal(int, float, int, object)  # generation, elapsed, total, error
    _enrich_failed = Signal(object)  # exc
    _rename_finished = Signal(object, object, object)  # success, errors, renamed_paths
    _rename_failed = Signal(object)  # exc
    _organize_finished = Signal(int, object, int, int, object)  # renamed, rename_errors, moved, skipped, move_errors

    def __init__(
        self, parent: Optional[QWidget] = None, settings: Optional[SettingsStore] = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RenameTab")
        self._settings = settings
        self._loading = False
        self.folder_path: Optional[Path] = None
        self.recursive = True
        self.tracks: List = list(make_demo_tracks())
        self.rules: List[Rule] = list(make_default_rules())
        self.demo_mode = True
        self._scan_generation = 0
        self._preview_generation = 0
        self._enrich_generation = 0
        self._enrich_cancel: threading.Event | None = None
        self._enrich_proc = None
        self._preview_stale = False
        self._busy = False
        self._destructive_busy = False
        self._applied_rules_fingerprint = self._rules_fingerprint(self.rules)

        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        self._scan_progress.connect(self._on_scan_progress)
        self._scan_finished.connect(self._on_scan_done)
        self._enrich_status.connect(self._on_enrich_status)
        self._enrich_progress.connect(self._on_enrich_progress)
        self._enrich_result.connect(self._on_enrich_result)
        self._enrich_finished.connect(self._on_enrich_done)
        self._enrich_failed.connect(self._on_enrich_error)
        self._rename_finished.connect(self._on_rename_done)
        self._rename_failed.connect(self._on_rename_error)
        self._organize_finished.connect(self._on_organize_done)

        self._build_ui()
        self._bind_autosave()
        self.load_settings()
        self._apply_preview()
        self._update_footer()

    # ----- build -----

    def _build_ui(self) -> None:
        self._docked = False
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(theme.ACTION_ROW_TOP_GAP)

        # Body holds left|side when undocked; side docks beside the host tab bar
        # in Rename mode so PATH aligns with the "Rename" tab label.
        self._body = QWidget()
        self._body_lay = QHBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 0, 0, 0)
        self._body_lay.setSpacing(0)

        # Left: description+? → presets + Clear/Apply → RULES
        self.left_panel = QWidget()
        left_lay = QVBoxLayout(self.left_panel)
        left_lay.setContentsMargins(theme.PAGE_CONTENT_INSET, 12, 0, 0)
        left_lay.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 8, 0)
        header.setSpacing(6)
        title = BodyLabel(
            "Scan a folder, build rules, and preview filename changes before applying"
        )
        title.setObjectName("HeaderDesc")
        title.setWordWrap(False)
        header.addWidget(title, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.help_icon = InfoIcon(self, on_click=lambda: show_rename_help_dialog(self))
        self.help_icon.setToolTip(TIPS["help"])
        header.addWidget(self.help_icon, 0, Qt.AlignVCenter)
        header.addStretch(1)
        left_lay.addLayout(header)

        left_lay.addSpacing(20)

        presets = QHBoxLayout()
        presets.setContentsMargins(0, 0, 8, 0)
        presets.setSpacing(6)
        self.preset_menu = ComboBox()
        self.preset_menu.setToolTip(TIPS["preset"])
        self.preset_menu.setMinimumWidth(180)
        self.preset_menu.setMaximumWidth(280)
        self.preset_menu.currentTextChanged.connect(self._on_preset_changed)
        presets.addWidget(self.preset_menu)
        self.save_preset_btn = PushButton("+")
        self.save_preset_btn.setFixedWidth(32)
        self.save_preset_btn.setToolTip(TIPS["save_preset"])
        self.save_preset_btn.clicked.connect(self._save_preset)
        presets.addWidget(self.save_preset_btn)
        self.delete_preset_btn = PushButton("−")
        self.delete_preset_btn.setFixedWidth(32)
        self.delete_preset_btn.setToolTip(TIPS["delete_preset"])
        self.delete_preset_btn.clicked.connect(self._delete_preset)
        presets.addWidget(self.delete_preset_btn)
        presets.addStretch(1)

        self.rules_panel = RulesPanel(
            self.left_panel,
            on_change=self._on_rules_changed,
            on_apply=self._apply_preview,
        )
        self.rules_panel.set_rules(self.rules)
        # Same row as template selector — no vertical gap to Clear / Apply
        presets.addWidget(self.rules_panel.clear_btn)
        presets.addWidget(self.rules_panel.apply_btn)
        left_lay.addLayout(presets)

        left_lay.addSpacing(10)
        left_lay.addWidget(self.rules_panel, stretch=1)

        # Right: PATH + PREVIEW (docked beside tab bar in Rename mode).
        # Top pad centers the PATH title in the tab-bar band (same height as "Rename").
        # Right inset must match footer_bar so PATH card ends with Rename button.
        self._rename_right_inset = theme.PAGE_CONTENT_INSET
        self.side_panel = QWidget()
        self.side_panel.setObjectName("RenameSidePanel")
        right_lay = QVBoxLayout(self.side_panel)
        _path_top = max(0, (theme.ACTION_BTN_HEIGHT - theme.SECTION_TITLE_PX) // 2 - 3)
        # Left 0 — sit flush against the rules scrollbar (no center seam/line)
        right_lay.setContentsMargins(0, _path_top, self._rename_right_inset, 0)
        right_lay.setSpacing(theme.SECTION_GAP)

        paths = Section(self.side_panel, "Path")
        paths.body.layout().setSpacing(12)
        self.folder_row = PathRow(
            paths.body,
            "Input folder",
            tip_text=TIPS["open_folder"],
            label_width=80,
            caption="Open folder",
        )
        self.folder_row.browse_btn.clicked.disconnect()
        self.folder_row.browse_btn.clicked.connect(self._open_folder)
        self.recursive_chk = CheckBox("Include subfolders")
        self.recursive_chk.setChecked(True)
        self.recursive_chk.setToolTip(TIPS["recursive"])
        self.recursive_chk.toggled.connect(self._on_recursive_toggle)
        paths.body.layout().addWidget(self.recursive_chk)
        # Recover 3px after PATH/card shift so card bottom still meets Apply.
        paths.body.layout().addSpacing(3)
        # Indent PATH to match PREVIEW left; right edge shares footer inset (no extra pad).
        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(0)
        path_row.addSpacing(8)
        path_row.addWidget(paths, stretch=1)
        right_lay.addLayout(path_row)

        # Pads PREVIEW down so its title lines up with RULES (measured after layout)
        self._preview_align_pad = QWidget()
        self._preview_align_pad.setFixedHeight(0)
        right_lay.addWidget(self._preview_align_pad)

        self.preview_panel = PreviewPanel(self.side_panel)
        # Drop PreviewPanel's right pad so table aligns with PATH / Rename button
        if self.preview_panel.layout() is not None:
            m = self.preview_panel.layout().contentsMargins()
            self.preview_panel.layout().setContentsMargins(m.left(), m.top(), 0, m.bottom())
        self.preview_panel.on_change = self._update_footer
        self.preview_panel.on_active = self._on_active_preview
        self.preview_panel.on_play_pause = self._toggle_audio_preview
        self.preview_panel.on_seek = self._seek_audio_preview
        self.preview_panel.on_override_rename = self._rename_from_prefix_override
        right_lay.addWidget(self.preview_panel, stretch=1)

        self._body_lay.addWidget(self.left_panel, stretch=1)
        self._body_lay.addWidget(self.side_panel, stretch=1)
        self._root.addWidget(self._body, stretch=1)
        QTimer.singleShot(0, self._align_preview_header_to_rules)

        # Footer — own widget so it can span under tabs|side when docked
        self.footer_bar = QWidget()
        footer = QHBoxLayout(self.footer_bar)
        # Top gap above wavebar; right inset matches PATH card / side panel
        footer.setContentsMargins(
            theme.PAGE_CONTENT_INSET,
            theme.RENAME_PLAYER_TOP_GAP,
            self._rename_right_inset,
            0,
        )
        self.file_count_label = CaptionLabel("0 files")
        self.file_count_label.setStyleSheet(f"color: {theme.DARK['text_dim']};")
        footer.addWidget(self.file_count_label)
        self.audio_player = AudioPlayerBar()
        footer.addWidget(self.audio_player, stretch=1)
        self.cancel_btn = PushButton("Cancel")
        self.cancel_btn.setToolTip(TIPS["cancel"])
        self.cancel_btn.clicked.connect(self._cancel)
        footer.addWidget(self.cancel_btn)
        self.rename_btn = PrimaryPushButton("Rename")
        self.rename_btn.setObjectName("Accent")
        self.rename_btn.setToolTip(TIPS["rename"])
        self.rename_btn.clicked.connect(self._apply_renames)
        footer.addWidget(self.rename_btn)
        self._root.addWidget(self.footer_bar)

        self._refresh_preset_menu()

    def dock_to_host(self, side_host: QWidget, footer_host: QWidget) -> None:
        """Place PATH+PREVIEW beside the tab bar; footer under the full row."""
        if self._docked:
            return
        self._docked = True
        self._body_lay.removeWidget(self.side_panel)
        self._root.removeWidget(self.footer_bar)
        side_host.layout().addWidget(self.side_panel)
        footer_host.layout().addWidget(self.footer_bar)
        self.side_panel.show()
        self.footer_bar.show()
        QTimer.singleShot(0, self._align_preview_header_to_rules)

    def undock_from_host(self) -> None:
        """Restore PATH+PREVIEW and footer inside the Rename tab page."""
        if not self._docked:
            return
        self._docked = False
        self.side_panel.setParent(self._body)
        self.footer_bar.setParent(self)
        self._body_lay.addWidget(self.side_panel, stretch=1)
        self._root.addWidget(self.footer_bar)
        self.side_panel.show()
        self.footer_bar.show()
        QTimer.singleShot(0, self._align_preview_header_to_rules)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._align_preview_header_to_rules)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._align_preview_header_to_rules)

    def _align_preview_header_to_rules(self) -> None:
        """Push PREVIEW down so its section title shares Y with RULES."""
        rt = getattr(self.rules_panel, "section_title", None)
        pt = getattr(self.preview_panel, "section_title", None)
        pad = getattr(self, "_preview_align_pad", None)
        if rt is None or pt is None or pad is None:
            return
        if not rt.isVisible() or not pt.isVisible():
            return
        win = self.window()
        origin = win if win is not None else self
        ry = rt.mapTo(origin, QPoint(0, 0)).y()
        py = pt.mapTo(origin, QPoint(0, 0)).y()
        # Adjust existing pad by the remaining gap (avoids reset flicker on resize)
        new_h = max(0, pad.height() + (ry - py))
        if new_h != pad.height():
            pad.setFixedHeight(new_h)

    # ----- rules / preview -----

    def _rules_fingerprint(self, rules: List[Rule]) -> str:
        return json.dumps([rule_to_dict(r) for r in rules], sort_keys=True)

    def _on_rules_changed(self) -> None:
        self.rules = self.rules_panel.get_rules()
        new_fp = self._rules_fingerprint(self.rules)
        if new_fp != self._applied_rules_fingerprint:
            if not self._preview_stale:
                self._preview_stale = True
                self.preview_panel.cancel_preview_work()
                self.rules_panel.set_apply_pending(True)
                self.preview_panel.set_preview_pending(True)
        self._update_footer()
        self._schedule_save()

    def _apply_preview(self) -> None:
        self.rules = self.rules_panel.get_rules()
        # Normalize categories
        try:
            changed1 = sort_rule_category_keywords(self.rules)
            changed2 = sync_category_names_from_affix(self.rules)
            if changed1 or changed2:
                self.rules_panel.set_rules(self.rules)
        except Exception:
            pass
        self._applied_rules_fingerprint = self._rules_fingerprint(self.rules)
        self._preview_stale = False
        self.rules_panel.set_apply_pending(False)
        self.preview_panel.set_preview_pending(False)
        self._refresh_preview()
        self._update_footer()

    def _refresh_preview(self) -> None:
        self._preview_generation += 1
        try:
            colors = applied_category_colors(self.rules)
        except Exception:
            colors = {}
        self.audio_player.set_category_colors(colors)
        self.preview_panel.model.set_category_colors(colors)
        try:
            self.preview_panel.set_category_options(list_category_rules(self.rules))
        except Exception:
            self.preview_panel.set_category_options([])
        root_label = self.folder_path.name if self.folder_path else "ROOT"
        self.preview_panel.begin_viewport_lazy(self.tracks, self.rules, root_label)
        self._update_footer()

    # ----- scan -----

    def _open_folder(self) -> None:
        if self._busy:
            return
        start = str(self.folder_path) if self.folder_path else self.folder_row.text()
        path = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if path:
            self._scan_folder(Path(path))

    def _set_source_path(self, path: Path | None) -> None:
        """Keep the PATH input in sync with the scanned folder."""
        if path is None:
            self.folder_row.set_text("")
            return
        self.folder_row.set_text(display_path(str(path)))

    def _scan_folder(self, path: Path) -> None:
        self._scan_generation += 1
        generation = self._scan_generation
        self.folder_path = path
        self._set_source_path(path)
        self.preview_panel.clear_active()
        self.audio_player.reset()
        self._set_busy(True, "Scanning…")

        def progress(count: int) -> None:
            if generation != self._scan_generation:
                return
            self._scan_progress.emit(generation, count)

        def work():
            try:
                tracks = scan_folder(path, recursive=self.recursive, progress=progress)
                error = None
            except Exception as exc:
                tracks = []
                error = exc
            self._scan_finished.emit(path, tracks, error, generation)

        threading.Thread(target=work, daemon=True).start()

    def _on_scan_progress(self, generation: int, count: int) -> None:
        if generation != self._scan_generation or not self._busy:
            return
        self._set_busy(True, f"Scanning… {count:,} files found")

    def _on_scan_done(self, path, tracks, error, generation: int | None = None) -> None:
        if generation is not None and generation != self._scan_generation:
            return
        if error is not None:
            show_info(self, "Rename Files", f"Scan failed:\n{error}")
            self._set_busy(False, "Idle")
            return
        if not tracks:
            show_info(self, "Rename Files", "All files are moved from this folder.")
            self.preview_panel.set_rows([])
            self._set_busy(False, "Idle")
            return
        self.tracks = list(tracks)
        self.folder_path = Path(path) if not isinstance(path, Path) else path
        self.demo_mode = False
        self._set_source_path(self.folder_path)
        try:
            from track_renamer.instrument_enrich import apply_cached_labels

            apply_cached_labels(self.tracks)
        except Exception:
            pass
        try:
            self._apply_preview()
        finally:
            # Always clear busy — even if preview setup raises — so elapsed stops.
            self._set_busy(False, "Idle")

    def _on_recursive_toggle(self, checked: bool) -> None:
        self.recursive = bool(checked)
        self._schedule_save()
        if self.folder_path is not None and not self._busy:
            self._scan_folder(self.folder_path)

    # ----- rename -----

    def _selected_need_instrument_ml(self) -> bool:
        """True when Auto-detect/Combo is on and selected tracks lack ML labels."""
        if not rules_need_instrument_ml(self.rules):
            return False
        for track in self.tracks:
            if track.selected and not str(getattr(track, "instrument", "") or "").strip():
                return True
        return False

    def _apply_renames(self) -> None:
        if self._busy:
            return
        if self.demo_mode:
            show_info(self, "Rename Files", "Browse & select a folder first.")
            return
        # Always re-read panel — Auto-detect toggle must be current.
        self.rules = self.rules_panel.get_rules()
        if rules_need_instrument_ml(self.rules):
            self._enrich_then_rename()
            return
        renames = self.preview_panel.selected_renames()
        if not renames:
            show_info(self, "Rename Files", "No files selected for rename.")
            return
        n = len(renames)
        if not ask_yes_no(self, "Rename Files", f"Rename {n} file(s)?"):
            return
        self._start_rename_job(renames)

    def _rename_from_prefix_override(self, renames: dict) -> None:
        """Immediate disk rename after preview Change to: (no second confirm)."""
        if self._busy or not renames:
            return
        if self.demo_mode:
            show_info(self, "Rename Files", "Browse & select a folder first.")
            return
        self._start_rename_job(renames)

    def _enrich_then_rename(self) -> None:
        selected = [t for t in self.tracks if t.selected]
        if not selected:
            show_info(self, "Rename Files", "No selected files.")
            return

        # Drop preview locks before the tagger opens the same files.
        self.preview_panel.clear_active()
        self.audio_player.release_for_file_ops(settle_s=0.05)

        self._enrich_generation += 1
        generation = self._enrich_generation
        self._enrich_cancel = threading.Event()
        self._enrich_proc = None
        cancel = self._enrich_cancel
        self.preview_panel.cancel_preview_work()
        total = len(selected)
        self.preview_panel.begin_analyze_log(total)
        self._set_busy(True, f"Analyzing instruments (0/{total:,})…")

        def on_status(msg: str) -> None:
            self._enrich_status.emit(generation, msg)

        def on_progress(done: int, total_count: int) -> None:
            self._enrich_progress.emit(generation, done, total_count)

        def on_result(row: dict) -> None:
            self._enrich_result.emit(generation, row)

        def on_process(proc) -> None:
            self._enrich_proc = proc

        def work():
            start = time.monotonic()
            try:
                _classified, error = enrich_tracks(
                    selected,
                    status=on_status,
                    on_progress=on_progress,
                    on_result=on_result,
                    cancel=cancel,
                    on_process=on_process,
                )
            except Exception as exc:
                self._enrich_failed.emit(exc)
                return
            finally:
                self._enrich_proc = None
            self._enrich_finished.emit(generation, time.monotonic() - start, total, error)

        threading.Thread(target=work, daemon=True).start()

    def _on_enrich_status(self, generation: int, message: str) -> None:
        if generation != self._enrich_generation:
            return
        self.preview_panel.append_analyze_status(message)

    def _on_enrich_progress(self, generation: int, done: int, total: int) -> None:
        if generation != self._enrich_generation:
            return
        self._set_busy(True, f"Analyzing instruments ({done:,}/{total:,})…")

    def _on_enrich_result(self, generation: int, row: object) -> None:
        if generation != self._enrich_generation or not isinstance(row, dict):
            return
        name = str(row.get("name") or "")
        label = str(row.get("label") or "")
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            second = float(row.get("second_score") or 0.0)
        except (TypeError, ValueError):
            second = 0.0
        err = str(row.get("error") or "")
        try:
            done = int(row.get("index") or 0)
        except (TypeError, ValueError):
            done = 0
        try:
            total = int(row.get("total") or 0)
        except (TypeError, ValueError):
            total = 0
        if err:
            self.preview_panel.append_analyze_log(
                filename=name,
                action="error",
                reason=err[:120],
                label=label,
                done=done,
                total=total,
            )
            return
        action, category = classify_decision(label, score, second_score=second)
        self.preview_panel.append_analyze_log(
            filename=name,
            action=action,
            category=category,
            score=score,
            label=label,
            done=done,
            total=total,
        )

    def _on_enrich_error(self, exc: Exception) -> None:
        # Cancel kills the tagger; ignore late failures from that shutdown.
        if self._enrich_cancel is not None and self._enrich_cancel.is_set():
            return
        self.preview_panel.end_analyze_log()
        show_info(
            self.window() or self,
            "Rename Files",
            f"Auto-detect failed:\n{exc}",
        )
        self._set_busy(False, "Idle")

    def _on_enrich_done(self, generation: int, elapsed: float, total: int, error) -> None:
        if generation != self._enrich_generation:
            return
        if error:
            self._set_busy(False, "Idle")
            self.preview_panel.end_analyze_log()
            show_info(
                self.window() or self,
                "Rename Files",
                f"Auto-detect had errors:\n{error}",
            )
            self._refresh_preview()
            return
        self.preview_panel.append_analyze_summary(elapsed_sec=elapsed, total=total)
        renames = self._compute_selected_renames()
        # Refresh table so Keyword shows <audio-determined> while ML fields are live.
        self._refresh_preview()
        self._set_busy(False, "Idle")
        # Defer so busy UI settles before the modal (same as Classify post-RMS).
        QTimer.singleShot(
            0, lambda g=generation, r=renames: self._offer_after_enrich(g, r)
        )

    def _offer_after_enrich(self, generation: int, renames: dict) -> None:
        """Post-Analyze confirm via opaque-card show_info / ask_yes_no (not Fluent MessageBox)."""
        if generation != self._enrich_generation:
            return
        parent = self.window() or self
        if not renames:
            show_info(
                parent,
                "Rename Files",
                "No selected files will change after instrument analysis.",
            )
            self.preview_panel.end_analyze_log()
            self._refresh_preview()
            return
        n = len(renames)
        if not ask_yes_no(parent, "Rename Files", f"Rename {n} file(s)?"):
            self.preview_panel.end_analyze_log()
            self._refresh_preview()
            return
        self.preview_panel.end_analyze_log()
        # Defer preview refresh until after rename — refreshing first can
        # re-select rows and reload the audio player onto files mid-rename.
        self._start_rename_job(renames)

    def _compute_selected_renames(self) -> dict:
        try:
            prepared = prepare_rules(self.rules)
        except Exception:
            prepared = self.rules
        renames = {}
        for idx, track in enumerate(self.tracks):
            if not track.selected:
                continue
            try:
                row = compute_preview_row(track, prepared, index=idx + 1)
            except Exception:
                continue
            if row.changed:
                renames[track.id] = row.new_name
        return renames

    def _start_rename_job(self, renames: dict) -> None:
        self.preview_panel.clear_active()
        # Drop ffplay/ffmpeg locks before staging renames (WinError 32 otherwise).
        # Short settle on UI thread; worker does a longer post-Analyze settle.
        self.audio_player.release_for_file_ops(settle_s=0.05)
        self._destructive_busy = True
        self._set_busy(True, "Renaming files…")

        def work():
            try:
                # Tagger / AV may still hold shares briefly after Analyze.
                self.audio_player.service.release_for_file_ops(settle_s=0.4)
                success, errors, renamed_paths = apply_file_renames_detailed(renames)
            except Exception as exc:
                self._rename_failed.emit(exc)
                return
            self._rename_finished.emit(success, errors, renamed_paths)

        threading.Thread(target=work, daemon=True).start()

    def _on_rename_error(self, exc: Exception) -> None:
        self._destructive_busy = False
        show_info(self, "Rename Files", f"Rename failed:\n{exc}")
        self._set_busy(False, "Idle")

    def _on_rename_done(self, success: int, errors: List[str], renamed_paths: List[Path]) -> None:
        self._destructive_busy = False
        if not renamed_paths or self.folder_path is None:
            self._show_file_operation_result(
                "Rename",
                f"{success} file(s) renamed.",
                errors,
            )
            self._finish_file_operation()
            return
        if not ask_yes_no(
            self,
            "Organize by prefix",
            "Move renamed files into category prefix folders (BASS, DRUMS, …)?",
        ):
            self._show_file_operation_result(
                "Rename",
                f"{success} file(s) renamed.",
                errors,
            )
            self._finish_file_operation()
            return
        dest = QFileDialog.getExistingDirectory(self, "Destination for organized files")
        if not dest:
            self._show_file_operation_result(
                "Rename",
                f"{success} file(s) renamed.",
                errors,
            )
            self._finish_file_operation()
            return

        # Ensure preview is not still pointing at paths about to move.
        self.audio_player.release_for_file_ops()

        def work():
            try:
                moved, skipped, move_errors = move_files_to_prefix_folders(renamed_paths, Path(dest))
            except Exception as exc:
                self._rename_failed.emit(exc)
                return
            self._organize_finished.emit(
                success, errors, moved, len(renamed_paths) - moved, move_errors
            )

        threading.Thread(target=work, daemon=True).start()

    def _on_organize_done(self, renamed: int, rename_errors: List[str], moved: int, skipped: int, move_errors: List[str]) -> None:
        summary = f"{renamed} renamed, {moved} moved into category folders."
        all_errors = list(rename_errors) + list(move_errors)
        self._show_file_operation_result("Rename + Organize", summary, all_errors)
        self._finish_file_operation()

    def _show_file_operation_result(self, title: str, summary: str, errors: List[str]) -> None:
        body = summary
        if errors:
            shown = 20
            tail = "\n".join(errors[:shown])
            more = "" if len(errors) <= shown else f"\n…and {len(errors) - shown} more."
            body += f"\n\nErrors:\n{tail}{more}"
        try:
            from done_sound import play_done_sound

            play_done_sound()
        except Exception:
            pass
        show_info(self, title, body)

    def _finish_file_operation(self) -> None:
        # Always rescan after rename/organize (match CTk). The busy flag is
        # still True here from the job — do not skip the refresh.
        if self.folder_path is not None:
            self._scan_folder(self.folder_path)
        else:
            self._set_busy(False, "Idle")

    # ----- presets -----

    def _preset_names(self) -> List[str]:
        # "Default" is a builtin (make_default_rules) — never also pull it from
        # disk, otherwise a stale Default.json produces a duplicate dropdown entry.
        names = ["Default"]
        try:
            for p in sorted(PRESETS_DIR.glob("*.json")):
                stem = p.stem
                if stem == "Default":
                    continue
                names.append(stem)
        except OSError:
            pass
        return names

    def _refresh_preset_menu(self) -> None:
        self.preset_menu.blockSignals(True)
        cur = self.preset_menu.currentText()
        self.preset_menu.clear()
        for name in self._preset_names():
            self.preset_menu.addItem(name)
        if cur and cur in self._preset_names():
            self.preset_menu.setCurrentText(cur)
        else:
            self.preset_menu.setCurrentText("Default")
        self.preset_menu.blockSignals(False)

    def _on_preset_changed(self, name: str) -> None:
        if not name:
            return
        self._load_preset(name)
        self._schedule_save()

    def _load_preset(self, name: str) -> None:
        if name == "Default":
            self.rules = list(make_default_rules())
        else:
            path = PRESETS_DIR / f"{name}.json"
            if not path.is_file():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.rules = [rule_from_dict(r) for r in data.get("rules", [])]
            except Exception:
                show_info(self, "Rename Files", f"Could not load preset:\n{path}")
                return
        normalize_rules_category_colors(self.rules)
        self.rules_panel.set_rules(self.rules)
        self._apply_preview()

    def _save_preset(self) -> None:
        name = _ask_preset_name(self)
        if not name:
            return
        if name == "Default":
            # "Default" is a builtin; refuse to shadow it with a file (would
            # otherwise be ignored on load AND duplicate the dropdown entry).
            show_info(
                self, "Rename Files",
                "The name \"Default\" is reserved for the built-in preset.\n"
                "Please choose a different name.",
            )
            return
        path = PRESETS_DIR / f"{name}.json"
        try:
            path.write_text(
                json.dumps({"rules": [rule_to_dict(r) for r in self.rules]}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            show_info(self, "Rename Files", f"Could not save preset:\n{exc}")
            return
        self._refresh_preset_menu()
        self.preset_menu.setCurrentText(name)
        self._schedule_save()

    def _delete_preset(self) -> None:
        name = self.preset_menu.currentText()
        if not name or name == "Default":
            return
        if not ask_yes_no(self, "Delete preset", f"Delete preset '{name}'?"):
            return
        path = PRESETS_DIR / f"{name}.json"
        try:
            path.unlink()
        except OSError as exc:
            show_info(self, "Rename Files", f"Could not delete preset:\n{exc}")
            return
        self._refresh_preset_menu()
        self._load_preset("Default")
        self._schedule_save()

    # ----- settings -----

    def settings_snapshot(self) -> dict:
        self.rules = self.rules_panel.get_rules()
        return {
            "rename_folder": display_path(self.folder_row.text()),
            "rename_recursive": bool(self.recursive_chk.isChecked()),
            "rename_preset": self.preset_menu.currentText() or "Default",
            "rename_rules": [rule_to_dict(r) for r in self.rules],
            "rename_only_changed": bool(self.preview_panel.only_changed_btn.isChecked()),
        }

    def load_settings(self) -> None:
        if self._settings is None:
            return
        self._loading = True
        try:
            d = self._settings.data
            self.recursive = bool(d.get("rename_recursive", True))
            self.recursive_chk.setChecked(self.recursive)

            folder = str(d.get("rename_folder") or "").strip()
            if folder:
                self.folder_row.set_text(display_path(folder))
            elif d.get("output_dir"):
                self.folder_row.set_text(d["output_dir"])

            rules_data = d.get("rename_rules")
            loaded_rules = False
            if isinstance(rules_data, list) and rules_data:
                try:
                    self.rules = [rule_from_dict(r) for r in rules_data]
                    normalize_rules_category_colors(self.rules)
                    self.rules_panel.set_rules(self.rules)
                    loaded_rules = True
                except Exception:
                    loaded_rules = False

            preset = str(d.get("rename_preset") or "Default")
            names = self._preset_names()
            if preset not in names:
                preset = "Default"
            self.preset_menu.blockSignals(True)
            self.preset_menu.setCurrentText(preset)
            self.preset_menu.blockSignals(False)
            if not loaded_rules:
                self._load_preset(preset)

            only = bool(d.get("rename_only_changed", False))
            self.preview_panel.only_changed_btn.blockSignals(True)
            self.preview_panel.only_changed_btn.setChecked(only)
            self.preview_panel.only_changed_btn.blockSignals(False)
            self.preview_panel._on_only_changed_toggled(only)
        finally:
            self._loading = False

    def _bind_autosave(self) -> None:
        if self._settings is None:
            return
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        self.folder_row.entry.textChanged.connect(self._schedule_save)
        self.preview_panel.only_changed_btn.toggled.connect(self._schedule_save)

    def _schedule_save(self, *_) -> None:
        if self._loading or self._settings is None:
            return
        if not hasattr(self, "_autosave_timer"):
            return
        self._autosave_timer.start()

    def _flush_settings(self) -> None:
        if self._loading or self._settings is None:
            return
        self._settings.merge(self.settings_snapshot())
        self._settings.flush()

    def flush_settings(self) -> None:
        """Immediate persist (used on app close)."""
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()

    # ----- audio -----

    def _on_active_preview(self, track, row) -> None:
        # Avoid reopening files while rename/organize holds the disk.
        if self._destructive_busy:
            return
        self.audio_player.set_active(track, row)

    def _toggle_audio_preview(self) -> None:
        if self._destructive_busy:
            return
        self.audio_player.toggle_playback()

    def _seek_audio_preview(self, seconds: float) -> None:
        if self._destructive_busy:
            return
        self.audio_player.seek(seconds)

    # ----- busy / status -----

    def _set_busy(self, busy: bool, message: str = "") -> None:
        was_busy = self._busy
        self._busy = busy
        for w in (
            self.folder_row,
            self.recursive_chk,
            self.preset_menu,
            self.save_preset_btn,
            self.delete_preset_btn,
            self.rules_panel,
            self.rules_panel.clear_btn,
            self.rules_panel.apply_btn,
        ):
            try:
                w.setEnabled(not busy)
            except Exception:
                pass
        if busy:
            # Keep folder path visible in PATH; status bar carries progress text.
            if not was_busy:
                self.status_running.emit()
            if message:
                self.status_text.emit(message)
        else:
            self._set_source_path(self.folder_path)
            if was_busy:
                self.status_idle.emit("Idle")
        self._update_footer()

    def _update_footer(self) -> None:
        total = len(self.tracks)
        selected_n = sum(1 for t in self.tracks if t.selected)
        rename_count = self.preview_panel.rename_count()
        complete = self.preview_panel.lazy_compute_complete()
        ml_on_rename = (
            rules_need_instrument_ml(self.rules)
            and not self.demo_mode
            and selected_n > 0
        )
        need_analyze = ml_on_rename and self._selected_need_instrument_ml()

        self.file_count_label.setText(f"{total:,} files")
        if complete:
            if need_analyze:
                self.rename_btn.setText(f"Analyze ({selected_n:,})")
                self.rename_btn.setToolTip(TIPS.get("analyze", TIPS["rename"]))
            else:
                self.rename_btn.setText(f"Rename {rename_count:,}")
                self.rename_btn.setToolTip(TIPS["rename"])
        else:
            done, lazy_total = self.preview_panel.lazy_compute_progress()
            self.rename_btn.setText(f"Preparing {done:,}/{lazy_total:,}")
            self.rename_btn.setToolTip(TIPS["rename"])

        can_rename = (
            complete
            and (rename_count > 0 or ml_on_rename)
            and not self._busy
            and not self._preview_stale
        )
        self.rename_btn.setEnabled(can_rename)

    def _cancel(self) -> None:
        self.preview_panel.clear_active()
        try:
            self.audio_player.reset()
        except Exception:
            pass

        # Mid rename/organize: selection clear only (disk op already running).
        if self._destructive_busy:
            return

        if not self._busy:
            return

        # Abandon Analyze (and in-flight scan): invalidate generations so
        # finish callbacks no-op, kill the tagger tree, restore idle UI.
        self._enrich_generation += 1
        self._scan_generation += 1
        if self._enrich_cancel is not None:
            self._enrich_cancel.set()
        if self._enrich_proc is not None:
            terminate_tagger_process(self._enrich_proc)
            self._enrich_proc = None
        self.preview_panel.end_analyze_log()
        self.preview_panel.cancel_preview_work()
        self._set_busy(False, "Idle")
        self._refresh_preview()

    # ----- host tab lifecycle (used by RenameTab) -----

    def on_tab_shown(self) -> None:
        if self._preview_stale:
            self._apply_preview()

    def on_tab_hidden(self) -> None:
        # Stop audio and clear source so file locks do not linger across tabs.
        try:
            self.audio_player.reset()
        except Exception:
            pass

    def shutdown(self) -> None:
        self.preview_panel.shutdown()
        self.audio_player.shutdown()

    @property
    def destructive_busy(self) -> bool:
        return self._destructive_busy

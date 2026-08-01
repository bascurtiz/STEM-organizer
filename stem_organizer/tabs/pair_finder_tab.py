"""Match & Align tab — port of pair_finder_panel.PairFinderPanel.

Two sub-tabs:
  Match  — Folders card (Acapella / Instrumental / Pairs output + reference radio
           + include subfolders), Matching card (strictness slider + filename
           fallback), Ignore card (parens / brackets / spaces + custom keywords).
  Align  — Stems root + Export list + Originals inbox + With/Without original +
           Analysis seconds + backup + skip-existing.

Action buttons: Find pairs / Organize / Play (Match), Export list / Distribute /
Sort / Align (Align).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    CompactDoubleSpinBox,
    LineEdit,
    RadioButton,
    ScrollArea,
    TabWidget,
)

from pair_matcher import (
    AUDIO_EXTS as PAIR_AUDIO_EXTS,
    IgnoreRules,
    find_pairs,
    organize_matched_folder,
    strictness_to_threshold,
)
from stem_align import (
    align_all_songs,
    default_with_original_dir,
    default_without_original_dir,
    distribute_originals,
    export_song_list,
    resolve_export_list_path,
    sort_folders_by_original,
)

from .. import theme
from ..run_summary import emit_run_summary
from ..settings_store import SettingsStore, app_dir, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..widgets.slider_field import SliderField
from ..workers.pair_worker import PairWorker


PANEL_TITLE = "Match & Align"
DEFAULT_PAIRS_OUTPUT = r"C:\Pairs"

TIPS = {
    "acapella": "Folder containing acapella FLAC or MP3 files.",
    "instrumental": "Folder containing instrumental FLAC or MP3 files.",
    "pairs_output": (
        "Matched pairs are moved here (flat). Organize folder then groups them "
        "into Artist - Title subfolders inside this same folder."
    ),
    "reference": (
        "Matching starts from the reference folder; each file looks for a partner "
        "in the other folder."
    ),
    "include_subfolders": "Scan every subfolder of Acapella / Instrumental recursively.",
    "strictness": (
        "Loose accepts minor tag differences (extra artists, & vs and, spacing).\n"
        "Strict requires near-exact tags."
    ),
    "filename_fallback": (
        'Parse artist/title from the filename only (e.g. Artist - Title (Acapella).mp3), '
        "even when metadata tags exist. Off = use tags only."
    ),
    "ignore_parentheses": "Strip text inside (parentheses) before comparing tags.",
    "ignore_square": "Strip text inside [square brackets] before comparing tags.",
    "ignore_spaces": "Collapse repeated spaces and trim edges before comparing.",
    "ignore_custom": "Remove these words or phrases from tags (case-insensitive) before comparing.",
    "add_keyword": "Add a custom keyword to ignore.",
    "remove_keyword": "Remove this custom keyword row.",
    "align_stems_root": "Root folder with one subfolder per song (each contains acapella + instrumental).",
    "align_inbox": "Folder where you drop downloaded originals before distributing.",
    "align_export": "Text file listing subfolder names to look up on Soundiiz / YouTube / Deezer.",
    "align_backup": "Copy the original stems into _backup_before_align before overwriting.",
    "align_skip_existing": "Skip songs whose output already exists (resume mode).",
    "align_analysis": (
        "How many seconds of audio to analyze for offset detection. "
        "Longer = more accurate but slower."
    ),
    "align_with_original": "Folders that received an original song (auto-filled as stems root / with_original).",
    "align_without_original": "Folders still missing an original (auto-filled as stems root / without_original).",
    "find_pairs": "Match instrumentals to vocals by tags (or filename) and move pairs into Output.",
    "organize": "Group matched files inside Output into Artist - Title song folders.",
    "export_list_btn": "Write song-folder names from Stems root into the export list file.",
    "distribute_btn": "Copy originals from the inbox into matching song folders under Stems root.",
    "sort_folders_btn": "Move song folders into with_original / without_original based on whether an original exists.",
    "align_stems_btn": "Align acapella and instrumental stems to the original song timeline.",
    "play_stems": (
        "Open the stem preview player on the current library "
        "(Match: Output; Align: With original)."
    ),
}

TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


class PairFinderTab(QWidget):
    """Match + Align sub-tabs and shared action bar."""

    request_status_running = Signal()
    request_status_idle = Signal(str)
    request_progress = Signal(float, object, int, int, str)
    request_status_text = Signal(str)
    request_log = Signal(str, str)
    request_clear_log = Signal()
    request_open_player = Signal()

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._worker: Optional[PairWorker] = None
        self._busy = False
        self._busy_panel: Optional[str] = None  # "match" | "align"
        self._loading = False
        self._custom_keyword_edits: list[LineEdit] = []

        # State — port of the tk var list.
        self.acapella_dir = ""
        self.instrumental_dir = ""
        self.pairs_output_dir = ""
        self.reference_side = "acapella"
        self.strictness = 75.0
        self.use_filename_fallback = True
        self.include_subfolders = True
        self.ignore_parentheses = True
        self.ignore_square_brackets = True
        self.ignore_extra_spaces = True
        self.custom_keywords: list[str] = []
        self.align_stems_root = ""
        self.align_originals_inbox = ""
        self.align_export_file = str(app_dir() / "songs_to_download.txt")
        self.align_backup = True
        self.align_skip_existing = True
        self.align_analysis_sec = 30
        self.align_with_original_dir = ""
        self.align_without_original_dir = ""

        self._build_ui()
        self.load_settings()
        self._bind_autosave()

    # ----------------- UI build -----------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._tabview = TabWidget()
        theme.configure_tab_widget(self._tabview)
        self._tabview.addTab(self._build_match_tab(), "Match")
        self._tabview.addTab(self._build_align_tab(), "Align")
        theme.configure_tab_widget(self._tabview)
        theme.inset_tab_bar(self._tabview)
        self._tabview.currentChanged.connect(self._on_subtab_changed)
        layout.addWidget(self._tabview)

    def _build_match_tab(self) -> QWidget:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(ScrollArea.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        v.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel("Match instrumental/vocal files by artist/title tags, then organize pairs into song folders")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=self._show_match_help))
        header.addStretch(1)
        v.addLayout(header)

        # Folders — Instrumental first, then Vocal (formerly Acapella)
        folders = Section(inner, "Paths")
        folders.body.layout().setSpacing(5)
        _path_lbl_w = 80  # fits "Instrumental" at body font
        self.instrumental_row = PathRow(
            folders.body, "Instrumental",
            tip_text=TIPS["instrumental"],
            label_width=_path_lbl_w,
        )
        self.acapella_row = PathRow(
            folders.body, "Vocal",
            tip_text=TIPS["acapella"],
            label_width=_path_lbl_w,
        )
        self.pairs_output_row = PathRow(
            folders.body, "Output",
            tip_text=TIPS["pairs_output"],
            label_width=_path_lbl_w,
        )
        self._add_reference_row(folders.body, label_width=_path_lbl_w)
        self.include_subfolders_chk = CheckBox("Include subfolders")
        self.include_subfolders_chk.setChecked(True)
        self.include_subfolders_chk.setToolTip(TIPS["include_subfolders"])
        folders.body.layout().addWidget(self.include_subfolders_chk)
        v.addWidget(folders)

        # Matching
        matching = Section(inner, "Matching")
        m = matching.body.layout().contentsMargins()
        matching.body.layout().setContentsMargins(m.left(), 14, m.right(), m.bottom())
        matching.body.layout().setSpacing(12)
        self.strictness_slider = SliderField(
            matching.body, "Tag strictness",
            minimum=0, maximum=100, value=int(self.strictness),
            format_value=self._strictness_label,
            label_width=90,
            readout_width=110,  # "Very loose (100%)" / "Balanced (84%)"
        )
        self.strictness_slider.setToolTip(TIPS["strictness"])
        matching.body.layout().addWidget(self.strictness_slider)

        self.use_filename_fallback_chk = CheckBox('Use filename instead of tags ("Artist - Title.ext")')
        self.use_filename_fallback_chk.setChecked(True)
        self.use_filename_fallback_chk.setToolTip(TIPS["filename_fallback"])
        matching.body.layout().addWidget(self.use_filename_fallback_chk)
        v.addWidget(matching)

        # Ignore
        ignore = Section(inner, "Ignore when matching")
        im = ignore.body.layout().contentsMargins()
        ignore.body.layout().setContentsMargins(im.left(), 14, im.right(), im.bottom())
        ignore.body.layout().setSpacing(10)
        self.ignore_parens_chk = CheckBox("Ignore (parentheses)")
        self.ignore_parens_chk.setChecked(True)
        self.ignore_parens_chk.setToolTip(TIPS["ignore_parentheses"])
        self.ignore_brackets_chk = CheckBox("Ignore [square brackets]")
        self.ignore_brackets_chk.setChecked(True)
        self.ignore_brackets_chk.setToolTip(TIPS["ignore_square"])
        self.ignore_spaces_chk = CheckBox("Ignore extra spaces")
        self.ignore_spaces_chk.setChecked(True)
        self.ignore_spaces_chk.setToolTip(TIPS["ignore_spaces"])
        for w in (self.ignore_parens_chk, self.ignore_brackets_chk, self.ignore_spaces_chk):
            ignore.body.layout().addWidget(w)

        kw_row = QHBoxLayout()
        kw_row.setContentsMargins(0, 4, 0, 0)
        kw_lbl = BodyLabel("Custom keywords to ignore")
        kw_row.addWidget(kw_lbl)
        kw_row.addStretch(1)
        self.add_kw_btn = action_button(
            "+ Add", on_click=lambda: self._add_custom_keyword_row(), tip=TIPS["add_keyword"]
        )
        kw_row.addWidget(self.add_kw_btn)
        ignore.body.layout().addLayout(kw_row)

        self._custom_keywords_host = QWidget()
        self._custom_keywords_layout = QVBoxLayout(self._custom_keywords_host)
        self._custom_keywords_layout.setContentsMargins(0, 0, 0, 0)
        self._custom_keywords_layout.setSpacing(6)
        ignore.body.layout().addWidget(self._custom_keywords_host)
        v.addWidget(ignore)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _add_reference_row(self, parent: QWidget, *, label_width: int = 80) -> None:
        row = QHBoxLayout()
        # Top: air under Output. Bottom: match Matching card gap (12) to Include subfolders
        # (folders body spacing is 5 → bottom margin 7)
        row.setContentsMargins(0, 12, 0, 7)
        row.setSpacing(0)
        lbl = BodyLabel("Reference")
        lbl.setFixedWidth(label_width)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row.addWidget(lbl)
        # 4px — same as PathRow label→LineEdit so the Acapella dot lines up with inputs
        row.addSpacing(4)
        self._ref_group = QButtonGroup(self)
        self.ref_acapella = RadioButton("Vocal")
        self.ref_instrumental = RadioButton("Instrumental")
        self.ref_acapella.setChecked(True)
        self._ref_group.addButton(self.ref_acapella, 0)
        self._ref_group.addButton(self.ref_instrumental, 1)
        row.addWidget(self.ref_acapella)
        row.addSpacing(theme.RADIO_OPTION_GAP)
        row.addWidget(self.ref_instrumental)
        row.addStretch(1)
        self.ref_acapella.setToolTip(TIPS["reference"])
        self.ref_instrumental.setToolTip(TIPS["reference"])
        parent.layout().addLayout(row)

    def _build_align_tab(self) -> QWidget:
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(ScrollArea.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        v.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel("Align instrumental/vocal files to the original song")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=self._show_align_help))
        header.addStretch(1)
        v.addLayout(header)

        root_card = Section(inner, "Path")
        _align_lbl_w = 100  # fits "Without original" / "Originals inbox"
        self.stems_root_row = PathRow(
            root_card.body, "Stems root",
            tip_text=TIPS["align_stems_root"],
            label_width=_align_lbl_w,
        )
        self.stems_root_row.entry.textChanged.connect(self._sync_align_sort_dirs)
        v.addWidget(root_card)

        step1 = Section(inner, "1 · Export list")
        self.export_file_row = PathRow(
            step1.body, "Export list",
            directory=False, save_dialog=True,
            filter_pattern="Text files (*.txt);;All files (*)",
            caption="Choose export list file",
            tip_text=TIPS["align_export"],
            label_width=_align_lbl_w,
        )
        v.addWidget(step1)

        step2 = Section(inner, "2 · Distribute")
        self.originals_inbox_row = PathRow(
            step2.body, "Originals inbox",
            tip_text=TIPS["align_inbox"],
            label_width=_align_lbl_w,
        )
        v.addWidget(step2)

        step3 = Section(inner, "3 · Sort")
        step3.body.layout().setSpacing(5)
        self.with_original_row = PathRow(
            step3.body, "With original",
            tip_text=TIPS["align_with_original"],
            label_width=_align_lbl_w,
        )
        self.without_original_row = PathRow(
            step3.body, "Without original",
            tip_text=TIPS["align_without_original"],
            label_width=_align_lbl_w,
        )
        v.addWidget(step3)

        step4 = Section(inner, "4 · Align")
        # Same vertical rhythm as Matching (Tag strictness → checkbox = 12)
        s4 = step4.body.layout().contentsMargins()
        step4.body.layout().setContentsMargins(s4.left(), 14, s4.right(), s4.bottom())
        step4.body.layout().setSpacing(12)
        arow = QHBoxLayout()
        arow.setContentsMargins(0, 0, 0, 0)
        arow.setSpacing(8)
        arow.addWidget(BodyLabel("Analysis"))
        self.analysis_sec_spin = CompactDoubleSpinBox()
        self.analysis_sec_spin.setObjectName("FilterDurationSpin")
        self.analysis_sec_spin.setRange(1.0, 600.0)
        self.analysis_sec_spin.setSingleStep(1.0)
        self.analysis_sec_spin.setDecimals(0)
        self.analysis_sec_spin.setValue(float(self.align_analysis_sec))
        self.analysis_sec_spin.setSuffix(" s")
        self.analysis_sec_spin.setFixedWidth(theme.COMPACT_SPIN_WIDTH)
        self.analysis_sec_spin.setToolTip(TIPS["align_analysis"])
        theme.sync_compact_spin_button(self.analysis_sec_spin, theme.COMPACT_SPIN_HEIGHT)
        theme._install_compact_spin_resize_sync(self.analysis_sec_spin)
        arow.addWidget(self.analysis_sec_spin)
        arow.addStretch(1)
        step4.body.layout().addLayout(arow)
        self.align_backup_chk = CheckBox("Backup stems before align")
        self.align_backup_chk.setChecked(True)
        self.align_backup_chk.setToolTip(TIPS["align_backup"])
        self.align_skip_existing_chk = CheckBox("Skip if output already exists")
        self.align_skip_existing_chk.setChecked(True)
        self.align_skip_existing_chk.setToolTip(TIPS["align_skip_existing"])
        step4.body.layout().addWidget(self.align_backup_chk)
        step4.body.layout().addWidget(self.align_skip_existing_chk)
        v.addWidget(step4)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    # ----------------- helpers -----------------

    def _strictness_label(self, value: int) -> str:
        if value >= 90:
            label = "Strict"
        elif value >= 65:
            label = "Balanced"
        elif value >= 35:
            label = "Loose"
        else:
            label = "Very loose"
        threshold = strictness_to_threshold(float(value))
        return f"{label} ({threshold:.0%})"

    def _add_custom_keyword_row(self, value: str = "") -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        edit = LineEdit()
        edit.setText(value if isinstance(value, str) else "")
        edit.setPlaceholderText("keyword to ignore")
        edit.setFixedHeight(theme.PATH_FIELD_HEIGHT)
        edit.setClearButtonEnabled(False)
        edit.setToolTip(TIPS["ignore_custom"])
        theme.style_line_edit(edit)
        edit.textChanged.connect(self._schedule_save)
        row.addWidget(edit, stretch=1)

        remove = action_button(
            "×",
            on_click=lambda: self._remove_custom_keyword_row(edit),
            height=theme.PATH_FIELD_HEIGHT,
            width=36,
            tip=TIPS["remove_keyword"],
        )
        remove.setObjectName("KeywordRemove")
        # Title-bar close hover: solid danger red fill
        t = theme.DARK
        remove.setStyleSheet(
            f"""
            PushButton#KeywordRemove {{
                background-color: {theme.CONTROL_BG};
                color: {t['text_dim']};
                border: 1px solid {t['border']};
                border-radius: 5px;
                font-size: 14px;
            }}
            PushButton#KeywordRemove:hover {{
                background-color: {t['danger']};
                color: {t['text']};
                border: 1px solid {t['danger']};
            }}
            """
        )
        row.addWidget(remove)
        host = QWidget()
        host.setLayout(row)
        self._custom_keywords_layout.addWidget(host)
        self._custom_keyword_edits.append(edit)

    def _remove_custom_keyword_row(self, edit: LineEdit, *, ensure_one: bool = True) -> None:
        host = edit.parentWidget()
        if host is None:
            return
        self._custom_keywords_layout.removeWidget(host)
        host.deleteLater()
        if edit in self._custom_keyword_edits:
            self._custom_keyword_edits.remove(edit)
        if ensure_one and not self._custom_keyword_edits:
            self._add_custom_keyword_row()
        self._schedule_save()

    def _collect_custom_keywords(self) -> list[str]:
        return [e.text().strip() for e in self._custom_keyword_edits if e.text().strip()]

    def _collect_ignore_rules(self) -> IgnoreRules:
        return IgnoreRules(
            ignore_parentheses=self.ignore_parens_chk.isChecked(),
            ignore_square_brackets=self.ignore_brackets_chk.isChecked(),
            ignore_extra_spaces=self.ignore_spaces_chk.isChecked(),
            custom_keywords=tuple(self._collect_custom_keywords()),
        )

    def _align_sort_dirs(self, root: Path) -> tuple[Path, Path]:
        with_dir = self.with_original_row.text().strip() or str(default_with_original_dir(root))
        without_dir = self.without_original_row.text().strip() or str(default_without_original_dir(root))
        return Path(with_dir), Path(without_dir)

    def _sync_align_sort_dirs(self, *_) -> None:
        root_text = self.stems_root_row.text().strip()
        if not root_text:
            return
        try:
            root = Path(root_text)
        except Exception:
            return
        with_default = str(default_with_original_dir(root))
        without_default = str(default_without_original_dir(root))
        cur_with = self.with_original_row.text().strip()
        cur_without = self.without_original_row.text().strip()
        if not cur_with:
            self.with_original_row.set_text(with_default)
        if not cur_without:
            self.without_original_row.set_text(without_default)

    def player_library_root(self) -> str:
        """Folder whose immediate children are song folders (Classify-style).

        Align → With original (song dirs with vocals/instrumental/original).
        Match → Pairs Output.
        """
        if getattr(self, "_tabview", None) is not None and self._tabview.currentIndex() == 1:
            with_dir = self.with_original_row.text().strip()
            if with_dir:
                return with_dir
            root = self.stems_root_row.text().strip()
            if root:
                try:
                    return str(default_with_original_dir(Path(root)))
                except Exception:
                    return ""
            return ""
        return self.pairs_output_row.text().strip()

    def _on_subtab_changed(self, idx: int) -> None:
        if not hasattr(self, "_action_page"):
            return
        if idx == 1:  # Align
            self._show_align_buttons()
        else:
            self._show_match_buttons()

    def _show_match_buttons(self) -> None:
        if not hasattr(self, "_action_page"):
            return
        for btn in (
            self.find_btn, self.organize_btn, self.play_stems_btn,
            self.export_list_btn, self.distribute_btn, self.sort_folders_btn, self.align_btn,
        ):
            btn.hide()
        self.find_btn.show()
        self.organize_btn.show()
        self.play_stems_btn.show()

    def _show_align_buttons(self) -> None:
        if not hasattr(self, "_action_page"):
            return
        for btn in (
            self.find_btn, self.organize_btn, self.play_stems_btn,
            self.export_list_btn, self.distribute_btn, self.sort_folders_btn, self.align_btn,
        ):
            btn.hide()
        self.export_list_btn.show()
        self.distribute_btn.show()
        self.sort_folders_btn.show()
        self.align_btn.show()
        self.play_stems_btn.show()

    # ----------------- action bar -----------------

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._action_page = page
        # Same metrics as Classify: action_button height/font + content width (no fixed widths).
        self.find_btn = action_button(
            "▶ Find pairs", on_click=self._start_find_pairs, accent=True, tip=TIPS["find_pairs"]
        )
        self.organize_btn = action_button(
            "▶ Organize folder", on_click=self._start_organize, tip=TIPS["organize"]
        )
        self.export_list_btn = action_button(
            "Export list", on_click=self._start_export_list, tip=TIPS["export_list_btn"]
        )
        self.distribute_btn = action_button(
            "Distribute originals",
            on_click=self._start_distribute_originals,
            tip=TIPS["distribute_btn"],
        )
        self.sort_folders_btn = action_button(
            "Sort folders", on_click=self._start_sort_folders, tip=TIPS["sort_folders_btn"]
        )
        self.align_btn = action_button(
            "Align stems", on_click=self._start_align_stems, tip=TIPS["align_stems_btn"]
        )
        self.play_stems_btn = action_button(
            "♫ Play",
            on_click=lambda: self.request_open_player.emit(),
            tip=TIPS["play_stems"],
        )
        # Layout (left → right): primary actions, stretch, Play at far end.
        # Visibility per subtab:
        #   Match → [Find pairs] [Organize]            [stretch] [♫ Play]
        #   Align → [Export list] [Distribute] [Sort folders] [Align] [stretch] [♫ Play]
        # First visible button always sits at the page's left inset.
        for btn in (
            self.find_btn, self.organize_btn,
            self.export_list_btn, self.distribute_btn,
            self.sort_folders_btn, self.align_btn,
        ):
            page.add_button(btn)
        page.add_stretch()
        page.add_button(self.play_stems_btn)
        # Initial visibility
        self._show_match_buttons()

    # ----------------- worker actions -----------------

    def _guard(self) -> bool:
        if self._busy:
            return False
        return True

    @staticmethod
    def _enable_widgets(enabled: bool, *widgets) -> None:
        for w in widgets:
            if w is None:
                continue
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def _set_match_settings_enabled(self, enabled: bool) -> None:
        self._enable_widgets(
            enabled,
            self.acapella_row,
            self.instrumental_row,
            self.pairs_output_row,
            self.ref_acapella,
            self.ref_instrumental,
            self.include_subfolders_chk,
            self.strictness_slider,
            self.use_filename_fallback_chk,
            self.ignore_parens_chk,
            self.ignore_brackets_chk,
            self.ignore_spaces_chk,
            self.add_kw_btn,
            self._custom_keywords_host,
        )
        for edit in self._custom_keyword_edits:
            host = edit.parentWidget()
            self._enable_widgets(enabled, host if host is not None else edit)

    def _set_align_settings_enabled(self, enabled: bool) -> None:
        self._enable_widgets(
            enabled,
            self.stems_root_row,
            self.export_file_row,
            self.originals_inbox_row,
            self.with_original_row,
            self.without_original_row,
            self.analysis_sec_spin,
            self.align_backup_chk,
            self.align_skip_existing_chk,
        )

    def _set_busy(
        self, busy: bool, status: str = "", *, panel: Optional[str] = None
    ) -> None:
        """Lock Match or Align settings for the started job only."""
        self._busy = busy
        self._busy_panel = panel if busy else None
        for btn in (
            self.find_btn, self.organize_btn, self.export_list_btn,
            self.distribute_btn, self.sort_folders_btn, self.align_btn,
        ):
            btn.setEnabled(not busy)
        if busy and panel == "match":
            self._set_match_settings_enabled(False)
            self._set_align_settings_enabled(True)
        elif busy and panel == "align":
            self._set_match_settings_enabled(True)
            self._set_align_settings_enabled(False)
        else:
            self._set_match_settings_enabled(True)
            self._set_align_settings_enabled(True)
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def _start_worker(
        self,
        action,
        status: str,
        *,
        starting: str | None = None,
        panel: str = "match",
    ) -> None:
        if not self._guard():
            return
        self.request_clear_log.emit()
        # Instant feedback before the worker thread does any I/O
        # Startup/config indented like Classify; === Summary headers stay flush.
        label = (starting or status).rstrip(".… ")
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            _PREFIX = {
                "Find pairs": "find_pairs",
                "Organize folder": "organize",
                "Export list": "export_list",
                "Distribute originals": "distribute",
                "Sort folders": "sort_folders",
                "Align stems": "align",
            }
            win.set_log_export_prefix(
                _PREFIX.get(
                    label,
                    "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label.lower())
                    .strip("_")
                    or "match",
                )
            )
        self.request_log.emit(f"  Starting {label}...", "info")
        self._set_busy(True, status, panel=panel)
        self._worker = PairWorker(action, parent=self)
        self._worker.log_line.connect(self.request_log)
        self._worker.progress.connect(self.request_progress)
        self._worker.status.connect(self.request_status_text)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, status: str) -> None:
        self._worker = None
        self._set_busy(False, status)

    # Find pairs
    def _start_find_pairs(self) -> None:
        if not self._guard():
            return
        acapella = Path(self.acapella_row.text().strip())
        instrumental = Path(self.instrumental_row.text().strip())
        output = Path(self.pairs_output_row.text().strip())
        if not acapella.is_dir():
            show_info(self, PANEL_TITLE, "Acapella folder is missing or invalid.")
            return
        if not instrumental.is_dir():
            show_info(self, PANEL_TITLE, "Instrumental folder is missing or invalid.")
            return
        if not output.exists():
            try:
                output.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                show_info(self, PANEL_TITLE, f"Could not create output folder:\n{exc}")
                return
        ref_acapella = self.ref_acapella.isChecked()
        strictness = float(self.strictness_slider.value())
        use_filename = self.use_filename_fallback_chk.isChecked()
        include_sub = self.include_subfolders_chk.isChecked()
        ignore_rules = self._collect_ignore_rules()

        def action(on_log, on_progress):
            t0 = time.monotonic()
            if ref_acapella:
                reference_dir, partner_dir = acapella, instrumental
                ref_label, partner_label = "acapella", "instrumental"
            else:
                reference_dir, partner_dir = instrumental, acapella
                ref_label, partner_label = "instrumental", "acapella"
            result = find_pairs(
                reference_dir, partner_dir,
                reference_is_acapella=ref_acapella,
                strictness=strictness,
                use_filename_fallback=use_filename,
                ignore_rules=ignore_rules,
                include_subfolders=include_sub,
                move_to=output,
                on_progress=on_progress,
                on_log=on_log,
            )
            threshold = strictness_to_threshold(strictness)
            fallback_note = "filename only" if use_filename else "tags only"
            self._log_feature_summary(
                on_log,
                "Find pairs",
                elapsed=time.monotonic() - t0,
                lines=[
                (f"Reference: {ref_label} · {fallback_note}", "info"),
                (f"Threshold: {threshold:.0%}", "info"),
                (f"Pairs: {len(result.pairs):,}", "ok"),
                (f"Unmatched reference: {len(result.unmatched_reference):,}", "warn" if result.unmatched_reference else "info"),
                (f"Unmatched partner: {len(result.unmatched_partner):,}", "warn" if result.unmatched_partner else "info"),
            ])
            self._worker.set_final_status(f"Done · {len(result.pairs)} pair(s) moved")

        self._start_worker(action, "Finding pairs…", starting="Find pairs", panel="match")

    # Organize
    def _start_organize(self) -> None:
        if not self._guard():
            return
        folder = Path(self.pairs_output_row.text().strip())
        if not folder.is_dir():
            show_info(
                self, PANEL_TITLE,
                "Pairs output folder is missing or invalid.\n"
                "Find pairs writes there; Organize folder groups files inside it.",
            )
            return
        strictness = float(self.strictness_slider.value())
        use_filename = self.use_filename_fallback_chk.isChecked()
        include_sub = self.include_subfolders_chk.isChecked()
        ignore_rules = self._collect_ignore_rules()

        def action(on_log, on_progress):
            t0 = time.monotonic()
            on_log(f"  Organizing matched files in:\n    {folder}", "info")
            moved = organize_matched_folder(
                folder, strictness=strictness, use_filename_fallback=use_filename,
                ignore_rules=ignore_rules, include_subfolders=include_sub,
                on_progress=on_progress, on_log=on_log,
            )
            if not moved:
                on_log("No FLAC or MP3 files found in the selected folder.", "warn")
                self._worker.set_final_status("Nothing to organize")
                return
            on_log(f"Created {len(moved):,} folder(s)", "ok")
            show = min(25, len(moved))
            for dest_dir, _files in moved[:show]:
                on_log(f"✓ {dest_dir.name}/", "ok")
            if len(moved) > show:
                on_log(f"… and {len(moved) - show:,} more folder(s)", "info")
            self._log_feature_summary(
                on_log,
                "Organize folder",
                elapsed=time.monotonic() - t0,
                lines=[(f"Folders created: {len(moved):,}", "ok")],
            )
            self._worker.set_final_status(f"Done · {len(moved)} folder(s) created")

        self._start_worker(action, "Organizing…", starting="Organize folder", panel="match")

    # Export list
    def _start_export_list(self) -> None:
        if not self._guard():
            return
        root = Path(self.stems_root_row.text().strip())
        try:
            export_path = resolve_export_list_path(Path(self.export_file_row.text().strip()))
        except Exception:
            show_info(self, PANEL_TITLE, "Export list path is invalid.")
            return
        if not root.is_dir():
            show_info(self, PANEL_TITLE, "Stems root folder is missing or invalid.")
            return
        if not export_path.parent.is_dir():
            show_info(self, PANEL_TITLE, "Export list folder does not exist.")
            return
        self.export_file_row.set_text(display_path(str(export_path)))

        def action(on_log, on_progress):
            t0 = time.monotonic()
            count = export_song_list(root, export_path)
            self._log_feature_summary(
                on_log,
                "Export list",
                elapsed=time.monotonic() - t0,
                lines=[
                (f"Names: {count:,}", "ok"),
                (f"File: {export_path}", "info"),
            ])
            self._worker.set_final_status(f"Done · {count:,} name(s) exported")

        self._start_worker(action, "Exporting song list…", starting="Export list", panel="align")

    # Distribute
    def _start_distribute_originals(self) -> None:
        if not self._guard():
            return
        inbox = Path(self.originals_inbox_row.text().strip())
        root = Path(self.stems_root_row.text().strip())
        if not inbox.is_dir():
            show_info(self, PANEL_TITLE, "Originals inbox folder is missing or invalid.")
            return
        if not root.is_dir():
            show_info(self, PANEL_TITLE, "Stems root folder is missing or invalid.")
            return
        with_dir, without_dir = self._align_sort_dirs(root)

        def action(on_log, on_progress):
            t0 = time.monotonic()
            moved, skipped, unmatched, rejected, _sw, _so = distribute_originals(
                inbox, root, on_log=on_log, on_progress=on_progress, sort_after=False,
                with_original_dir=with_dir, without_original_dir=without_dir,
            )
            self._log_feature_summary(
                on_log,
                "Distribute originals",
                elapsed=time.monotonic() - t0,
                lines=[
                (f"Moved: {moved:,}", "ok"),
                (f"Skipped: {skipped:,}", "warn" if skipped else "info"),
                (f"Unmatched: {unmatched:,}", "warn" if unmatched else "info"),
                (f"Rejected: {rejected:,}", "warn" if rejected else "info"),
            ])
            self._worker.set_final_status(f"Done · {moved:,} moved")

        self._start_worker(action, "Distributing originals…", starting="Distribute originals", panel="align")

    # Sort
    def _start_sort_folders(self) -> None:
        if not self._guard():
            return
        root = Path(self.stems_root_row.text().strip())
        if not root.is_dir():
            show_info(self, PANEL_TITLE, "Stems root folder is missing or invalid.")
            return
        with_dir, without_dir = self._align_sort_dirs(root)

        def action(on_log, on_progress):
            t0 = time.monotonic()
            moved_with, moved_without, skipped = sort_folders_by_original(
                root, with_original_dir=with_dir, without_original_dir=without_dir,
                on_log=on_log, on_progress=on_progress,
            )
            self._log_feature_summary(
                on_log,
                "Sort folders",
                elapsed=time.monotonic() - t0,
                lines=[
                (f"With original: {moved_with:,}", "ok"),
                (f"Without original: {moved_without:,}", "err"),
                (f"Skipped: {skipped:,}", "warn" if skipped else "info"),
            ])
            self._worker.set_final_status(f"Done · {moved_with + moved_without:,} sorted")

        self._start_worker(action, "Sorting folders…", starting="Sort folders", panel="align")

    # Align
    def _start_align_stems(self) -> None:
        if not self._guard():
            return
        root = Path(self.stems_root_row.text().strip())
        if not root.is_dir():
            show_info(self, PANEL_TITLE, "Stems root folder is missing or invalid.")
            return
        try:
            analysis_sec = float(self.analysis_sec_spin.value())
        except (ValueError, TypeError):
            show_info(self, PANEL_TITLE, "Analysis seconds must be a number.")
            return
        backup = self.align_backup_chk.isChecked()
        skip_existing = self.align_skip_existing_chk.isChecked()
        with_dir, _without_dir = self._align_sort_dirs(root)

        def action(on_log, on_progress):
            t0 = time.monotonic()
            results, skipped = align_all_songs(
                root, with_original_dir=with_dir, analysis_sec=analysis_sec,
                backup=backup, skip_existing=skip_existing,
                on_log=on_log, on_progress=on_progress,
            )
            self._log_feature_summary(
                on_log,
                "Align stems",
                elapsed=time.monotonic() - t0,
                lines=[
                (f"Aligned: {len(results):,}", "ok"),
                (f"Skipped (already aligned): {skipped:,}", "warn" if skipped else "info"),
            ])
            self._worker.set_final_status(f"Done · {len(results):,} aligned")

        self._start_worker(action, "Aligning stems…", starting="Align stems", panel="align")

    def _log_feature_summary(
        self,
        on_log,
        feature: str,
        *,
        lines=None,
        elapsed: float | None = None,
        files: int | None = None,
    ) -> None:
        emit_run_summary(
            on_log,
            feature,
            elapsed=elapsed,
            files=files,
            stat_lines=lines or (),
        )

    # ----------------- help -----------------

    def _show_match_help(self) -> None:
        help_dialog(
            self,
            title="Match help",
            heading="Match instrumentals & vocals",
            intro="Find matching versions by artist and title, then turn them into an organized library",
            sections=[
                ("Workflow", [
                    "1. Choose the Acapella and Instrumental source folders.",
                    "2. Choose where matched files should be moved.",
                    "3. Set the reference side and matching strictness.",
                    "4. Click Find pairs, review the log, then Organize folder.",
                ]),
                ("How matching works", [
                    "Artist and title tags are compared separately. The reference folder drives the scan: "
                    "each reference file searches for its best partner in the other folder. "
                    '"Use filename instead" parses artist/title from the filename only, ignoring tags.',
                ]),
                ("Tune the result", [
                    "Lower strictness accepts small differences such as extra artists, spacing, "
                    'or "&" versus "and". Ignore rules remove brackets, extra spaces, and custom words '
                    "before comparison; they do not rename the source files.",
                ]),
                ("File safety", [
                    "Find pairs moves only confirmed matches into Pairs output. "
                    "Organize folder then groups those files into Artist - Title subfolders inside Pairs output. "
                    "Unmatched files remain in their source folder.",
                ]),
            ],
        )

    def _show_align_help(self) -> None:
        help_dialog(
            self,
            title="Align help",
            heading="Align stems to the original",
            intro="Use the original song as the master timeline for its instrumental and acapella",
            sections=[
                ("Required layout", [
                    "Stems root contains one folder per song, with an instrumental and acapella inside. "
                    "Downloaded original songs first go into the Originals inbox.",
                ]),
                ("Four-step workflow", [
                    "1. Export the song-folder names as a download list.",
                    "2. Put downloaded originals in the inbox and distribute them.",
                    "3. Sort song folders into with_original and without_original.",
                    "4. Align folders that have an original.",
                ]),
                ("How alignment works", [
                    "The original song is the master timeline. Audio is analyzed to estimate the offset; "
                    "silence is added or the beginning is trimmed so the instrumental and acapella start at "
                    "the correct time. A longer analysis window can help difficult material but takes more time.",
                ]),
                ("File safety", [
                    "Keep Backup stems before align enabled to preserve the untouched files in _backup_before_align. "
                    "Skip if output already exists makes interrupted batches safe to resume.",
                ]),
            ],
        )

    # ----------------- settings -----------------

    def settings_snapshot(self) -> dict:
        return {
            "acapella_dir": display_path(self.acapella_row.text()),
            "instrumental_dir": display_path(self.instrumental_row.text()),
            "pairs_output_dir": display_path(self.pairs_output_row.text()),
            "reference_side": "acapella" if self.ref_acapella.isChecked() else "instrumental",
            "strictness": float(self.strictness_slider.value()),
            "use_filename_fallback": bool(self.use_filename_fallback_chk.isChecked()),
            "include_subfolders": bool(self.include_subfolders_chk.isChecked()),
            "ignore_rules": self._collect_ignore_rules().to_dict(),
            "align_stems_root": display_path(self.stems_root_row.text()),
            "align_originals_inbox": display_path(self.originals_inbox_row.text()),
            "align_export_file": display_path(self.export_file_row.text()),
            "align_backup": bool(self.align_backup_chk.isChecked()),
            "align_skip_existing": bool(self.align_skip_existing_chk.isChecked()),
            "align_analysis_sec": int(self.analysis_sec_spin.value()),
            "align_with_original_dir": display_path(self.with_original_row.text()),
            "align_without_original_dir": display_path(self.without_original_row.text()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("acapella_dir"):
                self.acapella_row.set_text(d["acapella_dir"])
            if d.get("instrumental_dir"):
                self.instrumental_row.set_text(d["instrumental_dir"])
            if d.get("pairs_output_dir"):
                self.pairs_output_row.set_text(d["pairs_output_dir"])
            elif d.get("organize_dir"):
                self.pairs_output_row.set_text(d["organize_dir"])
            side = d.get("reference_side", "acapella")
            self.ref_acapella.setChecked(side == "acapella")
            self.ref_instrumental.setChecked(side != "acapella")
            if d.get("strictness") is not None:
                self.strictness_slider.setValue(int(float(d["strictness"])))
            self.use_filename_fallback_chk.setChecked(bool(d.get("use_filename_fallback", True)))
            self.include_subfolders_chk.setChecked(bool(d.get("include_subfolders", True)))
            rules = IgnoreRules.from_dict(d.get("ignore_rules"))
            self.ignore_parens_chk.setChecked(rules.ignore_parentheses)
            self.ignore_brackets_chk.setChecked(rules.ignore_square_brackets)
            self.ignore_spaces_chk.setChecked(rules.ignore_extra_spaces)
            # Rebuild keyword rows
            for edit in list(self._custom_keyword_edits):
                self._remove_custom_keyword_row(edit, ensure_one=False)
            for kw in rules.custom_keywords:
                self._add_custom_keyword_row(kw)
            if not self._custom_keyword_edits:
                self._add_custom_keyword_row()
            if d.get("align_stems_root"):
                self.stems_root_row.set_text(d["align_stems_root"])
            if d.get("align_originals_inbox"):
                self.originals_inbox_row.set_text(d["align_originals_inbox"])
            if d.get("align_export_file"):
                self.export_file_row.set_text(d["align_export_file"])
            self.align_backup_chk.setChecked(bool(d.get("align_backup", True)))
            self.align_skip_existing_chk.setChecked(bool(d.get("align_skip_existing", True)))
            if d.get("align_analysis_sec") is not None:
                self.analysis_sec_spin.setValue(float(d["align_analysis_sec"]))
            if d.get("align_with_original_dir"):
                self.with_original_row.set_text(d["align_with_original_dir"])
            if d.get("align_without_original_dir"):
                self.without_original_row.set_text(d["align_without_original_dir"])
            self._sync_align_sort_dirs()
            self._apply_cross_tab_paths(d)
        finally:
            self._loading = False

    def on_tab_shown(self) -> None:
        """Pull Genre / Gender / Output takeovers when visiting Match & Align."""
        self._loading = True
        try:
            self._apply_cross_tab_paths(self._settings.data)
        finally:
            self._loading = False
        self._flush_settings()

    def _apply_cross_tab_paths(self, d: dict | None = None) -> None:
        """Instrumental←Genre, Vocal←Gender, Output default, Stems←Output."""
        d = d if d is not None else self._settings.data
        genre = (d.get("gg_genre_input_dir") or "").strip()
        gender = (d.get("gg_gender_input_dir") or "").strip()
        if genre:
            self.instrumental_row.set_text(genre)
        if gender:
            self.acapella_row.set_text(gender)
        if not self.pairs_output_row.text().strip():
            self.pairs_output_row.set_text(DEFAULT_PAIRS_OUTPUT)
        pairs_out = self.pairs_output_row.text().strip()
        if pairs_out:
            self.stems_root_row.set_text(pairs_out)
            self._sync_align_sort_dirs()

    def _autofill_from_instrumental(self, text: str) -> None:
        if self._loading:
            return
        text = text.strip()
        if not text:
            return
        if not self.pairs_output_row.text().strip():
            self.pairs_output_row.set_text(DEFAULT_PAIRS_OUTPUT)

    def _autofill_align_from_output(self, text: str) -> None:
        if self._loading:
            return
        text = text.strip()
        if not text:
            return
        # Stems root takes over Match Output.
        self.stems_root_row.set_text(text)
        self._sync_align_sort_dirs()

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        self.instrumental_row.entry.textChanged.connect(self._autofill_from_instrumental)
        self.pairs_output_row.entry.textChanged.connect(self._autofill_align_from_output)
        for sig in (
            self.acapella_row.entry.textChanged,
            self.instrumental_row.entry.textChanged,
            self.pairs_output_row.entry.textChanged,
            self.ref_acapella.toggled,
            self.ref_instrumental.toggled,
            self.strictness_slider.valueChanged,
            self.use_filename_fallback_chk.toggled,
            self.include_subfolders_chk.toggled,
            self.ignore_parens_chk.toggled,
            self.ignore_brackets_chk.toggled,
            self.ignore_spaces_chk.toggled,
            self.stems_root_row.entry.textChanged,
            self.originals_inbox_row.entry.textChanged,
            self.export_file_row.entry.textChanged,
            self.with_original_row.entry.textChanged,
            self.without_original_row.entry.textChanged,
            self.align_backup_chk.toggled,
            self.align_skip_existing_chk.toggled,
            self.analysis_sec_spin.valueChanged,
        ):
            sig.connect(self._schedule_save)

    def _schedule_save(self, *_) -> None:
        if self._loading:
            return
        if not hasattr(self, "_autosave_timer"):
            return
        self._autosave_timer.start()

    def _flush_settings(self) -> None:
        if self._loading:
            return
        self._settings.merge(self.settings_snapshot())
        self._settings.flush()

    def flush_settings(self) -> None:
        """Immediate persist (used on app close)."""
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()


def register(window, settings: SettingsStore) -> None:
    tab = PairFinderTab(settings)
    window.register_tab("Match & Align", tab)
    page = window.action_bar.make_page("Match & Align")
    tab.attach_action_bar(page, window)
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_status_text.connect(window.status_bar.set_status)
    tab.request_log.connect(window.append_log)
    tab.request_clear_log.connect(window.clear_log)
    tab.request_open_player.connect(lambda: window._open_player())

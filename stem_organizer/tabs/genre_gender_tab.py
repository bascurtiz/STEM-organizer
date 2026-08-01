"""Genre & Gender tab — port of genre_gender_panel.GenreGenderPanel.

Three sub-tabs:
  Genre  — Paths (input + include subfolders), Run mode (Batch/Per-file),
           Tag style (Combined/Split), Tag options (Write metadata, Skip if already tagged).
  Gender — same shape + Voice gender field (Comment/Gender) + Reverb mode
           (Combined/Split).
  Vocal type — PANNs Cnn14 (Singing / Speech / Rapping / Humming / Choir).

Action buttons: ▶ Tag genre / ▶ Tag gender / ▶ Tag vocal type / ■ Stop.
"""
from __future__ import annotations

import classify_backend as cb
import time
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    RadioButton,
    ScrollArea,
    TabWidget,
)

from .. import theme
from ..settings_store import SettingsStore, display_path
from ..tag_input import (
    TagPanel,
    collect_paths_for_panel,
    layout_log_line,
    resolve_tag_input,
    scan_mode_from_recursive,
    write_files_list,
)
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.panns_worker import PannsWorker
from ..workers.tagger_worker import TaggerWorker


PANEL_TITLE = "Genre & Gender"

TIPS = {
    "genre_input": (
        "Folder to scan for genre/style tagging. Uses the same layout rules as "
        "Classify: only instrumental-side files are tagged (instrumentals, bass/drums/other, "
        "or the instrumental stem in pairs — not vocals)."
    ),
    "gender_input": (
        "Folder to scan for gender/reverb tagging. Uses the same layout rules as "
        "Classify: only vocal files are tagged (vocals stem in pairs, not instrumentals)."
    ),
    "vocal_input": (
        "Folder to scan for vocal-type tagging (Singing / Speech / …). Uses the same "
        "layout rules as Classify: vocals only, not instrumentals."
    ),
    "include_subfolders": "Scan audio files in subfolders too, not just the selected folder itself.",
    "run_mode_batch": (
        "Batch decodes several files in parallel, then runs the model in GPU batches "
        "(or CPU if no CUDA). Fastest overall; LOG shows a single progress counter."
    ),
    "run_mode_per_file": (
        "Per-file processes one track at a time and prints each prediction live in the LOG. "
        "Still uses the GPU for inference when available; slower than Batch because it does not "
        "overlap decode across files."
    ),
    "tag_style_combined": "Combined writes a single GENRE tag as Genre/Style.",
    "tag_style_split": (
        "Split writes separate GENRE and STYLE tags. "
        "Genre/Style tags are read by Charts."
    ),
    "tag_field_comment": "Comment writes gender to the COMMENT tag.",
    "tag_field_gender": (
        "Gender (custom) writes to a GENDER tag. Gender tag is read by Charts."
    ),
    "vocal_tag_field_comment": "Comment writes the label (e.g. Singing) to COMMENT.",
    "vocal_tag_field_vocal": (
        "Vocal type (custom) writes to a VOCAL_TYPE tag. "
        "Vocal type tag is read by Charts."
    ),
    "reverb_mode_combined": (
        "Dry/wet from the bundled vocal mel-CNN. "
        "Combined writes gender/reverb into the chosen field."
    ),
    "reverb_mode_split": (
        "Dry/wet from the bundled vocal mel-CNN. "
        "Split writes gender alone and REVERB=wet|dry as a separate custom field. "
        "Gender/Reverb tags are read by Charts."
    ),
    "write_meta": "Write tags to FLAC/MP3/M4A/WAV. Disable to only generate the CSV.",
    "vocal_write_meta": "Write tags to FLAC/MP3/M4A/WAV. Disable to only log scores.",
    "skip_existing": (
        "On (default): skip files that already have genre/gender tags "
        "(resume-friendly). Off: re-tag every file."
    ),
    "vocal_skip_existing": (
        "On (default): skip files that already have a vocal-type tag. "
        "Off: re-tag every file."
    ),
    "vocal_segments": (
        "Also score 2-second windows and list Singing/Speech/… over time in the LOG."
    ),
    "tag_genre": "Run the genre/style tagger on the input folder.",
    "tag_gender": "Run the voice gender + reverb tagger on the input folder.",
    "tag_vocal": "Run PANNs vocal-type tagging (Singing / Speech / Rapping / Humming / Choir).",
    "stop": "Stop the running tagger.",
}

TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}

# Hint text beside radios — QLabel + stylesheet (BodyLabel polish resets to white)
_HINT_FONT_PX = theme.BODY_FONT_PX


def _load_skip_existing(d: dict, skip_key: str, overwrite_key: str) -> bool:
    """Prefer skip_existing; migrate legacy overwrite_tags (inverted). Default on."""
    if skip_key in d:
        return bool(d[skip_key])
    if overwrite_key in d:
        return not bool(d[overwrite_key])
    return True


def _split_option_label(text: str) -> tuple[str, str | None]:
    """'Batch (fast)' → ('Batch', 'fast'); no paren → (text, None)."""
    paren = text.find("(")
    if paren > 0:
        hint = text[paren:].strip().removeprefix("(").removesuffix(")").strip()
        return text[:paren].rstrip(), hint or None
    return text.strip(), None


def _style_radio_hint(lbl: QLabel) -> None:
    dim = theme.DARK["text_dim"]
    lbl.setObjectName("RadioHint")
    lbl.setStyleSheet(
        f"""
        QLabel#RadioHint {{
            color: {dim};
            font-family: "{theme.FONT_FAMILY}";
            font-size: {_HINT_FONT_PX}px;
            background: transparent;
        }}
        """
    )


class _RadioRow(QWidget):
    """Two radio options split evenly across the full card width."""

    def __init__(
        self,
        parent: QWidget,
        options,
        value: str,
        *,
        tooltip: str = "",
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
            main, hint = _split_option_label(label)
            key = str(val)

            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            cell_lay = QHBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(6)

            rb = RadioButton(main)
            rb.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            rb.setCursor(Qt.PointingHandCursor)
            tip_text = (tooltips or {}).get(key) or tooltip
            if tip_text:
                tip = theme.format_tooltip(tip_text)
                rb.setToolTip(tip)
            self._group.addButton(rb)
            self._buttons[key] = rb
            cell_lay.addWidget(rb, 0, Qt.AlignVCenter)

            if hint:
                hint_lbl = QLabel(hint)
                hint_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                _style_radio_hint(hint_lbl)
                hint_lbl.setCursor(Qt.PointingHandCursor)
                if tip_text:
                    hint_lbl.setToolTip(tip)
                hint_lbl.mousePressEvent = (  # type: ignore[method-assign]
                    lambda _e, v=key: self.set_value(v)
                )
                cell_lay.addWidget(hint_lbl, 0, Qt.AlignVCenter)
                self._extra.append(hint_lbl)

            cell_lay.addStretch(1)
            layout.addWidget(cell, 1)  # equal stretch → half-split
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

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 Qt name
        super().setEnabled(enabled)
        for rb in self._buttons.values():
            rb.setEnabled(enabled)
        for w in self._extra:
            w.setEnabled(enabled)


class GenreGenderTab(QWidget):
    """Genre + Gender + Vocal type sub-tabs sharing an action bar."""

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
        self._worker: Optional[Union[TaggerWorker, PannsWorker]] = None
        self._busy = False
        self._busy_panel: Optional[str] = None  # "genre" | "gender" | "vocal"
        self._busy_since = 0.0
        self._loading = False

        self._build_ui()
        self.load_settings()
        self._bind_autosave()

    # ----- UI -----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        self._tabview = TabWidget()
        theme.configure_tab_widget(self._tabview)
        self._tabview.addTab(self._build_genre_tab(), "Genre")
        self._tabview.addTab(self._build_gender_tab(), "Gender")
        self._tabview.addTab(self._build_vocal_tab(), "Vocal type")
        theme.configure_tab_widget(self._tabview)
        theme.inset_tab_bar(self._tabview)
        self._tabview.currentChanged.connect(self._on_subtab_changed)
        layout.addWidget(self._tabview)

    def _build_genre_tab(self) -> QWidget:
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
        title = BodyLabel("Tag instrumental/music files with Discogs genre / style tags")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=lambda: self._show_help("genre")))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.genre_input_row = PathRow(
            paths.body, "Input folder",
            tip_text=TIPS["genre_input"],
            label_width=80,
        )
        self.genre_include_subfolders = CheckBox("Include subfolders")
        self.genre_include_subfolders.setChecked(True)
        self.genre_include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths.body.layout().addWidget(self.genre_include_subfolders)
        v.addWidget(paths)

        run_card = Section(inner, "Run mode")
        self.genre_run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
            tooltips={
                "batch": TIPS["run_mode_batch"],
                "per_file": TIPS["run_mode_per_file"],
            },
        )
        run_card.body.layout().addWidget(self.genre_run_mode)
        v.addWidget(run_card)

        style_card = Section(inner, "Tag style")
        self.genre_tag_style = _RadioRow(
            style_card.body,
            [("Combined  (GENRE=Rock/Surf)", "combined"), ("Split  (GENRE=Rock, STYLE=Surf)", "split")],
            "split",
            tooltips={
                "combined": TIPS["tag_style_combined"],
                "split": TIPS["tag_style_split"],
            },
        )
        style_card.body.layout().addWidget(self.genre_tag_style)
        v.addWidget(style_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.genre_write_meta = CheckBox("Write metadata tags")
        self.genre_write_meta.setChecked(True)
        self.genre_write_meta.setToolTip(TIPS["write_meta"])
        self.genre_skip_existing = CheckBox("Skip if already tagged")
        self.genre_skip_existing.setChecked(True)
        self.genre_skip_existing.setToolTip(TIPS["skip_existing"])
        opts_lay.addWidget(self.genre_write_meta)
        opts_lay.addWidget(self.genre_skip_existing)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _build_gender_tab(self) -> QWidget:
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
        title = BodyLabel("Tag vocal audio files with voice gender (female/male) and reverb (wet/dry)")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=lambda: self._show_help("gender")))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.gender_input_row = PathRow(
            paths.body, "Input folder",
            tip_text=TIPS["gender_input"],
            label_width=80,
        )
        self.gender_include_subfolders = CheckBox("Include subfolders")
        self.gender_include_subfolders.setChecked(True)
        self.gender_include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths.body.layout().addWidget(self.gender_include_subfolders)
        v.addWidget(paths)

        run_card = Section(inner, "Run mode")
        self.gender_run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
            tooltips={
                "batch": TIPS["run_mode_batch"],
                "per_file": TIPS["run_mode_per_file"],
            },
        )
        run_card.body.layout().addWidget(self.gender_run_mode)
        v.addWidget(run_card)

        field_card = Section(inner, "Write gender to")



        self.gender_tag_field = _RadioRow(
            field_card.body,
            [("Comment", "comment"), ("Gender (custom)", "gender")],
            "gender",
            tooltips={
                "comment": TIPS["tag_field_comment"],
                "gender": TIPS["tag_field_gender"],
            },
        )
        field_card.body.layout().addWidget(self.gender_tag_field)
        v.addWidget(field_card)

        rev_card = Section(inner, "Write reverb to")



        self.gender_reverb_mode = _RadioRow(
            rev_card.body,
            [("Combined  (COMMENT=female/wet)", "combined"), ("Split  (GENDER=female, REVERB=wet)", "split")],
            "split",
            tooltips={
                "combined": TIPS["reverb_mode_combined"],
                "split": TIPS["reverb_mode_split"],
            },
        )
        rev_card.body.layout().addWidget(self.gender_reverb_mode)
        v.addWidget(rev_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.gender_write_meta = CheckBox("Write metadata tags")
        self.gender_write_meta.setChecked(True)
        self.gender_write_meta.setToolTip(TIPS["write_meta"])
        self.gender_skip_existing = CheckBox("Skip if already tagged")
        self.gender_skip_existing.setChecked(True)
        self.gender_skip_existing.setToolTip(TIPS["skip_existing"])
        opts_lay.addWidget(self.gender_write_meta)
        opts_lay.addWidget(self.gender_skip_existing)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _build_vocal_tab(self) -> QWidget:
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
        title = BodyLabel(
            "Classify vocal content: Singing / Speech / Rapping / Humming / Choir"
        )
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=lambda: self._show_help("vocal")))
        header.addStretch(1)
        v.addLayout(header)

        paths = Section(inner, "Path")
        paths.body.layout().setSpacing(12)
        self.vocal_input_row = PathRow(
            paths.body, "Input folder",
            tip_text=TIPS["vocal_input"],
            label_width=80,
        )
        self.vocal_include_subfolders = CheckBox("Include subfolders")
        self.vocal_include_subfolders.setChecked(True)
        self.vocal_include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths.body.layout().addWidget(self.vocal_include_subfolders)
        v.addWidget(paths)

        self.vocal_segments = CheckBox()
        self.vocal_segments.setChecked(False)
        self.vocal_segments.hide()

        run_card = Section(inner, "Run mode")
        self.vocal_run_mode = _RadioRow(
            run_card.body,
            [("Batch (fast)", "batch"), ("Per-file (live results)", "per_file")],
            "batch",
            tooltips={
                "batch": TIPS["run_mode_batch"],
                "per_file": TIPS["run_mode_per_file"],
            },
        )
        run_card.body.layout().addWidget(self.vocal_run_mode)
        v.addWidget(run_card)

        field_card = Section(inner, "Write label to")
        self.vocal_tag_field = _RadioRow(
            field_card.body,
            [("Comment", "comment"), ("Vocal type (custom)", "vocal")],
            "vocal",
            tooltips={
                "comment": TIPS["vocal_tag_field_comment"],
                "vocal": TIPS["vocal_tag_field_vocal"],
            },
        )
        field_card.body.layout().addWidget(self.vocal_tag_field)
        v.addWidget(field_card)

        opts_card = Section(inner, "Tag options")
        opts_lay = opts_card.body.layout()
        m = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(m.left(), 14, m.right(), m.bottom())
        opts_lay.setSpacing(12)
        self.vocal_write_meta = CheckBox("Write metadata tags")
        self.vocal_write_meta.setChecked(True)
        self.vocal_write_meta.setToolTip(TIPS["vocal_write_meta"])
        self.vocal_skip_existing = CheckBox("Skip if already tagged")
        self.vocal_skip_existing.setChecked(True)
        self.vocal_skip_existing.setToolTip(TIPS["vocal_skip_existing"])
        opts_lay.addWidget(self.vocal_write_meta)
        opts_lay.addWidget(self.vocal_skip_existing)
        v.addWidget(opts_card)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _on_subtab_changed(self, _idx: int) -> None:
        if not hasattr(self, "_action_page"):
            return
        idx = self._tabview.currentIndex()
        self.genre_btn.setVisible(idx == 0)
        self.gender_btn.setVisible(idx == 1)
        self.vocal_btn.setVisible(idx == 2)
        # Vocal type input follows Gender path.
        if idx == 2:
            self._sync_vocal_from_gender()

    def _sync_vocal_from_gender(self, *_args) -> None:
        """Vocal type uses the same input folder as Gender."""
        if self._loading:
            return
        gender = self.gender_input_row.text().strip()
        if not gender:
            return
        if self.vocal_input_row.text().strip() != gender:
            self.vocal_input_row.set_text(gender)

    # ----- action bar -----

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._action_page = page
        # Same metrics as Classify: action_button height/font + content width (no fixed widths).
        self.genre_btn = action_button(
            "▶ Tag genre", on_click=self._start_genre, accent=True, tip=TIPS["tag_genre"]
        )
        self.gender_btn = action_button(
            "▶ Tag gender", on_click=self._start_gender, accent=True, tip=TIPS["tag_gender"]
        )
        self.vocal_btn = action_button(
            "▶ Tag vocal type", on_click=self._start_vocal, accent=True, tip=TIPS["tag_vocal"]
        )
        self.stop_btn = action_button("■ Stop", on_click=self._stop, tip=TIPS["stop"])
        self.stop_btn.setEnabled(False)
        page.add_button(self.genre_btn)
        page.add_button(self.gender_btn)
        page.add_button(self.vocal_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()
        self._on_subtab_changed(0)

    # ----- worker lifecycle -----

    @staticmethod
    def _enable_widgets(enabled: bool, *widgets) -> None:
        for w in widgets:
            if w is None:
                continue
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def _panel_settings(self, panel: str) -> tuple:
        if panel == "genre":
            return (
                self.genre_input_row,
                self.genre_include_subfolders,
                self.genre_run_mode,
                self.genre_tag_style,
                self.genre_write_meta,
                self.genre_skip_existing,
            )
        if panel == "gender":
            return (
                self.gender_input_row,
                self.gender_include_subfolders,
                self.gender_run_mode,
                self.gender_tag_field,
                self.gender_reverb_mode,
                self.gender_write_meta,
                self.gender_skip_existing,
            )
        if panel == "vocal":
            return (
                self.vocal_input_row,
                self.vocal_include_subfolders,
                self.vocal_segments,
                self.vocal_tag_field,
                self.vocal_write_meta,
                self.vocal_skip_existing,
            )
        return ()

    def _set_busy(self, busy: bool, status: str = "", *, panel: Optional[str] = None) -> None:
        """Lock only the started sub-tab's settings; other Genre/Gender/Vocal panels stay editable."""
        self._busy = busy
        self._busy_panel = panel if busy else None
        self._busy_since = time.monotonic() if busy else 0.0
        self.genre_btn.setEnabled(not busy)
        self.gender_btn.setEnabled(not busy)
        self.vocal_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        for name in ("genre", "gender", "vocal"):
            locked = busy and self._busy_panel == name
            self._enable_widgets(not locked, *self._panel_settings(name))
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def _on_worker_done(self, status: str) -> None:
        self._worker = None
        self._set_busy(False, status)

    def _stop(self) -> None:
        if self._worker is None:
            return
        # Ignore Enter/Return that was meant for the layout dialog and arrives
        # just after Stop becomes enabled.
        if self._busy_since and (time.monotonic() - self._busy_since) < 0.45:
            return
        self._worker.stop()
        self.request_log.emit("[stopping] ...", "warn")

    def _prepare_tag_files(
        self,
        input_dir: str,
        *,
        panel: TagPanel,
        include_subfolders: bool,
        title_prefix: str,
    ) -> tuple[str, str] | None:
        """Detect layout, prompt when ambiguous, return (files_list_path, log_summary)."""
        root = Path(input_dir).expanduser().resolve()
        scan_mode = scan_mode_from_recursive(include_subfolders)
        options = resolve_tag_input(
            self.window() or self,
            root,
            scan_mode,
            title_prefix=title_prefix,
        )
        if options is None:
            return None

        paths = collect_paths_for_panel(root, scan_mode, options, panel)
        if not paths:
            if panel == "genre":
                msg = (
                    "No instrumental-side files found.\n\n"
                    "Genre tags apply to instrumentals, samples, and the instrumental "
                    "stem in pairs (not vocals)."
                )
            else:
                msg = (
                    "No vocal files found.\n\n"
                    "Gender, reverb, and vocal-type tags apply to vocals and the vocal "
                    "stem in pairs (not instrumentals)."
                )
            show_info(self, PANEL_TITLE, msg)
            return None

        total_audio = sum(1 for _ in cb.iter_sdr_audio_files(root, scan_mode))
        skipped = max(0, total_audio - len(paths))
        summary = f"{len(paths):,} file(s) to tag · {layout_log_line(options)}"
        if skipped and options.layout == cb.SDR_LAYOUT_MIXED_FLAT:
            summary += f" · {skipped:,} skipped (no vocals/instrumental keyword)"

        self.request_log.emit(f"  Layout: {layout_log_line(options)}", "info")
        self.request_log.emit(f"  {len(paths):,} file(s) selected for tagging", "info")
        if skipped and options.layout == cb.SDR_LAYOUT_MIXED_FLAT:
            self.request_log.emit(
                f"  {skipped:,} file(s) skipped (unrecognized vocals/instrumental name)",
                "detail",
            )

        return str(write_files_list(paths)), summary

    def _start_genre(self) -> None:
        if self._busy:
            return
        input_dir = self.genre_input_row.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            show_info(self, PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self.request_clear_log.emit()
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("genre")
        self.request_log.emit("  Starting genre tagger:", "info")
        self.request_log.emit(f"  {input_dir}", "info")
        prepared = self._prepare_tag_files(
            input_dir,
            panel="genre",
            include_subfolders=self.genre_include_subfolders.isChecked(),
            title_prefix="Genre",
        )
        if prepared is None:
            return
        files_from, _summary = prepared
        worker = TaggerWorker(
            "genre", input_dir,
            batch_mode=self.genre_run_mode.value() == "batch",
            tag_style=self.genre_tag_style.value(),
            gender_field="comment",
            write_meta=self.genre_write_meta.isChecked(),
            csv_path="",
            include_subfolders=self.genre_include_subfolders.isChecked(),
            overwrite_tags=not self.genre_skip_existing.isChecked(),
            settings=self._settings,
            files_from=files_from,
            parent=self,
        )
        self._wire(worker)
        self._set_busy(True, "Tagging genre…", panel="genre")
        worker.start()

    def _start_gender(self) -> None:
        if self._busy:
            return
        input_dir = self.gender_input_row.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            show_info(self, PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self.request_clear_log.emit()
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("gender")
        self.request_log.emit("  Starting gender tagger:", "info")
        self.request_log.emit(f"  {input_dir}", "info")
        prepared = self._prepare_tag_files(
            input_dir,
            panel="gender",
            include_subfolders=self.gender_include_subfolders.isChecked(),
            title_prefix="Gender",
        )
        if prepared is None:
            return
        files_from, _summary = prepared
        worker = TaggerWorker(
            "gender", input_dir,
            batch_mode=self.gender_run_mode.value() == "batch",
            tag_style=self.gender_reverb_mode.value(),
            gender_field=self.gender_tag_field.value(),
            write_meta=self.gender_write_meta.isChecked(),
            csv_path="",
            include_subfolders=self.gender_include_subfolders.isChecked(),
            overwrite_tags=not self.gender_skip_existing.isChecked(),
            settings=self._settings,
            files_from=files_from,
            parent=self,
        )
        self._wire(worker)
        self._set_busy(True, "Tagging gender…", panel="gender")
        worker.start()

    def _start_vocal(self) -> None:
        if self._busy:
            return
        input_dir = self.vocal_input_row.text().strip()
        if not input_dir or not Path(input_dir).is_dir():
            show_info(self, PANEL_TITLE, "Input folder is missing or invalid.")
            return
        self.request_clear_log.emit()
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("vocal")
        self.request_log.emit("  Starting PANNs vocal-type tagger:", "info")
        self.request_log.emit(f"  {input_dir}", "info")
        self.request_log.emit(
            "  Focus: Singing · Speech · Rapping · Humming · Choir",
            "info",
        )
        prepared = self._prepare_tag_files(
            input_dir,
            panel="vocal",
            include_subfolders=self.vocal_include_subfolders.isChecked(),
            title_prefix="Vocal type",
        )
        if prepared is None:
            return
        files_from, _summary = prepared
        worker = PannsWorker(
            input_dir,
            include_subfolders=self.vocal_include_subfolders.isChecked(),
            write_meta=self.vocal_write_meta.isChecked(),
            overwrite_tags=not self.vocal_skip_existing.isChecked(),
            tag_field=self.vocal_tag_field.value(),
            segment_sec=2.0 if self.vocal_segments.isChecked() else 0.0,
            batch_mode=self.vocal_run_mode.value() == "batch",
            files_from=files_from,
            parent=self,
        )
        self._wire(worker)
        self._set_busy(True, "Tagging vocal type…", panel="vocal")
        worker.start()

    def _wire(self, worker: Union[TaggerWorker, PannsWorker]) -> None:
        self._worker = worker
        worker.log_line.connect(self.request_log)
        worker.progress.connect(self.request_progress)
        worker.processed.connect(self.request_gg_processed)
        worker.status.connect(self.request_status_text)
        worker.finished_ok.connect(self._on_worker_done)

    # ----- help -----

    def _show_help(self, mode: str) -> None:
        if mode == "vocal":
            help_dialog(
                self,
                title="Vocal type help",
                heading="Classify Singing / Speech / Rapping / Humming / Choir",
                intro=(
                    "PANNs Cnn14 (AudioSet) scores vocal content without fine-tuning. "
                    "The five focus classes are softmax-renormalized to shares that sum "
                    "to 100% (same idea as Classify RMS energy shares)."
                ),
                sections=[
                    ("Workflow", [
                        "1. Choose a folder (vocals root, pairs root, or mixed library).",
                        "2. Layout is detected like Classify — only vocal files are tagged.",
                        "3. When many files lack clear names, you may be asked to treat all "
                        "as vocals or all as pairs (same prompts as SI-SDR).",
                        "4. Pick run mode (Batch or Per-file) and tag options.",
                        "5. Click ▶ Tag vocal type and watch scores in the LOG panel.",
                    ]),
                    ("Options", [
                        "Comment stores the bare label (e.g. Singing). Vocal type stores "
                        "the same label in a custom VOCAL_TYPE field. Confidence percentages "
                        "appear in the LOG only, not in tags.",
                        "Skip if already tagged (default on) is resume-friendly.",
                    ]),
                    ("Setup", [
                        "Frozen build: run install-deps.bat beside STEM-organizer.exe "
                        "(installs panns_inference into site-packages\\). "
                        "From source: root install-deps.bat or panns_tagger\\install-deps.bat.",
                    ]),
                    ("Sources", [
                        '<a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">'
                        "PANNs Cnn14</a> pretrained on AudioSet (panns-inference). "
                        "First run downloads ~470 MB weights into panns_tagger\\models\\.",
                    ]),
                ],
            )
            return

        is_genre = mode == "genre"
        help_dialog(
            self,
            title="Genre help" if is_genre else "Gender help",
            heading=(
                "Tag instrumentals with genre & style"
                if is_genre
                else "Tag vocals with voice gender + dry/wet reverb"
            ),
            intro=(
                "Classify instrumental tracks and write Discogs-style GENRE / STYLE tags"
                if is_genre
                else "Estimate singing voice gender and dry/wet reverb, then write tags"
            ),
            sections=[
                ("Workflow", [
                    (
                        "1. Choose a folder (instrumental root, pairs root, or mixed library)."
                        if is_genre
                        else "1. Choose a folder (vocal root, pairs root, or mixed library)."
                    ),
                    (
                        "2. Layout is detected like Classify — only instrumental-side files "
                        "are tagged (instrumentals, bass/drums/other, or the instrumental stem "
                        "in pairs — not vocals)."
                        if is_genre
                        else "2. Layout is detected like Classify — only vocal files are tagged "
                        "(vocals stem in pairs, not instrumentals)."
                    ),
                    "3. When many files lack clear names, you may be asked to treat all as "
                    "pairs or all as one stem type (same prompts as SI-SDR).",
                    "4. Pick run mode (Batch or Per-file) and tag options.",
                    "5. Click ▶ Tag genre (or Tag gender) and watch progress in the LOG panel.",
                ]),
                ("Options", [
                    "Batch is faster (best with a GPU). Per-file prints each prediction live. "
                    "Combined writes one tag as Genre/Style (or gender/reverb). Split writes separate tags.",
                    "Tags are written to FLAC, MP3, M4A, and WAV (ID3 / Vorbis / MP4 atoms as appropriate). "
                    "A CSV export is always written.",
                    "Skip if already tagged (default on) is resume-friendly.",
                ]),
                ("Setup", [
                    "Frozen build: run install-deps.bat beside STEM-organizer.exe "
                    "(wheels go into site-packages\\; no nested genre_gender_tagger\\venv). "
                    "From source: run root install-deps.bat or genre_gender_tagger\\install-deps.bat.",
                ]),
                ("Sources", [
                    (
                        'Genre: <a href="https://huggingface.co/mtg-upf/discogs-maest-30s-pw-129e">'
                        "Hugging Face MAEST</a> "
                        "mtg-upf/discogs-maest-30s-pw-129e-519l (Discogs519)."
                        if is_genre
                        else (
                            'Gender: <a href="https://essentia.upf.edu/models.html#voice-gender">'
                            "Discogs-EffNet</a> (ONNX Runtime DirectML). "
                            "Reverb: in-house trained vocal mel-CNN."
                        )
                    ),
                ]),
            ],
        )

    # ----- settings -----

    def settings_snapshot(self) -> dict:
        return {
            "gg_genre_input_dir": display_path(self.genre_input_row.text()),
            "gg_genre_include_subfolders": bool(self.genre_include_subfolders.isChecked()),
            "gg_genre_batch_mode": self.genre_run_mode.value() == "batch",
            "gg_genre_tag_style": self.genre_tag_style.value(),
            "gg_genre_write_meta": bool(self.genre_write_meta.isChecked()),
            "gg_genre_skip_existing": bool(self.genre_skip_existing.isChecked()),
            "gg_gender_input_dir": display_path(self.gender_input_row.text()),
            "gg_gender_include_subfolders": bool(self.gender_include_subfolders.isChecked()),
            "gg_gender_batch_mode": self.gender_run_mode.value() == "batch",
            "gg_gender_tag_field": self.gender_tag_field.value(),
            "gg_gender_reverb_mode": self.gender_reverb_mode.value(),
            "gg_gender_write_meta": bool(self.gender_write_meta.isChecked()),
            "gg_gender_skip_existing": bool(self.gender_skip_existing.isChecked()),
            "gg_vocal_input_dir": display_path(self.vocal_input_row.text()),
            "gg_vocal_include_subfolders": bool(self.vocal_include_subfolders.isChecked()),
            "gg_vocal_batch_mode": self.vocal_run_mode.value() == "batch",
            "gg_vocal_segments": bool(self.vocal_segments.isChecked()),
            "gg_vocal_tag_field": self.vocal_tag_field.value(),
            "gg_vocal_write_meta": bool(self.vocal_write_meta.isChecked()),
            "gg_vocal_skip_existing": bool(self.vocal_skip_existing.isChecked()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("gg_genre_input_dir"):
                self.genre_input_row.set_text(d["gg_genre_input_dir"])
            self.genre_include_subfolders.setChecked(bool(d.get("gg_genre_include_subfolders", True)))
            self.genre_run_mode.set_value("batch" if d.get("gg_genre_batch_mode", True) else "per_file")
            self.genre_tag_style.set_value(d.get("gg_genre_tag_style", "split"))
            self.genre_write_meta.setChecked(bool(d.get("gg_genre_write_meta", True)))
            self.genre_skip_existing.setChecked(
                _load_skip_existing(d, "gg_genre_skip_existing", "gg_genre_overwrite_tags")
            )
            if d.get("gg_gender_input_dir"):
                self.gender_input_row.set_text(d["gg_gender_input_dir"])
            self.gender_include_subfolders.setChecked(bool(d.get("gg_gender_include_subfolders", True)))
            self.gender_run_mode.set_value("batch" if d.get("gg_gender_batch_mode", True) else "per_file")
            self.gender_tag_field.set_value(d.get("gg_gender_tag_field", "gender"))
            self.gender_reverb_mode.set_value(d.get("gg_gender_reverb_mode", "split"))
            self.gender_write_meta.setChecked(bool(d.get("gg_gender_write_meta", True)))
            self.gender_skip_existing.setChecked(
                _load_skip_existing(d, "gg_gender_skip_existing", "gg_gender_overwrite_tags")
            )
            if d.get("gg_vocal_input_dir"):
                self.vocal_input_row.set_text(d["gg_vocal_input_dir"])
            self.vocal_include_subfolders.setChecked(bool(d.get("gg_vocal_include_subfolders", True)))
            self.vocal_run_mode.set_value("batch" if d.get("gg_vocal_batch_mode", True) else "per_file")
            self.vocal_segments.setChecked(bool(d.get("gg_vocal_segments", False)))
            self.vocal_tag_field.set_value(d.get("gg_vocal_tag_field", "vocal"))
            self.vocal_write_meta.setChecked(bool(d.get("gg_vocal_write_meta", True)))
            self.vocal_skip_existing.setChecked(
                _load_skip_existing(d, "gg_vocal_skip_existing", "gg_vocal_overwrite_tags")
            )
            # Genre / Gender: fall back to Classify output when empty.
            # Vocal type: always follow Gender (not Classify).
            classify_out = (d.get("output_dir") or "").strip()
            if classify_out:
                if not self.genre_input_row.text().strip():
                    self.genre_input_row.set_text(classify_out)
                if not self.gender_input_row.text().strip():
                    self.gender_input_row.set_text(classify_out)
            gender_path = self.gender_input_row.text().strip()
            if gender_path:
                self.vocal_input_row.set_text(gender_path)
        finally:
            self._loading = False

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        self.gender_input_row.entry.textChanged.connect(self._sync_vocal_from_gender)
        for sig in (
            self.genre_input_row.entry.textChanged,
            self.genre_include_subfolders.toggled,
            self.genre_run_mode.valueChanged,
            self.genre_tag_style.valueChanged,
            self.genre_write_meta.toggled,
            self.genre_skip_existing.toggled,
            self.gender_input_row.entry.textChanged,
            self.gender_include_subfolders.toggled,
            self.gender_run_mode.valueChanged,
            self.gender_tag_field.valueChanged,
            self.gender_reverb_mode.valueChanged,
            self.gender_write_meta.toggled,
            self.gender_skip_existing.toggled,
            self.vocal_input_row.entry.textChanged,
            self.vocal_include_subfolders.toggled,
            self.vocal_run_mode.valueChanged,
            self.vocal_segments.toggled,
            self.vocal_tag_field.valueChanged,
            self.vocal_write_meta.toggled,
            self.vocal_skip_existing.toggled,
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
    tab = GenreGenderTab(settings)
    window.register_tab("Genre & Gender", tab)
    page = window.action_bar.make_page("Genre & Gender")
    tab.attach_action_bar(page, window)
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_status_text.connect(window.status_bar.set_status)
    tab.request_log.connect(window.append_log)
    tab.request_clear_log.connect(window.clear_log)
    tab.request_gg_processed.connect(window.update_gg_processed)

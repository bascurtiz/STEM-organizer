"""Classify tab — port of the inline Classify tab from stem_organizer_ui._build_ui.

Sections: header, PATHS (input/output/scan/naming), OUTPUT FILTERS (delete_if_short
+ min_duration, skip_existing, delete_if_incomplete), OPTIONS (model/stems/
quality/ambig + CUDA), CLASSIFICATION (nested RMS / SI-SDR tabwidget).

Action bar: Start / Stop / Play.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QSignalBlocker, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    CompactSpinBox,
    ScrollArea,
    SegmentedWidget,
)

import classify_backend as cb
from .. import theme
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import ask_yes_no, help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..widgets.slider_field import SliderField
from ..workers.base import BaseWorker
from ..workers.classify_worker import (
    ClassifyWorker,
    SdrClassifyWorker,
    SdrPreflightWorker,
)


# Action-bar / mode tips (field tips are set inline beside each control).
TIPS = {
    "start": "Begin classifying and mixing. The UI stays responsive during the run.",
    "stop": "Request a clean stop after the current folder finishes.",
    "play_stems": (
        "Open the stem preview player.\n"
        "Load a folder with bass/drums/other/vocals or instrumental/vocals to audition mixes."
    ),
    "start_sdr": (
        "Run SI-SDR quality check on organized stem folders.\n"
        "Each stem file is processed through Demucs individually and compared to the model output."
    ),
    "stop_sdr": "Request a clean stop after the current folder finishes.",
    "cls_rms": "Classify stems by Demucs RMS energy share per category.",
    "cls_sdr": "Score already-organized stems with SI-SDR and recycle low-quality files.",
    "sdr_threshold": "Minimum SI-SDR (dB) for this stem. Files scoring below are moved to the Recycle Bin.",
}

TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


class ClassifyTab(QWidget):
    """Hosts the Classify UI + owns the RMS/SDR workers."""

    request_status_running = Signal()
    request_status_idle = Signal(str)
    request_progress = Signal(float, object, int, int, str)
    request_log = Signal(str, str)
    request_sdr_log = Signal(str, float, float)
    request_open_player = Signal()

    def __init__(self, settings: SettingsStore) -> None:
        super().__init__()
        self._settings = settings
        self._worker: Optional[object] = None  # BaseWorker | SdrPreflightWorker
        self._worker_kind: Optional[str] = None  # "rms" | "sdr" | "sdr_preflight"
        self._rms_saw_done = False
        self._is_sdr_mode = False
        self._loading = False
        self._pending_sdr_params: Optional[dict] = None
        self._sdr_threshold_widgets: dict[str, SliderField] = {}
        self._sdr_thresholds: dict[str, int] = dict(cb.SDR_DEFAULT_THRESHOLDS)

        self._build_ui()
        self._bind_autosave()
        self.load_settings()

    # ----- UI build ----------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        # Horizontal inset lives on the scroll *content* (not outer), so a vertical
        # scrollbar cannot steal the right margin and leave content flush to the LOG.
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(4)

        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(ScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        self._inner_layout.setSpacing(theme.SECTION_GAP)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        # Header — CTk text_dim + a bit of air above/below
        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel("Classifies stems, mixes originals into 2/4-stem folders")
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        info = InfoIcon(inner, on_click=self._show_about)
        header.addWidget(info)
        header.addStretch(1)
        self._inner_layout.addLayout(header)

        # PATHS — dense rows; spacing matches other cards (equal section gaps)
        paths = Section(inner, "Paths")
        paths.body.layout().setSpacing(5)
        self.input_row = PathRow(
            paths.body, "Input",
            tip_text="Folder containing your multi-track songs (one sub-folder per song).",
            label_width=56,
        )
        self.output_row = PathRow(
            paths.body, "Output",
            tip_text="Where the organized stems will be written.",
            label_width=56,
        )

        self.scan_combo = ComboBox()
        for k in cb.SCAN_MODES:
            self.scan_combo.addItem(k)
        self.scan_combo.setToolTip(
            "Each subfolder: one level of folders under input.\n"
            "Each leaf folder: every folder containing audio is scanned."
        )
        self.naming_combo = ComboBox()
        for k in cb.NAMING_MODES:
            self.naming_combo.addItem(k)
        self.naming_combo.setToolTip("How the output folder is named.")
        paths.body.layout().addLayout(
            self._labeled_combo_row("Scan", self.scan_combo)
        )
        paths.body.layout().addLayout(
            self._labeled_combo_row("Naming", self.naming_combo)
        )
        self._inner_layout.addWidget(paths)

        # OUTPUT FILTERS — three identical-height rows + fixed spacing (even gaps)
        filters = Section(inner, "Output filters")
        filters_wrap = QWidget()
        filters_body = QVBoxLayout(filters_wrap)
        filters_body.setContentsMargins(0, 0, 0, 0)
        filters_body.setSpacing(8)
        filters_body.setAlignment(Qt.AlignTop)

        self.delete_if_short = CheckBox("Delete output shorter than")
        self.delete_if_short.setChecked(True)  # CTk default ON
        self.delete_if_short.setToolTip("Recycle output folders whose duration is below the minimum.")
        self.min_duration_sec = CompactSpinBox()
        self.min_duration_sec.setObjectName("FilterDurationSpin")
        self.min_duration_sec.setRange(1, 3600)
        self.min_duration_sec.setValue(8)
        self.min_duration_sec.setSuffix(" s")
        self.min_duration_sec.setFixedWidth(theme.COMPACT_SPIN_WIDTH)
        self.min_duration_sec.setToolTip("Minimum duration, in seconds.")
        theme.sync_compact_spin_button(self.min_duration_sec, theme.COMPACT_SPIN_HEIGHT)
        theme._install_compact_spin_resize_sync(self.min_duration_sec)

        self.delete_if_incomplete = CheckBox("Delete output missing expected stems")
        self.delete_if_incomplete.setToolTip("Recycle output folders that don't contain every expected stem.")

        self.skip_existing = CheckBox("Skip songs that already have output stems")
        self.skip_existing.setChecked(True)  # CTk default ON
        self.skip_existing.setToolTip("Resume mode — skip any song whose output folder already exists.")

        _row_h = theme.COMPACT_SPIN_HEIGHT
        for _cb in (self.delete_if_short, self.delete_if_incomplete, self.skip_existing):
            _cb.setFixedHeight(_row_h)
            _cb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        def _filter_row(*widgets: QWidget) -> QWidget:
            host = QWidget()
            host.setFixedHeight(_row_h)
            host.setMinimumHeight(_row_h)
            host.setMaximumHeight(_row_h)
            host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            for w in widgets:
                row.addWidget(w, 0, Qt.AlignVCenter)
            row.addStretch(1)
            return host

        filters_body.addWidget(_filter_row(self.delete_if_short, self.min_duration_sec))
        filters_body.addWidget(_filter_row(self.delete_if_incomplete))
        filters_body.addWidget(_filter_row(self.skip_existing))
        filters_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        filters.body.layout().setSpacing(0)
        filters.body.layout().setAlignment(Qt.AlignTop)
        filters.body.layout().addWidget(filters_wrap)
        self._inner_layout.addWidget(filters)

        # OPTIONS — CTk 2-column grid
        opts = Section(inner, "Options")
        opts_grid_host = QWidget()
        opts_grid = QGridLayout(opts_grid_host)
        opts_grid.setContentsMargins(0, 0, 0, 0)
        opts_grid.setHorizontalSpacing(10)
        opts_grid.setVerticalSpacing(4)
        opts_grid.setColumnStretch(1, 1)
        opts_grid.setColumnStretch(3, 1)

        self.model_combo = ComboBox()
        for k in cb.MODELS:
            self.model_combo.addItem(k)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_combo.setToolTip(self._model_tooltip(self.model_combo.currentText()))
        self.stem_combo = ComboBox()
        for k in cb.STEM_MODES:
            self.stem_combo.addItem(k)
        self.stem_combo.setToolTip(
            "2: vocals + instrumental.\n"
            "4: bass / drums / other / vocals."
        )
        self.stem_combo.currentTextChanged.connect(lambda _t: self._rebuild_sdr_thresholds())
        self.quality_combo = ComboBox()
        for k in cb.QUALITY_PRESETS:
            self.quality_combo.addItem(k)
        self.quality_combo.setToolTip(
            "FLAC is lossless and smaller than WAV.\nWAV 32-bit float supports the mixture file."
        )
        self.ambig_combo = ComboBox()
        for k in cb.AMBIG_MODES:
            self.ambig_combo.addItem(k)
        self.ambig_combo.setToolTip(
            "What to do when a stem can't be confidently assigned to one category."
        )
        self.use_cuda = CheckBox("Use CUDA (GPU)")
        self._configure_cuda_checkbox()

        self._place_combo(opts_grid, 0, 0, "Model", self.model_combo)
        self._place_combo(opts_grid, 0, 2, "Stems", self.stem_combo)
        # qfluent ComboBox.setText() calls adjustSize(); pin width to the longest
        # label so 2↔4 selection does not reflow the Options grid. Use theme
        # body font (polish may apply it after construct).
        _stem_pad = 40  # left/right padding + dropdown chevron
        _stem_fm = QFontMetrics(theme.F_BODY)
        _stem_w = max(_stem_fm.horizontalAdvance(t) for t in cb.STEM_MODES) + _stem_pad
        self.stem_combo.setFixedWidth(_stem_w)
        # Filter stem choices to the default model's allowed set (htdemucs → 2/4).
        self._refresh_stem_combo_for_model(self.model_combo.currentText())
        self._place_combo(opts_grid, 1, 0, "Quality", self.quality_combo)
        opts_grid.addWidget(self.use_cuda, 1, 2, 1, 2, Qt.AlignLeft | Qt.AlignVCenter)
        self._place_combo(opts_grid, 2, 0, "Ambiguous", self.ambig_combo)

        opts.body.layout().addWidget(opts_grid_host)
        self._inner_layout.addWidget(opts)

        # CLASSIFICATION — breathe a bit (was over-tight after stack-height sync)
        cls_section = Section(inner, "Classification")
        self._cls_seg = SegmentedWidget()
        self._cls_seg.addItem("rms", "RMS")
        self._cls_seg.addItem("sdr", "SI-SDR")
        theme.configure_segmented_widget(self._cls_seg)
        for key, tip in (("rms", TIPS["cls_rms"]), ("sdr", TIPS["cls_sdr"])):
            item = self._cls_seg.widget(key) if hasattr(self._cls_seg, "widget") else None
            if item is not None:
                item.setToolTip(tip)
            else:
                # Fallback: tip the whole segmented control once
                self._cls_seg.setToolTip(TIPS["cls_rms"] + "\n\n" + TIPS["cls_sdr"])
                break
        self._cls_stack = QStackedWidget()
        self._cls_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cls_notebook = self._cls_stack  # keep name for _rebuild_sdr_thresholds
        self._cls_seg.currentItemChanged.connect(self._on_cls_seg_changed)

        rms_tab = QWidget()
        rms_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rms_lay = QVBoxLayout(rms_tab)
        # Shared top inset so RMS / SI-SDR content share the same start line under the tabs
        _cls_page_top = 12
        rms_lay.setContentsMargins(0, _cls_page_top, 0, 0)
        rms_lay.setSpacing(6)
        rms_lay.setAlignment(Qt.AlignTop)

        self.threshold_slider = SliderField(
            rms_tab, "Confidence",
            minimum=0, maximum=100, value=40,
            format_value=lambda v: f"{v/100:.0%}",
            label_width=78,
            readout_width=40,
        )
        self.threshold_slider.setToolTip(
            "Minimum top-category RMS share required to keep a stem (e.g. 40%)."
        )
        self.margin_slider = SliderField(
            rms_tab, "Min. margin",
            minimum=0, maximum=100, value=20,
            format_value=lambda v: f"{v/100:.0%}",
            label_width=78,
            readout_width=40,
        )
        self.margin_slider.setToolTip("Minimum gap between the top category and the runner-up.")
        rms_lay.addWidget(self.threshold_slider)
        rms_lay.addWidget(self.margin_slider)

        # Bottom: equal-height rows, airy spacing; Batch size right-aligned under slider %
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 4, 0, 0)
        bottom_lay.setSpacing(10)

        self.dedup = CheckBox("De-dupe identical stems")
        self.dedup.setToolTip("Detect near-identical stems (phase-inverted null test) and keep one.")
        self.peak_norm = CheckBox("Peak normalize to -1 dBFS")
        self.peak_norm.setChecked(True)  # CTk default ON
        self.peak_norm.setToolTip("Scale each category mix so its peak is -1 dBFS.")
        self.make_mixture = CheckBox("Write mixture (WAV only)")
        self.make_mixture.setToolTip("Also write a mixture.wav that sums all categorized stems.")

        self.batch_spin = CompactSpinBox()
        self.batch_spin.setObjectName("FilterDurationSpin")
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        self.batch_spin.setFixedWidth(theme.COMPACT_SPIN_WIDTH)
        theme.sync_compact_spin_button(self.batch_spin)
        theme._install_compact_spin_resize_sync(self.batch_spin)
        self.batch_spin.setToolTip(
            theme.format_tooltip(
                "How many stems Demucs processes in one forward pass. "
                "Lower if you hit CUDA OOM."
            )
        )

        _row_h = theme.COMPACT_SPIN_HEIGHT
        for _cb in (self.dedup, self.peak_norm, self.make_mixture):
            _cb.setFixedHeight(_row_h)

        def _rms_check_row(checkbox: QWidget, trailing: QWidget | None = None) -> QWidget:
            host = QWidget()
            host.setFixedHeight(_row_h)
            row = QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(checkbox, 0, Qt.AlignVCenter)
            row.addStretch(1)
            if trailing is not None:
                row.addWidget(trailing, 0, Qt.AlignVCenter)
            return host

        batch_host = QWidget()
        batch_lay = QHBoxLayout(batch_host)
        batch_lay.setContentsMargins(0, 0, 0, 0)
        batch_lay.setSpacing(8)
        batch_lay.addWidget(BodyLabel("Batch size"), 0, Qt.AlignVCenter)
        batch_lay.addWidget(self.batch_spin, 0, Qt.AlignVCenter)

        bottom_lay.addWidget(_rms_check_row(self.dedup, batch_host))
        bottom_lay.addWidget(_rms_check_row(self.peak_norm))
        bottom_lay.addWidget(_rms_check_row(self.make_mixture))
        rms_lay.addWidget(bottom)
        rms_lay.addStretch(1)

        # SI-SDR tab
        sdr_tab = QWidget()
        sdr_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sdr_layout = QVBoxLayout(sdr_tab)
        sdr_layout.setContentsMargins(0, _cls_page_top, 0, 0)
        sdr_layout.setSpacing(6)
        sdr_layout.setAlignment(Qt.AlignTop)
        # Match Confidence row height so the intro line sits on the same band as RMS
        sdr_intro = BodyLabel("SI-SDR threshold per stem (dB): delete output below threshold.")
        sdr_intro.setMinimumHeight(max(22, self.threshold_slider.sizeHint().height()))
        sdr_intro.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        sdr_layout.addWidget(sdr_intro)
        self._sdr_thresholds_holder = sdr_layout
        self.sdr_delete_folder = CheckBox("Delete entire folder when any stem fails")
        self.sdr_delete_folder.setChecked(True)
        self.sdr_delete_folder.setToolTip(
            theme.format_tooltip(
                "Off: delete only the failing stem file. "
                "On: delete the whole output folder."
            )
        )
        self.sdr_write_tags = CheckBox("Write SI-SDR to SDR metadata tag")
        self.sdr_write_tags.setChecked(True)
        self.sdr_write_tags.setToolTip(
            theme.format_tooltip(
                "Write each stem's SI-SDR score (dB) into the SDR tag "
                "(FLAC Vorbis / ID3 TXXX / M4A). SDR tag is read by Charts."
            )
        )

        self._cls_stack.addWidget(rms_tab)
        self._cls_stack.addWidget(sdr_tab)
        # Rebuild after stack pages exist
        self._rebuild_sdr_thresholds()
        sdr_opts = QHBoxLayout()
        sdr_opts.setContentsMargins(0, 0, 0, 0)
        sdr_opts.setSpacing(16)
        sdr_opts.addWidget(self.sdr_delete_folder, 0, Qt.AlignVCenter)
        sdr_opts.addWidget(self.sdr_write_tags, 0, Qt.AlignVCenter)
        sdr_opts.addStretch(1)
        sdr_layout.addLayout(sdr_opts)
        sdr_layout.addStretch(1)

        cls_body = cls_section.body.layout()
        cls_body.setSpacing(6)
        cls_body.setContentsMargins(10, 8, 10, 8)
        cls_body.setAlignment(Qt.AlignTop)
        cls_body.addWidget(self._cls_seg, 0, Qt.AlignTop)
        # Stack grows with the card so leftover height becomes breathing room
        self._cls_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cls_body.addWidget(self._cls_stack, stretch=1)
        self._cls_seg.setCurrentItem("rms")
        # Floor height to tallest page so RMS↔SI-SDR doesn't jump; allow grow for align-with-log
        self._sync_cls_stack_height()
        QTimer.singleShot(0, self._sync_cls_stack_height)

        # Classification fills leftover left-column height so its bottom matches the log view
        cls_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cls_section.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Extra height goes into the card body (not below the bordered panel)
        cls_section.layout().setStretchFactor(cls_section.body, 1)
        self._inner_layout.addWidget(cls_section, stretch=1)

    def _sync_cls_stack_height(self) -> None:
        """Keep stack at least as tall as the SI-SDR page; allow grow for card stretch.

        Hidden stack pages often report a bad sizeHint until given a real width;
        constrain width temporarily so the first paint matches later visits.
        """
        w = max(self._cls_stack.width(), self._cls_stack.sizeHint().width(), 400)
        h = 0
        for i in range(self._cls_stack.count()):
            page = self._cls_stack.widget(i)
            if page is None:
                continue
            prev_min, prev_max = page.minimumWidth(), page.maximumWidth()
            page.setFixedWidth(w)
            page.ensurePolished()
            lay = page.layout()
            if lay is not None:
                lay.invalidate()
                lay.activate()
                h = max(h, lay.sizeHint().height())
            h = max(h, page.sizeHint().height(), page.minimumSizeHint().height())
            page.setMinimumWidth(prev_min)
            page.setMaximumWidth(prev_max if prev_max > 0 else 16777215)
        if h > 0:
            self._cls_stack.setMinimumHeight(h)
            # Clear any prior fixed cap so the card can breathe / align with the log
            self._cls_stack.setMaximumHeight(16777215)

    @staticmethod
    def _labeled_combo_row(label: str, combo: ComboBox) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        lbl = BodyLabel(label)
        lbl.setFixedWidth(56)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setFixedHeight(theme.PATH_FIELD_HEIGHT)
        row.addWidget(lbl)
        row.addWidget(combo, stretch=1)
        return row

    @staticmethod
    def _place_combo(grid: QGridLayout, row: int, col: int, label: str, combo: ComboBox) -> None:
        lbl = BodyLabel(label)
        lbl.setFixedWidth(72 if col == 0 else 48)
        combo.setFixedHeight(theme.PATH_FIELD_HEIGHT)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid.addWidget(lbl, row, col, Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(combo, row, col + 1)

    def _configure_cuda_checkbox(self) -> None:
        """Enable GPU checkbox — CUDA preferred, then DirectML, else CPU."""
        has_dml = False
        has_cuda = False
        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
            has_dml = "DmlExecutionProvider" in available
        except Exception:
            available = set()
        try:
            from ort_util import cuda_ep_usable

            has_cuda = bool(cuda_ep_usable())
        except Exception:
            has_cuda = False

        self._cuda_enabled = bool(has_cuda or has_dml)
        if has_cuda:
            self.use_cuda.setText("Use GPU (CUDA)")
            self.use_cuda.setChecked(True)
            self.use_cuda.setEnabled(True)
            self.use_cuda.setToolTip(
                theme.format_tooltip(
                    "Classify separators + taggers via ONNX Runtime CUDA EP.\n"
                    "Requires a real NVIDIA GPU + cuDNN (bundled).\n"
                    "HTDemucs uses CUDA or CPU (DirectML not used).\n"
                    "Falls back to CPU if CUDA/cuDNN is unavailable."
                )
            )
            return
        if has_dml:
            self.use_cuda.setText("Use GPU (DirectML)")
            self.use_cuda.setChecked(True)
            self.use_cuda.setEnabled(True)
            self.use_cuda.setToolTip(
                theme.format_tooltip(
                    "ONNX Runtime DirectML (vendor-neutral GPU).\n"
                    "HTDemucs uses CUDA or CPU (DirectML historically rejected)."
                )
            )
            return

        self.use_cuda.setText("Use GPU   ·   unavailable")
        self.use_cuda.setChecked(False)
        self.use_cuda.setEnabled(False)
        self.use_cuda.setToolTip(
            theme.format_tooltip(
                "No CUDA or DirectML EP available in this ONNX Runtime build.\n"
                "Processing will use CPU."
            )
        )

    def _on_cls_seg_changed(self, key: str) -> None:
        idx = 1 if key == "sdr" else 0
        self._cls_stack.setCurrentIndex(idx)
        self._sync_cls_stack_height()
        self._on_subtab_changed(idx)
        self._schedule_save()

    def _allowed_stem_modes(self, model_label: str) -> list[str]:
        """Stem-mode labels selectable for a given model.

        HTDemucs produces 4 stems → 2/4 only (no guitar/piano).
        Vocal CNN6 classifies vocal/instrumental → 2-stem only.
        """
        all_modes = list(cb.STEM_MODES.keys())
        model_id = cb.MODELS.get(model_label, '')
        if model_id == 'vocal_cnn6':
            return [m for m in all_modes if m.startswith('2 ')]
        return [m for m in all_modes if not m.startswith('6 ')]

    def _model_tooltip(self, model_label: str) -> str:
        model_id = cb.MODELS.get(model_label, '')
        if model_id == 'vocal_cnn6':
            return theme.format_tooltip(
                "Vocal CNN6 — lightweight binary classifier (4.9M params). "
                "Classifies stems as vocal or instrumental in one fast forward pass "
                "(~10 ms per clip on GPU), no stem separation needed. "
                "2-stem mode only. For SI-SDR quality scoring, switch to HTDemucs "
                "or another separator after RMS classification completes."
            )
        return theme.format_tooltip(
            "htdemucs — Hybrid Transformer Demucs (StemSplit ONNX). "
            "~7.8 s chunks, CUDA EP (DirectML not used — VRAM/accuracy). "
            "4 stems: drums/bass/other/vocals."
        )

    def _refresh_stem_combo_for_model(self, model_label: str) -> None:
        """Restrict the Stems dropdown to the model's allowed modes.

        If the current selection is no longer allowed (e.g. 6-stem under
        htdemucs), fall back to the closest valid mode (6 → 4).
        """
        allowed = self._allowed_stem_modes(model_label)
        current = self.stem_combo.currentText()
        with QSignalBlocker(self.stem_combo):
            self.stem_combo.clear()
            for m in allowed:
                self.stem_combo.addItem(m)
        if current in allowed:
            self.stem_combo.setCurrentText(current)
        elif current.startswith('6 ') and any(m.startswith('4 ') for m in allowed):
            self.stem_combo.setCurrentText(next(m for m in allowed if m.startswith('4 ')))
        elif allowed:
            self.stem_combo.setCurrentText(allowed[0])
        self._rebuild_sdr_thresholds()

    def _on_model_changed(self, *_args) -> None:
        """React to Model dropdown: update tooltip + filter Stems choices."""
        label = self.model_combo.currentText()
        self.model_combo.setToolTip(self._model_tooltip(label))
        self._refresh_stem_combo_for_model(label)

    def _rebuild_sdr_thresholds(self) -> None:
        """Rebuild SI-SDR sliders for the categories of the selected Stems mode."""
        if getattr(self, 'cls_notebook', None) is None:
            return  # not constructed yet (early model-change during _setup_ui)
        sdr_tab = self.cls_notebook.widget(1)
        if sdr_tab is None:
            return
        layout = sdr_tab.layout()
        for w in list(self._sdr_threshold_widgets.values()):
            layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._sdr_threshold_widgets.clear()

        mode = self.stem_combo.currentText()
        mode_cfg = cb.STEM_MODES.get(mode) or next(iter(cb.STEM_MODES.values()))
        categories = mode_cfg["categories"]

        insert_at = 1  # after the intro label
        for cat in categories:
            default = int(
                self._sdr_thresholds.get(cat, cb.SDR_DEFAULT_THRESHOLDS.get(cat, 30))
            )
            slider = SliderField(
                sdr_tab, cat.capitalize(),
                minimum=0, maximum=40, value=default,
                format_value=lambda v, _c=cat: f"{v} dB",
            )
            slider.setToolTip(TIPS["sdr_threshold"])
            slider.valueChanged.connect(lambda v, c=cat: self._on_sdr_threshold_changed(c, v))
            layout.insertWidget(insert_at, slider)
            self._sdr_threshold_widgets[cat] = slider
            insert_at += 1
        self._sync_cls_stack_height()
        QTimer.singleShot(0, self._sync_cls_stack_height)

    def _on_sdr_threshold_changed(self, cat: str, value: int) -> None:
        self._sdr_thresholds[cat] = int(value)
        if not self._loading:
            self._settings.set("sdr_thresholds", dict(self._sdr_thresholds))
            self._settings.flush()

    def _on_subtab_changed(self, idx: int) -> None:
        self._is_sdr_mode = idx == 1
        if hasattr(self, "_action_page"):
            self._update_action_buttons()

    # ----- action bar --------------------------------------------------

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._action_page = page
        self.start_btn = action_button(
            "▶ Start RMS", on_click=self._on_start, accent=True, tip=TIPS["start"]
        )
        self.stop_btn = action_button(
            "■ Stop", on_click=self._on_stop, tip=TIPS["stop"]
        )
        self.stop_btn.setEnabled(False)
        self.play_btn = action_button(
            "♫ Play",
            on_click=lambda: self.request_open_player.emit(),
            tip=TIPS["play_stems"],
        )
        page.add_button(self.start_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()
        page.add_button(self.play_btn)
        self._update_action_buttons()

    def player_library_root(self) -> str:
        """Folder whose immediate children are song folders."""
        return self.output_row.text().strip()

    def _update_action_buttons(self) -> None:
        if self._is_sdr_mode:
            self.start_btn.setText("▶ Start SI-SDR")
            self.start_btn.setToolTip(TIPS["start_sdr"])
            self.stop_btn.setToolTip(TIPS["stop_sdr"])
        else:
            self.start_btn.setText("▶ Start RMS")
            self.start_btn.setToolTip(TIPS["start"])
            self.stop_btn.setToolTip(TIPS["stop"])

    # ----- worker lifecycle -------------------------------------------

    def _on_start(self) -> None:
        if self._worker is not None:
            return
        if self._is_sdr_mode:
            self._start_sdr()
        else:
            self._start_rms()

    def _start_rms(self) -> None:
        params = self._collect_params_rms()
        if params is None:
            return
        self._worker_kind = "rms"
        self._rms_saw_done = False
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("classify")
        # Immediate feedback before the worker thread spins up (CUDA probe /
        # model load can take seconds before the first backend log line).
        self.request_log.emit("  Starting…", "")
        self._worker = ClassifyWorker(params, parent=self)
        self._wire_worker(self._worker)
        self.set_running(True)
        self._worker.start()

    def _start_sdr(self) -> None:
        target_dir = self.output_row.text().strip()
        if not target_dir:
            self.request_log.emit("[error] Select an output folder first.", "err")
            return
        model_id = cb.MODELS.get(self.model_combo.currentText(), '')
        if model_id == 'vocal_cnn6':
            self.request_log.emit(
                "[error] Vocal CNN6 is a classifier, not a separator. "
                "Switch Model to HTDemucs for SI-SDR.",
                "err",
            )
            return
        params = {
            "target_dir": target_dir,
            "use_cuda": self.use_cuda.isChecked(),
            "model_id": cb.MODELS[self.model_combo.currentText()],
            "stem_mode": self.stem_combo.currentText(),
            "scan_mode": cb.SCAN_MODES[self.scan_combo.currentText()],
            "sdr_thresholds": dict(self._sdr_thresholds),
            "sdr_delete_folder": self.sdr_delete_folder.isChecked(),
            "write_sdr_tags": self.sdr_write_tags.isChecked(),
        }
        # Heavy tree walks used to run on the GUI thread here and froze Start
        # (Windows busy cursor + Not Responding title-bar line). Scan off-thread
        # first; dialogs stay on the main thread after the scan returns.
        self._pending_sdr_params = params
        self._worker_kind = "sdr_preflight"
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            win.set_log_export_prefix("sdr")
        self.request_log.emit("  Scanning for SI-SDR…", "")
        preflight = SdrPreflightWorker(params, parent=self)
        preflight.log_line.connect(self._forward_worker_log)
        preflight.finished_result.connect(self._on_sdr_preflight_done)
        self._worker = preflight
        self.set_running(True)
        preflight.start()

    def _on_sdr_preflight_done(self, result: object) -> None:
        """Apply process-all dialogs (fast) then launch the real SI-SDR worker."""
        if self._worker_kind != "sdr_preflight":
            return
        self._worker = None
        self._worker_kind = None
        params = self._pending_sdr_params
        self._pending_sdr_params = None
        if params is None or result is None:
            self.set_running(False)
            if result is None:
                self.request_log.emit("  [stopping] SI-SDR scan cancelled.", "warn")
            return
        if not isinstance(result, dict):
            self.set_running(False)
            self.request_log.emit("[error] SI-SDR scan returned unexpected data.", "err")
            return
        self._apply_sdr_process_all_prompts(params, result)
        self._worker_kind = "sdr"
        self.request_log.emit("  Starting…", "")
        self._worker = SdrClassifyWorker(params, parent=self)
        self._wire_worker(self._worker)
        # Already in running UI state from preflight.
        self._worker.start()

    def _apply_sdr_process_all_prompts(self, params: dict, preflight: dict) -> None:
        """Ask process-all questions using scan results gathered off-thread."""
        mode_cfg = cb.STEM_MODES[cb.resolve_stem_mode(params["stem_mode"])]
        preferred = mode_cfg["categories"]
        parent = self.window() or self

        # 2-stem: many song folders with exactly 2 files → offer process-all as pairs.
        if set(preferred) == {"instrumental", "vocals"}:
            pair_hint = preflight.get("pair_hint")
            if pair_hint and pair_hint.get("should_ask_process_all"):
                if ask_yes_no(
                    parent,
                    "SI-SDR · all as pairs?",
                    cb.pair_folder_process_all_message(pair_hint),
                    yes_text="Yes, all pairs",
                    no_text="Keywords only",
                ):
                    params["sdr_categories"] = ("instrumental", "vocals")
                    params["sdr_layout"] = cb.SDR_LAYOUT_MUSDB
                    params["sdr_pair_process_all"] = True
                    return
                # Keywords only: still use improved name matching (no force).

        layout = preflight.get("layout")
        # Loose-file layouts (and unknown flat dumps) can leave unmarked tracks behind.
        if layout not in (
            cb.SDR_LAYOUT_SINGLE_FLAT,
            cb.SDR_LAYOUT_MIXED_FLAT,
            None,
        ):
            return

        candidates = [
            h
            for h in (preflight.get("flat_hints") or [])
            if h and h.get("should_ask_process_all")
        ]
        if not candidates:
            return
        hint = max(candidates, key=lambda h: int(h.get("keyword_matches") or 0))
        kind = str(hint["kind"])
        title = (
            "SI-SDR · all as vocals?"
            if kind == "vocals"
            else "SI-SDR · all as instrumental?"
        )
        if ask_yes_no(
            parent,
            title,
            cb.single_stem_process_all_message(hint),
            yes_text="Yes, all files",
            no_text="Keywords only",
        ):
            params["sdr_categories"] = (kind,)
            params["sdr_layout"] = cb.SDR_LAYOUT_SINGLE_FLAT
            params["sdr_flat_process_all"] = True
            params["sdr_user_picked_category"] = True

    def _wire_worker(self, worker: BaseWorker) -> None:
        worker.log_line.connect(self._forward_worker_log)
        worker.progress.connect(self.request_progress)
        worker.sdr_line.connect(self.request_sdr_log)
        worker.finished_ok.connect(self._on_worker_done)

    def _forward_worker_log(self, text: str, tag: str = "") -> None:
        """Forward worker log lines; remember RMS completion for the SI-SDR offer."""
        if self._worker_kind == "rms" and text.strip() in ("DONE", "Done."):
            self._rms_saw_done = True
        self.request_log.emit(text, tag)

    def _on_worker_done(self) -> None:
        kind = self._worker_kind
        # Preflight uses finished_result, not finished_ok — ignore stray signals.
        if kind == "sdr_preflight":
            return
        saw_rms_done = self._rms_saw_done
        self._worker = None
        self._worker_kind = None
        self._rms_saw_done = False
        self.set_running(False)
        if kind == "rms" and saw_rms_done:
            # Defer so Start/Stop re-enable before the modal blocks the UI thread
            QTimer.singleShot(0, self._offer_sdr_after_rms)

    def _offer_sdr_after_rms(self) -> None:
        """After a successful RMS run, switch to SI-SDR then prompt (CTk order)."""
        # Switch first so the tab is ready even while the info dialog is up.
        self._switch_to_sdr_tab()
        parent = self.window() or self
        show_info(
            parent,
            "Calculate SI-SDR?",
            "RMS classification is complete.\n\n"
            "Check the SI-SDR thresholds and settings before you hit Start SI-SDR.",
        )

    def _switch_to_sdr_tab(self) -> None:
        """Select SI-SDR classification page and refresh Start / Stop labels."""
        self._cls_seg.setCurrentItem("sdr")
        # setCurrentItem may no-op if already selected; always sync stack + actions
        self._on_cls_seg_changed("sdr")
        self.set_running(False)

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        self._worker.stop()
        self.request_log.emit("[stopping] ...", "warn")

    def set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (
            self.input_row, self.output_row, self.scan_combo, self.naming_combo,
            self.model_combo, self.stem_combo, self.quality_combo, self.ambig_combo,
            self.threshold_slider, self.margin_slider, self.batch_spin,
            self.dedup, self.peak_norm, self.make_mixture,
            self.delete_if_short, self.min_duration_sec, self.skip_existing, self.delete_if_incomplete,
            self.sdr_delete_folder,
            self.sdr_write_tags,
            *self._sdr_threshold_widgets.values(),
        ):
            try:
                w.setEnabled(not running)
            except Exception:
                pass
        # Keep CUDA greyed out when no usable GPU — do not re-enable after a run.
        try:
            self.use_cuda.setEnabled(bool(getattr(self, "_cuda_enabled", False)) and not running)
        except Exception:
            pass
        if running:
            self.request_status_running.emit()
        else:
            self.request_status_idle.emit("Idle")

    def _collect_params_rms(self):
        in_dir = self.input_row.text().strip()
        out_dir = self.output_row.text().strip()
        if not in_dir:
            self.request_log.emit("[error] Select an input folder.", "err")
            return None
        if not out_dir:
            self.request_log.emit("[error] Select an output folder.", "err")
            return None
        return {
            "input_dir": in_dir,
            "output_dir": out_dir,
            "use_cuda": self.use_cuda.isChecked(),
            "model_id": cb.MODELS[self.model_combo.currentText()],
            "stem_mode": self.stem_combo.currentText(),
            "quality": self.quality_combo.currentText(),
            "threshold": self.threshold_slider.value() / 100.0,
            "min_margin": self.margin_slider.value() / 100.0,
            "batch_size": int(self.batch_spin.value()),
            "peak_norm": self.peak_norm.isChecked(),
            "make_mixture": self.make_mixture.isChecked(),
            "dedup": self.dedup.isChecked(),
            "ambig_mode": cb.AMBIG_MODES[self.ambig_combo.currentText()],
            "scan_mode": cb.SCAN_MODES[self.scan_combo.currentText()],
            "naming_mode": cb.NAMING_MODES[self.naming_combo.currentText()],
            "append_duration": False,
            "delete_if_short": self.delete_if_short.isChecked(),
            "min_duration_sec": int(self.min_duration_sec.value()),
            "delete_if_incomplete": self.delete_if_incomplete.isChecked(),
            "skip_existing": self.skip_existing.isChecked(),
        }

    # ----- help --------------------------------------------------------

    def _show_about(self) -> None:
        # Content ported from CTk show_about_dialog / ABOUT_* in stem_organizer_ui.py
        # Icon: same project-root logo as splash / window (CTk uses logo.ico → About)
        logo = Path(__file__).resolve().parents[2] / "logo.png"
        if not logo.exists():
            logo = Path(__file__).resolve().parents[2] / "logo.ico"
        help_dialog(
            self,
            title="About Classify",
            heading="STEM organizer",
            version_line=f"v{theme.APP_VERSION} — by Gilliaan & Bas Curtiz",
            intro="Organize, classify, prepare and balance audio datasets.",
            header_icon=logo if logo.exists() else None,
            repo_url=theme.STATUS_LINK_URL,
            rhythm="classify",
            sections=[
                ("How it works", [
                    (
                        "• Scans folders of audio stems and classifies each one via "
                        '<a href="https://github.com/facebookresearch/demucs">Demucs</a> '
                        "(vocals, drums, bass, other)"
                    ),
                    "• Mixes original files into cleanly-grouped outputs per folder",
                    (
                        "• Skips stems with ambiguous classification "
                        "(e.g. background vocals + guitar)"
                    ),
                    (
                        "• Supports 2-way (instrumental/vocals) and 4-way "
                        "(bass/drums/other/vocals) mixing modes"
                    ),
                    (
                        "Ideal for organizing ripped stems, unsorted libraries, or building "
                        "training datasets — without having to audition everything manually."
                    ),
                    (
                        "Accepted stems are summed from their original files (not AI-separated). "
                        "You can filter short or incomplete outputs, resume by skipping existing "
                        "results, and export an optional mixture.wav per song."
                    ),
                    (
                        "Additionally, you can play 2- or 4-stem folders to audition mixes, "
                        "using the STEM player with the Play button."
                    ),
                ]),
                ("Stem legend — 2-stem", [
                    "instrumental — Non-vocal content: drums, bass, keys, synths, and other "
                    "instruments combined.",
                    "vocals — Lead vocals, backing vocals, and vocal FX.",
                ]),
                ("Stem legend — 4-stem", [
                    "bass — Bass guitar, synth bass, low-end.",
                    "drums — Kick, snare, hats, percussion.",
                    "other — Keys, guitars, synths, and everything else non-vocal.",
                    "vocals — Lead vocals, backing vocals, and vocal FX.",
                ]),
                ("Skipped stems", [
                    "skip — Ambiguous or rejected stem — below confidence/margin thresholds, "
                    "or classification error.",
                ]),
                ("Classification controls", [
                    "Confidence threshold: minimum top-category RMS share required to keep "
                    "a stem.\n"
                    "Lower = more permissive; higher = stricter.",
                    "Min. margin: required gap between the top and runner-up categories. "
                    "Prevents ambiguous stems from being mis-categorized.",
                    "On ambiguous — Skip stem only: drop just that stem, keep the rest.\n"
                    "Skip the entire song: abort this folder; no outputs are written.",
                ]),
                ("(Optional) Calculate SI-SDR", [
                    (
                        "After organizing stems (or on an existing library), you can filter out "
                        "low-quality results using scale-invariant SDR:"
                    ),
                    (
                        "• Optional "
                        '<a href="https://source-separation.github.io/tutorial/basics/evaluation.html#si-sdr">'
                        "SI-SDR</a> quality check on organized stem folders."
                    ),
                    (
                        "• Each stem file is processed individually through "
                        '<a href="https://github.com/facebookresearch/demucs">Demucs</a> '
                        "and compared to the model output."
                    ),
                    (
                        "• Set per-stem thresholds (dB) — stems scoring below are moved to the "
                        "Recycle Bin."
                    ),
                    (
                        "• Optionally delete the whole folder ("
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md#type-1-musdb">'
                        "Type 1</a>) or all stems for a song ("
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md#type-2-stems">'
                        "Type 2</a>) when any stem fails."
                    ),
                    (
                        "• Uses the same 2-way or 4-way stem mode and thresholds as the main "
                        "classification settings."
                    ),
                    "Two input layouts are supported:",
                    (
                        "• "
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md#type-1-musdb">'
                        "Type 1</a>: one folder per song containing vocals.wav, bass.wav, etc."
                    ),
                    (
                        "• "
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md#type-2-stems">'
                        "Type 2</a>: one folder per stem category (vocals/, bass/, …) with a file "
                        "per song inside."
                    ),
                ]),
            ],
        )

    # ----- settings ----------------------------------------------------

    def settings_snapshot(self) -> dict:
        # When CUDA is unavailable the box is force-unchecked for display only —
        # do not persist that as a user "off" preference (keeps default-on when
        # CUDA later becomes usable). Respect a prior saved False if present.
        if getattr(self, "_cuda_enabled", False):
            use_cuda_pref = bool(self.use_cuda.isChecked())
        else:
            use_cuda_pref = bool(self._settings.get("use_cuda", True))
        snap = {
            "input_dir": display_path(self.input_row.text()),
            "output_dir": display_path(self.output_row.text()),
            "use_cuda": use_cuda_pref,
            "model_label": self.model_combo.currentText(),
            "stem_mode": self.stem_combo.currentText(),
            "quality": self.quality_combo.currentText(),
            "threshold": self.threshold_slider.value() / 100.0,
            "min_margin": self.margin_slider.value() / 100.0,
            "batch_size": int(self.batch_spin.value()),
            "peak_norm": bool(self.peak_norm.isChecked()),
            "make_mixture": bool(self.make_mixture.isChecked()),
            "dedup": bool(self.dedup.isChecked()),
            "ambig_label": self.ambig_combo.currentText(),
            "scan_label": self.scan_combo.currentText(),
            "naming_label": self.naming_combo.currentText(),
            "append_duration": False,
            "delete_if_short": bool(self.delete_if_short.isChecked()),
            "min_duration_sec": int(self.min_duration_sec.value()),
            "delete_if_incomplete": bool(self.delete_if_incomplete.isChecked()),
            "skip_existing": bool(self.skip_existing.isChecked()),
            "sdr_delete_folder": bool(self.sdr_delete_folder.isChecked()),
            "write_sdr_tags": bool(self.sdr_write_tags.isChecked()),
            "sdr_thresholds": dict(self._sdr_thresholds),
            # Always reopen on RMS — do not restore last SI-SDR session.
            "classify_mode": "rms",
        }
        return snap

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("input_dir"):
                self.input_row.set_text(d["input_dir"])
            if d.get("output_dir"):
                self.output_row.set_text(d["output_dir"])
            # Default True when CUDA is usable (CTk: data.get('use_cuda', True));
            # force off + disabled when cuda_effective() is False.
            if getattr(self, "_cuda_enabled", False):
                self.use_cuda.setChecked(bool(d.get("use_cuda", True)))
            else:
                self.use_cuda.setChecked(False)
            if d.get("model_label") in cb.MODELS:
                self.model_combo.setCurrentText(d["model_label"])
            stem_mode = cb.resolve_stem_mode(d.get("stem_mode") or "")
            allowed_modes = self._allowed_stem_modes(self.model_combo.currentText())
            if stem_mode in allowed_modes:
                self.stem_combo.setCurrentText(stem_mode)
            elif stem_mode.startswith('6 ') and any(m.startswith('4 ') for m in allowed_modes):
                self.stem_combo.setCurrentText(next(m for m in allowed_modes if m.startswith('4 ')))
            if d.get("quality") in cb.QUALITY_PRESETS:
                self.quality_combo.setCurrentText(d["quality"])
            else:
                self.quality_combo.setCurrentText("FLAC 16-bit")
            if d.get("threshold") is not None:
                self.threshold_slider.setValue(int(float(d["threshold"]) * 100))
            if d.get("min_margin") is not None:
                self.margin_slider.setValue(int(float(d["min_margin"]) * 100))
            if d.get("batch_size") is not None:
                self.batch_spin.setValue(int(d["batch_size"]))
            self.peak_norm.setChecked(bool(d.get("peak_norm", True)))
            self.make_mixture.setChecked(bool(d.get("make_mixture", False)))
            self.dedup.setChecked(bool(d.get("dedup", False)))
            if d.get("ambig_label") in cb.AMBIG_MODES:
                self.ambig_combo.setCurrentText(d["ambig_label"])
            if d.get("scan_label") in cb.SCAN_MODES:
                self.scan_combo.setCurrentText(d["scan_label"])
            if d.get("naming_label") in cb.NAMING_MODES:
                self.naming_combo.setCurrentText(d["naming_label"])
            self.delete_if_short.setChecked(bool(d.get("delete_if_short", True)))
            if d.get("min_duration_sec") is not None:
                self.min_duration_sec.setValue(int(d["min_duration_sec"]))
            self.delete_if_incomplete.setChecked(bool(d.get("delete_if_incomplete", False)))
            self.skip_existing.setChecked(bool(d.get("skip_existing", True)))
            self.sdr_delete_folder.setChecked(bool(d.get("sdr_delete_folder", True)))
            self.sdr_write_tags.setChecked(bool(d.get("write_sdr_tags", True)))
            stored = d.get("sdr_thresholds")
            if isinstance(stored, dict):
                for cat in cb.SDR_DEFAULT_THRESHOLDS:
                    if cat in stored:
                        self._sdr_thresholds[cat] = int(stored[cat])
            self._rebuild_sdr_thresholds()
            # Always start on RMS regardless of last session (SI-SDR is session-only).
            self._cls_seg.setCurrentItem("rms")
            self._on_cls_seg_changed("rms")
        finally:
            self._loading = False

    def _autofill_output_from_input(self, text: str) -> None:
        if self._loading:
            return
        if self.output_row.text().strip():
            return
        text = text.strip()
        if not text:
            return
        p = Path(text)
        self.output_row.set_text(str(Path(p.parent, p.name + "_organized")))

    def _bind_autosave(self) -> None:
        """Wire every settings-affecting widget to a debounced save."""
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        self.input_row.entry.textChanged.connect(self._autofill_output_from_input)
        for sig in (
            self.input_row.entry.textChanged,
            self.output_row.entry.textChanged,
            self.scan_combo.currentTextChanged,
            self.naming_combo.currentTextChanged,
            self.model_combo.currentTextChanged,
            self.stem_combo.currentTextChanged,
            self.quality_combo.currentTextChanged,
            self.ambig_combo.currentTextChanged,
            self.use_cuda.toggled,
            self.threshold_slider.valueChanged,
            self.margin_slider.valueChanged,
            self.batch_spin.valueChanged,
            self.dedup.toggled,
            self.peak_norm.toggled,
            self.make_mixture.toggled,
            self.delete_if_short.toggled,
            self.min_duration_sec.valueChanged,
            self.delete_if_incomplete.toggled,
            self.skip_existing.toggled,
            self.sdr_delete_folder.toggled,
            self.sdr_write_tags.toggled,
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


# Late import for typing in attach_action_bar; BaseWorker is from .base.
from ..workers.base import BaseWorker  # noqa: E402


def register(window, settings: SettingsStore) -> None:
    tab = ClassifyTab(settings)
    window.register_tab("Classify", tab)
    page = window.action_bar.make_page("Classify")
    tab.attach_action_bar(page, window)
    # Wire tab → window signals
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_log.connect(window.append_log)
    tab.request_sdr_log.connect(window.append_sdr_log)
    tab.request_open_player.connect(lambda: window._open_player())
    # Footer Device: tracks Classify ORT preference (not a one-shot torch probe).
    tab.use_cuda.toggled.connect(lambda checked: window.refresh_device_status(bool(checked)))
    window.refresh_device_status()
    window.show_action_bar("Classify")
    # Show the Classify action bar as the initial one
    window.tabs.setCurrentIndex(0)

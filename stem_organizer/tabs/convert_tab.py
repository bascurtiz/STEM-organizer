"""Convert tab — normalize libraries to FLAC @ target sample rate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    CompactDoubleSpinBox,
    CompactSpinBox,
    RadioButton,
)

from .. import theme
from ..dataset.convert_flac import run_convert_to_flac
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.dialogs import help_dialog, show_info
from ..widgets.info_icon import InfoIcon
from ..widgets.option_hint import style_how_it_works
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.overview_worker import OverviewWorker

PANEL_TITLE = "Convert"

TIPS = {
    "input": (
        "Source folder (WAV, FLAC, MP3, OGG, …). Nested folders included when checked. "
        "_backup_before_align folders are skipped."
    ),
    "output": (
        "Destination folder when writing elsewhere. Relative paths under Input "
        "are mirrored as .flac. A conversion_report.csv is written here."
    ),
    "dest_folder": "Write converted .flac files into a separate Output folder.",
    "dest_inplace": (
        "Overwrite in place: WAV/AIFF become .flac beside the source (source "
        "removed after success). FLAC that needs re-encode is replaced safely "
        "via a temp file."
    ),
    "include_subfolders": "Scan nested folders under the input root.",
    "skip_optimal": (
        "Leave (or copy) FLAC files that are already at the target sample rate, "
        "16/24-bit, and Stereo/Mono layout — no re-encode."
    ),
    "tag_lossy_filename": (
        "When converting MP3/OGG/… to FLAC, append a source-format suffix to the "
        "filename (e.g. What’s Up.mp3 → What’s Up_mp3-320.flac). Exact CBR rates "
        "keep the kbps; other rates use -vbr."
    ),
    "samplerate": (
        "Target sample rate for the whole dataset (default 44100). "
        "Other rates are resampled."
    ),
    "channels_stereo": "Force 2-channel output (mono is duplicated L=R).",
    "channels_mono": "Force 1-channel output (multi-channel is downmixed).",
    "headroom": (
        "Only used when a 32-bit float file peaks above 0 dBFS: uniform gain "
        "reduction so the peak lands at this level (e.g. −1.0 dB) before "
        "quantizing to ≤24-bit FLAC."
    ),
    "dither": (
        "Triangular probability density function (TPDF) dither when bit depth "
        "is reduced (e.g. 32-bit → 24-bit). No effect on lossless 16/24-bit "
        "passthroughs."
    ),
    "cores": (
        "How many files to convert in parallel (like dBpoweramp “Using N Cores”). "
        "Higher = faster, more CPU/RAM. Default is all logical cores."
    ),
    "convert": "Convert the library to consistent FLAC for MSST.",
    "stop": "Stop the running convert job.",
}
TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


class ConvertTab(QWidget):
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
        self._window = None
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
        inner.setObjectName("ConvertPage")
        inner.setStyleSheet(
            f"QWidget#ConvertPage {{ background: {theme.COLORS['bg']}; }}"
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
            "Normalize a library to FLAC at one sample rate"
        )
        title.setObjectName("HeaderDesc")
        header.addWidget(title)
        header.addWidget(InfoIcon(inner, on_click=self._show_help))
        header.addStretch(1)
        self._inner.addLayout(header)

        paths = Section(inner, "Path")
        # Same Path top padding as Genre (Section default 8)
        pl = paths.body.layout()
        pl.setSpacing(5)
        self.input_row = PathRow(
            paths.body, "Input", tip_text=TIPS["input"], label_width=80
        )
        self.output_row = PathRow(
            paths.body, "Output", tip_text=TIPS["output"], label_width=80
        )

        dest_row = QHBoxLayout()
        # Same vertical rhythm as Match Reference → Include subfolders
        dest_row.setContentsMargins(0, 12, 0, 7)
        dest_row.setSpacing(0)
        dest_lbl = BodyLabel("Destination")
        dest_lbl.setFixedWidth(80)
        dest_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._dest_group = QButtonGroup(self)
        self.dest_folder = RadioButton("Output folder")
        self.dest_inplace = RadioButton("Overwrite in place")
        self.dest_folder.setToolTip(TIPS["dest_folder"])
        self.dest_inplace.setToolTip(TIPS["dest_inplace"])
        self.dest_inplace.setChecked(True)
        self._dest_group.addButton(self.dest_folder)
        self._dest_group.addButton(self.dest_inplace)
        dest_row.addWidget(dest_lbl)
        dest_row.addSpacing(4)
        _first_radio_w = 108
        self.dest_folder.setMinimumWidth(_first_radio_w)
        dest_row.addWidget(self.dest_folder)
        dest_row.addSpacing(theme.RADIO_OPTION_GAP)
        dest_row.addWidget(self.dest_inplace)
        dest_row.addStretch(1)
        paths.body.layout().addLayout(dest_row)

        chk_lay = QVBoxLayout()
        chk_lay.setContentsMargins(0, 0, 0, 0)
        chk_lay.setSpacing(12)
        self.include_subfolders = CheckBox("Include subfolders")
        self.include_subfolders.setChecked(True)
        self.include_subfolders.setToolTip(TIPS["include_subfolders"])
        chk_lay.addWidget(self.include_subfolders)
        self.skip_optimal = CheckBox(
            "Skip / copy already-optimal FLAC (target rate · 16/24-bit · channels)"
        )
        self.skip_optimal.setChecked(True)
        self.skip_optimal.setToolTip(TIPS["skip_optimal"])
        chk_lay.addWidget(self.skip_optimal)
        self.tag_lossy_filename = CheckBox(
            "Append source suffix to lossy → FLAC names (e.g. _mp3-320)"
        )
        self.tag_lossy_filename.setChecked(True)
        self.tag_lossy_filename.setToolTip(TIPS["tag_lossy_filename"])
        chk_lay.addWidget(self.tag_lossy_filename)
        paths.body.layout().addLayout(chk_lay)
        self._inner.addWidget(paths)
        self.dest_folder.toggled.connect(self._sync_dest_ui)
        self.dest_inplace.toggled.connect(self._sync_dest_ui)
        self._sync_dest_ui()

        opts = Section(inner, "Convert options")
        opts_lay = opts.body.layout()
        om = opts_lay.contentsMargins()
        opts_lay.setContentsMargins(om.left(), 14, om.right(), om.bottom())
        opts_lay.setSpacing(12)

        ch_row = QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        ch_row.setSpacing(0)
        ch_lbl = BodyLabel("Channels")
        ch_lbl.setFixedWidth(80)
        ch_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._channels_group = QButtonGroup(self)
        self.channels_stereo = RadioButton("Stereo")
        self.channels_mono = RadioButton("Mono")
        self.channels_stereo.setToolTip(TIPS["channels_stereo"])
        self.channels_mono.setToolTip(TIPS["channels_mono"])
        self.channels_stereo.setChecked(True)
        self._channels_group.addButton(self.channels_stereo)
        self._channels_group.addButton(self.channels_mono)
        ch_row.addWidget(ch_lbl)
        ch_row.addSpacing(4)
        self.channels_stereo.setMinimumWidth(_first_radio_w)
        ch_row.addWidget(self.channels_stereo)
        ch_row.addSpacing(theme.RADIO_OPTION_GAP)
        ch_row.addWidget(self.channels_mono)
        ch_row.addStretch(1)
        opts_lay.addLayout(ch_row)

        sr_lbl = BodyLabel("Target sample rate")
        sr_lbl.setToolTip(TIPS["samplerate"])
        self.samplerate_spin = CompactSpinBox()
        self.samplerate_spin.setRange(8000, 192000)
        self.samplerate_spin.setSingleStep(100)
        self.samplerate_spin.setValue(44100)
        self.samplerate_spin.setSuffix(" Hz")
        self.samplerate_spin.setFixedWidth(120)
        self.samplerate_spin.setToolTip(TIPS["samplerate"])
        theme.sync_compact_spin_button(self.samplerate_spin, theme.COMPACT_SPIN_HEIGHT)
        theme._install_compact_spin_resize_sync(self.samplerate_spin)

        hr_lbl = BodyLabel("Float-over headroom")
        hr_lbl.setToolTip(TIPS["headroom"])
        self.headroom_spin = CompactDoubleSpinBox()
        # Signed dBFS peak target when gain-reducing float overs (default −1.0).
        self.headroom_spin.setRange(-12.0, 0.0)
        self.headroom_spin.setDecimals(1)
        self.headroom_spin.setSingleStep(0.1)
        self.headroom_spin.setValue(-1.0)
        self.headroom_spin.setSuffix(" dB")
        self.headroom_spin.setFixedWidth(120)
        self.headroom_spin.setToolTip(TIPS["headroom"])
        theme.sync_compact_spin_button(self.headroom_spin, theme.COMPACT_SPIN_HEIGHT)
        theme._install_compact_spin_resize_sync(self.headroom_spin)

        # Same label column width so sample-rate and headroom spins line up.
        _spin_lbl_w = max(sr_lbl.sizeHint().width(), hr_lbl.sizeHint().width())
        sr_lbl.setFixedWidth(_spin_lbl_w)
        hr_lbl.setFixedWidth(_spin_lbl_w)

        sr_row = QHBoxLayout()
        sr_row.setContentsMargins(0, 0, 0, 0)
        sr_row.setSpacing(8)
        sr_row.addWidget(sr_lbl, 0, Qt.AlignVCenter)
        sr_row.addWidget(self.samplerate_spin, 0, Qt.AlignVCenter)
        sr_row.addStretch(1)
        opts_lay.addLayout(sr_row)

        hr_row = QHBoxLayout()
        hr_row.setContentsMargins(0, 0, 0, 0)
        hr_row.setSpacing(8)
        hr_row.addWidget(hr_lbl, 0, Qt.AlignVCenter)
        hr_row.addWidget(self.headroom_spin, 0, Qt.AlignVCenter)
        hr_row.addStretch(1)
        opts_lay.addLayout(hr_row)

        self.dither = CheckBox("TPDF dither on bit-depth reduction")
        self.dither.setChecked(True)
        self.dither.setToolTip(TIPS["dither"])
        opts_lay.addWidget(self.dither)

        cores_row = QHBoxLayout()
        cores_row.setContentsMargins(0, 0, 0, 0)
        cores_row.setSpacing(8)
        cores_lbl = BodyLabel("CPU cores")
        cores_lbl.setToolTip(TIPS["cores"])
        self.cores_spin = CompactSpinBox()
        _ncpu = max(1, int(os.cpu_count() or 4))
        self.cores_spin.setRange(1, _ncpu)
        self.cores_spin.setValue(_ncpu)
        self.cores_spin.setFixedWidth(theme.COMPACT_SPIN_WIDTH)
        self.cores_spin.setToolTip(TIPS["cores"])
        theme.sync_compact_spin_button(self.cores_spin, theme.COMPACT_SPIN_HEIGHT)
        theme._install_compact_spin_resize_sync(self.cores_spin)
        cores_row.addWidget(cores_lbl, 0, Qt.AlignVCenter)
        cores_row.addWidget(self.cores_spin, 0, Qt.AlignVCenter)
        cores_row.addStretch(1)
        opts_lay.addLayout(cores_row)
        self._inner.addWidget(opts)

        note = Section(inner, "How it works")
        note_lay = note.body.layout()
        note_lay.setSpacing(10)
        for paragraph in (
            "MSST expects WAV or FLAC and one layout (same samplerate, stereo or mono).",
            "16/24-bit sources re-encode losslessly. 32-bit int / float is "
            "reduced to 24-bit (FLAC’s ceiling).",
            "Only float files that peak above 0 dBFS get a uniform gain "
            "reduction before quantizing.",
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

    def _sync_dest_ui(self, *_args) -> None:
        use_folder = self.dest_folder.isChecked()
        self.output_row.setEnabled(use_folder)
        self.output_row.setVisible(use_folder)

    def bind_action_buttons(self, convert_btn, stop_btn) -> None:
        self.convert_btn = convert_btn
        self.stop_btn = stop_btn

    def set_sibling_busy_hook(self, hook) -> None:
        self._sibling_busy_hook = hook

    def is_busy(self) -> bool:
        return bool(self._busy)

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        """Standalone registration (unused when hosted under Integrity)."""
        self.bind_window(window)
        self.convert_btn = action_button(
            "▶ Convert to FLAC",
            on_click=self.start_convert,
            accent=True,
            tip=TIPS["convert"],
        )
        self.stop_btn = action_button(
            "■ Stop", on_click=self.stop_convert, tip=TIPS["stop"]
        )
        self.stop_btn.setEnabled(False)
        page.add_button(self.convert_btn)
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
            self.dest_folder,
            self.dest_inplace,
            self.include_subfolders,
            self.skip_optimal,
            self.tag_lossy_filename,
            self.samplerate_spin,
            self.channels_stereo,
            self.channels_mono,
            self.headroom_spin,
            self.dither,
            self.cores_spin,
        )
        if enabled:
            self._sync_dest_ui()
        else:
            self.output_row.setEnabled(False)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self._set_settings_enabled(not busy)
        hook = getattr(self, "_sibling_busy_hook", None)
        if hook is not None:
            hook(busy)
        elif hasattr(self, "convert_btn"):
            self.convert_btn.setEnabled(not busy)
            self.stop_btn.setEnabled(busy)
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def start_convert(self) -> None:
        self._on_convert()

    def stop_convert(self) -> None:
        self._on_stop()

    def _on_convert(self) -> None:
        root = self.input_row.text().strip()
        out = self.output_row.text().strip()
        inplace = bool(self.dest_inplace.isChecked())
        if not root or not Path(root).is_dir():
            show_info(self, PANEL_TITLE, "Browse & select an input folder first.")
            return
        if not inplace:
            if not out:
                show_info(self, PANEL_TITLE, "Browse & select an output folder first.")
                return
            if Path(out).resolve() == Path(root).resolve():
                show_info(
                    self,
                    PANEL_TITLE,
                    "Output must be a different folder from Input, "
                    "or choose Overwrite in place.",
                )
                return

        recursive = bool(self.include_subfolders.isChecked())
        skip_optimal = bool(self.skip_optimal.isChecked())
        tag_lossy = bool(self.tag_lossy_filename.isChecked())
        target_sr = int(self.samplerate_spin.value())
        target_ch = 1 if self.channels_mono.isChecked() else 2
        headroom = float(self.headroom_spin.value())
        dither = bool(self.dither.isChecked())
        cores = int(self.cores_spin.value())

        def action(on_log, on_progress, stop_event):
            return run_convert_to_flac(
                root,
                None if inplace else out,
                inplace=inplace,
                recursive=recursive,
                skip_optimal=skip_optimal,
                tag_lossy_filename=tag_lossy,
                max_workers=cores,
                headroom_db=headroom,
                dither=dither,
                target_samplerate=target_sr,
                target_channels=target_ch,
                on_log=on_log,
                on_progress=on_progress,
                stop_event=stop_event,
            )

        if self._worker is not None:
            return
        self.request_clear_log.emit()
        if self._window is not None and hasattr(self._window, "set_log_export_prefix"):
            self._window.set_log_export_prefix("convert")
        self.request_log.emit("=== Convert to FLAC ===", "info")
        self._set_busy(True, "Converting…")
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
            self.request_log.emit("Convert stopping…", "warn")

    def _show_help(self) -> None:
        help_dialog(
            self,
            title="Convert help",
            heading="Consistent FLAC libraries",
            intro=(
                "Batch-convert audio to FLAC at one sample rate and channel layout "
                "so "
                '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training">'
                "MSST</a> gets uniform WAV/FLAC-ready inputs."
            ),
            sections=[
                (
                    "Path",
                    [
                        "Input root. Choose Output folder to mirror .flac elsewhere, "
                        "or Overwrite in place to replace sources (WAV→FLAC deletes "
                        "the original after a successful write).",
                        "Already-optimal FLAC can be skipped without re-encoding.",
                        "Optional: append a source-format suffix to lossy → FLAC "
                        "filenames (e.g. Track.mp3 → Track_mp3-320.flac).",
                    ],
                ),
                (
                    "Options",
                    [
                        "Target sample rate (default 44.1 kHz) — other rates are resampled.",
                        "Stereo (default) or Mono — "
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training">'
                        "MSST</a> expects one layout.",
                        "Float-over headroom (e.g. −1.0 dB) — only when 32-bit float "
                        "peaks above 0 dBFS.",
                        "TPDF dither (triangular PDF) when reducing bit depth to ≤24-bit.",
                        "CPU cores — convert multiple files in parallel "
                        "(like dBpoweramp “Using N Cores”).",
                    ],
                ),
                (
                    "Formats",
                    [
                        '<a href="https://github.com/ZFTurbo/Music-Source-Separation-Training">'
                        "MSST</a> expects WAV or FLAC. Other formats (MP3, OGG, …) are "
                        "decoded and written as FLAC at your target rate and channels.",
                        "Tags and cover art are copied onto the new FLAC "
                        "(including COMPRESSION, KEY, and other STEM fields).",
                        "Already-correct FLAC (right sample rate and bit depth) is "
                        "copied or skipped, not re-encoded.",
                    ],
                ),
                (
                    "Sources",
                    [
                        '<a href="https://ffmpeg.org/">FFmpeg</a> / ffprobe '
                        "(bundled via install-deps.bat) — probe + decode only. "
                        '<a href="https://python-soundfile.readthedocs.io/">'
                        "soundfile</a> / libsndfile writes the FLAC.",
                        "Probe (sample rate / channels / bit depth):",
                        "<pre>ffprobe -v error -select_streams a:0 "
                        "-show_entries stream=sample_rate,channels,sample_fmt,"
                        "bits_per_raw_sample,codec_name -of json INPUT</pre>",
                        "Probe (bitrate for optional _mp3-320 filename tag):",
                        "<pre>ffprobe -v error -select_streams a:0 "
                        "-show_entries stream=bit_rate:format=bit_rate "
                        "-of json INPUT</pre>",
                        "Decode to raw float32 PCM on stdout:",
                        "<pre>ffmpeg -nostdin -hide_banner -loglevel error "
                        "-i INPUT -vn -f f32le -acodec pcm_f32le pipe:1</pre>",
                        "Then STEM resamples / mono-stereo / dither / headroom in "
                        "Python and writes FLAC via soundfile "
                        "(format=FLAC, subtype=PCM_16 or PCM_24) — not ffmpeg encode.",
                    ],
                ),
            ],
        )

    def settings_snapshot(self) -> dict:
        return {
            "convert_input_dir": display_path(self.input_row.text()),
            "convert_output_dir": display_path(self.output_row.text()),
            "convert_inplace": bool(self.dest_inplace.isChecked()),
            "convert_include_subfolders": bool(self.include_subfolders.isChecked()),
            "convert_skip_optimal": bool(self.skip_optimal.isChecked()),
            "convert_tag_lossy_filename": bool(self.tag_lossy_filename.isChecked()),
            "convert_samplerate": int(self.samplerate_spin.value()),
            "convert_channels": (
                "mono" if self.channels_mono.isChecked() else "stereo"
            ),
            "convert_headroom_db": float(self.headroom_spin.value()),
            "convert_dither": bool(self.dither.isChecked()),
            "convert_cores": int(self.cores_spin.value()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("convert_input_dir"):
                self.input_row.set_text(d["convert_input_dir"])
            elif d.get("output_dir"):
                self.input_row.set_text(d["output_dir"])
            if d.get("convert_output_dir"):
                self.output_row.set_text(d["convert_output_dir"])
            if "convert_inplace" in d:
                if bool(d.get("convert_inplace")):
                    self.dest_inplace.setChecked(True)
                else:
                    self.dest_folder.setChecked(True)
            else:
                self.dest_inplace.setChecked(True)
            self.include_subfolders.setChecked(
                bool(d.get("convert_include_subfolders", True))
            )
            self.skip_optimal.setChecked(bool(d.get("convert_skip_optimal", True)))
            if "convert_tag_lossy_filename" in d:
                self.tag_lossy_filename.setChecked(
                    bool(d["convert_tag_lossy_filename"])
                )
            else:
                self.tag_lossy_filename.setChecked(True)
            if "convert_samplerate" in d:
                try:
                    self.samplerate_spin.setValue(int(d["convert_samplerate"]))
                except (TypeError, ValueError):
                    pass
            ch = str(d.get("convert_channels", "stereo") or "stereo").lower()
            if ch == "mono":
                self.channels_mono.setChecked(True)
            else:
                self.channels_stereo.setChecked(True)
            if "convert_headroom_db" in d:
                try:
                    v = float(d["convert_headroom_db"])
                    # Migrate old positive "1.0 below FS" → signed −1.0 dB.
                    if v > 0:
                        v = -v
                    self.headroom_spin.setValue(v)
                except (TypeError, ValueError):
                    pass
            # Default on; only honor an explicit saved False.
            if "convert_dither" in d:
                self.dither.setChecked(bool(d["convert_dither"]))
            else:
                self.dither.setChecked(True)
            if "convert_cores" in d:
                try:
                    n = int(d["convert_cores"])
                    self.cores_spin.setValue(
                        max(1, min(n, self.cores_spin.maximum()))
                    )
                except (TypeError, ValueError):
                    pass
            self._sync_dest_ui()
        finally:
            self._loading = False

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        for sig in (
            self.input_row.entry.textChanged,
            self.output_row.entry.textChanged,
            self.dest_folder.toggled,
            self.dest_inplace.toggled,
            self.include_subfolders.toggled,
            self.skip_optimal.toggled,
            self.tag_lossy_filename.toggled,
            self.samplerate_spin.valueChanged,
            self.channels_stereo.toggled,
            self.channels_mono.toggled,
            self.headroom_spin.valueChanged,
            self.dither.toggled,
            self.cores_spin.valueChanged,
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

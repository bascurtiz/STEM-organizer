"""Dataset overview tab — inventory charts, scan, balance."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPoint, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CheckBox, SegmentedWidget

from .. import theme
from ..dataset.balance import (
    BALANCE_FEATURES,
    BalanceMode,
    build_balance_plan,
    execute_balance,
    select_balanced_units,
)
from ..dataset.charts import (
    ChartHost,
    GenreStyleBreakdown,
    GenreVerticalBars,
    genre_colors_for,
    make_bar,
    make_donut,
)
from ..dataset.inventory import ScanResult, ScanUnit, scan_library
from ..dataset.models import (
    style_genres_from_stats,
    OverviewStats,
    VOCAL_TYPES,
    make_demo_overview,
)
from ..dataset.scan_cache import load_scan_result, save_scan_result
from ..dataset.report_export import PathsMeta, export_overview_report
from ..settings_store import SettingsStore, display_path
from ..widgets.action_bar import ActionBarPage
from ..widgets.action_button import action_button
from ..widgets.balance_dialog import (
    BalanceConfirmStats,
    ask_balance_confirm,
    ask_balance_finished,
    ask_balance_options,
)
from ..widgets.dialogs import ask_yes_no, help_dialog, show_info
from ..widgets.export_notes_dialog import ask_export_notes
from ..widgets.info_icon import InfoIcon
from ..widgets.path_row import PathRow
from ..widgets.section import Section
from ..workers.overview_worker import OverviewWorker

PANEL_TITLE = "Charts"
# Extra plot height beyond the fitted viewport floor (tweak ±px as needed).
_CHART_HEIGHT_BOOST = 2

TIPS = {
    "inst": "Folder of instrumental / music-only audio (subfolders included when checked).",
    "vocal": "Folder of vocal / acapella audio (subfolders included when checked).",
    "pairs": (
        "Song folders that each contain one instrumental + one vocal. "
        "Each song folder counts as +1 instrumental and +1 vocal. "
        "_backup_before_align folders are ignored."
    ),
    "samples": (
        "Sample / loop library. Own Samples role on the Roles chart; "
        "genre / style facets match Instrumental."
    ),
    "include_subfolders": "Scan nested folders under each root.",
    "scan": "Read metadata from the four roots and refresh charts.",
    "balance": (
        "Equalize Roles (always on) plus selected chart features (Genre, …), "
        "pick a destination folder/drive, then Copy / Move into {root}_BALANCED "
        "or write a CSV list."
    ),
    "stop": "Stop the running scan or balance job.",
    "export": (
        "Save a report of KPIs + charts. PNG = dark screen spread; "
        "PDF = same layout, one A4 landscape page for print."
    ),
}
TIPS = {k: theme.format_tooltip(v) for k, v in TIPS.items()}


def _fmt_bytes(n: int) -> str:
    tb = 1024**4
    gb = 1024**3
    if n >= tb:
        return f"{n / tb:.2f} TB"
    if n >= gb:
        return f"{n / gb:.1f} GB"
    mb = 1024**2
    if n >= mb:
        return f"{n / mb:.0f} MB"
    return f"{n} B"


def _fmt_hours(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    h = seconds / 3600.0
    if h >= 100:
        return f"{h:,.0f} h"
    return f"{h:,.1f} h"


class _KpiChip(QFrame):
    """One card: dark-grey label above, bold bright value below."""

    # Label: softer than section mute; value: softer than full fg.
    _LABEL_COLOR = "#7a8199"  # DARK text_mute
    _VALUE_COLOR = "#c4c8d6"

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("kpiChip")
        self.setStyleSheet(
            f"""
            QFrame#kpiChip {{
                background: {theme.COLORS['panel2']};
                border: none;
                border-radius: 8px;
            }}
            QFrame#kpiChip QLabel#kpiLabel {{
                color: {self._LABEL_COLOR};
                background: transparent;
                border: none;
                font-family: "{theme.FONT_FAMILY}";
                font-size: {theme.BODY_FONT_PX - 1}px;
                font-weight: 400;
            }}
            QFrame#kpiChip QLabel#kpiValue {{
                color: {self._VALUE_COLOR};
                background: transparent;
                border: none;
                font-family: "{theme.FONT_FAMILY_MONO}";
                font-size: {theme.LOG_FONT_PX + 4}px;
                font-weight: 700;
            }}
            """
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        # Plain QLabel — BodyLabel gets recolored by polish_fluent_controls.
        self._title = QLabel(label)
        self._title.setObjectName("kpiLabel")
        self._value = QLabel("—")
        self._value.setObjectName("kpiValue")
        bold_mono = QFont(theme.FONT_FAMILY_MONO)
        bold_mono.setPixelSize(theme.LOG_FONT_PX + 4)
        bold_mono.setWeight(QFont.Bold)
        self._value.setFont(bold_mono)
        lay.addWidget(self._title)
        lay.addWidget(self._value)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class DatasetOverviewTab(QWidget):
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
        self._demo_mode = True
        self._stats: OverviewStats = make_demo_overview()
        self._chart_key = "sdr"
        self._units: list[ScanUnit] = []
        self._worker: Optional[OverviewWorker] = None
        self._busy = False
        self._balance_dest: Optional[str] = None

        self._build_ui()
        self.load_settings()
        self._bind_autosave()
        cached = load_scan_result()
        if cached is not None:
            self._units = list(cached.units)
            self._apply_stats(cached.stats)
        else:
            self._apply_stats(self._stats)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Match Classify: slight top air so HeaderDesc lines up across tabs.
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        # No ScrollArea — charts are fitted to the viewport; scrolling is disabled.
        page = QWidget()
        page.setStyleSheet(f"background: {theme.COLORS['bg']};")
        self._page = page
        self._inner = QVBoxLayout(page)
        self._inner.setAlignment(Qt.AlignTop)
        self._inner.setContentsMargins(
            theme.PAGE_CONTENT_INSET, 0, theme.PAGE_CONTENT_INSET, 0
        )
        self._inner.setSpacing(theme.SECTION_GAP)

        header = QHBoxLayout()
        header.setContentsMargins(0, 8, 0, 10)
        header.setSpacing(6)
        title = BodyLabel(
            "Inventory charts for instrumentals, vocals & pairs — demo preview until you scan"
        )
        title.setObjectName("HeaderDesc")
        self._header_desc = title
        header.addWidget(title)
        header.addWidget(InfoIcon(page, on_click=self._show_help))
        header.addStretch(1)
        self._demo_badge = BodyLabel("DEMO")
        # Inset from the right like status-bar Device (STATUS_PAD_X rhythm).
        self._demo_badge.setStyleSheet(
            f"color: {theme.COLORS['accent']}; font-weight: 700; "
            f"font-size: {theme.BODY_FONT_PX - 1}px; letter-spacing: 1px; "
            f"margin-right: {theme.PAGE_EDGE_INSET}px;"
        )
        header.addWidget(self._demo_badge, 0, Qt.AlignVCenter)
        self._inner.addLayout(header)

        paths = Section(page, "Paths")
        paths_body = paths.body.layout()
        paths_body.setSpacing(5)
        _path_lbl_w = 80  # Match tab density — fits "Instrumental"

        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(12)
        left = QWidget(paths.body)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(5)
        right = QWidget(paths.body)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(5)

        self.inst_row = PathRow(
            left, "Instrumental", tip_text=TIPS["inst"], label_width=_path_lbl_w
        )
        self.vocal_row = PathRow(
            left, "Vocal", tip_text=TIPS["vocal"], label_width=_path_lbl_w
        )
        self.pairs_row = PathRow(
            right, "Pairs", tip_text=TIPS["pairs"], label_width=_path_lbl_w
        )
        self.samples_row = PathRow(
            right, "Samples", tip_text=TIPS["samples"], label_width=_path_lbl_w
        )
        cols.addWidget(left, 1)
        cols.addWidget(right, 1)
        paths_body.addLayout(cols)

        self.include_subfolders = CheckBox("Include subfolders")
        self.include_subfolders.setChecked(True)
        self.include_subfolders.setToolTip(TIPS["include_subfolders"])
        paths_body.addWidget(self.include_subfolders)
        self._inner.addWidget(paths)

        # Library KPIs — section title matches Paths / Charts
        library = Section(page, "Library")
        library.body.setObjectName("")  # no Card chrome; chips are the cards
        library.body.setStyleSheet("background: transparent; border: none;")
        lib_body = library.body.layout()
        lib_body.setContentsMargins(0, 0, 0, 0)
        lib_body.setSpacing(0)
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(8)
        self.kpi_files = _KpiChip("Files")
        self.kpi_size = _KpiChip("Size")
        self.kpi_dur = _KpiChip("Duration")
        self.kpi_inst = _KpiChip("Instrumental units")
        self.kpi_voc = _KpiChip("Vocal units")
        self.kpi_samples = _KpiChip("Sample units")
        for chip in (
            self.kpi_files,
            self.kpi_size,
            self.kpi_dur,
            self.kpi_inst,
            self.kpi_voc,
            self.kpi_samples,
        ):
            kpi_row.addWidget(chip, 1)
        lib_body.addLayout(kpi_row)
        self._inner.addWidget(library)

        # Charts hug content — no stretch under the footnote (that forced a scrollbar).
        charts_sec = Section(page, "Charts")
        charts_sec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        charts_sec.body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        charts_body = charts_sec.body.layout()
        charts_body.setContentsMargins(10, 8, 10, 4)
        charts_body.setSpacing(6)
        charts_body.setAlignment(Qt.AlignTop)
        self._chart_seg = SegmentedWidget()
        for key, label in (
            ("sdr", "SI-SDR"),
            ("roles", "Roles"),
            ("genre", "Genre"),
            ("style", "Style"),
            ("gender", "Gender"),
            ("reverb", "Reverb"),
            ("vocal", "Vocal type"),
            ("keys", "Keys"),
            ("compression", "Compression"),
        ):
            self._chart_seg.addItem(key, label)
        theme.configure_segmented_widget(self._chart_seg)
        self._chart_seg.currentItemChanged.connect(self._on_chart_seg)
        charts_body.addWidget(self._chart_seg, 0)

        self._charts_body = charts_body
        self._chart_stack = QWidget()
        self._chart_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._chart_stack.setMinimumHeight(180)
        self._chart_grid = QGridLayout(self._chart_stack)
        self._chart_grid.setContentsMargins(0, 0, 0, 0)
        self._chart_grid.setSpacing(0)

        self._donut_host = ChartHost()
        self._bar_host = ChartHost()
        self._sdr_host = ChartHost()
        self._genre_bars = GenreVerticalBars()
        self._style_breakdown = GenreStyleBreakdown()
        self._chart_note = BodyLabel("")
        self._chart_note.setObjectName("ChartNote")
        self._chart_note.setWordWrap(True)
        self._chart_note.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._chart_note.setContentsMargins(2, 2, 2, 0)
        self._chart_note.setStyleSheet(
            f"color: {theme.COLORS['fg_dim']}; background: transparent;"
        )
        if hasattr(self._chart_note, "setTextColor"):
            self._chart_note.setTextColor(
                theme.COLORS["fg_dim"], theme.COLORS["fg_dim"]
            )

        self._chart_grid.addWidget(self._donut_host, 0, 0, Qt.AlignTop)
        self._chart_grid.addWidget(self._bar_host, 0, 0, Qt.AlignTop)
        self._chart_grid.addWidget(self._sdr_host, 0, 0, Qt.AlignTop)
        self._chart_grid.addWidget(self._genre_bars, 0, 0)
        self._chart_grid.addWidget(self._style_breakdown, 0, 0)
        self._chart_grid.setRowStretch(0, 1)

        charts_body.addWidget(self._chart_stack, 0)
        charts_body.addWidget(self._chart_note, 0)
        self._inner.addWidget(charts_sec, 0)

        layout.addWidget(page, 1)
        self._chart_seg.setCurrentItem("sdr")
        self._chart_fit_timer = QTimer(self)
        self._chart_fit_timer.setSingleShot(True)
        self._chart_fit_timer.setInterval(0)
        self._chart_fit_timer.timeout.connect(self._fit_charts_height)
        self._schedule_charts_height()

    def attach_action_bar(self, page: ActionBarPage, window) -> None:
        self._window = window
        self._action_page = page
        self.scan_btn = action_button(
            "▶ Scan library", on_click=self._on_scan, accent=True, tip=TIPS["scan"]
        )
        self.balance_btn = action_button(
            "Balance…", on_click=self._on_balance, tip=TIPS["balance"]
        )
        self.export_btn = action_button(
            "Export…", on_click=self._on_export, tip=TIPS["export"]
        )
        self.stop_btn = action_button(
            "■ Stop", on_click=self._on_stop, tip=TIPS["stop"]
        )
        page.add_button(self.scan_btn)
        page.add_button(self.balance_btn)
        page.add_button(self.export_btn)
        page.add_button(self.stop_btn)
        page.add_stretch()
        # Cache may already have a real scan from __init__ — sync after buttons exist.
        self._sync_action_buttons()

    def _can_balance(self) -> bool:
        """Balance needs a real (non-demo) scan with unit inventory."""
        return (not self._busy) and (not self._demo_mode) and bool(self._units)

    def _sync_action_buttons(self) -> None:
        if not hasattr(self, "balance_btn"):
            return
        self.scan_btn.setEnabled(not self._busy)
        self.balance_btn.setEnabled(self._can_balance())
        self.export_btn.setEnabled(not self._busy)
        self.stop_btn.setEnabled(self._busy)

    def _set_log_busy(self, busy: bool) -> None:
        """Reveal host LOG only while scan / balance is running."""
        win = getattr(self, "_window", None)
        if win is not None and hasattr(win, "set_dataset_log_busy"):
            win.set_dataset_log_busy(busy)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self._busy = busy
        self._set_log_busy(busy)
        self._sync_action_buttons()
        # Lock Charts paths while scan / balance / export runs.
        for w in (
            self.inst_row,
            self.vocal_row,
            self.pairs_row,
            self.samples_row,
            self.include_subfolders,
        ):
            try:
                w.setEnabled(not busy)
            except Exception:
                pass
        if busy:
            self.request_status_running.emit()
            if status:
                self.request_status_text.emit(status)
        else:
            self.request_status_idle.emit(status or "Idle")

    def _paths(self) -> tuple[str, str, str, str, bool]:
        return (
            self.inst_row.text().strip(),
            self.vocal_row.text().strip(),
            self.pairs_row.text().strip(),
            self.samples_row.text().strip(),
            bool(self.include_subfolders.isChecked()),
        )

    def _start_worker(
        self,
        action,
        *,
        status: str,
        starting: str,
        balance_dest: str = "",
    ) -> None:
        if self._worker is not None:
            return
        self._balance_dest = balance_dest.strip() or None
        self.request_clear_log.emit()
        win = self.window()
        if win is not None and hasattr(win, "set_log_export_prefix"):
            prefix = "balance" if starting.lower().startswith("balance") else "scan"
            win.set_log_export_prefix(prefix)
        self.request_log.emit(f"=== {starting} ===", "info")
        self._set_busy(True, status)
        self._worker = OverviewWorker(action, parent=self)
        self._worker.log_line.connect(self.request_log)
        self._worker.progress.connect(self.request_progress)
        self._worker.status.connect(self.request_status_text)
        self._worker.result_ready.connect(self._on_worker_result)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_result(self, payload) -> None:
        self._pending_result = payload

    def _on_worker_done(self, status: str) -> None:
        worker = self._worker
        self._worker = None
        payload = getattr(self, "_pending_result", None)
        self._pending_result = None
        balance_dest = self._balance_dest
        self._balance_dest = None
        balance_ok = (
            balance_dest
            and isinstance(payload, dict)
            and status != "Stopped"
            and not str(status).startswith("Failed")
        )
        if isinstance(payload, ScanResult):
            self._units = list(payload.units)
            self._apply_stats(payload.stats)
            save_scan_result(payload)
            if worker is not None:
                worker.set_final_status(
                    f"Scanned · {payload.stats.total_files:,} units"
                )
                status = f"Scanned · {payload.stats.total_files:,} units"
        self._set_busy(False, status)
        # Re-enable balance after successful real scan (units already applied above).
        self._sync_action_buttons()
        if balance_ok:
            QTimer.singleShot(0, lambda: self._offer_open_balance_folder(balance_dest))

    @staticmethod
    def _open_path_in_explorer(path: str) -> None:
        target = Path(path)
        if not target.exists():
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(target)], check=False)
            else:
                subprocess.run(["xdg-open", str(target)], check=False)
        except Exception:
            pass

    def _offer_open_balance_folder(self, dest: str) -> None:
        path = Path(dest)
        if not path.is_dir():
            return
        if ask_balance_finished(self, dest):
            self._open_path_in_explorer(dest)

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.request_status_text.emit("Stopping…")

    def _on_scan(self) -> None:
        inst, voc, pairs, samples, recursive = self._paths()
        if not any(
            Path(p).is_dir() for p in (inst, voc, pairs, samples) if p
        ):
            show_info(
                self,
                PANEL_TITLE,
                "Browse & select at least one Instrumental, Vocal, Pairs, or Samples folder.",
            )
            return

        def action(on_log, on_progress, stop_event):
            result = scan_library(
                instrumental=inst,
                vocal=voc,
                pairs=pairs,
                samples=samples,
                include_subfolders=recursive,
                on_log=on_log,
                on_progress=on_progress,
                stop_event=stop_event,
            )
            return result

        self._start_worker(action, status="Scanning library…", starting="Scan library")

    def _on_balance(self) -> None:
        if self._demo_mode or not self._units:
            show_info(
                self,
                PANEL_TITLE,
                "Run Scan library first so Balance has a real inventory.",
            )
            return

        # Prefer parent of the first configured scan root as destination default.
        default_dest = ""
        for root in (
            self.inst_row.text().strip(),
            self.vocal_row.text().strip(),
            self.pairs_row.text().strip(),
            self.samples_row.text().strip(),
        ):
            if root and Path(root).is_dir():
                default_dest = str(Path(root).parent)
                break

        opts = ask_balance_options(self, default_dest=default_dest)
        if not opts:
            return

        selected, target_n = select_balanced_units(
            self._units, features=opts.features
        )
        if not selected or target_n <= 0:
            show_info(
                self,
                PANEL_TITLE,
                "Nothing to balance for the selected features "
                "(no overlapping categories with counts).",
            )
            return

        labels = dict(BALANCE_FEATURES)
        feat_txt = ", ".join(labels.get(f, f) for f in opts.features)
        n_pairs = sum(1 for u in selected if u.role == "pair")
        n_inst = sum(1 for u in selected if u.role == "instrumental")
        n_samples = sum(1 for u in selected if u.role == "sample")
        n_voc = sum(1 for u in selected if u.role == "vocal")
        n_strata = (
            len(selected) // target_n if target_n else 0
        )
        mode_label = {"copy": "Copy", "move": "Move", "csv": "CSV list"}[opts.mode]
        if not ask_balance_confirm(
            self,
            BalanceConfirmStats(
                features_label=feat_txt,
                mode_label=mode_label,
                categories=n_strata,
                per_category=target_n,
                total=len(selected),
                pairs=n_pairs,
                instrumental=n_inst,
                vocal=n_voc,
                samples=n_samples,
                dest=opts.dest,
            ),
        ):
            return

        mode = opts.mode
        if mode == "move":
            if not ask_yes_no(
                self,
                PANEL_TITLE,
                "Move is destructive — files leave their current folders.\n\nContinue?",
                yes_text="Move",
                no_text="Cancel",
            ):
                return
        dest = opts.dest
        dest_path = Path(dest)
        if not dest or (dest_path.exists() and not dest_path.is_dir()):
            show_info(self, PANEL_TITLE, "Destination folder is missing or invalid.")
            return
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            show_info(self, PANEL_TITLE, f"Could not create destination:\n{exc}")
            return
        if not dest_path.is_dir():
            show_info(self, PANEL_TITLE, "Destination folder is missing or invalid.")
            return

        mode_typed: BalanceMode = mode  # type: ignore[assignment]
        plan = build_balance_plan(
            self._units,
            mode=mode_typed,
            dest=str(dest_path),
            features=opts.features,
            inst_root=self.inst_row.text().strip(),
            vocal_root=self.vocal_row.text().strip(),
            pairs_root=self.pairs_row.text().strip(),
            samples_root=self.samples_row.text().strip(),
        )

        def action(on_log, on_progress, stop_event):
            return execute_balance(
                plan,
                on_log=on_log,
                on_progress=on_progress,
                stop_event=stop_event,
            )

        label = {"copy": "Copy", "move": "Move", "csv": "CSV"}[mode]
        self._start_worker(
            action,
            status=f"Balancing ({label})…",
            starting=f"Balance · {label}",
            balance_dest=str(dest_path),
        )

    def _on_export(self) -> None:
        default = f"STEM-dataset-overview-{datetime.now():%Y%m%d}.png"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export Charts report",
            default,
            "PNG (*.png);;PDF (*.pdf)",
        )
        if not path:
            return
        # Ensure suffix matches the chosen filter when the user omits it.
        lower = path.lower()
        if selected.startswith("PNG") and not lower.endswith(".png"):
            path = path + ".png"
        elif selected.startswith("PDF") and not lower.endswith(".pdf"):
            path = path + ".pdf"
        elif not (lower.endswith(".pdf") or lower.endswith(".png")):
            path = path + ".png"

        notes = ""
        saved_notes = str(self._settings.data.get("ds_export_notes") or "")
        typed = ask_export_notes(self, initial_text=saved_notes)
        if typed is None:
            return
        notes = typed
        self._settings.merge({"ds_export_notes": notes})
        self._settings.flush()

        meta = PathsMeta(
            instrumental=self.inst_row.text().strip(),
            vocal=self.vocal_row.text().strip(),
            pairs=self.pairs_row.text().strip(),
            samples=self.samples_row.text().strip(),
            include_subfolders=bool(self.include_subfolders.isChecked()),
        )
        try:
            self.request_status_text.emit("Exporting report…")
            export_overview_report(path, self._stats, meta, notes=notes)
        except Exception as exc:
            self.request_status_idle.emit("Export failed")
            show_info(self, PANEL_TITLE, f"Export failed:\n{exc}")
            return
        self.request_status_idle.emit(f"Exported · {path}")
        show_info(self, PANEL_TITLE, "Report saved:", link_path=path)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Defer until layout settles — measuring mid-resize (esp. max→restore)
        # under-sizes the plot; switching chart tabs re-fit and "fixed" it.
        self._schedule_charts_height()

    def _schedule_charts_height(self) -> None:
        timer = getattr(self, "_chart_fit_timer", None)
        if timer is None:
            QTimer.singleShot(0, self._fit_charts_height)
            return
        timer.start()

    def _fit_charts_height(self) -> None:
        """Size plot to the page — no scrolling on this tab."""
        page = getattr(self, "_page", None)
        stack = getattr(self, "_chart_stack", None)
        if page is None or stack is None:
            return
        if page.height() < 120:
            return

        # Drop prior fixed/min heights so mapTo + sizeHints reflect the new
        # viewport (stale max-size mins made restore collapse the panel).
        floor = 180
        stack.setMinimumHeight(floor)
        stack.setMaximumHeight(16777215)
        for host in (self._donut_host, self._bar_host, self._sdr_host):
            host.setMinimumHeight(floor)
            host.setMaximumHeight(16777215)
        self._genre_bars.setMinimumHeight(floor)
        self._genre_bars.setMaximumHeight(16777215)
        self._style_breakdown.setMinimumHeight(floor)
        self._style_breakdown.setMaximumHeight(16777215)

        above = stack.mapTo(page, QPoint(0, 0)).y()
        note_h = max(18, self._chart_note.sizeHint().height())
        # Fill toward the page floor, plus a small bump —
        # change _CHART_HEIGHT_BOOST if too much / too little.
        avail = int(page.height() - above - note_h) + _CHART_HEIGHT_BOOST
        avail = max(floor, avail)

        for host in (self._donut_host, self._bar_host, self._sdr_host):
            host.set_plot_height(avail)
        self._genre_bars.setFixedHeight(avail)
        self._style_breakdown.setFixedHeight(avail)
        stack.setFixedHeight(avail)
        stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._charts_body.setStretchFactor(stack, 0)

    def _on_chart_seg(self, key: str) -> None:
        self._chart_key = key or "vocal"
        self._render_chart(self._chart_key)
        self._schedule_charts_height()

    def _apply_stats(self, stats: OverviewStats) -> None:
        self._stats = stats
        self._demo_mode = bool(stats.demo)
        self._demo_badge.setVisible(self._demo_mode)
        if hasattr(self, "_header_desc"):
            self._header_desc.setText(
                "Inventory charts for instrumentals, vocals & pairs — demo preview until you scan"
                if self._demo_mode
                else "Inventory charts for instrumentals, vocals & pairs"
            )
        self.kpi_files.set_value(f"{stats.total_files:,}")
        self.kpi_size.set_value(_fmt_bytes(stats.total_bytes))
        self.kpi_dur.set_value(_fmt_hours(stats.duration.total))
        self.kpi_inst.set_value(f"{stats.roles.instrumental:,}")
        self.kpi_voc.set_value(f"{stats.roles.vocal:,}")
        self.kpi_samples.set_value(f"{stats.roles.samples:,}")
        self._render_chart(self._chart_key)
        self._sync_action_buttons()

    def _render_chart(self, key: str) -> None:
        stats = self._stats
        self._chart_key = key
        self._donut_host.hide()
        self._bar_host.hide()
        self._sdr_host.hide()
        self._genre_bars.hide()
        self._style_breakdown.hide()
        self._schedule_charts_height()

        if key == "vocal":
            items = []
            total = sum(b.count for b in stats.vocal_type.values()) or 1
            for name in VOCAL_TYPES:
                b = stats.vocal_type.get(name)
                pct = 100.0 * (b.count if b else 0) / total
                items.append((name, pct))
            self._donut_host.set_chart(
                make_donut(
                    "Vocal type",
                    items,
                    colors=theme.CHART_VOCAL_TYPE_COLORS,
                ),
                series_grow=False,
            )
            self._donut_host.show()
            self._chart_note.setText(
                "Share of tagged vocal files by type. Demo until you scan."
                if stats.demo
                else "Share of files with a VOCAL_TYPE tag."
            )
            return

        if key == "keys":
            from ..musical_keys import CHART_KEY_COLORS, KEY_DISPLAY_ORDER, key_chart_label

            items: list[tuple[str, float]] = []
            colors: dict[str, str] = {}
            for name in KEY_DISPLAY_ORDER:
                b = stats.key.get(name)
                if b and b.count > 0:
                    label = key_chart_label(name)
                    items.append((label, float(b.count)))
                    colors[label] = CHART_KEY_COLORS.get(name, theme.COLORS["accent"])
            if not items:
                self._chart_note.setText(
                    "No Initial key tags yet — run Key Detect, then Scan library."
                )
                return
            self._genre_bars.set_data(
                "Keys (vs largest)",
                items,
                colors=colors,
            )
            self._genre_bars.show()
            self._chart_note.setText(
                f"{len(items)} keys — bar height relative to the largest key; "
                f"labels show file counts. Order follows the circle of fifths "
                f"display sequence (C … Abm / G#m)."
            )
            return

        if key == "roles":
            inst = stats.roles.instrumental
            voc = stats.roles.vocal
            samples = stats.roles.samples
            total = max(inst + voc + samples, 1)
            chart = make_donut(
                "Instrumental / Vocal / Samples",
                [
                    ("Instrumental", 100.0 * inst / total),
                    ("Vocal", 100.0 * voc / total),
                    ("Samples", 100.0 * samples / total),
                ],
                colors=theme.CHART_ROLE_COLORS,
            )
            self._donut_host.set_chart(chart)
            self._donut_host.show()
            pairs = stats.roles.pair_folders
            self._chart_note.setText(
                f"Pairs contribute equally to Instrumental and Vocal "
                f"({pairs:,} song-folder pairs). Samples are a separate slice."
            )
            return

        if key in ("gender", "reverb", "compression"):
            store = {
                "gender": stats.gender,
                "reverb": stats.reverb,
                "compression": stats.compression,
            }[key]
            title = {
                "gender": "Gender",
                "reverb": "Reverb",
                "compression": "Compression",
            }[key]
            color_map = {
                "gender": theme.CHART_GENDER_COLORS,
                "reverb": theme.CHART_REVERB_COLORS,
                "compression": theme.CHART_COMPRESSION_COLORS,
            }[key]
            total = sum(b.count for b in store.values()) or 1
            # Title-case labels to match Roles (Female / Dry / Lossless).
            items = [
                (name[:1].upper() + name[1:], 100.0 * b.count / total)
                for name, b in store.items()
            ]
            # Color map keys stay lowercase — make_donut resolves both.
            self._donut_host.set_chart(make_donut(title, items, colors=color_map))
            self._donut_host.show()
            if key == "compression":
                self._chart_note.setText(
                    "Percent of tagged audio files. Pair folders count both stems."
                )
            else:
                self._chart_note.setText("Percent of tagged files in this facet.")
            return

        if key == "genre":
            items = sorted(
                ((n, b.count) for n, b in stats.genre.items()),
                key=lambda x: -x[1],
            )
            self._genre_bars.set_data(
                "Genre (vs largest)",
                [(n, float(c)) for n, c in items],
                colors=genre_colors_for([n for n, _ in items]),
            )
            self._genre_bars.show()
            self._chart_note.setText(
                f"{len(items)} genres — bar height relative to the largest genre; "
                f"labels show file counts."
            )
            return

        if key == "style":
            genres = style_genres_from_stats(stats)
            if not genres:
                self._chart_note.setText(
                    "No genre tags yet — run Genre tagger, then Scan library."
                )
                return
            colors = genre_colors_for([n for n, _ in genres])
            self._style_breakdown.set_data(
                "Styles by genre",
                genres,
                stats.styles_by_genre,
                genre_colors=colors,
            )
            self._style_breakdown.show()
            n_styles = sum(len(v) for v in stats.styles_by_genre.values())
            self._chart_note.setText(
                f"{len(genres)} genres · {n_styles} styles — "
                f"% is share within that genre."
            )
            return

        if key == "sdr":
            # Histogram bins
            vals = stats.sdr.values
            if not vals:
                self._chart_note.setText("No SDR tags yet — enable Write SI-SDR on Classify.")
                return
            if stats.demo:
                bins = list(range(30, 56, 5))  # demo: 30–35 … 50–55
            else:
                lo = int(min(vals) // 5 * 5)
                hi = int((max(vals) + 4) // 5 * 5)
                if hi <= lo:
                    hi = lo + 5
                bins = list(range(lo, hi + 1, 5))
                if len(bins) < 2:
                    bins = [lo, lo + 5]
            counts = [0] * (len(bins) - 1)
            for v in vals:
                for i in range(len(counts)):
                    if bins[i] <= v < bins[i + 1]:
                        counts[i] += 1
                        break
            items = [
                (f"{bins[i]}–{bins[i+1]}", float(counts[i]))
                for i in range(len(counts))
            ]
            sdr_color = {name: theme.SDR_PASS_COLOR for name, _ in items}
            self._genre_bars.set_data(
                "SI-SDR (dB) distribution",
                items,
                colors=sdr_color,
                tooltip_unit="stems",
            )
            self._genre_bars.show()
            self._chart_note.setText(
                f"n={stats.sdr.count:,}  avg={stats.sdr.average:.1f} dB  "
                f"median={stats.sdr.median:.1f} dB — hover bars for counts."
            )
            return

    def _show_help(self) -> None:
        help_dialog(
            self,
            title="Charts help",
            heading="Library inventory & balance",
            intro=(
                "Scan your library roots to see role counts, tag distributions, "
                "and export a charts report. Demo charts show the layout until you scan."
            ),
            sections=[
                ("Paths", [
                    "Set Instrumental, Vocal, Pairs, and/or Samples roots. "
                    "Each role counts separately. Pairs add one unit to both Instrumental and Vocal.",
                ]),
                ("Scan", [
                    "Reads tags from the four roots and refreshes KPIs + charts. "
                    "Genre / Style from instrumentals and samples; Gender / Reverb / Vocal type from vocals.",
                ]),
                ("Balance", [
                    "Equalize roles (and optionally tags) into {OriginalRoot}_BALANCED folders. "
                    "Copy, Move, or export a CSV list.",
                ]),
                ("Charts", [
                    "Donuts for gender, reverb, vocal type, compression. "
                    "Bars for genre, keys, and SI-SDR. Hover bars for counts.",
                ]),
                ("Export", [
                    "Export… saves a PNG (dark-mode) or PDF (bright-mode, print-friendly) "
                    "charts spread. Notes are remembered for the next export.",
                ]),
                ("Sources", [
                    "SI-SDR scores: (SDR tag in metadata).",
                    "Genre / Style: (GENRE, STYLE tags in metadata).",
                    "Gender / Reverb: (GENDER, REVERB tags in metadata).",
                    "Vocal type: (VOCAL_TYPE tag in metadata).",
                    "Keys: (KEY tag in metadata).",
                    "Compression: (COMPRESSION tag in metadata).",
                ]),
            ],
        )

    def settings_snapshot(self) -> dict:
        return {
            "ds_inst_dir": display_path(self.inst_row.text()),
            "ds_vocal_dir": display_path(self.vocal_row.text()),
            "ds_pairs_dir": display_path(self.pairs_row.text()),
            "ds_samples_dir": display_path(self.samples_row.text()),
            "ds_include_subfolders": bool(self.include_subfolders.isChecked()),
        }

    def load_settings(self) -> None:
        self._loading = True
        try:
            d = self._settings.data
            if d.get("ds_inst_dir"):
                self.inst_row.set_text(d["ds_inst_dir"])
            if d.get("ds_vocal_dir"):
                self.vocal_row.set_text(d["ds_vocal_dir"])
            if d.get("ds_pairs_dir"):
                self.pairs_row.set_text(d["ds_pairs_dir"])
            if d.get("ds_samples_dir"):
                self.samples_row.set_text(d["ds_samples_dir"])
            self.include_subfolders.setChecked(
                bool(d.get("ds_include_subfolders", True))
            )
            # Cross-tab takeovers (Genre / Gender / Align / Rename).
            self._apply_cross_tab_paths(d)
        finally:
            self._loading = False

    def on_tab_shown(self) -> None:
        """Pull Genre / Gender / Align With-original / Rename into Charts paths."""
        self._loading = True
        try:
            self._apply_cross_tab_paths(self._settings.data)
        finally:
            self._loading = False
        self._flush_settings()

    def _apply_cross_tab_paths(self, d: dict | None = None) -> None:
        """Instrumental←Genre, Vocal←Gender, Pairs←Align With original, Samples←Rename."""
        d = d if d is not None else self._settings.data
        genre = (d.get("gg_genre_input_dir") or "").strip()
        gender = (d.get("gg_gender_input_dir") or "").strip()
        with_original = (d.get("align_with_original_dir") or "").strip()
        rename = (d.get("rename_folder") or "").strip()
        if genre:
            self.inst_row.set_text(genre)
        if gender:
            self.vocal_row.set_text(gender)
        if with_original:
            self.pairs_row.set_text(with_original)
        if rename:
            self.samples_row.set_text(rename)

    def _bind_autosave(self) -> None:
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(200)
        self._autosave_timer.timeout.connect(self._flush_settings)
        for sig in (
            self.inst_row.entry.textChanged,
            self.vocal_row.entry.textChanged,
            self.pairs_row.entry.textChanged,
            self.samples_row.entry.textChanged,
            self.include_subfolders.toggled,
        ):
            sig.connect(self._schedule_save)

    def _schedule_save(self, *_) -> None:
        if self._loading:
            return
        self._autosave_timer.start()

    def _flush_settings(self) -> None:
        if self._loading:
            return
        self._settings.merge(self.settings_snapshot())
        self._settings.flush()

    def flush_settings(self) -> None:
        if hasattr(self, "_autosave_timer"):
            self._autosave_timer.stop()
        self._flush_settings()


def register(window, settings: SettingsStore) -> None:
    tab = DatasetOverviewTab(settings)
    window.register_tab("Charts", tab)
    page = window.action_bar.make_page("Charts")
    tab.attach_action_bar(page, window)
    tab.request_status_running.connect(lambda: window.status_bar.show_running())
    tab.request_status_idle.connect(window.status_bar.show_idle)
    tab.request_status_text.connect(window.status_bar.set_status)
    tab.request_progress.connect(window.status_bar.update_progress)
    tab.request_log.connect(window.append_log)
    tab.request_clear_log.connect(window.clear_log)

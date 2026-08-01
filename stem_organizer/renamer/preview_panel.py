"""Preview panel — QTableView + QAbstractTableModel + lazy-compute worker.

Replaces the hand-rolled virtualized canvas list from
track_renamer.gui.preview_panel with idiomatic Qt model/view.

Columns: ✓ | Category | Keyword | Original | New
ANALYZE LOG (instrument ML output) lives in a hidden QTextEdit swapped in via
QStackedLayout during an enrich run.

Folder section headers (CTk port): when scanned paths include relative folders,
the view interleaves non-selectable folder bars above the tracks in each folder.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

from PySide6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QRect,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QStackedLayout,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    CaptionLabel,
    PushButton,
    RoundMenu,
    TableItemDelegate,
    TableView,
    TextEdit,
    ToggleButton,
)

from track_renamer.category_palette import default_category_color, parse_category_prefix_display
from track_renamer.engine.models import CategoryRule, PreviewRow, Track
from track_renamer.engine.ops import override_category_affix
from track_renamer.engine.processor import compute_preview_row, prepare_rules

from .. import theme
from ..widgets.log_panel import LOG_INDENT
from .theme import TIPS


# Columns: ✓ | Category | Keyword | Original | New
COL_CHECK = 0
COL_CATEGORY = 1
COL_KEYWORD = 2
COL_ORIGINAL = 3
COL_NEW = 4
N_COLS = 5

KEYWORD_COL_WIDTH = 100
EMPTY_KEYWORD = "—"

RESULT_BATCH_SIZE = 64
RESULT_POLL_MS = 40
LAZY_BUFFER_ROWS = 60
FOLDER_HEADER_FONT_PX = 10


@dataclass(frozen=True)
class _FolderHeader:
    """Non-track display row: folder section bar (matches CTk _FolderHeader)."""

    label: str


_ViewEntry = Union[int, _FolderHeader]


@dataclass
class _PreviewJob:
    generation: int
    tracks: List[Track]
    rules: object  # PreparedRulePlan
    cancel: threading.Event = field(default_factory=threading.Event)
    priority: "queue.SimpleQueue[int]" = field(default_factory=queue.SimpleQueue)
    results: "queue.SimpleQueue" = field(default_factory=queue.SimpleQueue)
    requested: set = field(default_factory=set)


def _build_folder_view(
    tracks: List[Track],
    source_indices: Iterable[int],
    root_label: str,
) -> List[_ViewEntry]:
    """Interleave folder headers with track indices — port of CTk _build_folder_view."""
    indices = list(source_indices)
    if not indices or not any(tracks[i].relative_path for i in indices):
        return list(indices)

    entries: List[_ViewEntry] = []
    previous_folder: Optional[str] = None
    for index in indices:
        relative_path = tracks[index].relative_path
        parent = Path(relative_path).parent
        if str(parent) == ".":
            folder = root_label or "ROOT"
        else:
            folder = " › ".join(parent.parts)
        if folder != previous_folder:
            entries.append(_FolderHeader(folder))
            previous_folder = folder
        entries.append(index)
    return entries


class PreviewModel(QAbstractTableModel):
    """Table model: display rows = folder headers + tracks; previews keyed by track index."""

    def __init__(self) -> None:
        super().__init__()
        self._tracks: List[Track] = []
        self._rows: List[Optional[PreviewRow]] = []
        self._view: List[_ViewEntry] = []
        self._track_to_display: dict[int, int] = {}
        self._root_label: str = "ROOT"
        self._category_colors: dict = {}
        self._sort_column: Optional[int] = None
        self._sort_order: Qt.SortOrder = Qt.AscendingOrder

    # ----- model API -----

    def set_tracks(self, tracks: List[Track], root_label: str = "") -> None:
        self.beginResetModel()
        self._tracks = list(tracks)
        self._rows = [None] * len(self._tracks)
        self._root_label = root_label or "ROOT"
        self._sort_column = None
        self._sort_order = Qt.AscendingOrder
        self._set_view(_build_folder_view(self._tracks, range(len(self._tracks)), self._root_label))
        self.endResetModel()

    @property
    def sort_column(self) -> Optional[int]:
        return self._sort_column

    @property
    def sort_order(self) -> Qt.SortOrder:
        return self._sort_order

    def sort_by(self, column: int) -> None:
        """Sort tracks within each folder group by *column*; toggle order on re-click.

        Folder header rows stay interleaved via ``_build_folder_view`` — never
        sorted as data rows (QSortFilterProxy would break that).
        """
        if not (0 <= column < N_COLS):
            return
        if self._sort_column == column:
            self._sort_order = (
                Qt.DescendingOrder
                if self._sort_order == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        else:
            self._sort_column = column
            self._sort_order = Qt.AscendingOrder

        indices = [e for e in self._view if isinstance(e, int)]
        if not indices:
            return

        # Keep folder groups contiguous; sort only inside each group.
        groups: List[List[int]] = []
        previous_folder: Optional[str] = None
        for track_idx in indices:
            folder = self._folder_key(track_idx)
            if not groups or folder != previous_folder:
                groups.append([])
                previous_folder = folder
            groups[-1].append(track_idx)

        reverse = self._sort_order == Qt.DescendingOrder
        sorted_indices: List[int] = []
        for group in groups:
            group.sort(key=lambda i: self._sort_key(i, column), reverse=reverse)
            sorted_indices.extend(group)

        self.beginResetModel()
        self._set_view(
            _build_folder_view(self._tracks, sorted_indices, self._root_label)
        )
        self.endResetModel()

    def _folder_key(self, track_idx: int) -> str:
        """Same folder label as ``_build_folder_view`` for a track index."""
        relative_path = self._tracks[track_idx].relative_path
        parent = Path(relative_path).parent
        if str(parent) == ".":
            return self._root_label or "ROOT"
        return " › ".join(parent.parts)

    def _sort_key(self, track_idx: int, column: int):
        """Stable, case-insensitive key matching the visible cell value."""
        track = self._tracks[track_idx]
        preview = self._rows[track_idx] if 0 <= track_idx < len(self._rows) else None

        if column == COL_CHECK:
            return (0 if track.selected else 1,)

        if column == COL_CATEGORY:
            if preview is None:
                return ("",)
            known = {k: k for k in self._category_colors} or None
            parsed = parse_category_prefix_display(preview.new_display, known=known)
            return ((parsed[0] if parsed else "").lower(),)

        if column == COL_KEYWORD:
            if preview is None:
                return ("",)
            return ((preview.matched_keyword or "").strip().lower(),)

        if column == COL_ORIGINAL:
            return (track.display_name.lower(),)

        if column == COL_NEW:
            # Display shows "unchanged" when deselected or not changing — use
            # empty key so those rows cluster together (not mixed with real names).
            if preview is None or not track.selected or not preview.changed:
                return ("",)
            return (preview.new_display.lower(),)

        return ("",)

    def set_category_colors(self, colors: dict) -> None:
        self._category_colors = dict(colors or {})
        n = self.rowCount()
        if n:
            # Repaint Category badges when palette changes (Apply / live refresh).
            top = self.index(0, COL_CATEGORY)
            bottom = self.index(n - 1, COL_CATEGORY)
            self.dataChanged.emit(top, bottom, [Qt.DisplayRole, Qt.BackgroundRole])

    def is_folder_header(self, display_row: int) -> bool:
        if 0 <= display_row < len(self._view):
            return isinstance(self._view[display_row], _FolderHeader)
        return False

    def folder_label_at(self, display_row: int) -> str:
        if 0 <= display_row < len(self._view):
            entry = self._view[display_row]
            if isinstance(entry, _FolderHeader):
                return entry.label
        return ""

    def track_index_at(self, display_row: int) -> Optional[int]:
        if 0 <= display_row < len(self._view):
            entry = self._view[display_row]
            if isinstance(entry, int):
                return entry
        return None

    def display_row_for_track(self, track_idx: int) -> Optional[int]:
        return self._track_to_display.get(track_idx)

    def track_at(self, display_row: int) -> Optional[Track]:
        track_idx = self.track_index_at(display_row)
        if track_idx is None:
            return None
        if 0 <= track_idx < len(self._tracks):
            return self._tracks[track_idx]
        return None

    def row_at(self, display_row: int) -> Optional[PreviewRow]:
        track_idx = self.track_index_at(display_row)
        if track_idx is None:
            return None
        if 0 <= track_idx < len(self._rows):
            return self._rows[track_idx]
        return None

    def preview_for_track(self, track_idx: int) -> Optional[PreviewRow]:
        if 0 <= track_idx < len(self._rows):
            return self._rows[track_idx]
        return None

    def set_row(self, track_idx: int, preview: PreviewRow) -> None:
        """Store a computed preview by track index (not display row)."""
        if not (0 <= track_idx < len(self._rows)):
            return
        self._rows[track_idx] = preview
        display = self._track_to_display.get(track_idx)
        if display is None:
            return
        ix = self.index(display, 0)
        ix2 = self.index(display, N_COLS - 1)
        self.dataChanged.emit(ix, ix2, [Qt.DisplayRole, Qt.BackgroundRole, Qt.FontRole])

    def update_selection(self, display_row: int, value: bool) -> None:
        track = self.track_at(display_row)
        if track is None:
            return
        track.selected = bool(value)
        # CheckState + name styling (strikethrough clears when deselected)
        left = self.index(display_row, COL_CHECK)
        right = self.index(display_row, COL_NEW)
        self.dataChanged.emit(
            left,
            right,
            [Qt.CheckStateRole, Qt.FontRole, Qt.ForegroundRole, Qt.DisplayRole],
        )

    def update_selection_for_track(self, track_idx: int, value: bool) -> None:
        if not (0 <= track_idx < len(self._tracks)):
            return
        self._tracks[track_idx].selected = bool(value)
        display = self._track_to_display.get(track_idx)
        if display is None:
            return
        left = self.index(display, COL_CHECK)
        right = self.index(display, COL_NEW)
        self.dataChanged.emit(
            left,
            right,
            [Qt.CheckStateRole, Qt.FontRole, Qt.ForegroundRole, Qt.DisplayRole],
        )

    def _set_view(self, entries: List[_ViewEntry]) -> None:
        self._view = entries
        self._track_to_display = {
            entry: i for i, entry in enumerate(entries) if isinstance(entry, int)
        }

    # ----- QAbstractTableModel -----

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._view)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else N_COLS

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation != Qt.Horizontal:
            return None
        if role == Qt.TextAlignmentRole:
            if section == COL_CHECK:
                return Qt.AlignHCenter | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        if role == Qt.ToolTipRole:
            if section == COL_CHECK:
                return TIPS["file_checkbox"]
            if section == COL_KEYWORD:
                return TIPS.get("detected_keyword", "Keyword that triggered the category match.")
            return None
        if role != Qt.DisplayRole:
            return None
        return ("✓", "Category", "Keyword", "Original", "New")[section]

    def data(self, index, role=Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()

        if self.is_folder_header(row):
            if role == Qt.BackgroundRole:
                return QColor(theme.DARK["panel_2"])
            if role == Qt.ForegroundRole:
                return QColor(theme.DARK["text_mute"])
            if role == Qt.FontRole:
                font = QFont(theme.FONT_FAMILY)
                font.setPixelSize(FOLDER_HEADER_FONT_PX)
                font.setBold(True)
                return font
            if role == Qt.TextAlignmentRole:
                return Qt.AlignLeft | Qt.AlignVCenter
            if role == Qt.DisplayRole and col == COL_CHECK:
                # Spanned across all columns; text lives in the first cell.
                return f"  {self.folder_label_at(row)}"
            return None

        track = self.track_at(row)
        if track is None:
            return None
        preview = self.row_at(row)

        if role == Qt.CheckStateRole and col == COL_CHECK:
            return Qt.Checked if track.selected else Qt.Unchecked
        if role == Qt.ToolTipRole and col == COL_CHECK:
            return TIPS["file_checkbox"]
        if role == Qt.TextAlignmentRole:
            if col == COL_CHECK:
                return Qt.AlignHCenter | Qt.AlignVCenter
            if col == COL_CATEGORY:
                return Qt.AlignCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        if role == Qt.BackgroundRole:
            # Active-row highlight handled by selection model; just alternate rows
            if row % 2 == 1:
                return QColor(theme.COLORS["panel"])
            return None
        if role == Qt.ForegroundRole:
            # COL_CATEGORY is painted by _CategoryBadgeDelegate (badge bg + white text).
            if col == COL_CATEGORY:
                return None
            # Strikethrough / emphasis only while selected — deselected rows look normal.
            active_change = (
                preview is not None and preview.changed and track.selected
            )
            if col == COL_ORIGINAL and active_change:
                return QColor(theme.DARK["text_mute"])
            if col == COL_NEW and active_change:
                return QColor(theme.DARK["text"])
            if col == COL_NEW:
                # Deselected or not changing — same dim as former "unchanged" status
                return QColor(theme.DARK["unchanged"])
            if col == COL_KEYWORD:
                return QColor(theme.DARK["text_mute"])
            if col == COL_ORIGINAL:
                return QColor(theme.DARK["text"])
            return QColor(theme.DARK["text"])
        if role == Qt.FontRole:
            font = QFont(theme.FONT_FAMILY)
            font.setPixelSize(theme.BODY_FONT_PX)
            active_change = (
                preview is not None and preview.changed and track.selected
            )
            if col == COL_NEW and active_change:
                font.setBold(True)
            if col == COL_NEW and (
                preview is None or not track.selected or not preview.changed
            ):
                font.setItalic(True)
            if col == COL_ORIGINAL and active_change:
                font.setStrikeOut(True)
            return font
        if role == Qt.DisplayRole:
            if col == COL_ORIGINAL:
                return track.display_name
            if col == COL_NEW:
                if preview is None:
                    return "…"
                # Deselected → treat as not renaming (dimmed "unchanged" label)
                if not track.selected:
                    return "unchanged"
                if not preview.changed:
                    return "unchanged"
                return preview.new_display
            if col == COL_KEYWORD:
                if preview is None:
                    return ""
                kw = (preview.matched_keyword or "").strip()
                return kw if kw else EMPTY_KEYWORD
            if col == COL_CATEGORY:
                if preview is None:
                    return ""
                known = {k: k for k in self._category_colors} or None
                parsed = parse_category_prefix_display(
                    preview.new_display, known=known
                )
                return parsed[0].upper() if parsed else ""
        return None

    def flags(self, index):  # noqa: N802
        if not index.isValid():
            return Qt.NoItemFlags
        if self.is_folder_header(index.row()):
            # Visible but not selectable / checkable — skip with keyboard & click.
            return Qt.ItemIsEnabled
        # Check column is toggled by PreviewPanel click handler — Fluent's painted
        # checkbox sits outside Qt's ItemIsUserCheckable hit box, so relying on
        # setData alone makes uncheck work and re-check fail.
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def setData(self, index, value, role=Qt.EditRole):  # noqa: N802
        if role == Qt.CheckStateRole and index.column() == COL_CHECK:
            if self.is_folder_header(index.row()):
                return False
            checked = value in (
                Qt.Checked,
                getattr(Qt.CheckState, "Checked", Qt.Checked),
                2,
            )
            self.update_selection(index.row(), bool(checked))
            return True
        return False


class _CenteredCheckDelegate(TableItemDelegate):
    """Fluent TableItemDelegate pins the checkbox at rect.x+15; center it.

    Active-row left indicator is white (Fluent default falls back to accent).
    Checkbox checked fill stays theme accent via light/darkCheckedColor.
    Folder header rows paint as a full-width muted bar (CTk panel_2).
    """

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        model = index.model()
        if model is not None and getattr(model, "is_folder_header", lambda _r: False)(index.row()):
            self._paint_folder_header(painter, option, index)
            return
        super().paint(painter, option, index)

    @staticmethod
    def _paint_folder_header(painter: QPainter, option, index) -> None:
        rect = option.rect
        painter.save()
        painter.fillRect(rect, QColor(theme.DARK["panel_2"]))
        text = index.data(Qt.DisplayRole) or ""
        font = QFont(theme.FONT_FAMILY)
        font.setPixelSize(FOLDER_HEADER_FONT_PX)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(theme.DARK["text_mute"]))
        painter.drawText(rect.adjusted(6, 0, -4, 0), Qt.AlignLeft | Qt.AlignVCenter, str(text))
        painter.restore()

    def _drawIndicator(self, painter: QPainter, option, index) -> None:  # noqa: N802
        # Thin left bar for the focused/selected row — soft white, not accent lavender.
        y, h = option.rect.y(), option.rect.height()
        ph = round(0.35 * h if self.pressedRow == index.row() else 0.257 * h)
        painter.setBrush(QColor(theme.DARK["text"]))
        painter.drawRoundedRect(4, ph + y, 3, h - 2 * ph, 1.5, 1.5)

    def _drawCheckBox(self, painter: QPainter, option, index) -> None:  # noqa: N802
        size = 19
        # Fluent draws at option.rect.x() + 15 — shift rect so that lands centered.
        centered = option.rect.x() + (option.rect.width() - size) / 2.0
        opt = QStyleOptionViewItem(option)
        opt.rect = QRect(
            int(round(centered - 15)),
            option.rect.y(),
            option.rect.width(),
            option.rect.height(),
        )
        super()._drawCheckBox(painter, opt, index)


class _CategoryBadgeDelegate(QStyledItemDelegate):
    """Paints COL_CATEGORY with the same ChipRenderer pixmap as Rename/Classify
    log chips: Arial Bold 11px, white text, solid fill, fixed height, sharp
    corners (no radius — log pixmaps are filled rects). Centered with inset so
    dark row/grid lines stay visible between adjacent cells.
    """

    # Keep a dark gap around the chip so row separators remain visible.
    _BADGE_INSET_H = 2
    _BADGE_INSET_V = 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from ..widgets.log_panel import ChipRenderer

        self._chips = ChipRenderer()

    @staticmethod
    def _chip_font() -> QFont:
        from ..widgets.log_panel import LOG_CHIP_FONT_PX

        font = QFont("Arial")
        font.setBold(True)
        font.setPixelSize(LOG_CHIP_FONT_PX)
        if not QFontMetrics(font).inFont(ord("A")):
            font = QFont(theme.FONT_FAMILY_SEMIBOLD)
            font.setBold(True)
            font.setPixelSize(LOG_CHIP_FONT_PX)
        return font

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        model = index.model()
        if model is not None and getattr(model, "is_folder_header", lambda _r: False)(index.row()):
            return
        label = (index.data(Qt.DisplayRole) or "").strip()
        if not label:
            return
        color_hex = ""
        if hasattr(model, "_category_colors"):
            preview = model.row_at(index.row())
            if preview is not None:
                known = {k: k for k in model._category_colors} or None
                parsed = parse_category_prefix_display(
                    preview.new_display, known=known
                )
                if parsed:
                    cat = parsed[0]
                    color_hex = model._category_colors.get(cat) or default_category_color(cat)
        if not color_hex:
            color_hex = theme.DARK.get("text_mute", "#6b7080")

        chip_text = label.lower()
        pix = self._chips.category_pixmap(chip_text, color_hex)
        cell = option.rect
        # Erase default cell text (DisplayRole) before drawing the chip pixmap.
        bg = option.backgroundBrush.color() if option.backgroundBrush.style() != Qt.NoBrush else QColor(
            theme.COLORS["panel"] if index.row() % 2 else theme.COLORS["panel2"]
        )
        if option.state & QStyle.State_Selected:
            bg = option.palette.color(option.palette.ColorRole.Highlight)
        avail = cell.adjusted(
            self._BADGE_INSET_H,
            self._BADGE_INSET_V,
            -self._BADGE_INSET_H,
            -self._BADGE_INSET_V,
        )
        # Log chips are fixed compact height/width — center them; do not stretch
        # into a full-cell bar (that read as sharper/blockier than the log).
        x = avail.x() + max(0, (avail.width() - pix.width()) // 2)
        y = avail.y() + max(0, (avail.height() - pix.height()) // 2)

        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.fillRect(cell, bg)
        painter.setClipRect(avail)
        painter.drawPixmap(x, y, pix)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802
        from PySide6.QtCore import QSize

        label = (index.data(Qt.DisplayRole) or "").strip().lower() or "xxxx"
        pix = self._chips.category_pixmap(
            label, theme.DARK.get("text_mute", "#6b7080")
        )
        return QSize(
            pix.width() + 2 * self._BADGE_INSET_H,
            pix.height() + 2 * self._BADGE_INSET_V,
        )


class PreviewPanel(QWidget):
    """Right side: preview table + tools + analyze log."""

    on_change: Optional[Callable[[], None]] = None
    on_active: Optional[Callable[[Optional[Track], Optional[PreviewRow]], None]] = None
    on_play_pause: Optional[Callable[[], None]] = None
    on_seek: Optional[Callable[[float], None]] = None
    # Fired after Change to: prefix override with {track_id: new_stem} to rename on disk.
    on_override_rename: Optional[Callable[[dict], None]] = None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("PreviewPanel")
        self._tracks: List[Track] = []
        self._root_label: str = ""
        self._lazy_generation = 0
        self._lazy_done = 0
        self._lazy_total = 0
        self._lazy_changed = 0
        self._lazy_selected_changed = 0
        self._job: Optional[_PreviewJob] = None
        self._only_changed = False
        self._active_index: Optional[int] = None
        self._pending_state = False
        self._name_cols_balanced = False
        self._category_options: List[CategoryRule] = []

        self._build_ui()

        # Result poll timer
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(RESULT_POLL_MS)
        self._poll_timer.timeout.connect(self._drain_results)

    # ----- build -----

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # Top 0 — RULES title is the vertical anchor; TrackRenamerApp pads above us
        layout.setContentsMargins(8, 0, 8, 8)
        layout.setSpacing(6)

        # Header — match RULES row height / dim section title
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = CaptionLabel("PREVIEW")
        title.setObjectName("SectionTitle")
        title.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-size: {theme.SECTION_TITLE_PX}px; "
            f'font-family: "{theme.FONT_FAMILY}"; font-weight: 600; background: transparent;'
        )
        title.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.section_title = title
        # Explicit VCenter: SectionTitle QSS max-height vs fixed 30px otherwise
        # shifts this label a few px below the stats CaptionLabel in the row.
        header.addWidget(title, 0, Qt.AlignVCenter)
        header.addStretch(1)
        self.stats_label = CaptionLabel("0 files")
        self.stats_label.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-size: {theme.SECTION_TITLE_PX}px; "
            f'font-family: "{theme.FONT_FAMILY}"; background: transparent; padding: 0px;'
        )
        self.stats_label.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self.stats_label, 0, Qt.AlignVCenter)
        layout.addLayout(header)

        # Tools
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        self.select_all_btn = PushButton("Select all")
        self.select_all_btn.setToolTip(TIPS["select_all"])
        self.select_all_btn.clicked.connect(self._select_all)
        self.deselect_all_btn = PushButton("Deselect all")
        self.deselect_all_btn.setToolTip(TIPS["deselect_all"])
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        self.only_changed_btn = ToggleButton("✓ Only changed")
        self.only_changed_btn.setCheckable(True)
        self.only_changed_btn.setToolTip(TIPS["only_changed"])
        self.only_changed_btn.toggled.connect(self._on_only_changed_toggled)
        for b in (self.select_all_btn, self.deselect_all_btn, self.only_changed_btn):
            tools.addWidget(b)
        tools.addStretch(1)
        layout.addLayout(tools)

        # Stacked: table / analyze log
        stacked_host = QWidget()
        self._stack = QStackedLayout(stacked_host)

        self.table = TableView()
        self.model = PreviewModel()
        self.table.setModel(self.model)
        # Replace Fluent default so ✓ checkboxes center in the narrow first column
        # (keeps TableBase.delegate / hover+selection wiring via setItemDelegate).
        self.table.setItemDelegate(_CenteredCheckDelegate(self.table))
        # TableView subclasses QTableView; prefer TableView enums when present
        _select_rows = getattr(TableView, "SelectRows", QAbstractItemView.SelectRows)
        _extended = getattr(
            TableView, "ExtendedSelection", QAbstractItemView.ExtendedSelection
        )
        _no_edit = getattr(TableView, "NoEditTriggers", QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(_select_rows)
        # Shift = contiguous range; Ctrl = toggle individual rows (Windows/Linux).
        self.table.setSelectionMode(_extended)
        self.table.setEditTriggers(_no_edit)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # QTableView.wordWrap defaults True: names wrap at " - " and only the first
        # line is visible in our single-line row height → looks like early truncate.
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.setMinimumSectionSize(48)
        # Check Fixed; Category + Keyword + Original Interactive; New Stretch
        # fills leftover so viewport stays full without a horizontal scrollbar.
        # Dragging Original|New resizes both (Interactive next to Stretch).
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_CHECK, QHeaderView.Fixed)
        hdr.resizeSection(COL_CHECK, 34)
        hdr.resizeSection(COL_CATEGORY, 80)
        hdr.resizeSection(COL_KEYWORD, KEYWORD_COL_WIDTH)
        hdr.resizeSection(COL_ORIGINAL, 240)  # seed; balanced to ~½ remaining on first layout
        hdr.setSectionResizeMode(COL_CATEGORY, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_KEYWORD, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_ORIGINAL, QHeaderView.Interactive)
        hdr.setSectionResizeMode(COL_NEW, QHeaderView.Stretch)
        hdr.setStretchLastSection(False)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hdr.setSortIndicatorShown(False)
        hdr.sectionClicked.connect(self._on_header_clicked)
        hdr.sectionDoubleClicked.connect(self._on_header_double_clicked)
        self.table.verticalHeader().setDefaultSectionSize(26)
        # Category cell → log-style chip fill (inset so row lines show)
        self.table.setItemDelegateForColumn(COL_CATEGORY, _CategoryBadgeDelegate(self.table))
        self.table.clicked.connect(self._on_table_clicked)
        # Painted ✓ cells are not QCheckBox widgets — hand cursor on COL_CHECK hover.
        self.table.setMouseTracking(True)
        self.table.viewport().setMouseTracking(True)
        self.table.viewport().installEventFilter(self)
        self.table.setToolTip(TIPS.get("change_prefix", ""))
        self._stack.addWidget(self.table)
        QTimer.singleShot(0, self._fit_category_column)
        QTimer.singleShot(0, self._balance_filename_columns)

        # Auto-detect log — TextEdit (not PlainTextEdit) so painted chip images
        # render (QPlainTextEdit only shows a placeholder glyph for insertImage).
        # Same LogView styling + Fluent focus-layer suppression as widgets.LogPanel.
        self.analyze_log = TextEdit()
        self.analyze_log.setObjectName("LogView")
        self.analyze_log.setReadOnly(True)
        self.analyze_log.setFont(theme.F_LOG)
        layer = getattr(self.analyze_log, "layer", None)
        if layer is not None:
            layer.hide()
            layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        t = theme.DARK
        c = theme.COLORS
        log_qss = f"""
            TextEdit#LogView,
            TextEdit#LogView:hover,
            TextEdit#LogView:focus {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                font-family: "{theme.FONT_FAMILY_MONO}";
                font-size: {theme.LOG_FONT_PX}px;
                border: 1px solid {t['border']};
                border-bottom: 1px solid {t['border']};
                border-radius: {theme.LOG_VIEW_CORNER_RADIUS}px;
                padding: 8px;
                selection-background-color: {c['accent']};
                selection-color: {c['log_fg']};
            }}
        """
        self.analyze_log.setStyleSheet(log_qss)
        try:
            from qfluentwidgets import setCustomStyleSheet
            setCustomStyleSheet(self.analyze_log, log_qss, log_qss)
        except Exception:
            pass
        
        # Shared chip / format / line-spacing renderer — identical to Classify LOG.
        from ..widgets.log_panel import ChipRenderer

        self._chips = ChipRenderer()
        self._stack.addWidget(self.analyze_log)
        self._stack.setCurrentWidget(self.table)

        layout.addWidget(stacked_host, stretch=1)

        # Keyboard shortcuts (documented in the Rename help dialog).
        # QShortcut with WidgetWithChildrenShortcut works regardless of which
        # child (table, header) currently holds focus — a bare keyPressEvent on
        # the panel never fires because the QTableView eats Up/Down/PageUp/Down.
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence(Qt.Key_Up), self, self._kb_prev, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Down), self, self._kb_next, context=ctx)
        QShortcut(QKeySequence(Qt.Key_PageUp), self, self._kb_page_up, context=ctx)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, self._kb_page_down, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._kb_toggle_play, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Left), self, self._kb_seek_back, context=ctx)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._kb_seek_fwd, context=ctx)
        # Enter toggles ✓ for the active row — scoped to the table so line edits
        # / dialogs / tool buttons keep their own Enter behavior.
        table_ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence(Qt.Key_Return), self.table, self._kb_toggle_check, context=table_ctx)
        QShortcut(QKeySequence(Qt.Key_Enter), self.table, self._kb_toggle_check, context=table_ctx)

    # ----- public API (port method names) -----

    def set_rows(self, rows: List[PreviewRow]) -> None:
        self.model.set_tracks([r.track for r in rows], self._root_label)
        for i, r in enumerate(rows):
            self.model.set_row(i, r)
        self._sync_folder_spans()
        self._apply_only_changed_filter()
        self._clear_sort_indicator()
        self._update_stats()
        self._fit_category_column()

    def _sync_folder_spans(self) -> None:
        """Span folder header rows across all columns (full-width bar)."""
        try:
            self.table.clearSpans()
        except RuntimeError:
            return
        for i in range(self.model.rowCount()):
            if self.model.is_folder_header(i):
                self.table.setSpan(i, COL_CHECK, 1, N_COLS)

    def _clear_sort_indicator(self) -> None:
        """Hide header sort arrow after a full track reload (sort state reset)."""
        try:
            hdr = self.table.horizontalHeader()
        except RuntimeError:
            return
        hdr.setSortIndicatorShown(False)

    def _fit_category_column(self) -> None:
        """Widen Category so the longest chip label (and header) fit."""
        from track_renamer.category_palette import DEFAULT_CATEGORY_COLORS

        fm = QFontMetrics(_CategoryBadgeDelegate._chip_font())
        labels = {"category"}  # header baseline (lowercase measure)
        for name in DEFAULT_CATEGORY_COLORS:
            labels.add(name.lower())
        for i in range(self.model.rowCount()):
            if self.model.is_folder_header(i):
                continue
            preview = self.model.row_at(i)
            if preview is None:
                continue
            parsed = parse_category_prefix_display(preview.new_display)
            if parsed:
                labels.add(parsed[0].lower())
        text_w = max((fm.horizontalAdvance(t) for t in labels), default=40)
        try:
            hdr = self.table.horizontalHeader()
        except RuntimeError:
            return
        header_w = hdr.fontMetrics().horizontalAdvance("Category")
        # Same geometry as ChipRenderer.category_pixmap (shared min width, grows).
        chip_w = max(getattr(self._chips, "chip_width_px", 0), text_w + 16)
        width = max(chip_w + 2 * _CategoryBadgeDelegate._BADGE_INSET_H, header_w + 20, 56)
        hdr.resizeSection(COL_CATEGORY, width)
        # Category width changed remaining space — re-split Original/New once if needed.
        if not self._name_cols_balanced:
            self._balance_filename_columns()

    def _balance_filename_columns(self) -> None:
        """Give Original ~half of space after ✓+Category+Keyword; New Stretch takes the rest.

        Runs once so user drag-resizes are not overwritten on later data updates.
        """
        if self._name_cols_balanced:
            return
        try:
            hdr = self.table.horizontalHeader()
            vw = self.table.viewport().width()
        except RuntimeError:
            return
        if vw < 120:
            return
        fixed = (
            hdr.sectionSize(COL_CHECK)
            + hdr.sectionSize(COL_CATEGORY)
            + hdr.sectionSize(COL_KEYWORD)
        )
        remaining = max(0, vw - fixed)
        # Half for Original; Stretch New fills the other half (and any slack).
        orig = max(160, remaining // 2)
        hdr.resizeSection(COL_ORIGINAL, orig)
        self._name_cols_balanced = True

    def set_category_options(self, categories: List[CategoryRule]) -> None:
        """Category Macro rows for the right-click 'Change to:' override menu."""
        self._category_options = list(categories or [])

    def set_loading(self, loading: bool) -> None:
        # Simple: just dim the stats label
        self.stats_label.setText("Loading…" if loading else self.stats_label.text())

    def set_preview_pending(self, pending: bool) -> None:
        self._pending_state = pending
        if pending:
            self.stats_label.setText("Rules changed — click Apply")
        else:
            self._update_stats()

    def clear_active(self) -> None:
        self._active_index = None
        self.table.clearSelection()
        if self.on_active:
            self.on_active(None, None)

    def cancel_preview_work(self) -> None:
        if self._job is not None:
            self._job.cancel.set()

    def begin_viewport_lazy(self, tracks: List[Track], rules: List[Rule], root_label: str = "") -> None:
        self.cancel_preview_work()
        self._tracks = list(tracks)
        self._root_label = root_label or "ROOT"
        self._lazy_generation += 1
        gen = self._lazy_generation
        self._lazy_done = 0
        self._lazy_total = len(tracks)
        self._lazy_changed = 0
        self._lazy_selected_changed = 0

        try:
            prepared = prepare_rules(rules)
        except Exception:
            prepared = rules

        self.model.set_tracks(tracks, self._root_label)
        self._sync_folder_spans()
        self._apply_only_changed_filter()
        self._clear_sort_indicator()
        self._fit_category_column()

        # Seed priority with first ~LAZY_BUFFER_ROWS
        self._job = _PreviewJob(generation=gen, tracks=self._tracks, rules=prepared)
        # Priority seed only — do NOT mark requested yet or the worker skips them
        for i in range(min(LAZY_BUFFER_ROWS, len(tracks))):
            self._job.priority.put(i)

        worker = threading.Thread(target=self._preview_worker, args=(self._job,), daemon=True)
        worker.start()
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        self._update_stats()

    def lazy_compute_complete(self) -> bool:
        return self._lazy_done >= self._lazy_total

    def lazy_compute_progress(self):
        return (self._lazy_done, self._lazy_total)

    def selected_renames(self) -> dict:
        if not self.lazy_compute_complete():
            return {}
        result = {}
        for i, track in enumerate(self._tracks):
            row = self.model.preview_for_track(i)
            if row is None or not row.changed or not track.selected:
                continue
            result[track.id] = row.new_name
        return result

    def rename_count(self) -> int:
        if not self.lazy_compute_complete():
            return self._lazy_selected_changed
        n = 0
        for i, track in enumerate(self._tracks):
            row = self.model.preview_for_track(i)
            if row is not None and row.changed and track.selected:
                n += 1
        return n

    def shutdown(self) -> None:
        self._poll_timer.stop()
        if self._job is not None:
            self._job.cancel.set()

    # ----- ANALYZE LOG -----
    # Rendering mirrors widgets.LogPanel (Classify LOG): painted chip pixmaps,
    # Consolas mono tag colors, LOG_LINE_EXTRA_PX line spacing. Categories use
    # the same color source as the preview table badge (model._category_colors
    # / default_category_color) so log chips and table chips stay in sync.

    def begin_analyze_log(self, total: int) -> None:
        self._stack.setCurrentWidget(self.analyze_log)
        self.analyze_log.clear()
        # Startup/config indented like Classify; === Summary stays flush.
        self._append_line(f"{LOG_INDENT}Starting Auto-detect (PaSST OpenMIC)…", "info")
        self._append_line(f"{LOG_INDENT}Selected {total:,} file(s).", "info")
        self._append_line(f"{LOG_INDENT}Checking cache / starting tagger…", "info")
        self._append_line("", "info")

    def append_analyze_status(self, message: str) -> None:
        text = message or ""
        if text and not text.startswith((" ", "=", "-")):
            text = f"{LOG_INDENT}{text}"
        self._append_line(text, "info")

    def append_analyze_log(self, *, filename: str, action: str, category: str = "",
                           score: float = 0.0, label: str = "", total: int = 0,
                           done: int = 0, reason: str = "") -> None:
        from ..run_summary import file_progress_header

        # Header matches Classify / Genre: === [i/n] name ===
        self._append_line(file_progress_header(filename, done, total), "info")

        # Chip row under header (no repeated filename)
        cursor = self.analyze_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._chips.apply_line_spacing(cursor)
        self._chips.insert(cursor, "  ")

        if action == "apply":
            cat = (category or "—").strip()
            self._chips.insert_category_chip(cursor, cat, self._category_color(cat))
            self._chips.insert(cursor, f"  {score * 100:>3.0f}%", "log_pct")
        elif action == "error":
            self._chips.insert_chip(cursor, "skip")
            self._chips.insert(cursor, "   —   ", "log_pct")
            if reason:
                self._chips.insert(cursor, f"  ({reason})", "err")
        else:
            self._chips.insert_chip(cursor, "skip")
            self._chips.insert(cursor, f"  {score * 100:>3.0f}%", "log_pct")
            extra = reason or ""
            if label:
                extra = f"{extra} · {label}" if extra else label
            if extra:
                self._chips.insert(cursor, f"  ({extra})", "warn")

        self._chips.insert(cursor, "\n")
        self.analyze_log.setTextCursor(cursor)
        self.analyze_log.ensureCursorVisible()

    def append_analyze_summary(self, *, elapsed_sec: float, total: int) -> None:
        from ..run_summary import emit_run_summary

        def log(msg: str, tag: str) -> None:
            self._append_line(msg, tag)

        emit_run_summary(log, "Instrument", elapsed=elapsed_sec, files=total)
        try:
            from done_sound import play_done_sound
            play_done_sound()
        except Exception:
            pass

    def end_analyze_log(self) -> None:
        self._stack.setCurrentWidget(self.table)

    # ----- internals -----

    def _category_color(self, category: str) -> str:
        """Color for a category chip — same source as the table badge."""
        colors = getattr(self.model, "_category_colors", {}) or {}
        hex_val = colors.get(category)
        if hex_val:
            return hex_val
        return default_category_color(category)

    def _append_line(self, text: str, tag: str) -> None:
        """Append one status/summary line with Classify-LOG styling + spacing."""
        from ..widgets.log_panel import GG_PROGRESS_HEADER_RE

        cursor = self.analyze_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._chips.apply_line_spacing(cursor)
        prog_h = GG_PROGRESS_HEADER_RE.match((text or "").strip())
        if prog_h:
            counter, rest = prog_h.group(1), prog_h.group(2)
            self._chips.insert(cursor, "=== ", "info")
            self._chips.insert(cursor, counter, "detail")  # log_fg
            self._chips.insert(cursor, f" {rest} ===\n", "info")
        else:
            self._chips.insert(cursor, text + "\n", tag)
        self.analyze_log.setTextCursor(cursor)
        self.analyze_log.ensureCursorVisible()

    @staticmethod
    def _preview_worker(job: _PreviewJob) -> None:
        batch: List[tuple] = []
        next_idx = 0
        last_flush = time.monotonic()
        while not job.cancel.is_set():
            try:
                idx = job.priority.get_nowait()
            except queue.Empty:
                if next_idx >= len(job.tracks):
                    break
                idx = next_idx
                next_idx += 1
                if idx in job.requested:
                    continue
            if idx >= len(job.tracks) or idx in job.requested:
                continue
            try:
                row = compute_preview_row(job.tracks[idx], job.rules, index=idx + 1)
            except Exception:
                # Mark done so we don't spin forever; leave an empty slot in the model.
                job.requested.add(idx)
                continue
            job.requested.add(idx)
            batch.append((idx, row))
            if len(batch) >= RESULT_BATCH_SIZE or (time.monotonic() - last_flush) > 0.025:
                job.results.put((list(batch), False))
                batch.clear()
                last_flush = time.monotonic()
        if batch:
            job.results.put((list(batch), False))
        job.results.put(([], True))

    def _drain_results(self) -> None:
        if self._job is None:
            self._poll_timer.stop()
            return
        finished = False
        drained = 0
        while drained < 64:
            try:
                batch, done = self._job.results.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if done:
                finished = True
                continue
            for idx, row in batch:
                if self._job.generation != self._lazy_generation:
                    continue
                self.model.set_row(idx, row)
                self._lazy_done += 1
                if row.changed:
                    self._lazy_changed += 1
                    if row.track.selected:
                        self._lazy_selected_changed += 1
                # Active-row callback: deliver the freshly-computed PreviewRow
                if idx == self._active_index and self.on_active is not None:
                    self.on_active(row.track, row)
        if self._only_changed and drained:
            self._apply_only_changed_filter()
        # Re-seed priority for the visible viewport
        self._seed_visible_priority()
        self._update_stats()
        if self.on_change:
            self.on_change()
        job = self._job
        if job is None:
            self._poll_timer.stop()
            return
        # Only stop when *this* generation's worker is done (avoid killing a fresh job)
        if job.generation != self._lazy_generation:
            return
        if finished or self._lazy_done >= self._lazy_total:
            self._poll_timer.stop()
            self._fit_category_column()
            if self._only_changed:
                self._apply_only_changed_filter()
            # Worker finished but some rows were skipped — treat as complete for UI
            if self._lazy_done < self._lazy_total and finished:
                self._lazy_done = self._lazy_total
                self._update_stats()

    def _seed_visible_priority(self) -> None:
        if self._job is None or not self._tracks:
            return
        first = self.table.rowAt(0)
        last = self.table.rowAt(self.table.viewport().height() - 1)
        if first < 0:
            first = 0
        if last < 0:
            last = min(first + LAZY_BUFFER_ROWS, self.model.rowCount())
        track_indices: List[int] = []
        for display in range(max(0, first - 5), min(self.model.rowCount(), last + LAZY_BUFFER_ROWS)):
            track_idx = self.model.track_index_at(display)
            if track_idx is not None:
                track_indices.append(track_idx)
        for i in track_indices:
            if i not in self._job.requested:
                self._job.priority.put(i)
                # requested is filled by the worker when it actually computes the row

    def _update_stats(self) -> None:
        if self._lazy_done >= self._lazy_total:
            total = len(self._tracks)
            changed = sum(
                1 for i in range(total)
                if (p := self.model.preview_for_track(i)) is not None and p.changed
            )
            self.stats_label.setText(
                f"{changed} will change · {total - changed} unchanged"
            )
        else:
            self.stats_label.setText(
                f"Computing preview… {self._lazy_done}/{self._lazy_total}"
            )

    # ----- selection / interaction -----

    def eventFilter(self, obj, event):  # noqa: N802
        try:
            viewport = self.table.viewport()
        except RuntimeError:
            return super().eventFilter(obj, event)

        if obj is viewport and event.type() == QEvent.Type.MouseMove:
            idx = self.table.indexAt(event.pos())
            if (
                idx.isValid()
                and idx.column() == COL_CHECK
                and not self.model.is_folder_header(idx.row())
            ):
                viewport.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                viewport.unsetCursor()
        elif obj is viewport and event.type() == QEvent.Type.Resize:
            # First real width may arrive after the post-build singleShot.
            if not self._name_cols_balanced:
                self._balance_filename_columns()
        return super().eventFilter(obj, event)

    def _on_header_clicked(self, logical_index: int) -> None:
        """Sort tracks within each folder group by the clicked column (toggle asc/desc)."""
        if not (0 <= logical_index < N_COLS):
            return
        self.model.sort_by(logical_index)
        self._sync_folder_spans()
        self._apply_only_changed_filter()
        hdr = self.table.horizontalHeader()
        order = self.model.sort_order
        hdr.setSortIndicator(logical_index, order)
        hdr.setSortIndicatorShown(True)
        # Keep active track selection / highlight after model reset.
        if self._active_index is not None:
            display = self.model.display_row_for_track(self._active_index)
            if display is not None and not self.table.isRowHidden(display):
                self.table.selectRow(display)

    def _on_header_double_clicked(self, logical_index: int) -> None:
        """Auto-fit column to contents (Excel-style double-click on header)."""
        if logical_index == COL_CHECK:
            return
        hdr = self.table.horizontalHeader()
        hint = hdr.sectionSizeFromContents(logical_index).width()
        hdr.resizeSection(logical_index, max(48, min(hint + 12, 720)))

    def _on_table_clicked(self, index) -> None:
        if not index.isValid():
            return
        if self.model.is_folder_header(index.row()):
            return
        if index.column() == COL_CHECK:
            self._toggle_row_selected(index.row())
            return
        # ExtendedSelection already updated the highlight; don't wipe multi-select
        # with selectRow when Ctrl/Shift is held.
        mods = QApplication.keyboardModifiers()
        multi = bool(mods & (Qt.ControlModifier | Qt.ShiftModifier | Qt.MetaModifier))
        self._set_active_display(index.row(), sync_selection=not multi)

    def _toggle_row_selected(self, display_row: int) -> None:
        track = self.model.track_at(display_row)
        if track is None:
            return
        preview = self.model.row_at(display_row)
        was = bool(track.selected)
        now = not was
        self.model.update_selection(display_row, now)
        if preview is not None and preview.changed:
            if now and not was:
                self._lazy_selected_changed += 1
            elif was and not now:
                self._lazy_selected_changed = max(0, self._lazy_selected_changed - 1)
        if self.on_change:
            self.on_change()

    def _set_active_display(
        self, display_row: int, *, sync_selection: bool = True
    ) -> None:
        track_idx = self.model.track_index_at(display_row)
        if track_idx is None:
            return
        self._active_index = track_idx
        track = self._tracks[track_idx]
        preview = self.model.preview_for_track(track_idx)
        if sync_selection:
            self.table.selectRow(display_row)
        if self.on_active is not None:
            self.on_active(track, preview)

    def _selected_track_display_rows(self) -> List[int]:
        """Display rows for selected track entries (skips folder headers / hidden)."""
        sm = self.table.selectionModel()
        if sm is None:
            return []
        rows: List[int] = []
        for index in sm.selectedRows():
            row = index.row()
            if self.model.is_folder_header(row):
                continue
            if self.table.isRowHidden(row):
                continue
            if self.model.track_index_at(row) is None:
                continue
            rows.append(row)
        return rows

    def _on_context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid() or self.model.is_folder_header(index.row()):
            return
        if self.model.track_index_at(index.row()) is None:
            return

        sm = self.table.selectionModel()
        if sm is not None and not sm.isSelected(index):
            # Right-click outside the current selection → select only this row.
            self.table.selectRow(index.row())
        self._set_active_display(index.row(), sync_selection=False)

        categories = [c for c in self._category_options if getattr(c, "enabled", True)]
        if not categories:
            return

        menu = RoundMenu(parent=self)
        # Use a disabled Action as the header — RoundMenu.addWidget() sizes
        # unshown widgets via widget.size(), which can be huge and stretches
        # the menu (empty gap under "Change to:").
        header = Action("Change to:")
        header.setEnabled(False)
        menu.addAction(header)
        for cat in categories:
            label = (cat.name or "").strip() or (cat.affix or "").strip()
            if not label:
                continue
            action = Action(label, self)
            action.triggered.connect(
                lambda checked=False, c=cat: self._override_selected_category(c)
            )
            menu.addAction(action)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _override_selected_category(self, category: CategoryRule) -> None:
        """Strip existing category affix and apply *category* on selected rows.

        Then rename those files on disk immediately (via ``on_override_rename``).
        """
        display_rows = self._selected_track_display_rows()
        if not display_rows:
            return
        options = self._category_options or [category]
        touched = False
        renames: dict = {}
        for display_row in display_rows:
            track_idx = self.model.track_index_at(display_row)
            if track_idx is None:
                continue
            track = self._tracks[track_idx]
            preview = self.model.preview_for_track(track_idx)
            source_name = preview.new_name if preview is not None else track.name
            new_name = override_category_affix(source_name, category, options)
            original = preview.original_name if preview is not None else track.name
            changed = new_name != original
            new_preview = PreviewRow(
                track=track,
                original_name=original,
                new_name=new_name,
                changed=changed,
                matched_keyword="",  # manual override — clear keyword match
            )
            self.model.set_row(track_idx, new_preview)
            if not track.selected:
                self.model.update_selection_for_track(track_idx, True)
            # Keep Track.category in sync for audio-player / organize helpers.
            track.category = (category.name or "").strip()
            touched = True
            if changed:
                renames[track.id] = new_name
            if track_idx == self._active_index and self.on_active is not None:
                self.on_active(track, new_preview)

        if not touched:
            return
        self._recount_change_stats()
        if self._only_changed:
            self._apply_only_changed_filter()
        self._update_stats()
        self._fit_category_column()
        if self.on_change:
            self.on_change()
        if renames and self.on_override_rename is not None:
            self.on_override_rename(renames)

    def _recount_change_stats(self) -> None:
        changed = 0
        selected_changed = 0
        for i, track in enumerate(self._tracks):
            preview = self.model.preview_for_track(i)
            if preview is not None and preview.changed:
                changed += 1
                if track.selected:
                    selected_changed += 1
        self._lazy_changed = changed
        self._lazy_selected_changed = selected_changed

    def _select_all(self) -> None:
        for i, track in enumerate(self._tracks):
            if not track.selected:
                self.model.update_selection_for_track(i, True)
        self._lazy_selected_changed = self._lazy_changed
        if self.on_change:
            self.on_change()

    def _deselect_all(self) -> None:
        for i, track in enumerate(self._tracks):
            if track.selected:
                self.model.update_selection_for_track(i, False)
        self._lazy_selected_changed = 0
        if self.on_change:
            self.on_change()

    def _on_only_changed_toggled(self, checked: bool) -> None:
        self._only_changed = bool(checked)
        self._apply_only_changed_filter()

    def _apply_only_changed_filter(self) -> None:
        """Hide unchanged track rows; hide folder headers with no visible children."""
        checked = self._only_changed
        current_header: Optional[int] = None
        header_has_visible: dict[int, bool] = {}
        for i in range(self.model.rowCount()):
            if self.model.is_folder_header(i):
                current_header = i
                header_has_visible[i] = False
                self.table.setRowHidden(i, False)
                continue
            preview = self.model.row_at(i)
            hide = checked and preview is not None and not preview.changed
            self.table.setRowHidden(i, hide)
            if not hide and current_header is not None:
                header_has_visible[current_header] = True
        for header_row, visible in header_has_visible.items():
            self.table.setRowHidden(header_row, checked and not visible)

    # ----- keyboard -----
    # Handled via QShortcut (see _build_ui) — the panel's keyPressEvent never
    # fires because the embedded QTableView consumes navigation keys first.

    def _kb_step(self, delta: int) -> None:
        total = self.model.rowCount()
        if total <= 0:
            return
        # Start from the display row of the active track (skip folder headers).
        if self._active_index is not None:
            base = self.model.display_row_for_track(self._active_index)
            if base is None:
                base = -1 if delta > 0 else total
        else:
            base = -1 if delta > 0 else total
        target = base + delta
        while 0 <= target < total:
            if self.table.isRowHidden(target):
                target += 1 if delta > 0 else -1
                continue
            if self.model.is_folder_header(target):
                target += 1 if delta > 0 else -1
                continue
            self._set_active_display(target)
            self.table.scrollTo(self.model.index(target, 0))
            self.table.setFocus(Qt.ShortcutFocusReason)
            return

    def _kb_prev(self) -> None:
        self._kb_step(-1)

    def _kb_next(self) -> None:
        self._kb_step(1)

    def _kb_page_up(self) -> None:
        # Scroll roughly a viewport of rows.
        step = max(1, self.table.viewport().height() // max(1, self.table.rowHeight(0)))
        self._kb_step(-step)

    def _kb_page_down(self) -> None:
        step = max(1, self.table.viewport().height() // max(1, self.table.rowHeight(0)))
        self._kb_step(step)

    def _kb_toggle_play(self) -> None:
        if self.on_play_pause is not None:
            self.on_play_pause()

    def _kb_toggle_check(self) -> None:
        """Toggle include/✓ for the current preview row (Enter / Return)."""
        display_row: Optional[int] = None
        if self._active_index is not None:
            display_row = self.model.display_row_for_track(self._active_index)
        if display_row is None:
            idx = self.table.currentIndex()
            if not idx.isValid() or self.model.is_folder_header(idx.row()):
                return
            display_row = idx.row()
        self._toggle_row_selected(display_row)

    def _kb_seek_back(self) -> None:
        if self.on_seek is not None:
            self.on_seek(-3.0)

    def _kb_seek_fwd(self) -> None:
        if self.on_seek is not None:
            self.on_seek(3.0)

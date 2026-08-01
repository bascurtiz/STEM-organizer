"""Stem player window — port of stem_player.StemPlayerWindow.

Non-modal QWidget. Header (Load / title / time / transport / master volume /
meter), timeline + per-track rows (S / M / volume / waveform), and a
shortcuts footer. Loader threads use a ThreadPoolExecutor; results handed
back via a queue drained on the UI tick (33 ms QTimer).

Audio engine + TrackState are pure Python (see audio_engine.py, track_state.py).
Waveform drawing uses QPainter (see waveform_widget.py).

Keyboard shortcuts:
  Space       play / pause
  ← / →       seek ±15s
  [  /  ]     prev / next song
  P / F       mark folder [pass] / [fail]
  1..4        solo stem 1..4   (Shift or !@#$ = mute)
"""
from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    QToolButton,
    QSizePolicy,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon,
    PushButton,
    ScrollArea,
    Slider,
)

from .. import theme
from ..renamer.audio_player_bar import _make_transport_button, _set_transport_icon
from .audio_engine import AudioEngine, PLAYER_SR
from .meter_widget import MeterWidget
from .track_state import TrackState, _to_stereo
from .waveform_widget import WaveformWidget


# ---------------------------------------------------------------------------
# Constants (port of stem_player module-level)
# ---------------------------------------------------------------------------

SEEK_JUMP_SEC = 15
UI_TICK_MS = 33
STEM_ORDER_4 = ("bass", "drums", "other", "vocals")
DEMUCS_LAYOUT_STEMS = ("other", "drums", "bass")
FILE_STEM_NAMES = STEM_ORDER_4 + ("instrumental",)
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".opus")
SF_READ_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3", ".m4a", ".opus"}
PLAYER_WIN_W = 1180
PLAYER_WIN_H = 960
PLAYER_MIN_W = 900
PLAYER_MIN_H = 520
CONTROLS_W = 168
TIMELINE_H = 28
HEADER_H = 52
WAVE_ZOOM_MIN = 1.0
WAVE_ZOOM_MAX = 64.0
WAVE_ZOOM_STEP = 1.22
WAVE_PEAK_BINS_FULL = 4096  # full-zoom peak cache (load-time)
WAVE_FOLLOW_POS = 0.22
BACKUP_DIR_NAME = "_backup_before_align"
# Current song + prev/next neighbors. Larger caches hold multi-hundred-MB
# float32 buffers and amplify GC pauses during the audio callback.
FOLDER_CACHE_MAX = 3
WAVEFORM_DIM_BLEND = 0.68

STEM_COLORS = {
    "bass": "#ef4444",
    "drums": "#f59e0b",
    "other": "#10b981",
    "vocals": "#a855f7",
    "instrumental": "#60A5FA",
    "acapella": "#a855f7",
    "original": "#7c5cff",
}
STEM_LABELS = {
    "bass": "Bass",
    "drums": "Drums",
    "other": "Other",
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "acapella": "Acapella",
    "original": "Original",
}


# ---------------------------------------------------------------------------
# Audio decoding (port of _ensure_player_audio_deps + load_player_audio)
# ---------------------------------------------------------------------------

_np = None
_sf = None
_ffmpeg = None
_audio_deps_ready = False


def _ensure_player_audio_deps() -> None:
    global _np, _sf, _ffmpeg, _audio_deps_ready
    if _audio_deps_ready:
        return
    from ffmpeg_bootstrap import ffmpeg_path, subprocess_kwargs  # noqa: F401

    import numpy as np
    import soundfile as sf

    _np = np
    _sf = sf
    _ffmpeg = ffmpeg_path()
    _audio_deps_ready = True


def _normalize_player_audio(audio, file_sr: int, sr: int, ch: int):
    _ensure_player_audio_deps()
    if audio.shape[0] == 1:
        audio = _np.repeat(audio, ch, axis=0)
    elif audio.shape[0] > ch:
        audio = audio[:ch]
    if file_sr == sr:
        # Contiguous RAM copy — never keep a soundfile mmap view on the
        # mix hot path (page faults under a cold OS cache after listing
        # thousands of sibling folders cause audible underruns).
        return _np.ascontiguousarray(audio, dtype=_np.float32)
    try:
        from audio_resample import resample_audio

        audio = resample_audio(audio, file_sr, sr, axis=1)
    except ImportError:
        raise RuntimeError(
            f"Sample rate mismatch ({file_sr} Hz vs {sr} Hz) and scipy is not installed."
        )
    return _np.ascontiguousarray(audio, dtype=_np.float32)


def _read_soundfile_player(path: str, sr: int, ch: int):
    _ensure_player_audio_deps()
    # Read into RAM (no mmap). Playback must not depend on page faults from
    # a library root that may contain thousands of sibling folders.
    try:
        data, file_sr = _sf.read(path, dtype="float32", always_2d=True)
        return _normalize_player_audio(data.T, file_sr, sr, ch)
    except Exception:
        return None


def _read_via_ffmpeg_player(path: str, sr: int, ch: int):
    _ensure_player_audio_deps()
    if not _ffmpeg:
        return None
    from ffmpeg_bootstrap import subprocess_kwargs

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [_ffmpeg, "-y", "-loglevel", "error", "-i", path,
             "-ar", str(sr), "-ac", str(ch), tmp_path],
            check=True, capture_output=True,
            **subprocess_kwargs(),
        )
        return _read_soundfile_player(tmp_path, sr, ch)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def load_player_audio(path: str, sr: int = PLAYER_SR, ch: int = 2):
    _ensure_player_audio_deps()
    p = Path(path)
    ext = p.suffix.lower()
    if ext in SF_READ_EXTS:
        audio = _read_soundfile_player(str(p), sr, ch)
        if audio is not None:
            return audio
    audio = _read_via_ffmpeg_player(str(p), sr, ch)
    if audio is not None:
        return audio
    try:
        from demucs.audio import AudioFile
        return AudioFile(path).read(streams=0, samplerate=sr, channels=ch).numpy().astype(_np.float32)
    except Exception as exc:
        hint = (
            "Re-run install-deps.bat if packages are missing. "
            "For FLAC without ffmpeg, ensure soundfile/libsndfile supports FLAC."
        )
        raise RuntimeError(f"Could not decode audio ({p.name}): {exc}\n{hint}") from exc


# ---------------------------------------------------------------------------
# Waveform peaks (port of compute_waveform_peaks + _compute_peaks_full_fast)
# ---------------------------------------------------------------------------

_PEAKS_FAST_MAX_SAMPLES = 600_000


def compute_waveform_peaks(mono, num_bins: int):
    _ensure_player_audio_deps()
    if num_bins < 1:
        return _np.zeros(1, dtype=_np.float32)
    mono = _np.asarray(mono, dtype=_np.float32).ravel()
    if mono.size == 0:
        return _np.zeros(num_bins, dtype=_np.float32)
    n = mono.size
    step = max(1, n // num_bins)
    count = min(num_bins, (n + step - 1) // step)
    usable = count * step
    if usable <= 0:
        return _np.zeros(num_bins, dtype=_np.float32)
    peaks = _np.max(_np.abs(mono[:usable].reshape(count, step)), axis=1)
    if count < num_bins:
        peaks = _np.pad(peaks, (0, num_bins - count))
    return peaks.astype(_np.float32)


def resample_peak_bins(peaks, num_bins: int):
    """Resample a peak envelope to ``num_bins``.

    Downsample with per-column max (preserves spikes). Upsample with linear
    interp when the visible window is denser than the load-time cache.
    Linspace / plain interp on downsample drops peaks and draws chunky polygons.
    """
    _ensure_player_audio_deps()
    bins = max(1, int(num_bins))
    arr = _np.asarray(peaks, dtype=_np.float32).ravel()
    n = int(arr.size)
    if n == 0:
        return _np.zeros(bins, dtype=_np.float32)
    if n == bins:
        return arr
    if n < bins:
        src_x = _np.arange(n, dtype=_np.float32)
        dst_x = _np.linspace(0, n - 1, bins, dtype=_np.float32)
        return _np.interp(dst_x, src_x, arr).astype(_np.float32)
    # Max-pool each display column over its source span.
    bucket = _np.minimum((_np.arange(n, dtype=_np.int64) * bins) // n, bins - 1)
    out = _np.zeros(bins, dtype=_np.float32)
    _np.maximum.at(out, bucket, arr)
    return out


def _compute_peaks_full_fast(mono):
    _ensure_player_audio_deps()
    mono = _np.asarray(mono, dtype=_np.float32).ravel()
    if mono.size > _PEAKS_FAST_MAX_SAMPLES:
        step = max(1, mono.size // _PEAKS_FAST_MAX_SAMPLES)
        mono = mono[::step]
    return compute_waveform_peaks(mono, WAVE_PEAK_BINS_FULL)


# ---------------------------------------------------------------------------
# Time formatters (port)
# ---------------------------------------------------------------------------

def format_time_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}:{ms:03d}"
    return f"{mins:02d}:{secs:02d}:{ms:03d}"


def format_ruler_time(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Stem folder detection (port)
# ---------------------------------------------------------------------------

def find_stem_file(folder: Path, stem: str) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    stem_l = stem.lower()
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS and f.stem.lower() == stem_l:
            return f
    return None


def _collect_stem_roles(folder: Path) -> dict:
    from stem_align import classify_audio_file

    roles: dict = {}
    for path in sorted(folder.iterdir()):
        if path.name == BACKUP_DIR_NAME or not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        role = classify_audio_file(path)
        if role and role not in roles:
            roles[role] = path
    for stem in FILE_STEM_NAMES:
        if stem not in roles:
            found = find_stem_file(folder, stem)
            if found is not None:
                roles[stem] = found
    return roles


def _vocal_stem_role(roles: dict) -> Optional[str]:
    if "acapella" in roles:
        return "acapella"
    if "vocals" in roles:
        return "vocals"
    return None


def _order_stem_roles(roles: dict) -> List[str]:
    vocal = _vocal_stem_role(roles)
    names = set(roles)
    if vocal and all(k in names for k in DEMUCS_LAYOUT_STEMS):
        return [vocal, "other", "drums", "bass"]
    if vocal and "instrumental" in names and "original" in names:
        return [vocal, "instrumental", "original"]
    if vocal and "instrumental" in names:
        return [vocal, "instrumental"]
    if vocal and names.intersection(DEMUCS_LAYOUT_STEMS):
        order = [vocal]
        for stem in DEMUCS_LAYOUT_STEMS:
            if stem in names:
                order.append(stem)
        return order
    fallback = ("acapella", "vocals", "instrumental", "other", "drums", "bass", "original")
    order = [name for name in fallback if name in names]
    for name in sorted(names):
        if name not in order:
            order.append(name)
    return order


def detect_stem_folder(folder: Path):
    roles = _collect_stem_roles(folder)
    if len(roles) < 2:
        return []
    return [(name, roles[name]) for name in _order_stem_roles(roles)]


def _stem_row_label(name: str, stem_roles: set) -> str:
    if name not in ("acapella", "vocals"):
        return STEM_LABELS.get(name, name.title())
    if "instrumental" in stem_roles or "original" in stem_roles:
        return "Acapella"
    if {"other", "drums", "bass"}.issubset(stem_roles):
        return "Vocals"
    return STEM_LABELS.get(name, name.title())


# ---------------------------------------------------------------------------
# Song library + review rename (port)
# ---------------------------------------------------------------------------

def _strip_review_tag(name: str) -> str:
    text = name.strip()
    text = re.sub(r"_(?:\[pass\]|\[fail\])\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\[(?:pass|fail)\]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def list_player_song_folders(library_root: Path) -> List[Path]:
    """Immediate song-folder children of *library_root* (sorted, case-insensitive).

    Uses ``os.scandir`` so ``is_dir`` is free on Windows (DirEntry cache) —
    ``Path.iterdir`` + ``Path.is_dir`` costs an extra stat per entry and was
    ~6× slower at ~4000 folders in local benchmarks.
    """
    root = os.fspath(library_root)
    if not os.path.isdir(root):
        return []
    folders: List[Path] = []
    with os.scandir(root) as it:
        for entry in it:
            if entry.name == BACKUP_DIR_NAME:
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    folders.append(Path(entry.path))
            except OSError:
                continue
    folders.sort(key=lambda path: _strip_review_tag(path.name).casefold())
    return folders


def rename_folder_review(folder: Path, verdict: str) -> Path:
    verdict = verdict.strip().lower()
    if verdict not in {"pass", "fail"}:
        raise ValueError(f"Invalid verdict: {verdict}")
    clean = _strip_review_tag(folder.name)
    new_name = f"{clean}_[{verdict}]"
    dest = folder.parent / new_name
    if folder.name == new_name:
        return folder.resolve()
    if dest.exists():
        raise FileExistsError(f"Folder already exists: {new_name}")
    last_exc: Optional[OSError] = None
    for attempt in range(8):
        try:
            folder.rename(dest)
            return dest.resolve()
        except OSError as exc:
            last_exc = exc
            denied = (
                getattr(exc, "winerror", None) == 5
                or exc.errno in {13, 32}
            )
            if not denied or attempt >= 7:
                raise
            gc.collect()
            time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Could not rename folder: {folder.name}")


# ---------------------------------------------------------------------------
# Timeline widget (port of _draw_timeline)
# ---------------------------------------------------------------------------

class TimelineWidget(QWidget):
    """Click-to-seek timeline ruler with adaptive ticks."""

    seek_requested = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(TIMELINE_H)
        self._duration = 0.0
        self._view_start = 0.0
        self._view_duration = 0.0
        self._playhead_x: Optional[float] = None

    def set_view(self, duration: float, view_start: float, view_duration: float) -> None:
        self._duration = duration
        self._view_start = view_start
        self._view_duration = view_duration
        self.update()

    def set_playhead(self, x: Optional[float]) -> None:
        if x is self._playhead_x:
            return
        if (
            x is not None
            and self._playhead_x is not None
            and abs(x - self._playhead_x) < 0.5
        ):
            return
        self._playhead_x = x
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._view_duration > 0:
            rect = self.contentsRect()
            w = max(1.0, float(rect.width()))
            x = float(event.position().x()) - float(rect.x())
            frac = max(0.0, min(1.0, x / w))
            t = self._view_start + frac * self._view_duration
            self.seek_requested.emit(max(0.0, min(self._duration, t)))

    def paintEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtGui import QColor, QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w = max(1, self.width())
        h = self.height()
        p.fillRect(0, 0, w, h, QColor(theme.COLORS["log_bg"]))
        p.setPen(QPen(QColor(theme.COLORS["border"]), 1))
        p.drawLine(0, h - 1, w, h - 1)
        duration = self._duration
        vis = self._view_duration
        if duration <= 0 or vis <= 0:
            return
        vis_start = self._view_start
        vis_end = min(duration, vis_start + vis)
        interval = 30.0
        if vis > 600:
            interval = 60.0
        elif vis < 60:
            interval = 10.0
        if vis < 15:
            interval = 5.0
        if vis < 5:
            interval = 1.0
        p.setPen(QPen(QColor(theme.COLORS["fg_dim"]), 1))
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        t = vis_start
        while t <= vis_end + 0.01:
            if vis > 0:
                x = ((t - vis_start) / vis) * w
            else:
                x = 0
            p.drawLine(int(x), h - 12, int(x), h - 1)
            p.drawText(
                QRect(int(x) + 2, 2, 60, h - 4),
                Qt.AlignLeft | Qt.AlignTop,
                format_ruler_time(t),
            )
            t += interval
        if self._playhead_x is not None and 0 <= self._playhead_x <= w:
            p.setPen(QPen(QColor("#ffffff"), 1))
            p.drawLine(int(self._playhead_x), 0, int(self._playhead_x), h)


# ---------------------------------------------------------------------------
# Track row widget (port of _build_one_track_row)
# ---------------------------------------------------------------------------

def _sm_button_style(*, active: bool, danger: bool = False) -> str:
    """Plain S/M chrome — Fluent ToggleButton paints a glitched indicator at 28px."""
    if active:
        bg = theme.COLORS["danger"] if danger else theme.COLORS["accent"]
        fg = theme.COLORS["log_fg"]
        hover = bg
    else:
        bg = theme.CONTROL_BG
        fg = theme.DARK["text_dim"]
        hover = theme.CONTROL_BG_HOVER
    return f"""
        QToolButton {{
            background-color: {bg};
            color: {fg};
            border: 1px solid {theme.DARK["border"]};
            border-radius: 4px;
            font-weight: 700;
            font-size: 11px;
            padding: 0px;
            margin: 0px;
        }}
        QToolButton:hover {{
            background-color: {hover};
        }}
        QToolButton:checked {{
            background-color: {bg};
            color: {fg};
        }}
    """


def _make_sm_button(text: str, parent: QWidget) -> QToolButton:
    btn = QToolButton(parent)
    btn.setText(text)
    btn.setCheckable(True)
    btn.setFixedSize(28, 24)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFocusPolicy(Qt.NoFocus)
    btn.setStyleSheet(_sm_button_style(active=False))
    return btn


def _shortcut_key_chip(text: str, parent: QWidget) -> QLabel:
    """CTk-style key cap: bordered rectangle with the key glyph."""
    chip = QLabel(text, parent)
    chip.setAlignment(Qt.AlignCenter)
    chip.setStyleSheet(
        f"""
        QLabel {{
            background-color: {theme.CONTROL_BG};
            color: {theme.DARK["text"]};
            border: 1px solid {theme.DARK["border"]};
            border-radius: 3px;
            padding: 1px 6px;
            font-family: Consolas, 'Segoe UI';
            font-size: 9pt;
        }}
        """
    )
    chip.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return chip


class _FolderNameBar(BodyLabel):
    """Header chip showing the loaded song folder; click opens it in Explorer."""

    clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        # Parent-only super().__init__: FluentLabelBase's text overload does
        # self.__init__(parent), which re-enters this subclass __init__ forever.
        super().__init__(parent)
        self.setObjectName("StemFolderChip")
        self.setText("(no folder loaded)")
        self._interactive = False
        self._hovered = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Same tip path as Load (native QToolTip). Scope chip QSS to #StemFolderChip
        # so unqualified color/background do not leak into the tooltip.
        self._apply_style()

    def set_interactive(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._interactive == enabled:
            return
        self._interactive = enabled
        if not enabled:
            self._hovered = False
        if enabled:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(
                theme.format_tooltip("Open folder in Explorer")
            )
        else:
            self.unsetCursor()
            self.setToolTip("")
        self._apply_style()

    def _apply_style(self) -> None:
        if self._interactive and self._hovered:
            bg = theme.CONTROL_BG_HOVER
        elif self._interactive:
            bg = theme.CONTROL_BG
        else:
            bg = "transparent"
        # Type+objectName selector only — bare color/bg rules restyle QToolTip.
        self.setStyleSheet(
            f"QLabel#StemFolderChip {{"
            f" color: {theme.DARK['text']}; background-color: {bg};"
            f" padding: 4px 8px; border-radius: 4px;"
            f"}}"
        )

    def enterEvent(self, event) -> None:
        if self._interactive:
            self._hovered = True
            self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hovered:
            self._hovered = False
            self._apply_style()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            self._interactive
            and event.button() == Qt.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class TrackRow(QWidget):
    """One stem row: label + S/M + volume + waveform."""

    solo_toggled = Signal(object)   # track
    mute_toggled = Signal(object)   # track
    volume_changed = Signal(object, float)  # track, value 0..1
    waveform_clicked = Signal(float)        # fraction 0..1

    def __init__(
        self,
        track: TrackState,
        parent: Optional[QWidget] = None,
        *,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.track = track
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Left controls — vertically centered in the row
        ctrl = QFrame()
        ctrl.setFixedWidth(CONTROLS_W - 8)
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(4)
        ctrl_layout.addStretch(1)

        self.label = BodyLabel(label if label is not None else track.name)
        self.label.setStyleSheet(f"color: {track.color}; font-weight: 600;")
        ctrl_layout.addWidget(self.label, alignment=Qt.AlignLeft)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        self.solo_btn = _make_sm_button("S", self)
        self.solo_btn.setToolTip("Solo this stem")
        self.solo_btn.clicked.connect(lambda: self.solo_toggled.emit(self.track))
        self.mute_btn = _make_sm_button("M", self)
        self.mute_btn.setToolTip("Mute this stem")
        self.mute_btn.clicked.connect(lambda: self.mute_toggled.emit(self.track))
        btn_row.addWidget(self.solo_btn)
        btn_row.addWidget(self.mute_btn)
        btn_row.addStretch(1)
        ctrl_layout.addLayout(btn_row)

        self.vol_slider = Slider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(int(round(track.volume * 100)))
        self.vol_slider.setToolTip("Volume")
        self.vol_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(self.track, v / 100.0)
        )
        ctrl_layout.addWidget(self.vol_slider)
        ctrl_layout.addStretch(1)
        layout.addWidget(ctrl)

        # Waveform
        self.wave = WaveformWidget(self)
        self.wave.clicked.connect(self._on_wave_click)
        layout.addWidget(self.wave, stretch=1)

    def _on_wave_click(self, frac: float) -> None:
        self.waveform_clicked.emit(frac)


# ---------------------------------------------------------------------------
# Main Stem Player window
# ---------------------------------------------------------------------------

class StemPlayerWindow(QWidget):
    """Stem preview player. Singleton per host (see open_stem_player)."""

    review_done = Signal(object, int, str)   # new_path, idx, verdict
    review_error = Signal(object)            # exc

    def __init__(self, parent=None, *, library_root: Optional[str] = None) -> None:
        # Independent top-level window (parent=None). Parenting to MainWindow
        # creates a native child HWND that can cover the host 1:1 and leave the
        # OS pointing-hand cursor stuck on the main GUI after close.
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        # Before any resize/show-related work: changeEvent can fire during
        # Qt construction (same pattern as MainWindow).
        self._custom_maximized = False
        self._restore_geometry: Optional[QRect] = None
        self._was_minimized = False
        self._default_w = theme.WIN_DEFAULT_W
        self._default_h = theme.WIN_DEFAULT_H
        self.setWindowTitle("STEM Player")
        self._host = parent
        # Match main GUI size (host when available); safe now that we are not a
        # native child of MainWindow.
        pw, ph = theme.WIN_DEFAULT_W, theme.WIN_DEFAULT_H
        if parent is not None:
            try:
                host = parent.window()
                if host is not None and host.isVisible():
                    hg = host.frameGeometry()
                    if hg.width() > 0 and hg.height() > 0:
                        pw, ph = hg.width(), hg.height()
            except Exception:
                pass
        self.resize(pw, ph)
        self.setMinimumSize(PLAYER_MIN_W, PLAYER_MIN_H)
        # Opening size used for minimize → taskbar restore (match main GUI).
        self._default_w = pw
        self._default_h = ph

        from ..widgets.titlebar import (
            install_frame_resize,
            install_rounded_corner_watcher,
            prepare_dark_frameless_chrome,
        )
        install_rounded_corner_watcher(self, radius=theme.WINDOW_CORNER_RADIUS)
        install_frame_resize(self)
        # Same dark chrome fill as MainWindow (shared thick-frame path).
        self.setObjectName("AppRoot")
        prepare_dark_frameless_chrome(self)

        self._colors = theme.COLORS
        # Center once after first show (frame / thick-frame geometry known).
        self._center_pending = True
        # Closed windows destroy their HWND; reopening must construct a new
        # StemPlayerWindow (show() on a closed instance → CreateWindowEx fail).
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # State
        self._engine: Optional[AudioEngine] = None
        self._tracks: List[TrackState] = []
        self._folder: Optional[Path] = None
        self._library_root: Optional[Path] = Path(library_root) if library_root else None
        self._song_folders: List[Path] = []
        self._folder_index = -1
        self._view_zoom = WAVE_ZOOM_MIN
        self._view_start = 0.0
        self._wave_w = 0
        self._redraw_sig = None
        self._folder_job_active = False
        self._busy_generation = 0
        # Library scans use a separate generation so folder loads (which bump
        # _busy_generation) cannot mark a just-finished scan as stale and leave
        # _song_folders empty — that permanently breaks [ ] / prev/next.
        self._library_gen = 0
        self._library_scan_pending = False
        # Queued prev/next steps while a load is in flight or the library list
        # is still scanning (sum of +1 / -1).
        self._pending_folder_delta = 0
        # Absolute library index to open once an in-flight load finishes
        # (set when _apply_library_scan cannot call _open_folder yet).
        self._pending_open_index: Optional[int] = None
        self._stop_after_load = False
        self._folder_cache: "dict[Path, List[TrackState]]" = {}
        self._cache_order: List[Path] = []
        self._cache_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._main_jobs: "queue.SimpleQueue" = __import__("queue").SimpleQueue()
        self._prefetch_lock = threading.Lock()
        self._prefetch_inflight: set = set()
        self._prefetch_pending = False
        self._shortcuts: List[QShortcut] = []

        self.review_done.connect(self._on_review_done)
        self.review_error.connect(self._on_review_error)

        self._build_ui()
        self._bind_keys()

        # UI tick (33 ms)
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(UI_TICK_MS)
        self._tick_timer.setTimerType(Qt.PreciseTimer)
        self._tick_timer.timeout.connect(self._ui_tick_loop)
        self._tick_timer.start()

        # Redraw debounce (60 ms)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(60)
        self._redraw_timer.timeout.connect(lambda: self._redraw_wave_view(force=True))

        if self._library_root is not None:
            QTimer.singleShot(0, lambda: self._prepare_library(self._library_root))

    # ----- UI build -----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        from ..widgets.titlebar import CustomTitleBar
        self.title_bar = CustomTitleBar(self, height=theme.TITLE_BAR_HEIGHT)
        self.title_bar.close_requested = self.close
        self.title_bar.minimize_requested = self.showMinimized
        self.title_bar.maximize_requested = self._toggle_maximize
        root.addWidget(self.title_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(6)
        root.addWidget(content, stretch=1)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        # FluentIcon (theme fg) — not 📁 emoji (Windows yellow color glyph).
        self.load_btn = PushButton(FluentIcon.FOLDER, "Load")
        self.load_btn.setToolTip(
            theme.format_tooltip(
                "Choose a song folder to load stems. Use [ and ] for prev/next song."
            )
        )
        self.load_btn.clicked.connect(self._load_folder_dialog)
        header.addWidget(self.load_btn)
        self.title_lbl = _FolderNameBar(self)
        self.title_lbl.clicked.connect(self._reveal_loaded_folder)
        header.addWidget(self.title_lbl, stretch=1)
        self.time_lbl = CaptionLabel("00:00:000")
        self.time_lbl.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-family: Consolas; font-size: 14px;"
        )
        self.time_lbl.pixelFontSize = 14
        header.addWidget(self.time_lbl)

        # Transport
        self.prev_song_btn = PushButton("◀")
        self.prev_song_btn.setFixedWidth(36)
        self.prev_song_btn.setToolTip("Previous song ([)")
        self.prev_song_btn.clicked.connect(self._prev_song_folder)
        # _make_transport_button already connects on_click — do not connect again
        # (double-connect toggles play+pause in one click → appears broken).
        self.play_btn = _make_transport_button(self, self._toggle_play)
        self.play_btn.setToolTip("Play / Pause (Space)")
        _set_transport_icon(self.play_btn, playing=False)
        self.play_btn._is_playing = False
        self.stop_btn = PushButton("■")
        self.stop_btn.setFixedWidth(36)
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.next_song_btn = PushButton("▶")
        self.next_song_btn.setFixedWidth(36)
        self.next_song_btn.setToolTip("Next song (])")
        self.next_song_btn.clicked.connect(self._next_song_folder)
        for b in (self.prev_song_btn, self.play_btn, self.stop_btn, self.next_song_btn):
            header.addWidget(b)

        # Master volume + meter
        self.master_slider = Slider(Qt.Horizontal)
        self.master_slider.setRange(0, 100)
        self.master_slider.setValue(85)
        self.master_slider.setFixedWidth(120)
        self.master_slider.setToolTip("Master volume")
        self.master_slider.valueChanged.connect(self._on_master_volume)
        header.addWidget(self.master_slider)
        self.meter = MeterWidget(self)
        header.addWidget(self.meter)
        content_layout.addLayout(header)

        # Timeline
        self.timeline = TimelineWidget(self)
        self.timeline.seek_requested.connect(self._seek_to)
        
        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(0, 0, 0, 0)
        timeline_row.setSpacing(6)
        spacer = QWidget()
        spacer.setFixedWidth(CONTROLS_W)
        timeline_row.addWidget(spacer)
        timeline_row.addWidget(self.timeline)
        content_layout.addLayout(timeline_row)

        # Track rows (scroll area)
        self.tracks_scroll = ScrollArea()
        self.tracks_scroll.setWidgetResizable(True)
        self.tracks_scroll.setFrameShape(ScrollArea.NoFrame)
        self.tracks_host = QWidget()
        self.tracks_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tracks_layout = QVBoxLayout(self.tracks_host)
        self.tracks_layout.setContentsMargins(0, 0, 0, 0)
        self.tracks_layout.setSpacing(6)
        # No trailing stretch — stem rows share height equally (CTk uniform='track').
        self.tracks_scroll.setWidget(self.tracks_host)
        content_layout.addWidget(self.tracks_scroll, stretch=1)

        # Shortcuts footer (key chips + plain action labels)
        self._shortcuts_host = QWidget()
        self._shortcuts_layout = QHBoxLayout(self._shortcuts_host)
        self._shortcuts_layout.setContentsMargins(4, 2, 4, 4)
        self._shortcuts_layout.setSpacing(12)
        content_layout.addWidget(self._shortcuts_host)
        self._populate_shortcuts_bar(stem_count=0)

    def _populate_shortcuts_bar(self, *, stem_count: int = 0) -> None:
        """CTk-style legend: [key] chips + plain action text."""
        while self._shortcuts_layout.count():
            item = self._shortcuts_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        def _group(keys: tuple[str, ...], label: str, *, join: str = "gap") -> QWidget:
            cell = QWidget()
            row = QHBoxLayout(cell)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for i, key in enumerate(keys):
                if i > 0:
                    sep = QLabel("+" if join == "plus" else "")
                    if join == "plus":
                        sep.setStyleSheet(f"color: {theme.DARK['text_dim']}; font-size: 9pt;")
                        row.addWidget(sep)
                    else:
                        row.addSpacing(6)
                row.addWidget(_shortcut_key_chip(key, cell))
            action = CaptionLabel(label)
            action.setStyleSheet(f"color: {theme.DARK['text_dim']}; font-size: 9pt;")
            row.addWidget(action)
            return cell

        groups: list[tuple[tuple[str, ...], str, str]] = [
            (("Space",), "Play / Pause", "plus"),
        ]
        if stem_count > 0:
            n = min(stem_count, 4)
            groups.append((tuple(str(i + 1) for i in range(n)), "Solo stem", "gap"))
            groups.append((tuple(f"⇧{i + 1}" for i in range(n)), "Mute stem", "gap"))
            groups.append((("←", "→"), "Seek ±15s", "gap"))
            groups.append((("[", "]"), "Prev / Next", "gap"))
            groups.append((("P", "F"), "Pass / Fail", "gap"))
        else:
            groups.append((("Ctrl", "scroll"), "Zoom in / out", "plus"))
            groups.append((("[", "]"), "Prev / Next", "gap"))
            groups.append((("P", "F"), "Pass / Fail", "gap"))

        # Equal stretches at ends and between groups → even spread across width.
        self._shortcuts_layout.addStretch(1)
        for i, (keys, label, join) in enumerate(groups):
            if i > 0:
                self._shortcuts_layout.addStretch(1)
            self._shortcuts_layout.addWidget(_group(keys, label, join=join), stretch=0)
        self._shortcuts_layout.addStretch(1)

    # ----- keyboard shortcuts -----

    def _bind_keys(self) -> None:
        def _sc(seq, handler) -> None:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(handler)
            # Keep a Python ref — parented QObjects can still be GC'd in PySide.
            self._shortcuts.append(shortcut)

        _sc(Qt.Key_Space, self._toggle_play)
        _sc(Qt.Key_Left, lambda: self._seek_relative(-SEEK_JUMP_SEC))
        _sc(Qt.Key_Right, lambda: self._seek_relative(SEEK_JUMP_SEC))
        # Prefer key enums — string "[" / "]" is unreliable across layouts.
        _sc(Qt.Key_BracketLeft, self._prev_song_folder)
        _sc(Qt.Key_BracketRight, self._next_song_folder)
        # Explicit letter sequences — Qt.Key_P alone can miss lowercase keypresses
        # when focus sits on Fluent controls.
        _sc("P", lambda: self._mark_folder_review("pass"))
        _sc("F", lambda: self._mark_folder_review("fail"))
        for i in range(4):
            _sc(str(i + 1), lambda idx=i: self._stem_shortcut(idx, mute=False))
            _sc(f"Shift+{i + 1}", lambda idx=i: self._stem_shortcut(idx, mute=True))

    def _stem_shortcut(self, idx: int, *, mute: bool) -> None:
        if idx >= len(self._tracks):
            return
        track = self._tracks[idx]
        if mute:
            self._toggle_mute(track)
        else:
            self._toggle_solo(track)

    # ----- library -----

    def _prepare_library(self, library_root) -> None:
        """Discover song folders off the UI thread, then open the first one."""
        try:
            library_root = Path(library_root)
        except Exception as exc:
            self.title_lbl.setText(f"library error: {exc}")
            self._sync_folder_name_bar()
            return
        self._library_root = library_root
        self._song_folders = []
        self._folder_index = -1
        self._pending_folder_delta = 0
        self._pending_open_index = None
        self._library_gen += 1
        self._library_scan_pending = True
        # Invalidate any in-flight folder load from a previous library.
        self._busy_generation += 1
        self._folder_job_active = False
        self.title_lbl.setText(f"Scanning {library_root.name}…")
        self._sync_folder_name_bar()
        if self._executor is None:
            # Keep pool small: library scan + at most one neighbor prefetch.
            self._executor = ThreadPoolExecutor(max_workers=2)
        scan_gen = self._library_gen

        def work():
            try:
                folders = list_player_song_folders(library_root)
                self._main_jobs.put((scan_gen, "library", (library_root, folders, None)))
            except Exception as exc:
                self._main_jobs.put((scan_gen, "library", (library_root, [], str(exc))))

        self._executor.submit(work)

    def _apply_library_scan(self, library_root: Path, folders: List[Path], error) -> None:
        self._library_scan_pending = False
        if error:
            self.title_lbl.setText(f"library error: {error}")
            self._sync_folder_name_bar()
            self._pending_folder_delta = 0
            self._pending_open_index = None
            return
        if self._library_root is not None and Path(library_root) != self._library_root:
            return
        self._song_folders = folders
        if not self._song_folders:
            self.title_lbl.setText(f"(no song folders under {library_root.name})")
            self._sync_folder_name_bar()
            self._pending_folder_delta = 0
            self._pending_open_index = None
            return
        # Prefer the already-open song if it belongs to this library; otherwise
        # start at the first folder. Pending [ ] / arrow steps apply afterward.
        idx = self._index_in_library(self._folder)
        if idx < 0:
            idx = 0
        if self._pending_folder_delta:
            idx = max(0, min(len(self._song_folders) - 1, idx + self._pending_folder_delta))
            self._pending_folder_delta = 0
        target = self._song_folders[idx]
        # Manual Load may already be opening / displaying this song while the
        # new parent library was scanning — sync index only, do not reload.
        if self._same_path(self._folder, target):
            self._folder_index = idx
            self._pending_open_index = None
            if self._folder_job_active:
                return
            QTimer.singleShot(200, self._schedule_prefetch_adjacent)
            return
        if self._folder_job_active:
            # A different load is in flight. Open the resolved library slot as
            # soon as that job clears.
            self._pending_open_index = idx
            return
        self._pending_open_index = None
        self._open_folder(target, library_index=idx)

    def _same_path(self, a: Optional[Path], b: Optional[Path]) -> bool:
        if a is None or b is None:
            return False
        if a == b:
            return True
        try:
            return a.resolve() == b.resolve()
        except OSError:
            return False

    def _index_in_library(self, folder: Optional[Path]) -> int:
        if folder is None or not self._song_folders:
            return -1
        try:
            return self._song_folders.index(folder)
        except ValueError:
            pass
        try:
            resolved = folder.resolve()
        except OSError:
            return -1
        for i, cand in enumerate(self._song_folders):
            try:
                if cand.resolve() == resolved:
                    return i
            except OSError:
                continue
        return -1

    def _load_folder_dialog(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start = str(self._folder.parent) if self._folder else ""
        folder = QFileDialog.getExistingDirectory(self, "Open song folder", start)
        if not folder:
            return
        path = Path(folder)
        new_root = path.parent
        # Same parent library already scanned: open in place (prev/next unchanged).
        if (
            self._same_path(self._library_root, new_root)
            and self._song_folders
            and not self._library_scan_pending
        ):
            idx = self._index_in_library(path)
            self._open_folder(path, library_index=idx)
            return
        # Manual Load of a song under a different parent (or no library yet):
        # rebind library root to that parent, then open the chosen song.
        # Scan uses self._folder to land on the loaded index without breaking
        # Align/Classify initial Play (which still goes through _prepare_library alone).
        self._folder = path
        self._prepare_library(new_root)
        self._open_folder(path)

    def _reveal_loaded_folder(self) -> None:
        folder = self._folder
        if folder is None or not folder.is_dir():
            return
        path = str(folder)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception:
            pass

    def _sync_folder_name_bar(self) -> None:
        folder = self._folder
        self.title_lbl.set_interactive(folder is not None and folder.is_dir())

    def _open_folder(self, folder: Path, *, library_index: int = -1) -> None:
        if self._folder_job_active:
            return
        self._folder_job_active = True
        self._busy_generation += 1
        gen = self._busy_generation
        self._stop_playback_only()
        self._folder = folder
        if library_index >= 0:
            self._folder_index = library_index
        else:
            # Manual Load / reopen: keep prev/next working when the path is in
            # the already-scanned library list; otherwise clear the index.
            self._folder_index = self._index_in_library(folder)
        self.title_lbl.setText(folder.name)
        self._sync_folder_name_bar()
        self._clear_track_rows()

        # Async load
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2)

        def work():
            try:
                tracks = self._get_folder_cache(folder)
                if tracks is None:
                    tracks = self._load_tracks_from_stems(folder)
                    self._put_folder_cache(folder, tracks)
                self._main_jobs.put((gen, "loaded", (folder, tracks)))
            except Exception as exc:
                self._main_jobs.put((gen, "error", str(exc)))

        self._executor.submit(work)
        # Prefetch neighbors only when not playing (avoids disk/GIL contention
        # with the PortAudio callback on large libraries).
        QTimer.singleShot(200, self._schedule_prefetch_adjacent)

    def _load_tracks_from_stems(self, folder: Path) -> List[TrackState]:
        _ensure_player_audio_deps()
        pairs = detect_stem_folder(folder)
        if not pairs:
            return []
        tracks: List[TrackState] = []
        for name, path in pairs:
            try:
                audio = load_player_audio(str(path))
                mono = _to_stereo(audio).mean(axis=0)
                peaks_full = _compute_peaks_full_fast(mono)
            except Exception:
                continue
            track = TrackState(name, path, audio, STEM_COLORS.get(name, theme.COLORS["accent"]))
            track.peaks_full = peaks_full
            # Keep track.name as the role key (vocals/bass/…) for order + colors.
            # Display label is applied in _build_track_rows via _stem_row_label.
            tracks.append(track)
        return tracks

    # ----- folder cache -----

    def _get_folder_cache(self, folder: Path) -> Optional[List[TrackState]]:
        with self._cache_lock:
            return self._folder_cache.get(folder)

    def _put_folder_cache(self, folder: Path, tracks: List[TrackState]) -> None:
        with self._cache_lock:
            if folder in self._folder_cache:
                try:
                    self._cache_order.remove(folder)
                except ValueError:
                    pass
            self._folder_cache[folder] = tracks
            self._cache_order.append(folder)
            guard = 0
            while len(self._cache_order) > FOLDER_CACHE_MAX and guard < FOLDER_CACHE_MAX + 2:
                guard += 1
                oldest = self._cache_order.pop(0)
                # Keep the currently displayed song resident.
                if oldest == self._folder:
                    self._cache_order.append(oldest)
                    continue
                self._folder_cache.pop(oldest, None)

    def _schedule_prefetch_adjacent(self) -> None:
        """Prefetch neighbors when idle; remember intent if currently playing."""
        if self._engine is not None and self._engine.playing:
            self._prefetch_pending = True
            return
        self._prefetch_pending = False
        self._prefetch_adjacent_songs()

    def _prefetch_adjacent_songs(self) -> None:
        if self._executor is None or self._library_root is None:
            return
        if self._engine is not None and self._engine.playing:
            self._prefetch_pending = True
            return
        idx = self._folder_index
        if idx < 0:
            return
        for cand in (idx - 1, idx + 1):
            if 0 <= cand < len(self._song_folders):
                folder = self._song_folders[cand]
                with self._prefetch_lock:
                    if folder in self._prefetch_inflight:
                        continue
                    if self._get_folder_cache(folder) is not None:
                        continue
                    self._prefetch_inflight.add(folder)

                def work(f=folder):
                    try:
                        if self._engine is not None and self._engine.playing:
                            self._prefetch_pending = True
                            return
                        tracks = self._load_tracks_from_stems(f)
                        self._put_folder_cache(f, tracks)
                    except Exception:
                        pass
                    finally:
                        with self._prefetch_lock:
                            self._prefetch_inflight.discard(f)

                self._executor.submit(work)

    # ----- engine / track rows -----

    def _stop_playback_only(self) -> None:
        if self._engine is not None:
            self._engine.set_playing(False)
            self._engine.stop_stream()
            self._engine = None

    def _clear_track_rows(self) -> None:
        # Remove every row. Leaving a leftover widget made next-track loads
        # show a duplicate previous stem. No trailing stretch — rows themselves
        # expand to fill the scroll viewport equally.
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._tracks = []
        self._wave_w = 0

    def _install_loaded_folder(self, folder: Path, tracks: List[TrackState]) -> None:
        self._clear_track_rows()
        self._tracks = tracks
        self._engine = AudioEngine(tracks, sr=PLAYER_SR)
        try:
            self._engine.start_stream()
        except Exception as exc:
            self.title_lbl.setText(f"audio error: {exc}")
        self._view_zoom = WAVE_ZOOM_MIN
        self._view_start = 0.0
        self._build_track_rows()
        self._populate_shortcuts_bar(stem_count=len(tracks))
        self._redraw_wave_view(force=True)
        # Child layout can settle after this call without a window resizeEvent;
        # debounce another pass so bins match the final pixel width.
        self._redraw_timer.start()

    def _build_track_rows(self) -> None:
        stem_roles = {t.name for t in self._tracks}
        for track in self._tracks:
            row = TrackRow(
                track,
                self.tracks_host,
                label=_stem_row_label(track.name, stem_roles),
            )
            row.solo_toggled.connect(self._toggle_solo)
            row.mute_toggled.connect(self._toggle_mute)
            row.volume_changed.connect(self._on_track_volume)
            row.wave.clicked.connect(self._on_wave_click)
            # Waveform width changes on row layout, not only window resize.
            row.wave.width_changed.connect(self._redraw_timer.start)
            # Equal stretch → 50/50 for 2 stems, ~25% each for 4 (CTk uniform rows).
            self.tracks_layout.addWidget(row, stretch=1)
            track.row_widget = row
            track.wave_widget = row.wave
            track.solo_btn = row.solo_btn
            track.mute_btn = row.mute_btn
            track.vol_slider = row.vol_slider
            row.solo_btn.setChecked(track.solo)
            row.solo_btn.setStyleSheet(_sm_button_style(active=track.solo))
            row.mute_btn.setChecked(track.muted)
            row.mute_btn.setStyleSheet(_sm_button_style(active=track.muted, danger=True))
            row.wave.set_color(track.color)
            row.wave.set_filename(track.path.name)
            row.wave.set_peaks(None)

    # ----- transport -----

    def _toggle_play(self) -> None:
        if self._engine is None:
            return
        if self._engine.playing:
            self._engine.set_playing(False)
            _set_transport_icon(self.play_btn, playing=False)
            self.play_btn._is_playing = False
            if self._prefetch_pending:
                QTimer.singleShot(0, self._schedule_prefetch_adjacent)
        else:
            if self._engine.position >= self._engine.duration - 0.01:
                self._engine.position = 0.0
            self._engine.set_playing(True)
            _set_transport_icon(self.play_btn, playing=True)
            self.play_btn._is_playing = True

    def _stop(self) -> None:
        if self._engine is None:
            return
        self._engine.set_playing(False)
        self._engine.position = 0.0
        _set_transport_icon(self.play_btn, playing=False)
        self.play_btn._is_playing = False
        self.time_lbl.setText("00:00:000")
        self._update_playhead(0.0)
        if self._prefetch_pending:
            QTimer.singleShot(0, self._schedule_prefetch_adjacent)

    def _seek_to(self, seconds: float) -> None:
        if self._engine is None:
            return
        self._engine.position = seconds
        self.time_lbl.setText(format_time_ms(self._engine.position))
        self._update_playhead(self._engine.position)

    def _seek_relative(self, delta: float) -> None:
        if self._engine is None:
            return
        self._seek_to(self._engine.position + delta)

    def _on_master_volume(self, value: int) -> None:
        if self._engine is not None:
            self._engine.master_volume = value / 100.0

    def _on_track_volume(self, track: TrackState, value: float) -> None:
        track.volume = value

    # ----- mute / solo -----

    def _toggle_solo(self, track: TrackState) -> None:
        enabling = not track.solo
        if enabling:
            # Exclusive solo: only one stem can be solo at a time.
            for t in self._tracks:
                if t is track:
                    continue
                if t.solo:
                    t.solo = False
                    if t.solo_btn is not None:
                        t.solo_btn.setChecked(False)
                        t.solo_btn.setStyleSheet(_sm_button_style(active=False))
            track.solo = True
            # Solo and mute are mutually exclusive on a track.
            if track.muted:
                track.muted = False
        else:
            track.solo = False
        if track.solo_btn is not None:
            track.solo_btn.setChecked(track.solo)
            track.solo_btn.setStyleSheet(_sm_button_style(active=track.solo))
        self._refresh_waveform_colors()

    def _toggle_mute(self, track: TrackState) -> None:
        any_solo = any(t.solo for t in self._tracks)
        if any_solo:
            # While solo is active, M toggles membership in the audible solo group
            # (add/remove stems alongside the soloed track(s)).
            if track.solo:
                track.solo = False
            else:
                track.solo = True
                track.muted = False
            if track.solo_btn is not None:
                track.solo_btn.setChecked(track.solo)
                track.solo_btn.setStyleSheet(_sm_button_style(active=track.solo))
            self._refresh_waveform_colors()
            return

        enabling = not track.muted
        track.muted = enabling
        if enabling and track.solo:
            # Solo and mute are mutually exclusive on a track.
            track.solo = False
            if track.solo_btn is not None:
                track.solo_btn.setChecked(False)
                track.solo_btn.setStyleSheet(_sm_button_style(active=False))
        self._refresh_waveform_colors()

    def _effectively_muted(self, track: TrackState) -> bool:
        """True if muted for audio: real mute, or silenced by another track's solo."""
        if track.muted:
            return True
        if any(t.solo for t in self._tracks) and not track.solo:
            return True
        return False

    def _waveform_dimmed(self, track: TrackState) -> bool:
        return self._effectively_muted(track)

    def _refresh_mute_styles(self) -> None:
        """Mute chrome follows effective mute; track.muted stays the stored state."""
        for track in self._tracks:
            if track.mute_btn is None:
                continue
            on = self._effectively_muted(track)
            track.mute_btn.setChecked(on)
            track.mute_btn.setStyleSheet(_sm_button_style(active=on, danger=True))

    def _refresh_waveform_colors(self) -> None:
        for track in self._tracks:
            if track.wave_widget is not None:
                track.wave_widget.set_color(track.color, dimmed=self._waveform_dimmed(track))
        self._refresh_mute_styles()

    # ----- view math -----

    def _duration(self) -> float:
        return self._engine.duration if self._engine else 0.0

    def _view_duration(self) -> float:
        dur = self._duration()
        if dur <= 0:
            return 0.0
        return dur / max(WAVE_ZOOM_MIN, self._view_zoom)

    def _view_end(self) -> float:
        return min(self._duration(), self._view_start + self._view_duration())

    def _clamp_wave_view(self) -> None:
        dur = self._duration()
        if dur <= 0:
            self._view_zoom = WAVE_ZOOM_MIN
            self._view_start = 0.0
            return
        vis = self._view_duration()
        if vis >= dur - 1e-9:
            self._view_zoom = WAVE_ZOOM_MIN
            self._view_start = 0.0
            return
        max_start = max(0.0, dur - vis)
        self._view_start = max(0.0, min(self._view_start, max_start))

    def _time_to_x(self, seconds: float, width: float) -> float:
        vis = self._view_duration()
        if width <= 0 or vis <= 0:
            return 0.0
        return ((seconds - self._view_start) / vis) * width

    def _x_to_time(self, x: float, width: float) -> float:
        vis = self._view_duration()
        if width <= 0 or vis <= 0:
            return 0.0
        frac = max(0.0, min(1.0, x / width))
        t = self._view_start + frac * vis
        return max(0.0, min(self._duration(), t))

    def _peaks_for_view(self, track: TrackState, bins: int):
        _ensure_player_audio_deps()
        full = track.peaks_full
        dur = self._duration()
        if full is None or dur <= 0:
            return _np.zeros(max(1, bins), dtype=_np.float32)
        n = len(full)
        i0 = int(max(0, min(n - 1, (self._view_start / dur) * n)))
        i1 = int(max(i0 + 1, min(n, (self._view_end() / dur) * n)))
        region = full[i0:i1]
        if region.size == 0:
            return _np.zeros(max(1, bins), dtype=_np.float32)
        return resample_peak_bins(region, bins)

    def _follow_playhead(self, pos: float) -> bool:
        if self._view_zoom <= WAVE_ZOOM_MIN + 1e-9:
            return False
        if self._engine is None or not self._engine.playing:
            return False
        vis = self._view_duration()
        dur = self._duration()
        if vis <= 0 or dur <= 0:
            return False
        target = pos - vis * WAVE_FOLLOW_POS
        max_start = max(0.0, dur - vis)
        new_start = max(0.0, min(target, max_start))
        if abs(new_start - self._view_start) < 1e-4:
            return False
        self._view_start = new_start
        return True

    # ----- waveform redraw -----

    def _redraw_wave_view(self, *, force: bool = False) -> None:
        if not self._tracks:
            self.timeline.set_view(0.0, 0.0, 0.0)
            return
        # Measure drawable waveform width from first track (contentsRect matches paint)
        first = self._tracks[0].wave_widget
        if first is not None:
            w = first.contentsRect().width()
        else:
            w = 0
        if w < 2:
            # Schedule another attempt
            QTimer.singleShot(50, lambda: self._redraw_wave_view(force=force))
            return
        self._wave_w = w
        # Redraw signature cache (skip if unchanged and not forced)
        sig = (w, round(self._view_zoom, 4), round(self._view_start, 4),
               tuple((t.muted, t.solo) for t in self._tracks))
        if not force and sig == self._redraw_sig:
            return
        self._redraw_sig = sig
        # One display bin per pixel column; pixmap cache keeps playhead ticks
        # from rebuilding paths. Max-pool from WAVE_PEAK_BINS_FULL in
        # resample_peak_bins (not linspace/interp).
        bins = max(1, int(w))
        for track in self._tracks:
            track.peaks = self._peaks_for_view(track, bins)
            if track.wave_widget is not None:
                track.wave_widget.set_peaks(track.peaks)
                track.wave_widget.set_color(track.color, dimmed=self._waveform_dimmed(track))
        self.timeline.set_view(self._duration(), self._view_start, self._view_duration())
        self._update_playhead(self._engine.position if self._engine else 0.0)

    def _update_playhead(self, position: float) -> None:
        if not self._tracks:
            return
        vis = self._view_duration()
        frac: Optional[float] = None
        if vis > 0:
            raw = (position - self._view_start) / vis
            if 0.0 <= raw <= 1.0:
                frac = raw
        for track in self._tracks:
            if track.wave_widget is not None:
                track.wave_widget.set_playhead(frac)
        # Timeline still uses pixel X against its own live width.
        tw = self.timeline.width()
        self.timeline.set_playhead(
            self._time_to_x(position, tw) if tw > 0 and frac is not None else None
        )

    def _on_wave_click(self, frac: float) -> None:
        """Seek from waveform click fraction (0..1 of drawable width)."""
        vis = self._view_duration()
        if vis <= 0:
            return
        t = self._view_start + max(0.0, min(1.0, float(frac))) * vis
        self._seek_to(max(0.0, min(self._duration(), t)))

    # ----- UI tick -----

    def _ui_tick_loop(self) -> None:
        # Drain up to 4 main jobs
        drained = 0
        while drained < 4:
            try:
                gen, kind, payload = self._main_jobs.get_nowait()
            except Exception:
                break
            if kind == "library":
                # Library gen is independent of folder-load _busy_generation.
                if gen != self._library_gen:
                    drained += 1
                    continue
                root, folders, error = payload
                self._apply_library_scan(root, folders, error)
            elif gen == self._busy_generation:
                if kind == "loaded":
                    self._apply_loaded(gen, payload)
                elif kind == "error":
                    self.title_lbl.setText(f"load error: {payload}")
                    self._sync_folder_name_bar()
                    self._folder_job_active = False
                    self._flush_pending_folder_nav()
            drained += 1
        self._ui_tick()

    def _apply_loaded(self, gen: int, payload) -> None:
        folder, tracks = payload
        try:
            if not tracks:
                self._clear_track_rows()
                self._populate_shortcuts_bar(stem_count=0)
                self.title_lbl.setText(f"(no stems in {folder.name})")
                self._sync_folder_name_bar()
                return
            self._install_loaded_folder(folder, tracks)
        finally:
            self._folder_job_active = False
            self._flush_pending_folder_nav()

    def _ui_tick(self) -> None:
        if self._engine is None:
            self.meter.reset()
            return
        pos = self._engine.position
        self.time_lbl.setText(format_time_ms(pos))
        # Follow + playhead
        if self._follow_playhead(pos):
            self._redraw_wave_view(force=True)
        else:
            self._update_playhead(pos)
        # Meter
        self.meter.set_level(self._engine.meter_level())
        # Play button sync
        if not self._engine.playing and getattr(self.play_btn, "_is_playing", False):
            _set_transport_icon(self.play_btn, playing=False)
            self.play_btn._is_playing = False
            if self._prefetch_pending:
                QTimer.singleShot(0, self._schedule_prefetch_adjacent)

    # ----- song navigation -----

    def _queue_folder_step(self, delta: int) -> None:
        """Remember a prev/next intent until the library list / load is ready."""
        self._pending_folder_delta += int(delta)
        if self._pending_folder_delta == 0:
            return
        if self._library_scan_pending or self._folder_job_active:
            return
        self._flush_pending_folder_nav()

    def _flush_pending_folder_nav(self) -> None:
        if self._library_scan_pending or self._folder_job_active:
            return
        if self._pending_open_index is not None:
            target = self._pending_open_index
            self._pending_open_index = None
            if self._song_folders and 0 <= target < len(self._song_folders):
                if self._pending_folder_delta:
                    target = max(
                        0,
                        min(len(self._song_folders) - 1, target + self._pending_folder_delta),
                    )
                    self._pending_folder_delta = 0
                self._open_folder(self._song_folders[target], library_index=target)
                return
        if self._pending_folder_delta == 0:
            return
        if not self._song_folders:
            self._pending_folder_delta = 0
            return
        idx = self._folder_index
        if idx < 0:
            idx = self._index_in_library(self._folder)
        if idx < 0:
            idx = 0
        target = max(0, min(len(self._song_folders) - 1, idx + self._pending_folder_delta))
        self._pending_folder_delta = 0
        if target == self._folder_index and self._folder is not None:
            return
        self._open_folder(self._song_folders[target], library_index=target)

    def _prev_song_folder(self) -> None:
        if self._library_scan_pending or not self._song_folders:
            self._queue_folder_step(-1)
            return
        if self._folder_job_active:
            self._queue_folder_step(-1)
            return
        if self._folder_index <= 0:
            return
        self._open_folder(
            self._song_folders[self._folder_index - 1],
            library_index=self._folder_index - 1,
        )

    def _next_song_folder(self) -> None:
        if self._library_scan_pending or not self._song_folders:
            self._queue_folder_step(+1)
            return
        if self._folder_job_active:
            self._queue_folder_step(+1)
            return
        if self._folder_index < 0:
            self._open_folder(self._song_folders[0], library_index=0)
            return
        if self._folder_index >= len(self._song_folders) - 1:
            return
        self._open_folder(
            self._song_folders[self._folder_index + 1],
            library_index=self._folder_index + 1,
        )

    def _mark_folder_review(self, verdict: str) -> None:
        if self._folder_job_active or self._folder is None:
            return
        self._folder_job_active = True
        folder = self._folder
        idx = self._folder_index

        def work():
            try:
                new_path = rename_folder_review(folder, verdict)
                # Signal is thread-safe; do not QTimer from a worker thread.
                self.review_done.emit(new_path, idx, verdict)
            except Exception as exc:
                self.review_error.emit(exc)

        threading.Thread(target=work, daemon=True).start()

    def _on_review_done(self, new_path: Path, idx: int, verdict: str) -> None:
        # Update the in-memory library slot in place — do NOT rescan thousands
        # of sibling folders on the UI thread (blocks the audio callback GIL).
        old_path = self._folder
        new_path = Path(new_path)
        self._folder = new_path
        with self._cache_lock:
            if old_path is not None:
                self._folder_cache.pop(old_path, None)
                try:
                    self._cache_order.remove(old_path)
                except ValueError:
                    pass
            self._folder_cache.pop(new_path, None)
            try:
                self._cache_order.remove(new_path)
            except ValueError:
                pass
        new_idx = idx
        if 0 <= idx < len(self._song_folders):
            self._song_folders[idx] = new_path
            new_idx = idx
        else:
            # Fallback: locate by path without a full directory re-list.
            try:
                new_idx = self._song_folders.index(new_path)
            except ValueError:
                if old_path is not None:
                    try:
                        new_idx = self._song_folders.index(old_path)
                        self._song_folders[new_idx] = new_path
                    except ValueError:
                        new_idx = idx
        bg = "#7ee0a0" if verdict == "pass" else "#ff7a7a"
        self.title_lbl.setStyleSheet(
            f"QLabel#StemFolderChip {{ background: {bg}; color: #000; "
            f"padding: 4px 8px; border-radius: 4px; }}"
        )
        QTimer.singleShot(1500, self.title_lbl._apply_style)
        self._folder_job_active = False
        if 0 <= new_idx < len(self._song_folders):
            self._open_folder(self._song_folders[new_idx], library_index=new_idx)
        else:
            self.title_lbl.setText(self._folder.name)
            self._sync_folder_name_bar()

    def _on_review_error(self, exc: Exception) -> None:
        self.title_lbl.setText(f"rename error: {exc}")
        self._folder_job_active = False

    # ----- close -----

    def closeEvent(self, event) -> None:  # noqa: N802
        # Drop resize mouse-grab before hide — otherwise the grip's cursor can
        # stick on the host window underneath.
        from ..widgets.titlebar import disarm_win32_thick_frame

        disarm_win32_thick_frame(self)
        handler = getattr(self, "_frame_resize_handler", None)
        if handler is not None and getattr(handler, "is_resizing", False):
            try:
                handler.end_resize()
            except Exception:
                pass
        self._tick_timer.stop()
        if self._engine is not None:
            self._engine.set_playing(False)
            self._engine.stop_stream()
            self._engine = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self.unsetCursor()
        _release_cursor_overrides()
        _forget_player_window(self)
        super().closeEvent(event)
        # After this HWND is gone, re-apply cursor for whatever is under the mouse
        # (Qt often skips leave/enter when a covering window is closed).
        QTimer.singleShot(0, _force_cursor_resync)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._redraw_timer.start()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        def _after_show():
            from ..widgets.titlebar import enable_win32_thick_frame
            enable_win32_thick_frame(self)
            handler = getattr(self, "_frame_resize_handler", None)
            if handler is not None:
                handler._layout_grips()
                handler._raise_grips()
            # After thick-frame so frameGeometry is valid (exe Win11 top-left bug).
            if self._center_pending:
                self._center_pending = False
                self._center_on_parent_or_screen()
        QTimer.singleShot(0, _after_show)

    def _center_on_parent_or_screen(self) -> None:
        """Center over parent MainWindow (or its screen). Skip if maximized.

        Matches dialogs.py / CTk: prefer host ``frameGeometry``, else the
        screen that contains the host, else the primary screen work area.
        """
        if self.isMaximized() or getattr(self, "_custom_maximized", False):
            return

        host = None
        if self._host is not None:
            try:
                host = self._host.window()
            except Exception:
                host = self._host

        if host is not None and host is not self and host.isVisible():
            hg = host.frameGeometry()
            fg = self.frameGeometry()
            self.move(
                hg.x() + max(0, (hg.width() - fg.width()) // 2),
                hg.y() + max(0, (hg.height() - fg.height()) // 2),
            )
            return

        screen = None
        if host is not None:
            try:
                screen = host.screen()
            except Exception:
                screen = None
        if screen is None:
            screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        fg = self.frameGeometry()
        self.move(
            avail.x() + max(0, (avail.width() - fg.width()) // 2),
            avail.y() + max(0, (avail.height() - fg.height()) // 2),
        )

    def changeEvent(self, event) -> None:  # noqa: N802
        # Same as MainWindow: minimize → taskbar restore → opening/default size.
        from ..widgets.titlebar import (
            note_activation_chrome_refresh,
            note_minimize_restore_to_default,
        )

        dw = getattr(self, "_default_w", None)
        dh = getattr(self, "_default_h", None)
        if dw is not None and dh is not None:
            note_minimize_restore_to_default(self, event, width=dw, height=dh)
        note_activation_chrome_refresh(self, event)
        super().changeEvent(event)

    def nativeEvent(self, eventType, message):  # noqa: N802
        # Must match MainWindow: unpack MSG, then handle_native_frame_message(window, msg).
        # Passing (eventType, message) raises TypeError inside the Win32 callback and
        # kills the whole process (STATUS_FATAL_USER_CALLBACK_EXCEPTION / 0xC000041D).
        if sys.platform == "win32" and eventType in (b"windows_generic_MSG", "windows_generic_MSG"):
            try:
                from ctypes import wintypes

                msg = wintypes.MSG.from_address(int(message))
                from ..widgets.titlebar import handle_native_frame_message

                handled = handle_native_frame_message(self, msg)
                if handled is not None:
                    return handled
            except Exception:
                pass
        return super().nativeEvent(eventType, message)

    def _toggle_maximize(self) -> None:
        """CTk-style work-area fill — same path as MainWindow (not OS showMaximized)."""
        from ..widgets.titlebar import toggle_work_area_maximize

        toggle_work_area_maximize(self)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._view_zoom = min(WAVE_ZOOM_MAX, self._view_zoom * WAVE_ZOOM_STEP)
            elif delta < 0:
                self._view_zoom = max(WAVE_ZOOM_MIN, self._view_zoom / WAVE_ZOOM_STEP)
            self._clamp_wave_view()
            self._redraw_wave_view(force=True)
            event.accept()
        else:
            super().wheelEvent(event)


# ---------------------------------------------------------------------------
# Module-level singleton holder + entry point
# ---------------------------------------------------------------------------

_PLAYER_WINDOW: Optional[StemPlayerWindow] = None


def _forget_player_window(win: Optional[StemPlayerWindow]) -> None:
    """Drop singleton / host refs so the next Play constructs a fresh window."""
    global _PLAYER_WINDOW
    if win is None:
        return
    if _PLAYER_WINDOW is win:
        _PLAYER_WINDOW = None
    host = getattr(win, "_host", None)
    if host is not None and getattr(host, "_player_window", None) is win:
        try:
            host._player_window = None
        except Exception:
            pass


def _release_cursor_overrides() -> None:
    app = QApplication.instance()
    if app is None:
        return
    while app.overrideCursor() is not None:
        app.restoreOverrideCursor()


def _force_cursor_resync() -> None:
    """Re-apply the cursor for the widget under the mouse after player close."""
    from PySide6.QtGui import QCursor

    _release_cursor_overrides()
    app = QApplication.instance()
    if app is None:
        return
    w = QApplication.widgetAt(QCursor.pos())
    cursor = w.cursor() if w is not None else QCursor(Qt.ArrowCursor)
    app.setOverrideCursor(cursor)
    app.restoreOverrideCursor()


def open_stem_player(parent=None, library_root: Optional[str] = None):
    """Open (or focus) the singleton Stem Player window."""
    global _PLAYER_WINDOW
    _ensure_player_audio_deps()
    if _PLAYER_WINDOW is not None:
        try:
            # Only reuse a still-living, visible window. After close the HWND is
            # gone (WA_DeleteOnClose); show() on that instance → CreateWindowEx fail.
            if _PLAYER_Window_visible(_PLAYER_WINDOW):
                if library_root and (
                    not _PLAYER_WINDOW._library_root
                    or Path(library_root) != _PLAYER_WINDOW._library_root
                ):
                    _PLAYER_WINDOW._prepare_library(library_root)
                _PLAYER_WINDOW.raise_()
                _PLAYER_WINDOW.activateWindow()
                if parent is not None:
                    try:
                        parent._player_window = _PLAYER_WINDOW  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return _PLAYER_WINDOW
        except Exception:
            pass
        _forget_player_window(_PLAYER_WINDOW)

    win = StemPlayerWindow(parent, library_root=library_root)
    _PLAYER_WINDOW = win
    if parent is not None:
        try:
            parent._player_window = win  # type: ignore[attr-defined]
        except Exception:
            pass
    win.show()
    return win


def close_stem_player() -> None:
    """Close the singleton player if open (used from MainWindow shutdown)."""
    win = _PLAYER_WINDOW
    if win is None:
        return
    try:
        win.close()
    except Exception:
        _forget_player_window(win)


def _PLAYER_Window_visible(win: StemPlayerWindow) -> bool:
    try:
        from shiboken6 import isValid

        if not isValid(win):
            return False
    except Exception:
        pass
    try:
        return bool(win.isVisible()) and win.windowHandle() is not None
    except RuntimeError:
        return False
    except Exception:
        return False

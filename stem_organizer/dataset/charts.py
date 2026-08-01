"""Dark-themed chart widgets for Dataset overview (QtCharts).

Uniform chrome for all charts:
  panel2 background → title → plot (donut % / bars) → legend / axes
  with breathing room under the legend.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QPieSeries,
    QStackedBarSeries,
    QValueAxis,
)
from PySide6.QtCore import (
    QEasingCurve,
    QMargins,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Property,
    Qt,
)
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import theme

# Labels sit on the dark chart panel — always bright (never slice-contrast dark).
_LABEL_BRIGHT = theme.COLORS["fg"]          # #e8eaf0
_LABEL_MUTED = theme.COLORS["log_fg"]       # #d6dae8 — axes / legend
# GenreVerticalBars measure ~66px wide in the default GUI — SI-SDR bars and
# export donuts target that stroke; GUI donuts use a slightly thinner ring.
CHART_STROKE_PX = 66.0
CHART_DONUT_STROKE_PX = 50.0  # GUI only; export keeps CHART_STROKE_PX
CHART_DONUT_PIE = 0.78  # outer size; hole derived so ring == stroke_px
# QtCharts hosts size to remaining viewport (tab fits height so legend +
# footnote stay on-screen without scrolling). These are fallbacks only.
_CHART_MIN_H = 200
_CHART_HOST_H = 260
_CHART_MARGINS = QMargins(10, 6, 10, 28)  # bars: tighter top, room for axis
_DONUT_MARGINS = QMargins(22, 18, 22, 14)  # callout room without starving the pie
_BAR_GROW_MS = 450


def sync_chart_stroke_px(
    chart: QChart,
    *,
    stroke_px: float = CHART_STROKE_PX,
    donut_stroke_px: float | None = None,
) -> None:
    """Set SI-SDR bar width and donut ring thickness in pixels."""
    if getattr(chart, "_stroke_syncing", False):
        return
    chart._stroke_syncing = True  # type: ignore[attr-defined]
    try:
        plot = chart.plotArea()
        pw = float(plot.width())
        ph = float(plot.height())
        if pw < 8.0 or ph < 8.0:
            return
        side = min(pw, ph)
        ring_px = float(stroke_px if donut_stroke_px is None else donut_stroke_px)
        for series in chart.series():
            if hasattr(series, "setPieSize") and hasattr(series, "setHoleSize"):
                # ring_px = (pie − hole) / 2 * side  →  pie − hole = 2 * ring / side
                try:
                    pie = float(series.pieSize())
                except Exception:
                    pie = CHART_DONUT_PIE
                if pie <= 0.05:
                    pie = CHART_DONUT_PIE
                pie = min(0.86, max(0.50, pie))
                delta = 2.0 * ring_px / side
                if delta >= pie - 0.06:
                    # Plot too small for the target ring — use the thickest workable.
                    hole = 0.06
                    pie = min(0.88, hole + delta)
                else:
                    hole = pie - delta
                series.setPieSize(pie)
                series.setHoleSize(max(0.04, hole))
            if hasattr(series, "setBarWidth"):
                n_cats = 0
                for axis in chart.axes():
                    if hasattr(axis, "categories"):
                        n_cats = len(axis.categories())
                        break
                if n_cats <= 0 and hasattr(series, "barSets") and series.barSets():
                    n_cats = series.barSets()[0].count()
                if n_cats > 0:
                    cat_w = pw / float(n_cats)
                    series.setBarWidth(min(1.0, max(0.05, float(stroke_px) / cat_w)))
    finally:
        chart._stroke_syncing = False  # type: ignore[attr-defined]


def _bind_stroke_sync(chart: QChart) -> None:
    """Keep stroke correct when the plot area changes (GUI: thinner donuts)."""

    def _sync(_rect=None) -> None:
        sync_chart_stroke_px(chart, donut_stroke_px=CHART_DONUT_STROKE_PX)

    chart.plotAreaChanged.connect(_sync)
    # Initial layout often lands after the first show — nudge once.
    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, _sync)


def genre_colors_for(names: Sequence[str]) -> dict[str, str]:
    """Map genre names → Rename custom palette (stable by DEMO order / cycle)."""
    from track_renamer.category_palette import next_unused_palette_color

    from .models import MAEST_GENRE_COLORS

    fixed = dict(MAEST_GENRE_COLORS)
    out: dict[str, str] = {}
    used = {c.lower() for c in fixed.values()}
    for name in names:
        if name in fixed:
            out[name] = fixed[name]
        elif name in out:
            continue
        else:
            color = next_unused_palette_color(used)
            out[name] = color
            used.add(color.lower())
    return out


def _accent_palette(n: int) -> list[QColor]:
    """Fallback multi-slice palette."""
    base = [QColor(c) for c in theme.CHART_GENRE_PALETTE]
    if n <= len(base):
        return base[:n]
    out = list(base)
    while len(out) < n:
        out.append(base[len(out) % len(base)])
    return out


def _style_chart(chart: QChart, *, plot_bg: str | None = None) -> None:
    """Shared Roles-style chrome: panel2, bright title, legend/axis room."""
    chart.setBackgroundBrush(QColor(theme.COLORS["panel2"]))
    chart.setBackgroundRoundness(8)
    has_pie = any(hasattr(s, "slices") for s in chart.series())
    chart.setMargins(_DONUT_MARGINS if has_pie else _CHART_MARGINS)
    chart.setTitleBrush(QColor(_LABEL_BRIGHT))
    title_font = QFont(theme.FONT_FAMILY, 11)
    title_font.setBold(True)
    chart.setTitleFont(title_font)
    legend = chart.legend()
    legend.setLabelColor(QColor(_LABEL_BRIGHT))
    legend.setFont(QFont(theme.FONT_FAMILY, 9))
    legend.setAlignment(Qt.AlignBottom)
    chart.setAnimationOptions(QChart.NoAnimation)
    # Keep plot area on the same panel tone (no second bg).
    bg = plot_bg or theme.COLORS["panel2"]
    chart.setPlotAreaBackgroundBrush(QBrush(QColor(bg)))
    chart.setPlotAreaBackgroundVisible(True)


def _slice_callout_parts(slice_) -> tuple[str, str]:
    """Return (name, pct_text) from slice properties or label."""
    name = slice_.property("so_name")
    pct = slice_.property("so_pct")
    if isinstance(name, str) and name.strip() and isinstance(pct, str) and pct.strip():
        return name.strip(), pct.strip()
    raw = (slice_.label() or "").strip()
    if "  " in raw:
        left, right = raw.rsplit("  ", 1)
        if right.endswith("%"):
            return left.strip() or raw, right
    if raw.endswith("%") and " " in raw:
        left, right = raw.rsplit(" ", 1)
        return left.strip() or raw, right
    return raw, ""


def paint_donut_callouts(
    p: QPainter,
    chart: QChart,
    *,
    name_px: int = 10,
    pct_px: int = 13,
    ink: QColor | None = None,
) -> None:
    """Draw callouts: category above the leader, larger % below.

    Qt Charts cannot do multiline outside labels, so we hide built-in labels and
    paint these ourselves (GUI + PDF/PNG export). ``plotArea()`` must be in the
    current painter coordinate system (view / render target space).
    """
    plot = chart.plotArea()
    if plot.width() < 8 or plot.height() < 8:
        return
    cx = plot.center().x()
    cy = plot.center().y()
    bright = QColor(ink) if ink is not None else QColor(_LABEL_BRIGHT)
    name_font = QFont(theme.FONT_FAMILY, name_px)
    pct_font = QFont(theme.FONT_FAMILY, pct_px)
    pct_font.setBold(True)
    name_fm = QFontMetrics(name_font)
    pct_fm = QFontMetrics(pct_font)

    for series in chart.series():
        if not hasattr(series, "slices") or not hasattr(series, "pieSize"):
            continue
        pie_size = float(series.pieSize())
        r = 0.5 * min(plot.width(), plot.height()) * pie_size
        if r < 4:
            continue
        for slice_ in series.slices():
            if float(slice_.value()) <= 0:
                continue
            name, pct = _slice_callout_parts(slice_)
            if not name and not pct:
                continue
            mid = float(slice_.startAngle()) + float(slice_.angleSpan()) / 2.0
            rad = math.radians(mid)
            dx = math.sin(rad)
            dy = -math.cos(rad)
            arm = max(12.0, r * max(0.12, float(slice_.labelArmLengthFactor())) * 1.35)
            rim_x = cx + r * dx
            rim_y = cy + r * dy
            elbow_x = cx + (r + arm * 0.65) * dx
            elbow_y = cy + (r + arm * 0.65) * dy
            right = dx >= 0.0
            stub = 16.0
            hx = elbow_x + (stub if right else -stub)
            hy = elbow_y

            p.setPen(QPen(bright, 1.0))
            p.drawLine(QPointF(rim_x, rim_y), QPointF(elbow_x, elbow_y))
            p.drawLine(QPointF(elbow_x, elbow_y), QPointF(hx, hy))

            gap = 5.0
            if name:
                p.setFont(name_font)
                p.setPen(bright)
                nw = name_fm.horizontalAdvance(name)
                nx = hx + gap if right else hx - gap - nw
                # Baseline sits just above the horizontal arm.
                p.drawText(int(round(nx)), int(round(hy - 4)), name)
            if pct:
                p.setFont(pct_font)
                p.setPen(bright)
                pw = pct_fm.horizontalAdvance(pct)
                px_ = hx + gap if right else hx - gap - pw
                # Larger % below the arm.
                p.drawText(
                    int(round(px_)),
                    int(round(hy + pct_fm.ascent() + 3)),
                    pct,
                )


class DonutChartView(QChartView):
    """QChartView that paints name / % callouts for pie series."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._callout_name_px = 10
        self._callout_pct_px = 13
        self._callout_ink: QColor | None = None

    def set_callout_fonts(self, name_px: int, pct_px: int) -> None:
        self._callout_name_px = max(6, int(name_px))
        self._callout_pct_px = max(7, int(pct_px))

    def set_callout_ink(self, color: QColor | None) -> None:
        """Override callout/leader color (e.g. black for print PDF)."""
        self._callout_ink = QColor(color) if color is not None else None

    def drawForeground(self, painter, rect) -> None:  # noqa: N802
        super().drawForeground(painter, rect)
        chart = self.chart()
        if chart is None:
            return
        if not any(hasattr(s, "slices") for s in chart.series()):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        paint_donut_callouts(
            painter,
            chart,
            name_px=self._callout_name_px,
            pct_px=self._callout_pct_px,
            ink=self._callout_ink,
        )


class ChartHost(QWidget):
    """QChartView wrapper — fixed height keeps legend + footnote visible."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._view = DonutChartView()
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setStyleSheet(
            f"background: {theme.COLORS['panel2']}; border: none; border-radius: 8px;"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)
        self.setFixedHeight(_CHART_HOST_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"background: {theme.COLORS['panel2']}; border-radius: 8px;"
        )

    def set_plot_height(self, h: int) -> None:
        """Fit plot to remaining viewport (keeps legend + footnote on-screen)."""
        self.setFixedHeight(max(180, int(h)))

    def set_chart(
        self,
        chart: QChart,
        *,
        plot_bg: str | None = None,
        series_grow: bool = True,
    ) -> None:
        """Attach chart with series grow. Donuts use full pie SeriesAnimations."""
        from PySide6.QtCore import QTimer

        _style_chart(chart, plot_bg=plot_bg)
        has_bars = any(hasattr(s, "barSets") for s in chart.series())
        has_pie = any(hasattr(s, "slices") for s in chart.series())

        if series_grow and has_pie:
            # Full donut effect: attach empty slices, then animate to real values
            # with SeriesAnimations already on (matches pre-change Qt pie feel).
            saved: list[tuple[object, float]] = []
            for series in chart.series():
                if not hasattr(series, "slices"):
                    continue
                for slice_ in series.slices():
                    saved.append((slice_, float(slice_.value())))
                    slice_.setValue(0.0001)
            chart.setAnimationDuration(max(_BAR_GROW_MS, 600))
            chart.setAnimationOptions(QChart.SeriesAnimations)
            self._view.setChart(chart)

            def _grow_pie() -> None:
                if self._view.chart() is not chart:
                    return
                chart.setAnimationOptions(QChart.SeriesAnimations)
                for slice_, value in saved:
                    slice_.setValue(value)

            QTimer.singleShot(20, _grow_pie)
            return

        # Bars: attach still, then nudge — avoids whole-plot fly-in.
        chart.setAnimationOptions(QChart.NoAnimation)
        self._view.setChart(chart)
        if not series_grow or not has_bars:
            return

        def _enable_grow() -> None:
            if self._view.chart() is not chart:
                return
            chart.setAnimationDuration(_BAR_GROW_MS)
            chart.setAnimationOptions(QChart.SeriesAnimations)
            for series in chart.series():
                if not hasattr(series, "barSets"):
                    continue
                for bar_set in series.barSets():
                    vals = [bar_set.at(i) for i in range(bar_set.count())]
                    for i in range(len(vals)):
                        bar_set.replace(i, 0.0)
                    for i, v in enumerate(vals):
                        bar_set.replace(i, v)

        QTimer.singleShot(0, _enable_grow)


def make_donut(
    title: str,
    items: Sequence[tuple[str, float]],
    *,
    colors: Optional[Mapping[str, str]] = None,
) -> QChart:
    """Title → donut; name / % callouts painted by DonutChartView (not Qt labels)."""
    chart = QChart()
    chart.setTitle(title)
    series = QPieSeries()
    series.setPieSize(CHART_DONUT_PIE)
    series.setHoleSize(0.20)  # placeholder; sync_chart_stroke_px sets real hole
    fallback = _accent_palette(len(items))
    for i, (name, value) in enumerate(items):
        if value <= 0:
            continue
        slice_ = series.append(str(name), value)
        hex_c = None
        if colors:
            hex_c = colors.get(name) or colors.get(name.lower())
        brush = QColor(hex_c) if hex_c else fallback[i % len(fallback)]
        slice_.setBrush(brush)
        slice_.setPen(QPen(QColor(theme.COLORS["panel2"]), 2))
        # Built-in outside labels are single-line and clip in tight cells —
        # DonutChartView / export paint name above + larger % below the arm.
        slice_.setProperty("so_name", str(name))
        slice_.setProperty("so_pct", f"{value:.0f}%")
        slice_.setLabelVisible(False)
        slice_.setLabelArmLengthFactor(0.16 if value < 10 else 0.12)
    chart.addSeries(series)
    chart.legend().setVisible(False)
    _bind_stroke_sync(chart)
    return chart


def make_bar(
    title: str,
    items: Sequence[tuple[str, float]],
    *,
    unit: str = "",
    bar_color: str | None = None,
    per_bar_colors: Optional[Mapping[str, str]] = None,
    show_legend: bool = False,
) -> QChart:
    """Same chrome as donuts; bars for counts / histograms."""
    chart = QChart()
    chart.setTitle(title)
    cats = [name for name, _ in items]

    if per_bar_colors:
        # Stacked + one set per category (value only at that index) so each
        # genre gets a full-width bar in its own color. Plain QBarSeries with
        # one-value sets would cluster every bar under the first category.
        series: QBarSeries | QStackedBarSeries = QStackedBarSeries()
        n = len(items)
        for i, (name, value) in enumerate(items):
            s = QBarSet(name)
            for j in range(n):
                s.append(float(value) if j == i else 0.0)
            hex_c = per_bar_colors.get(name) or per_bar_colors.get(name.lower())
            s.setColor(QColor(hex_c or theme.COLORS["accent"]))
            series.append(s)
        chart.legend().setVisible(show_legend)
        if show_legend:
            chart.legend().setAlignment(Qt.AlignBottom)
    else:
        series = QBarSeries()
        bar = QBarSet("")
        for _, value in items:
            bar.append(value)
        bar.setColor(QColor(bar_color or theme.COLORS["accent"]))
        series.append(bar)
        chart.legend().setVisible(False)

    # Placeholder; sync_chart_stroke_px sets width to CHART_STROKE_PX (66).
    series.setBarWidth(0.5)
    chart.addSeries(series)

    axis_x = QBarCategoryAxis()
    axis_x.append(cats)
    axis_x.setLabelsColor(QColor(_LABEL_MUTED))
    axis_x.setLabelsFont(QFont(theme.FONT_FAMILY, 9))
    chart.addAxis(axis_x, Qt.AlignBottom)
    series.attachAxis(axis_x)

    axis_y = QValueAxis()
    axis_y.setLabelFormat("%.0f" + (f" {unit}" if unit else ""))
    axis_y.setLabelsColor(QColor(_LABEL_MUTED))
    axis_y.setLabelsFont(QFont(theme.FONT_FAMILY, 9))
    chart.addAxis(axis_y, Qt.AlignLeft)
    series.attachAxis(axis_y)
    _bind_stroke_sync(chart)
    return chart



class _StylePctBar(QWidget):
    """Thin horizontal share bar for a style row."""

    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._pct = 0.0
        self._color = QColor(color)
        self._anim: QPropertyAnimation | None = None
        self.setFixedHeight(6)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _get_pct(self) -> float:
        return self._pct

    def _set_pct(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, float(pct)))
        self.update()

    pct = Property(float, _get_pct, _set_pct)

    def set_pct(self, pct: float, *, animate: bool = False) -> None:
        target = max(0.0, min(100.0, float(pct)))
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        if not animate:
            self._set_pct(target)
            return
        self._set_pct(0.0)
        anim = QPropertyAnimation(self, b"pct", self)
        anim.setStartValue(0.0)
        anim.setEndValue(target)
        anim.setDuration(_BAR_GROW_MS)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim = anim
        anim.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.setPen(Qt.NoPen)
        track = QColor(theme.COLORS["border"])
        track.setAlpha(120)
        p.setBrush(track)
        p.drawRoundedRect(r, 3, 3)
        fill_w = int(r.width() * self._pct / 100.0)
        if fill_w > 0:
            p.setBrush(self._color)
            p.drawRoundedRect(0, 0, max(fill_w, 4), r.height(), 3, 3)


class _GenreVBar(QWidget):
    """Vertical share bar — same track/fill look as Style, upright."""

    def __init__(self, color: str, parent=None) -> None:
        super().__init__(parent)
        self._pct = 0.0
        self._color = QColor(color)
        self._anim: QPropertyAnimation | None = None
        self.setMinimumWidth(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _get_pct(self) -> float:
        return self._pct

    def _set_pct(self, pct: float) -> None:
        self._pct = max(0.0, min(100.0, float(pct)))
        self.update()

    pct = Property(float, _get_pct, _set_pct)

    def set_pct(self, pct: float, *, animate: bool = False) -> None:
        target = max(0.0, min(100.0, float(pct)))
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        if not animate:
            self._set_pct(target)
            return
        self._set_pct(0.0)
        anim = QPropertyAnimation(self, b"pct", self)
        anim.setStartValue(0.0)
        anim.setEndValue(target)
        anim.setDuration(_BAR_GROW_MS)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim = anim
        anim.start()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.setPen(Qt.NoPen)
        track = QColor(theme.COLORS["border"])
        track.setAlpha(120)
        p.setBrush(track)
        p.drawRoundedRect(r, 3, 3)
        fill_h = int(r.height() * self._pct / 100.0)
        if fill_h > 0:
            p.setBrush(self._color)
            top = r.height() - fill_h
            p.drawRoundedRect(0, top, r.width(), max(fill_h, 4), 3, 3)


class GenreVerticalBars(QWidget):
    """Genre counts as vertical Style-style bars (full width, scaled to max)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(_CHART_MIN_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background: {theme.COLORS['panel2']}; border-radius: 8px;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 6)
        root.setSpacing(8)

        self._title = QLabel("Genre")
        title_font = QFont(theme.FONT_FAMILY, 11)
        title_font.setBold(True)
        self._title.setFont(title_font)
        self._title.setStyleSheet(f"color: {_LABEL_BRIGHT}; background: transparent;")
        self._title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        root.addWidget(self._title)

        self._row = QWidget()
        self._row.setStyleSheet("background: transparent;")
        self._row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._row_lay = QHBoxLayout(self._row)
        self._row_lay.setContentsMargins(4, 2, 4, 2)
        self._row_lay.setSpacing(8)
        root.addWidget(self._row, stretch=1)

    def set_data(
        self,
        title: str,
        items: Sequence[tuple[str, float]],
        *,
        colors: Optional[Mapping[str, str]] = None,
        tooltip_unit: str = "files",
    ) -> None:
        """items: (name, count). Bar height = count / max(count) so the largest fills."""
        self._title.setText(title)
        while self._row_lay.count():
            item = self._row_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not items:
            return

        color_map = colors or genre_colors_for([n for n, _ in items])
        counts = [float(v) for _, v in items]
        peak = max(counts) or 1.0
        total = sum(counts) or 1.0
        # Fill relative to largest genre (max bar = 100%).
        fills = [100.0 * c / peak for c in counts]
        # Same name-band height for every column so bars share one baseline
        # (two-line key labels like Abm\nG#m must not lift those bars).
        name_band_h = 40 if any("\n" in n for n, _ in items) else 28

        for (name, raw), fill in zip(items, fills):
            share = 100.0 * float(raw) / total
            col = QWidget()
            col.setStyleSheet("background: transparent;")
            col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            col_lay = QVBoxLayout(col)
            col_lay.setContentsMargins(0, 0, 0, 0)
            col_lay.setSpacing(4)

            # Show count (mono) — denser signal than tiny share %
            top_lbl = QLabel(f"{raw:,.0f}")
            top_lbl.setAlignment(Qt.AlignHCenter)
            top_lbl.setFont(QFont(theme.FONT_FAMILY_MONO, theme.LOG_FONT_PX - 1))
            top_lbl.setStyleSheet(
                f"color: {theme.COLORS['fg_dim']}; background: transparent;"
            )
            col_lay.addWidget(top_lbl, 0)

            hex_c = color_map.get(name) or color_map.get(name.lower()) or theme.COLORS["accent"]
            bar = _GenreVBar(hex_c)
            bar.set_pct(fill, animate=True)
            col_lay.addWidget(bar, 1)

            name_lbl = QLabel(name)
            name_lbl.setWordWrap(True)
            name_lbl.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            name_lbl.setFixedHeight(name_band_h)
            name_lbl.setStyleSheet(
                f"color: {_LABEL_MUTED}; background: transparent; font-size: 10px;"
            )
            tip_name = name.replace("\n", " / ")
            col_lay.addWidget(name_lbl, 0)

            col.setToolTip(
                f"{tip_name}: {raw:,.0f} {tooltip_unit}  ·  {share:.0f}% of library  ·  "
                f"{fill:.0f}% of largest"
            )
            self._row_lay.addWidget(col, stretch=1)


class GenreStyleBreakdown(QWidget):
    """Styles nested under each tagged genre (MAEST order when present)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(_CHART_MIN_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background: {theme.COLORS['panel2']}; border-radius: 8px;"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 6)
        root.setSpacing(6)

        self._title = QLabel("Styles by genre")
        title_font = QFont(theme.FONT_FAMILY, 11)
        title_font.setBold(True)
        self._title.setFont(title_font)
        self._title.setStyleSheet(f"color: {_LABEL_BRIGHT}; background: transparent;")
        self._title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        root.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        self._body = QWidget()
        self._body.setStyleSheet("background: transparent;")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(4, 4, 8, 8)
        self._body_lay.setSpacing(14)
        self._body_lay.addStretch(1)
        scroll.setWidget(self._body)
        root.addWidget(scroll, stretch=1)

    def set_data(
        self,
        title: str,
        genres: Sequence[tuple[str, int]],
        styles_by_genre: Mapping[str, Mapping[str, object]],
        *,
        genre_colors: Optional[Mapping[str, str]] = None,
    ) -> None:
        """genres: (name, count). styles: genre → {style → bucket-like with .count}."""
        self._title.setText(title)
        # Clear previous rows (keep trailing stretch).
        while self._body_lay.count() > 1:
            item = self._body_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        colors = genre_colors or genre_colors_for([n for n, _ in genres])
        for genre_name, genre_count in genres:
            block = QWidget()
            block.setStyleSheet("background: transparent;")
            block_lay = QVBoxLayout(block)
            block_lay.setContentsMargins(0, 0, 0, 0)
            block_lay.setSpacing(4)

            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 2)
            header.setSpacing(8)
            swatch = QLabel()
            swatch.setFixedSize(10, 10)
            hex_c = colors.get(genre_name, theme.COLORS["accent"])
            swatch.setStyleSheet(
                f"background: {hex_c}; border-radius: 2px;"
            )
            name_lbl = QLabel(genre_name)
            name_font = QFont(theme.FONT_FAMILY, 10)
            name_font.setBold(True)
            name_lbl.setFont(name_font)
            name_lbl.setStyleSheet(f"color: {_LABEL_BRIGHT}; background: transparent;")
            count_lbl = QLabel(f"{genre_count:,}")
            count_lbl.setStyleSheet(
                f"color: {_LABEL_MUTED}; background: transparent; font-size: 11px;"
            )
            header.addWidget(swatch, 0, Qt.AlignVCenter)
            header.addWidget(name_lbl, 0, Qt.AlignVCenter)
            header.addStretch(1)
            header.addWidget(count_lbl, 0, Qt.AlignVCenter)
            block_lay.addLayout(header)

            nested = styles_by_genre.get(genre_name) or {}
            # Sort styles by count desc
            rows: list[tuple[str, int]] = []
            for style_name, bucket in nested.items():
                c = int(getattr(bucket, "count", 0) or 0)
                if c > 0:
                    rows.append((style_name, c))
            rows.sort(key=lambda x: -x[1])
            base = genre_count or sum(c for _, c in rows) or 1

            if not rows:
                empty = QLabel("  No styles tagged")
                empty.setStyleSheet(
                    f"color: {theme.COLORS['fg_dim']}; background: transparent; "
                    f"font-size: 11px; padding-left: 18px;"
                )
                block_lay.addWidget(empty)
            else:
                for style_name, count in rows:
                    pct = 100.0 * count / base
                    row = QHBoxLayout()
                    row.setContentsMargins(18, 0, 0, 0)
                    row.setSpacing(8)
                    s_lbl = QLabel(style_name)
                    s_lbl.setStyleSheet(
                        f"color: {_LABEL_MUTED}; background: transparent; "
                        f"font-size: 11px;"
                    )
                    s_lbl.setMinimumWidth(110)
                    bar = _StylePctBar(hex_c)
                    bar.set_pct(pct, animate=True)
                    pct_lbl = QLabel(f"{pct:.0f}%")
                    pct_lbl.setFixedWidth(36)
                    pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    pct_lbl.setStyleSheet(
                        f"color: {_LABEL_BRIGHT}; background: transparent; "
                        f"font-size: 11px;"
                    )
                    row.addWidget(s_lbl, 0)
                    row.addWidget(bar, 1)
                    row.addWidget(pct_lbl, 0)
                    block_lay.addLayout(row)

            self._body_lay.insertWidget(self._body_lay.count() - 1, block)


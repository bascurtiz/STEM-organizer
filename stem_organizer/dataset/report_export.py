"""Export Dataset overview KPIs + charts to PDF (multi-page) or PNG (spread)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCharts import QAbstractAxis, QChart
from PySide6.QtCore import QMargins, QMarginsF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPageLayout, QPageSize, QPainter, QPen
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import QApplication

from .. import theme
from .charts import (
    DonutChartView,
    _style_chart,
    genre_colors_for,
    make_bar,
    make_donut,
    sync_chart_stroke_px,
)
from .models import OverviewStats, VOCAL_TYPES, style_genres_from_stats


@dataclass
class PathsMeta:
    instrumental: str = ""
    vocal: str = ""
    pairs: str = ""
    samples: str = ""
    include_subfolders: bool = True


@dataclass(frozen=True)
class _Ink:
    """Screen (dark) vs print (B&W chrome) palette for report painting."""

    bg: str
    panel: str | None  # None → no card fill
    fg: str
    dim: str
    track: str
    border: str | None
    chart_bg: str
    slice_pen: str


_SCREEN_INK = _Ink(
    bg=theme.COLORS["bg"],
    panel=theme.COLORS["panel2"],
    fg=theme.COLORS["fg"],
    dim=theme.COLORS["fg_dim"],
    track=theme.COLORS["border"],
    border=None,
    chart_bg=theme.COLORS["panel2"],
    slice_pen=theme.COLORS["panel2"],
)
_PRINT_INK = _Ink(
    bg="#ffffff",
    panel=None,
    fg="#000000",
    dim="#000000",
    track="#dddddd",
    border=None,  # no card chrome on paper — gutters alone define cells
    chart_bg="#ffffff",
    slice_pen="#ffffff",
)


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


def _kpi_values(stats: OverviewStats) -> list[tuple[str, str]]:
    inst = stats.roles.instrumental
    voc = stats.roles.vocal
    samples = stats.roles.samples
    return [
        ("Files", f"{stats.total_files:,}"),
        ("Size", _fmt_bytes(stats.total_bytes)),
        ("Duration", _fmt_hours(stats.duration.total)),
        ("Instrumental units", f"{inst:,}"),
        ("Vocal units", f"{voc:,}"),
        ("Sample units", f"{samples:,}"),
    ]


def _sdr_bar_items(stats: OverviewStats) -> list[tuple[str, float]]:
    vals = stats.sdr.values
    if not vals:
        return []
    if stats.demo:
        bins = list(range(30, 56, 5))
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
    return [
        (f"{bins[i]}–{bins[i + 1]}", float(counts[i]))
        for i in range(len(counts))
    ]


def build_export_charts(stats: OverviewStats) -> dict[str, Optional[QChart]]:
    """Rebuild every facet chart off-screen (independent of UI segment)."""
    out: dict[str, Optional[QChart]] = {}

    # SI-SDR
    sdr_items = _sdr_bar_items(stats)
    out["sdr"] = (
        make_bar("SI-SDR (dB) distribution", sdr_items, bar_color=theme.SDR_PASS_COLOR)
        if sdr_items
        else None
    )

    # Roles
    inst = stats.roles.instrumental
    voc = stats.roles.vocal
    samples = stats.roles.samples
    total = max(inst + voc + samples, 1)
    out["roles"] = make_donut(
        "Instrumental / Vocal / Samples",
        [
            ("Instrumental", 100.0 * inst / total),
            ("Vocal", 100.0 * voc / total),
            ("Samples", 100.0 * samples / total),
        ],
        colors=theme.CHART_ROLE_COLORS,
    )

    # Genre is painted like the in-GUI GenreVerticalBars (not QtCharts).
    out["genre"] = None
    out["keys"] = None

    # Gender / Reverb / Compression
    for key, store, title, colors in (
        ("gender", stats.gender, "Gender", theme.CHART_GENDER_COLORS),
        ("reverb", stats.reverb, "Reverb", theme.CHART_REVERB_COLORS),
        ("compression", stats.compression, "Compression", theme.CHART_COMPRESSION_COLORS),
    ):
        total_c = sum(b.count for b in store.values()) or 1
        items = [
            (name[:1].upper() + name[1:], 100.0 * b.count / total_c)
            for name, b in store.items()
        ]
        out[key] = make_donut(title, items, colors=colors) if items else None

    # Vocal type — same multi-slice donut as the Charts GUI
    vt_total = sum(b.count for b in stats.vocal_type.values()) or 1
    vt_items = [
        (
            name,
            100.0
            * (stats.vocal_type[name].count if name in stats.vocal_type else 0)
            / vt_total,
        )
        for name in VOCAL_TYPES
    ]
    out["vocal"] = (
        make_donut("Vocal type", vt_items, colors=theme.CHART_VOCAL_TYPE_COLORS)
        if any(v > 0 for _, v in vt_items)
        else None
    )

    return out


def render_chart_to_painter(
    p: QPainter,
    chart: QChart,
    target: QRectF,
    *,
    ink: _Ink = _SCREEN_INK,
) -> None:
    """Paint a chart into ``target`` via QPainter (vector text on PDF).

    Avoids ``grab()`` pixmaps which rasterize labels and go blurry when zoomed.
    Always uses compact grid styling so donut callouts fit the 4×3 cells.
    """
    _style_chart(chart, plot_bg=ink.chart_bg)
    chart.setBackgroundBrush(QColor(ink.chart_bg))
    chart.setBackgroundRoundness(0 if ink.panel is None else 8)
    chart.setTitleBrush(QColor(ink.fg))
    legend = chart.legend()
    legend.setLabelColor(QColor(ink.fg))
    for axis in chart.axes():
        if isinstance(axis, QAbstractAxis):
            axis.setLabelsBrush(QColor(ink.fg))
            axis.setTitleBrush(QColor(ink.fg))
            axis.setLinePen(QPen(QColor(ink.dim), 1))
            axis.setGridLineColor(QColor(ink.track))
    for series in chart.series():
        if not hasattr(series, "slices"):
            continue
        for slice_ in series.slices():
            slice_.setPen(QPen(QColor(ink.slice_pen), 2))
    chart.setAnimationOptions(QChart.NoAnimation)

    # Soft cap only for pathological sizes (keeps off-screen widgets sane).
    w = max(120, min(1000, int(round(target.width()))))
    h = max(90, min(800, int(round(target.height()))))
    print_mode = ink.panel is None
    _style_chart_for_export_grid(chart, print_mode=print_mode)

    view = DonutChartView(chart)
    view.setRenderHint(QPainter.Antialiasing, True)
    view.setRenderHint(QPainter.TextAntialiasing, True)
    view.set_callout_ink(QColor(ink.fg))
    if print_mode:
        view.set_callout_fonts(7, 9)
    else:
        view.set_callout_fonts(8, 11)
    view.setStyleSheet(f"background: {ink.chart_bg}; border: none;")
    view.resize(w, h)
    view.setAttribute(Qt.WA_DontShowOnScreen, True)
    view.show()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    # Plot area is final after resize — lock bars/donut ring to 66px.
    sync_chart_stroke_px(chart)
    if print_mode:
        for series in chart.series():
            if hasattr(series, "setPieSize"):
                try:
                    series.setPieSize(min(0.58, float(series.pieSize())))
                except Exception:
                    series.setPieSize(0.58)
    if app is not None:
        app.processEvents()

    p.save()
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)
    p.translate(target.left(), target.top())
    # Scale widget render into the (possibly different) target size.
    if w > 0 and h > 0 and (
        abs(target.width() - w) > 0.5 or abs(target.height() - h) > 0.5
    ):
        p.scale(target.width() / w, target.height() / h)
    view.render(p, QRectF(0, 0, w, h), view.rect())
    p.restore()

    view.hide()
    view.deleteLater()


def _style_chart_for_export_grid(chart: QChart, *, print_mode: bool = False) -> None:
    """Smaller title + pie so custom callouts fit in grid cells."""
    title_font = QFont(theme.FONT_FAMILY, 8)
    title_font.setBold(True)
    chart.setTitleFont(title_font)
    # Print cells are physically smaller on A4 — more margin, shorter arms.
    if print_mode:
        chart.setMargins(QMargins(16, 10, 16, 12))
        arm = 0.18
    else:
        chart.setMargins(QMargins(10, 4, 10, 8))
        arm = 0.14
    axis_font = QFont(theme.FONT_FAMILY, 7)
    for axis in chart.axes():
        if isinstance(axis, QAbstractAxis):
            axis.setLabelsFont(axis_font)
    for series in chart.series():
        if not hasattr(series, "slices"):
            continue
        for slice_ in series.slices():
            slice_.setLabelVisible(False)
            slice_.setLabelArmLengthFactor(arm)
        if hasattr(series, "setPieSize"):
            series.setPieSize(0.62 if print_mode else 0.72)


def _genre_list(stats: OverviewStats) -> list[tuple[str, int]]:
    return style_genres_from_stats(stats)


# ---------------------------------------------------------------------------
# Painting helpers (shared PDF / PNG)
# ---------------------------------------------------------------------------

_BG = theme.COLORS["bg"]
_PANEL = theme.COLORS["panel2"]
_FG = theme.COLORS["fg"]
_DIM = theme.COLORS["fg_dim"]
_ACCENT = theme.COLORS["accent"]
_NOTES_LABEL_PT = 12
_NOTES_BODY_PT = 14
# PNG grid: match QtChart title baseline in the left-hand cards.
_EMBEDDED_TITLE_DY = 18.0
_EMBEDDED_TITLE_H = 20.0
_EMBEDDED_TOP_BAND = 44.0


def _fill_page(p: QPainter, rect: QRectF, *, ink: _Ink = _SCREEN_INK) -> None:
    # Bleed past the paper rect — matching DevicePixel bounds exactly often
    # leaves a 1px white hairline on the right/bottom of the PDF page.
    p.fillRect(rect.adjusted(-2, -2, 4, 4), QColor(ink.bg))


def _paint_card(p: QPainter, rect: QRectF, ink: _Ink, *, radius: float = 8.0) -> None:
    """Card chrome: filled panel on screen, light outline only when printing."""
    if ink.panel is not None:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ink.panel))
        p.drawRoundedRect(rect, radius, radius)
    elif ink.border is not None:
        p.setPen(QPen(QColor(ink.border), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


def _paper_rect(printer: QPrinter) -> QRectF:
    r = QRectF(printer.paperRect(QPrinter.DevicePixel))
    # Normalize to origin so paint coords match the printer device.
    return QRectF(0, 0, r.width(), r.height())


def _draw_footer(
    p: QPainter,
    rect: QRectF,
    page: int | None = None,
    total: int | None = None,
    *,
    ink: _Ink = _SCREEN_INK,
) -> None:
    p.save()
    font = QFont(theme.FONT_FAMILY, 8)
    p.setFont(font)
    p.setPen(QColor(ink.dim))
    left = "STEM organizer · Charts"
    y = rect.bottom() - 10
    p.drawText(QRectF(rect.left() + 24, y - 14, rect.width() - 48, 16), Qt.AlignLeft, left)
    if page is not None:
        right = f"{page} / {total}" if total else str(page)
        p.drawText(
            QRectF(rect.center().x(), y - 14, rect.width() / 2 - 24, 16),
            Qt.AlignRight,
            right,
        )
    p.restore()


def _draw_title_block(
    p: QPainter,
    rect: QRectF,
    *,
    demo: bool,
    stamp: str,
) -> float:
    """Returns y after the title block."""
    x = rect.left() + 28
    y = rect.top() + 28
    title_font = QFont(theme.FONT_FAMILY, 16)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(QColor(_FG))
    p.drawText(int(x), int(y), "Charts")
    y += 22
    if demo:
        badge = QFont(theme.FONT_FAMILY, 9)
        badge.setBold(True)
        p.setFont(badge)
        p.setPen(QColor(_ACCENT))
        p.drawText(int(x), int(y), "DEMO")
        y += 16
    meta = QFont(theme.FONT_FAMILY, 9)
    p.setFont(meta)
    p.setPen(QColor(_DIM))
    p.drawText(int(x), int(y), stamp)
    return y + 18


def _draw_paths(p: QPainter, rect: QRectF, y: float, paths: PathsMeta) -> float:
    x = rect.left() + 28
    label_font = QFont(theme.FONT_FAMILY, 9)
    label_font.setBold(True)
    value_font = QFont(theme.FONT_FAMILY_MONO, theme.LOG_FONT_PX - 1)
    rows = [
        ("Instrumental", paths.instrumental or "—"),
        ("Vocal", paths.vocal or "—"),
        ("Pairs", paths.pairs or "—"),
        ("Samples", paths.samples or "—"),
        (
            "Subfolders",
            "included" if paths.include_subfolders else "top-level only",
        ),
    ]
    for label, value in rows:
        p.setFont(label_font)
        p.setPen(QColor(_DIM))
        p.drawText(int(x), int(y), f"{label}:")
        p.setFont(value_font)
        p.setPen(QColor(_FG))
        # Truncate long paths
        text = value
        fm = p.fontMetrics()
        max_w = int(rect.width() - 160)
        if fm.horizontalAdvance(text) > max_w:
            while text and fm.horizontalAdvance(text + "…") > max_w:
                text = text[:-1]
            text = text + "…"
        p.drawText(int(x + 110), int(y), text)
        y += 16
    return y + 10


def _draw_kpi_strip(
    p: QPainter,
    rect: QRectF,
    y: float,
    stats: OverviewStats,
    *,
    ink: _Ink = _SCREEN_INK,
    layout_scale: float = 1.0,
) -> float:
    kpis = _kpi_values(stats)
    s = max(0.5, float(layout_scale))
    # Same outer inset as the chart grid below so edges line up.
    margin_x = 24.0 * s
    gap = 12.0 * s
    avail = rect.width() - 2 * margin_x
    chip_w = (avail - gap * (len(kpis) - 1)) / len(kpis)
    chip_h = 58.0 * s
    x = rect.left() + margin_x
    title_font = QFont(theme.FONT_FAMILY, 9)
    value_font = QFont(theme.FONT_FAMILY_MONO, theme.LOG_FONT_PX + 1)
    for label, value in kpis:
        chip = QRectF(x, y, chip_w, chip_h)
        _paint_card(p, chip, ink, radius=6.0)
        p.setPen(QColor(ink.dim))
        p.setFont(title_font)
        p.drawText(
            QRectF(chip.left() + 10 * s, chip.top() + 7 * s, chip.width() - 20 * s, 16 * s),
            Qt.AlignLeft | Qt.AlignVCenter,
            label,
        )
        p.setPen(QColor(ink.fg))
        p.setFont(value_font)
        p.drawText(
            QRectF(chip.left() + 10 * s, chip.top() + 26 * s, chip.width() - 20 * s, 24 * s),
            Qt.AlignLeft | Qt.AlignVCenter,
            value,
        )
        x += chip_w + gap
    return y + chip_h + 16 * s


def _draw_style_pages(
    painter: QPainter,
    printer: QPrinter,
    page_rect: QRectF,
    stats: OverviewStats,
    *,
    page_no: list[int],
) -> None:
    """Paginated genre → style breakdown (paint, not widget grab)."""
    genres = _genre_list(stats)
    colors = genre_colors_for([n for n, _ in genres])
    y = _draw_title_block(
        painter,
        page_rect,
        demo=stats.demo,
        stamp="Styles by genre — % is share within that genre",
    )
    x0 = page_rect.left() + 28
    bottom = page_rect.bottom() - 36
    title_font = QFont(theme.FONT_FAMILY, 10)
    title_font.setBold(True)
    style_font = QFont(theme.FONT_FAMILY, 8)
    mono = QFont(theme.FONT_FAMILY_MONO, 9)

    for genre_name, genre_count in genres:
        nested = stats.styles_by_genre.get(genre_name) or {}
        rows = sorted(
            (
                (sn, int(getattr(b, "count", 0) or 0))
                for sn, b in nested.items()
                if int(getattr(b, "count", 0) or 0) > 0
            ),
            key=lambda t: -t[1],
        )
        block_h = 22 + max(14, 14 * len(rows)) + 10
        if y + block_h > bottom:
            _draw_footer(painter, page_rect, page_no[0])
            printer.newPage()
            page_no[0] += 1
            _fill_page(painter, page_rect)
            y = _draw_title_block(
                painter,
                page_rect,
                demo=stats.demo,
                stamp="Styles by genre (continued)",
            )

        hex_c = colors.get(genre_name, theme.COLORS["accent"])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(hex_c))
        painter.drawRoundedRect(QRectF(x0, y - 9, 10, 10), 2, 2)
        painter.setFont(title_font)
        painter.setPen(QColor(_FG))
        painter.drawText(int(x0 + 16), int(y), genre_name)
        painter.setFont(mono)
        painter.setPen(QColor(_DIM))
        painter.drawText(
            QRectF(page_rect.right() - 120, y - 12, 90, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{genre_count:,}",
        )
        y += 16
        base = genre_count or sum(c for _, c in rows) or 1
        if not rows:
            painter.setFont(style_font)
            painter.setPen(QColor(_DIM))
            painter.drawText(int(x0 + 18), int(y), "No styles tagged")
            y += 14
        else:
            bar_left = x0 + 130
            bar_right = page_rect.right() - 70
            bar_w = max(40.0, bar_right - bar_left)
            for style_name, count in rows:
                pct = 100.0 * count / base
                painter.setFont(style_font)
                painter.setPen(QColor(theme.COLORS["log_fg"]))
                painter.drawText(int(x0 + 18), int(y), style_name)
                track = QRectF(bar_left, y - 7, bar_w, 6)
                painter.setPen(Qt.NoPen)
                track_c = QColor(theme.COLORS["border"])
                track_c.setAlpha(120)
                painter.setBrush(track_c)
                painter.drawRoundedRect(track, 3, 3)
                fill_w = max(3.0, bar_w * pct / 100.0) if pct > 0 else 0.0
                if fill_w:
                    painter.setBrush(QColor(hex_c))
                    painter.drawRoundedRect(
                        QRectF(bar_left, y - 7, fill_w, 6), 3, 3
                    )
                painter.setFont(mono)
                painter.setPen(QColor(_FG))
                painter.drawText(
                    QRectF(bar_right + 4, y - 12, 40, 16),
                    Qt.AlignRight | Qt.AlignVCenter,
                    f"{pct:.0f}%",
                )
                y += 14
        y += 8

    _draw_footer(painter, page_rect, page_no[0])


def _key_bar_metrics(stats: OverviewStats) -> tuple[list[tuple[str, float]], float, float, float]:
    from ..musical_keys import KEY_DISPLAY_ORDER

    items = [
        (name, float(stats.key[name].count))
        for name in KEY_DISPLAY_ORDER
        if name in stats.key and stats.key[name].count > 0
    ]
    if not items:
        return [], 40.0, 0.0, 1.0
    n = len(items)
    gap = 8.0
    gui_row_w = float(theme.WIN_DEFAULT_W + 320)
    col_w = (gui_row_w - gap * max(0, n - 1)) / max(n, 1)
    total_bars_w = n * col_w + gap * max(0, n - 1)
    peak = max(c for _, c in items) or 1.0
    return items, col_w, total_bars_w, peak


def _paint_key_bars_page(
    p: QPainter,
    rect: QRectF,
    stats: OverviewStats,
    *,
    page: int,
    stamp: str,
    panel_top: float | None = None,
    panel_bottom: float | None = None,
    draw_footer: bool = True,
    ink: _Ink = _SCREEN_INK,
    layout_scale: float = 1.0,
) -> float:
    """Match in-GUI Keys bars (GenreVerticalBars style + key colors)."""
    from ..musical_keys import CHART_KEY_COLORS, key_chart_label

    s = max(0.5, float(layout_scale))
    embedded = panel_top is not None
    if not embedded:
        _fill_page(p, rect, ink=ink)
        y = _draw_title_block(
            p,
            rect,
            demo=stats.demo,
            stamp=stamp,
        )
    else:
        y = float(panel_top)
    margin = 24.0 * s
    footer_pad = 36.0 * s
    pad_x = 12.0 * s
    gap = 8.0 * s
    top = panel_top if panel_top is not None else y
    bottom = (
        panel_bottom
        if panel_bottom is not None
        else rect.bottom() - footer_pad
    )
    avail_w = max(40.0 * s, rect.width() - 2 * margin)

    items, col_w, total_bars_w, peak = _key_bar_metrics(stats)
    if embedded:
        panel_w = avail_w
        if items:
            n = len(items)
            col_w = (panel_w - 2 * pad_x - gap * max(0, n - 1)) / max(n, 1)
            total_bars_w = n * col_w + gap * max(0, n - 1)
    else:
        content_w = total_bars_w + 2 * pad_x if items else avail_w
        panel_w = min(avail_w, max(120.0 * s, content_w))
        if items and panel_w + 0.5 >= avail_w:
            natural_col = (panel_w - 2 * pad_x - gap * (len(items) - 1)) / max(len(items), 1)
            col_w = min(col_w, natural_col)
            total_bars_w = len(items) * col_w + gap * max(0, len(items) - 1)
            panel_w = min(avail_w, total_bars_w + 2 * pad_x)

    panel = QRectF(
        rect.left() + margin,
        top,
        panel_w,
        max(40.0 * s, bottom - top),
    )
    _paint_card(p, panel, ink)

    if not items:
        p.setPen(QColor(ink.dim))
        p.setFont(QFont(theme.FONT_FAMILY, 11))
        p.drawText(panel, Qt.AlignCenter, "No key data")
        if draw_footer:
            _draw_footer(p, rect, page, ink=ink)
        return panel.right()

    title_dy = (_EMBEDDED_TITLE_DY if embedded else 10.0) * s
    top_band = (_EMBEDDED_TOP_BAND if embedded else 36.0) * s
    title_font = QFont(theme.FONT_FAMILY, 11)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(QColor(ink.fg))
    p.drawText(
        QRectF(
            panel.left(),
            panel.top() + title_dy,
            panel.width(),
            _EMBEDDED_TITLE_H * s,
        ),
        Qt.AlignHCenter | Qt.AlignTop,
        "Keys (vs largest)",
    )

    count_h = 18.0 * s
    name_h = 52.0 * s  # room for two stacked enharmonic lines
    col_top = panel.top() + top_band
    col_bottom = panel.bottom() - 10 * s
    bar_area_top = col_top + count_h + 4 * s
    bar_area_bottom = col_bottom - name_h - 4 * s
    bar_area_h = max(40.0 * s, bar_area_bottom - bar_area_top)
    origin_x = panel.left() + pad_x

    count_font = QFont(theme.FONT_FAMILY_MONO, 9)
    name_font = QFont(theme.FONT_FAMILY, 10)
    for i, (name, count) in enumerate(items):
        x = origin_x + i * (col_w + gap)
        fill = 100.0 * float(count) / peak
        hex_c = CHART_KEY_COLORS.get(name, theme.COLORS["accent"])
        # Count
        p.setFont(count_font)
        p.setPen(QColor(ink.dim))
        p.drawText(
            QRectF(x, col_top, col_w, count_h),
            Qt.AlignHCenter | Qt.AlignVCenter,
            f"{count:,.0f}",
        )
        # Bar
        bar_h = bar_area_h * (fill / 100.0)
        bar_rect = QRectF(
            x + col_w * 0.15,
            bar_area_bottom - bar_h,
            col_w * 0.7,
            bar_h,
        )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(hex_c))
        p.drawRoundedRect(bar_rect, 3 * s, 3 * s)
        # Name — stack enharmonics (Abm\nG#m), no slash
        label = key_chart_label(name)
        p.setFont(name_font)
        p.setPen(QColor(ink.dim))
        p.drawText(
            QRectF(x, col_bottom - name_h, col_w, name_h),
            Qt.AlignHCenter | Qt.AlignTop,
            label,
        )

    if draw_footer:
        _draw_footer(p, rect, page, ink=ink)
    return panel.right()


def _genre_bar_metrics(stats: OverviewStats) -> tuple[list[tuple[str, float]], float, float, float]:
    """items, col_w, total_bars_w, peak — shared by paint + PNG crop."""
    items = sorted(
        ((n, float(b.count)) for n, b in stats.genre.items() if b.count > 0),
        key=lambda x: -x[1],
    )
    if not items:
        return [], 0.0, 0.0, 1.0
    n = len(items)
    gap = 8.0
    # Same bar width policy as _paint_genre_bars_page.
    gui_row_w = float(theme.WIN_DEFAULT_W + 320)
    col_w = (gui_row_w - gap * max(0, n - 1)) / max(n, 1)
    total_bars_w = n * col_w + gap * max(0, n - 1)
    peak = max(c for _, c in items) or 1.0
    return items, col_w, total_bars_w, peak


def _paint_genre_bars_page(
    p: QPainter,
    rect: QRectF,
    stats: OverviewStats,
    *,
    page: int,
    stamp: str,
    panel_top: float | None = None,
    panel_bottom: float | None = None,
    draw_footer: bool = True,
    ink: _Ink = _SCREEN_INK,
    layout_scale: float = 1.0,
) -> float:
    """Match in-GUI GenreVerticalBars: count · vertical bar vs max · name.

    Panel is left-aligned and only as wide as the bars (no empty right padding).
    Returns the panel's right edge (for PNG crop / notes width).

    Optional ``panel_top`` / ``panel_bottom`` pin the Genre card (e.g. PNG
    spread: one grid row). When pinned, skips page fill / stamp so it can
    sit beside the donut grid without wiping neighbors.
    """
    s = max(0.5, float(layout_scale))
    embedded = panel_top is not None
    if not embedded:
        _fill_page(p, rect, ink=ink)
        y = _draw_title_block(
            p,
            rect,
            demo=stats.demo,
            stamp=stamp,
        )
    else:
        y = float(panel_top)
    margin = 24.0 * s
    footer_pad = 36.0 * s
    pad_x = 12.0 * s
    gap = 8.0 * s
    top = panel_top if panel_top is not None else y
    bottom = (
        panel_bottom
        if panel_bottom is not None
        else rect.bottom() - footer_pad
    )
    avail_w = max(40.0 * s, rect.width() - 2 * margin)

    items, col_w, total_bars_w, peak = _genre_bar_metrics(stats)
    if embedded:
        # Fill the 2-col span; scale bar columns to the panel width.
        panel_w = avail_w
        if items:
            n = len(items)
            col_w = (panel_w - 2 * pad_x - gap * max(0, n - 1)) / max(n, 1)
            total_bars_w = n * col_w + gap * max(0, n - 1)
    else:
        content_w = total_bars_w + 2 * pad_x if items else avail_w
        panel_w = min(avail_w, max(120.0 * s, content_w))
        if items and panel_w + 0.5 >= avail_w:
            natural_col = (panel_w - 2 * pad_x - gap * (len(items) - 1)) / max(len(items), 1)
            col_w = min(col_w, natural_col)
            total_bars_w = len(items) * col_w + gap * max(0, len(items) - 1)
            panel_w = min(avail_w, total_bars_w + 2 * pad_x)

    panel = QRectF(
        rect.left() + margin,
        top,
        panel_w,
        max(40.0 * s, bottom - top),
    )
    _paint_card(p, panel, ink)

    if not items:
        p.setPen(QColor(ink.dim))
        p.setFont(QFont(theme.FONT_FAMILY, 11))
        p.drawText(panel, Qt.AlignCenter, "No genre data")
        if draw_footer:
            _draw_footer(p, rect, page, ink=ink)
        return panel.right()

    # Title inside the panel (same treatment as GenreVerticalBars).
    title_dy = (_EMBEDDED_TITLE_DY if embedded else 10.0) * s
    top_band = (_EMBEDDED_TOP_BAND if embedded else 36.0) * s
    title_font = QFont(theme.FONT_FAMILY, 11)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(QColor(ink.fg))
    p.drawText(
        QRectF(
            panel.left(),
            panel.top() + title_dy,
            panel.width(),
            _EMBEDDED_TITLE_H * s,
        ),
        Qt.AlignHCenter | Qt.AlignTop,
        "Genre (vs largest)",
    )

    colors = genre_colors_for([n for n, _ in items])
    count_h = 18.0 * s
    name_h = 48.0 * s
    col_top = panel.top() + top_band
    col_bottom = panel.bottom() - 10 * s
    bar_area_top = col_top + count_h + 4 * s
    bar_area_bottom = col_bottom - name_h - 4 * s
    bar_area_h = max(40.0 * s, bar_area_bottom - bar_area_top)
    # Left-align bars inside the tight panel.
    origin_x = panel.left() + pad_x

    count_font = QFont(theme.FONT_FAMILY_MONO, 9)
    name_font = QFont(theme.FONT_FAMILY, 10)
    for i, (name, count) in enumerate(items):
        x = origin_x + i * (col_w + gap)
        # Count above bar
        p.setFont(count_font)
        p.setPen(QColor(ink.dim))
        p.drawText(
            QRectF(x, col_top, col_w, count_h),
            Qt.AlignHCenter | Qt.AlignVCenter,
            f"{count:,.0f}",
        )
        # Track + fill (bottom-up, relative to largest)
        track = QRectF(x + 2 * s, bar_area_top, max(4.0 * s, col_w - 4 * s), bar_area_h)
        track_c = QColor(ink.track)
        if ink is _SCREEN_INK:
            track_c.setAlpha(120)
        p.setPen(Qt.NoPen)
        p.setBrush(track_c)
        p.drawRoundedRect(track, 3 * s, 3 * s)
        fill_frac = float(count) / peak
        fill_h = max(4.0 * s, bar_area_h * fill_frac) if count > 0 else 0.0
        if fill_h > 0:
            hex_c = colors.get(name) or theme.COLORS["accent"]
            p.setBrush(QColor(hex_c))
            p.drawRoundedRect(
                QRectF(track.left(), track.bottom() - fill_h, track.width(), fill_h),
                3 * s,
                3 * s,
            )
        # Genre name under bar
        p.setFont(name_font)
        p.setPen(QColor(ink.fg if ink is _PRINT_INK else theme.COLORS["log_fg"]))
        p.drawText(
            QRectF(x, col_bottom - name_h, col_w, name_h),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            name,
        )

    if draw_footer:
        _draw_footer(p, rect, page, ink=ink)
    return panel.right()


def _draw_notes_block(
    p: QPainter,
    rect: QRectF,
    *,
    top: float,
    bottom: float,
    width: float | None = None,
    text: str = "",
    ink: _Ink = _SCREEN_INK,
    layout_scale: float = 1.0,
) -> None:
    """NOTES panel filling ``[top, bottom]`` × width (title inside, like Genre/Keys)."""
    s = max(0.5, float(layout_scale))
    margin = 24.0 * s
    x = rect.left() + margin
    box_w = (
        float(width)
        if width is not None
        else max(40.0 * s, rect.width() - 2 * margin)
    )
    box = QRectF(x, top, box_w, max(48.0 * s, bottom - top))
    _paint_card(p, box, ink)

    label_font = QFont(theme.FONT_FAMILY, _NOTES_LABEL_PT)
    label_font.setBold(True)
    p.setFont(label_font)
    p.setPen(QColor(ink.dim))
    title_h = 28.0 * s
    title_dy = _EMBEDDED_TITLE_DY * s
    p.drawText(
        QRectF(
            box.left() + 16.0 * s,
            box.top() + title_dy,
            box.width() - 32.0 * s,
            title_h,
        ),
        Qt.AlignLeft | Qt.AlignVCenter,
        "NOTES:",
    )
    body = (text or "").strip()
    if not body:
        return
    pad = 16.0 * s
    text_rect = QRectF(
        box.left() + pad,
        box.top() + title_dy + title_h,
        box.width() - 2 * pad,
        max(24.0 * s, box.height() - title_dy - title_h - pad),
    )
    body_font = QFont(theme.FONT_FAMILY, _NOTES_BODY_PT)
    p.setFont(body_font)
    body_color = ink.fg if ink is _PRINT_INK else theme.COLORS["log_fg"]
    p.setPen(QColor(body_color))
    p.drawText(
        text_rect,
        int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap),
        body,
    )


def _paint_notes_page(
    p: QPainter,
    rect: QRectF,
    *,
    page: int,
    stamp: str,
    notes: str = "",
) -> None:
    """Full-page NOTES card (PDF)."""
    _fill_page(p, rect)
    title_font = QFont(theme.FONT_FAMILY, 16)
    title_font.setBold(True)
    p.setFont(title_font)
    p.setPen(QColor(theme.COLORS["fg"]))
    p.drawText(
        QRectF(rect.left() + 24, rect.top() + 28, rect.width() - 48, 28),
        Qt.AlignLeft | Qt.AlignVCenter,
        "Notes",
    )
    stamp_font = QFont(theme.FONT_FAMILY, 9)
    p.setFont(stamp_font)
    p.setPen(QColor(_DIM))
    p.drawText(
        QRectF(rect.left() + 24, rect.top() + 56, rect.width() - 48, 18),
        Qt.AlignLeft | Qt.AlignVCenter,
        stamp,
    )
    _draw_notes_block(
        p,
        rect,
        top=rect.top() + 72.0,
        bottom=rect.bottom() - 36.0,
        text=notes,
    )
    _draw_footer(p, rect, page)


def _paint_summary_page(
    p: QPainter,
    rect: QRectF,
    stats: OverviewStats,
    paths: PathsMeta,
    *,
    page: int | None,
    stamp: str,
    chart_grid: Sequence[tuple[str, Optional[QChart]]] | None = None,
    include_header: bool = True,
    draw_footer: bool = True,
) -> tuple[float, float, float]:
    """Page 1: optional title/paths, KPIs, then a 2×3 chart grid below.

    Returns ``(charts_y, slot_h, gap_y)`` for the chart grid so callers
    (PNG spread) can align the Genre card to the first two rows.
    """
    _fill_page(p, rect)
    if include_header:
        y = _draw_title_block(p, rect, demo=stats.demo, stamp=stamp)
        y = _draw_paths(p, rect, y, paths)
    else:
        y = rect.top() + 20.0
    y = _draw_kpi_strip(p, rect, y, stats)

    margin = 20.0
    gap_x = 10.0
    gap_y = 10.0
    footer_pad = 32.0
    cols, rows = 2, 3
    avail_h = rect.bottom() - footer_pad - y
    avail_w = rect.width() - 2 * margin
    slot_w = (avail_w - gap_x) / cols
    slot_h = (avail_h - gap_y * (rows - 1)) / rows
    charts_y = y

    if not chart_grid:
        if draw_footer:
            _draw_footer(p, rect, page)
        return charts_y, slot_h, gap_y

    for i, (label, chart) in enumerate(list(chart_grid)[: cols * rows]):
        c = i % cols
        r = i // cols
        slot = QRectF(
            rect.left() + margin + c * (slot_w + gap_x),
            y + r * (slot_h + gap_y),
            slot_w,
            slot_h,
        )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(_PANEL))
        p.drawRoundedRect(slot, 8, 8)
        target = QRectF(
            slot.left() + 3,
            slot.top() + 3,
            slot.width() - 6,
            slot.height() - 6,
        )
        if chart is None:
            p.setPen(QColor(_DIM))
            p.setFont(QFont(theme.FONT_FAMILY, 9))
            p.drawText(slot, Qt.AlignCenter, f"{label}\n(no data)")
            continue
        render_chart_to_painter(p, chart, target)
    if draw_footer:
        _draw_footer(p, rect, page)
    return charts_y, slot_h, gap_y


def _paint_chart_pair_page(
    p: QPainter,
    rect: QRectF,
    charts: Sequence[Optional[QChart]],
    titles_fallback: Sequence[str],
    *,
    page: int,
    stamp: str,
    demo: bool,
) -> None:
    _fill_page(p, rect)
    y = _draw_title_block(p, rect, demo=demo, stamp=stamp)
    margin = 24.0
    gap = 12.0
    avail_h = rect.bottom() - 36 - y
    present = [c for c in charts if c is not None]
    # One chart → full height; two → stacked halves.
    if len(present) <= 1 and charts and charts[0] is not None:
        n_slots = 1
        draw_list: list[Optional[QChart]] = [charts[0]]
        fb = [titles_fallback[0] if titles_fallback else ""]
    else:
        draw_list = list(charts[:2])
        while len(draw_list) < 2:
            draw_list.append(None)
        fb = list(titles_fallback[:2])
        while len(fb) < 2:
            fb.append("")
        n_slots = 2

    slot_h = (avail_h - gap * (n_slots - 1)) / n_slots
    slot_w = rect.width() - 2 * margin

    for i, chart in enumerate(draw_list[:n_slots]):
        slot = QRectF(
            rect.left() + margin,
            y + i * (slot_h + gap),
            slot_w,
            slot_h,
        )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(_PANEL))
        p.drawRoundedRect(slot, 8, 8)
        if chart is None:
            p.setPen(QColor(_DIM))
            p.setFont(QFont(theme.FONT_FAMILY, 10))
            label = fb[i] if i < len(fb) else "No data"
            p.drawText(slot, Qt.AlignCenter, f"{label}\n(no data)")
            continue
        target = QRectF(
            slot.left() + 4,
            slot.top() + 4,
            slot.width() - 8,
            slot.height() - 8,
        )
        render_chart_to_painter(p, chart, target)
    _draw_footer(p, rect, page)


def _set_pdf_page(printer: QPrinter, orientation: QPageLayout.Orientation) -> QRectF:
    """Switch orientation with small printable margins; return paint rect at origin."""
    printer.setPageLayout(
        QPageLayout(
            QPageSize(QPageSize.A4),
            orientation,
            QMarginsF(8, 8, 8, 8),
            QPageLayout.Millimeter,
        )
    )
    r = QRectF(printer.pageRect(QPrinter.DevicePixel))
    return QRectF(0, 0, r.width(), r.height())


def _page1_chart_grid(
    charts: dict[str, Optional[QChart]],
) -> list[tuple[str, Optional[QChart]]]:
    """2×3 order: SDR | Roles / Gender | Reverb / Vocal type | Compression."""
    return [
        ("SI-SDR", charts.get("sdr")),
        ("Roles", charts.get("roles")),
        ("Gender", charts.get("gender")),
        ("Reverb", charts.get("reverb")),
        ("Vocal type", charts.get("vocal")),
        ("Compression", charts.get("compression")),
    ]


def export_overview_report(
    path: str | Path,
    stats: OverviewStats,
    paths: PathsMeta,
    *,
    notes: str = "",
) -> None:
    """Write a Charts report as PNG (screen) or PDF (print).

    Both use the same 4×3 landscape spread (KPIs + charts | Genre / Keys /
    NOTES). PDF is one A4 landscape page with white paper, no card fills,
    and black text. PNG keeps the dark UI theme.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    charts = build_export_charts(stats)
    grid = _page1_chart_grid(charts)

    if suffix == ".png":
        _export_png_spread(
            path, stats, paths, stamp=stamp, chart_grid=grid, notes=notes
        )
        return

    if suffix != ".pdf":
        raise ValueError("Export path must end with .pdf or .png")

    _export_pdf_spread(
        path, stats, stamp=stamp, chart_grid=grid, notes=notes
    )


def _export_pdf_spread(
    path: Path,
    stats: OverviewStats,
    *,
    stamp: str,
    chart_grid: Sequence[tuple[str, Optional[QChart]]],
    notes: str = "",
) -> None:
    """One A4 landscape PDF page — same layout as PNG, print ink.

    Paint at a fixed pixel size first (QtCharts text metrics are reliable
    there), then scale the image onto the PDF page. Avoids HighResolution
    device-pixel layout collapsing gutters / scrambling callouts.
    """
    # ~200 dpi on A4 landscape printable area — sharp enough to print.
    width = 2340
    height = 1654  # ≈ 297:210

    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(_PRINT_INK.bg))
    ip = QPainter(img)
    ip.setRenderHint(QPainter.Antialiasing)
    ip.setRenderHint(QPainter.TextAntialiasing)
    _paint_charts_spread(
        ip,
        QRectF(0, 0, width, height),
        stats,
        chart_grid=chart_grid,
        stamp=stamp,
        notes=notes,
        ink=_PRINT_INK,
    )
    ip.end()

    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(str(path))
    page_rect = _set_pdf_page(printer, QPageLayout.Landscape)

    painter = QPainter(printer)
    if not painter.isActive():
        raise OSError(f"Could not open PDF for writing: {path}")
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.fillRect(page_rect, QColor(_PRINT_INK.bg))

    # Letterbox into the printable area, preserve aspect.
    sx = page_rect.width() / width
    sy = page_rect.height() / height
    scale = min(sx, sy)
    tw = width * scale
    th = height * scale
    target = QRectF(
        page_rect.left() + (page_rect.width() - tw) / 2.0,
        page_rect.top() + (page_rect.height() - th) / 2.0,
        tw,
        th,
    )
    painter.drawImage(target, img)
    painter.end()


def _paint_charts_spread(
    p: QPainter,
    full: QRectF,
    stats: OverviewStats,
    *,
    chart_grid: Sequence[tuple[str, Optional[QChart]]],
    stamp: str,
    notes: str = "",
    ink: _Ink = _SCREEN_INK,
    layout_scale: float = 1.0,
) -> None:
    """4-col × 3-row Charts spread into ``full`` (PNG image or PDF raster)."""
    del layout_scale  # kept for call-site compat; pixel canvases use scale 1
    margin = 24.0
    gap = 12.0
    footer_pad = 40.0
    grid_w = full.width() - 2 * margin
    slot_w = (grid_w - 3 * gap) / 4.0

    _fill_page(p, full, ink=ink)

    p.setPen(QColor(ink.dim))
    p.setFont(QFont(theme.FONT_FAMILY, 9))
    p.drawText(
        QRectF(full.left() + margin, full.top() + 10, grid_w, 16),
        Qt.AlignRight | Qt.AlignVCenter,
        f"Charts  ·  {stamp}",
    )
    y = _draw_kpi_strip(p, full, full.top() + 32.0, stats, ink=ink)

    rows = 3
    avail_h = full.bottom() - footer_pad - y
    slot_h = (avail_h - gap * (rows - 1)) / rows
    charts_y = y

    def _cell(c: int, r: int, cspan: int = 1) -> QRectF:
        return QRectF(
            full.left() + margin + c * (slot_w + gap),
            charts_y + r * (slot_h + gap),
            cspan * slot_w + (cspan - 1) * gap,
            slot_h,
        )

    for i, (label, chart) in enumerate(list(chart_grid)[:6]):
        slot = _cell(i % 2, i // 2)
        _paint_card(p, slot, ink)
        target = QRectF(
            slot.left() + 4,
            slot.top() + 6,
            slot.width() - 8,
            slot.height() - 10,
        )
        if chart is None:
            p.setPen(QColor(ink.dim))
            p.setFont(QFont(theme.FONT_FAMILY, 9))
            p.drawText(slot, Qt.AlignCenter, f"{label}\n(no data)")
            continue
        render_chart_to_painter(p, chart, target, ink=ink)

    span = _cell(2, 0, cspan=2)
    host = QRectF(
        span.left() - 24.0,
        full.top(),
        span.width() + 48.0,
        full.height(),
    )
    _paint_genre_bars_page(
        p,
        host,
        stats,
        page=1,
        stamp="",
        panel_top=span.top(),
        panel_bottom=span.bottom(),
        draw_footer=False,
        ink=ink,
    )
    keys = _cell(2, 1, cspan=2)
    _paint_key_bars_page(
        p,
        host,
        stats,
        page=1,
        stamp="",
        panel_top=keys.top(),
        panel_bottom=keys.bottom(),
        draw_footer=False,
        ink=ink,
    )
    notes_cell = _cell(2, 2, cspan=2)
    _draw_notes_block(
        p,
        host,
        top=notes_cell.top(),
        bottom=notes_cell.bottom(),
        width=notes_cell.width(),
        text=notes,
        ink=ink,
    )
    _draw_footer(p, full, page=None, ink=ink)


def _export_png_spread(
    path: Path,
    stats: OverviewStats,
    paths: PathsMeta,
    *,
    stamp: str,
    chart_grid: Sequence[tuple[str, Optional[QChart]]],
    notes: str = "",
) -> None:
    """PNG spread: dark-theme 4×3 grid (same layout as print PDF)."""
    del paths  # reserved for future path strip on PNG
    height = 1754
    margin = 20.0
    gap = 10.0
    span2_w = 980.0
    slot_w = (span2_w - gap) / 2.0
    grid_w = 4 * slot_w + 3 * gap
    width = int(round(grid_w + 2 * margin))

    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(_SCREEN_INK.bg))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    _paint_charts_spread(
        painter,
        QRectF(0, 0, width, height),
        stats,
        chart_grid=chart_grid,
        stamp=stamp,
        notes=notes,
        ink=_SCREEN_INK,
    )
    painter.end()
    if not img.save(str(path), "PNG"):
        raise OSError(f"Could not write PNG: {path}")

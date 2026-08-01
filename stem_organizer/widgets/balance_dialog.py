"""Balance options dialog — features + copy/move/csv mode + destination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CheckBox, RadioButton, StrongBodyLabel

from .. import theme
from ..dataset.balance import BALANCE_FEATURES, BALANCE_OPTIONAL_FEATURES
from .action_button import action_button
from .dialogs import dim_behind, style_help_label, HELP_BODY_PX, HELP_HEADING_PX, HELP_SECTION_TITLE_PX
from .path_row import PathRow


@dataclass(frozen=True)
class BalanceDialogResult:
    mode: str  # copy | move | csv
    features: tuple[str, ...]
    dest: str


@dataclass(frozen=True)
class BalanceConfirmStats:
    features_label: str
    mode_label: str
    categories: int
    per_category: int
    total: int
    pairs: int
    instrumental: int
    vocal: int
    samples: int
    dest: str


def _kv_label(text: str, *, dim: bool = True) -> QLabel:
    t = theme.DARK
    lbl = QLabel(text)
    style_help_label(
        lbl,
        HELP_BODY_PX,
        t["text_dim"] if dim else t["text"],
        bold=not dim,
    )
    return lbl


def _section_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    """Inset card with section title — panel2 fill, soft border."""
    t = theme.DARK
    card = QFrame()
    card.setObjectName("BalanceConfirmCard")
    card.setStyleSheet(
        f"""
        QFrame#BalanceConfirmCard {{
            background-color: {t['panel_2']};
            border: 1px solid {t['border_soft']};
            border-radius: 8px;
        }}
        """
    )
    outer = QVBoxLayout(card)
    outer.setContentsMargins(14, 12, 14, 12)
    outer.setSpacing(10)
    head = QLabel(title.upper())
    style_help_label(head, HELP_SECTION_TITLE_PX, t["accent_hover"], bold=True)
    outer.addWidget(head)
    return card, outer


def ask_balance_confirm(parent: QWidget, stats: BalanceConfirmStats) -> bool:
    """Wide confirm card: meta + selection grids, themed path, Start / Cancel."""
    host = parent.window() if parent is not None else parent
    t = theme.DARK
    r = theme.DIALOG_CORNER_RADIUS

    dlg = QDialog(host)
    dlg.setWindowTitle("Balance")
    dlg.setModal(True)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(14, 14, 14, 14)

    shell = QFrame()
    shell.setObjectName("BalanceConfirmShell")
    shell.setStyleSheet(
        f"""
        QFrame#BalanceConfirmShell {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(shell)
    lay.setContentsMargins(24, 20, 24, 18)
    lay.setSpacing(14)

    title = StrongBodyLabel("Balance", shell)
    style_help_label(title, HELP_HEADING_PX, t["text"], bold=True)
    lay.addWidget(title)

    # —— Plan ——
    plan_card, plan_body = _section_card("Plan")
    plan_grid = QGridLayout()
    plan_grid.setContentsMargins(0, 0, 0, 0)
    plan_grid.setHorizontalSpacing(20)
    plan_grid.setVerticalSpacing(8)
    plan_grid.setColumnStretch(1, 1)

    def _add_plan_row(row: int, key: str, value: str) -> None:
        k = _kv_label(key)
        v = _kv_label(value, dim=False)
        plan_grid.addWidget(k, row, 0, Qt.AlignLeft | Qt.AlignVCenter)
        plan_grid.addWidget(v, row, 1, Qt.AlignLeft | Qt.AlignVCenter)

    _add_plan_row(0, "Balance on", stats.features_label)
    _add_plan_row(1, "Mode", stats.mode_label)
    _add_plan_row(
        2,
        "Scale",
        f"{stats.categories:,} categories  ·  "
        f"{stats.per_category:,} each  ·  "
        f"{stats.total:,} total",
    )
    plan_body.addLayout(plan_grid)
    lay.addWidget(plan_card)

    # —— Selection ——
    sel_card, sel_body = _section_card("Selection")
    sel_grid = QGridLayout()
    sel_grid.setContentsMargins(0, 0, 0, 0)
    sel_grid.setHorizontalSpacing(28)
    sel_grid.setVerticalSpacing(8)
    sel_grid.setColumnStretch(0, 1)
    sel_grid.setColumnStretch(2, 1)
    for col in (1, 3):
        sel_grid.setColumnMinimumWidth(col, 48)

    def _add_count(row: int, col: int, key: str, n: int) -> None:
        sel_grid.addWidget(
            _kv_label(key),
            row,
            col,
            Qt.AlignLeft | Qt.AlignVCenter,
        )
        num = QLabel(f"{n:,}")
        style_help_label(num, HELP_BODY_PX, t["text"], bold=True)
        num.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sel_grid.addWidget(num, row, col + 1, Qt.AlignRight | Qt.AlignVCenter)

    _add_count(0, 0, "Pairs", stats.pairs)
    _add_count(0, 2, "Instrumental", stats.instrumental)
    _add_count(1, 0, "Vocal", stats.vocal)
    _add_count(1, 2, "Samples", stats.samples)
    sel_body.addLayout(sel_grid)
    lay.addWidget(sel_card)

    # —— Destination ——
    dest_card, dest_body = _section_card("Destination")
    try:
        dest_display = str(Path(stats.dest).expanduser().resolve(strict=False))
    except OSError:
        dest_display = stats.dest
    dest_link = QLabel()
    dest_link.setObjectName("BalanceDestLink")
    dest_link.setWordWrap(True)
    dest_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    dest_link.setOpenExternalLinks(False)
    dest_link.setCursor(Qt.PointingHandCursor)
    url = QUrl.fromLocalFile(dest_display).toString()
    dest_link.setText(
        f'<a href="{url}" style="color: {t["log_fg"]}; text-decoration: none;">'
        f"{dest_display}</a>"
    )
    dest_link.setStyleSheet(
        f"""
        QLabel#BalanceDestLink {{
            background: transparent;
            color: {t["log_fg"]};
            font-family: "{theme.FONT_FAMILY_MONO}";
            font-size: {theme.LOG_FONT_PX}px;
            padding: 2px 0;
        }}
        """
    )
    dest_link.setToolTip("Open in File Explorer")

    def _open_dest(_href: str = "") -> None:
        target = Path(dest_display)
        if target.is_dir() or target.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        elif target.parent.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    dest_link.linkActivated.connect(_open_dest)
    dest_body.addWidget(dest_link)
    lay.addWidget(dest_card)

    # —— Footer ——
    footer = QHBoxLayout()
    footer.setContentsMargins(0, 4, 0, 0)
    footer.setSpacing(10)
    footer.addStretch(1)
    cancel_btn = action_button(
        "Cancel", on_click=dlg.reject, accent=False, parent=shell
    )
    cancel_btn.setMinimumWidth(88)
    start_btn = action_button(
        "Start", on_click=dlg.accept, accent=True, parent=shell
    )
    start_btn.setMinimumWidth(88)
    footer.addWidget(cancel_btn)
    footer.addWidget(start_btn)
    lay.addLayout(footer)

    outer.addWidget(shell)
    shell.setMinimumWidth(560)
    dlg.adjustSize()
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(parent if parent is not None else host):
        return dlg.exec() == QDialog.DialogCode.Accepted


def ask_balance_finished(parent: QWidget, dest: str) -> bool:
    """Same chrome as confirm: title, status card, destination path, Open / Close."""
    host = parent.window() if parent is not None else parent
    t = theme.DARK
    r = theme.DIALOG_CORNER_RADIUS

    dlg = QDialog(host)
    dlg.setWindowTitle("Balance")
    dlg.setModal(True)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(14, 14, 14, 14)

    shell = QFrame()
    shell.setObjectName("BalanceFinishedShell")
    shell.setStyleSheet(
        f"""
        QFrame#BalanceFinishedShell {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(shell)
    lay.setContentsMargins(24, 20, 24, 18)
    lay.setSpacing(14)

    title = StrongBodyLabel("Balance", shell)
    style_help_label(title, HELP_HEADING_PX, t["text"], bold=True)
    lay.addWidget(title)

    status_card, status_body = _section_card("Status")
    status_lbl = QLabel("Balance finished.")
    style_help_label(status_lbl, HELP_BODY_PX, t["text"], bold=False)
    status_body.addWidget(status_lbl)
    prompt_lbl = QLabel("Open the destination folder in File Explorer?")
    style_help_label(prompt_lbl, HELP_BODY_PX, t["text_dim"], bold=False)
    status_body.addWidget(prompt_lbl)
    lay.addWidget(status_card)

    dest_card, dest_body = _section_card("Destination")
    try:
        dest_display = str(Path(dest).expanduser().resolve(strict=False))
    except OSError:
        dest_display = dest
    dest_link = QLabel()
    dest_link.setObjectName("BalanceDestLink")
    dest_link.setWordWrap(True)
    dest_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
    dest_link.setOpenExternalLinks(False)
    dest_link.setCursor(Qt.PointingHandCursor)
    url = QUrl.fromLocalFile(dest_display).toString()
    dest_link.setText(
        f'<a href="{url}" style="color: {t["log_fg"]}; text-decoration: none;">'
        f"{dest_display}</a>"
    )
    dest_link.setStyleSheet(
        f"""
        QLabel#BalanceDestLink {{
            background: transparent;
            color: {t["log_fg"]};
            font-family: "{theme.FONT_FAMILY_MONO}";
            font-size: {theme.LOG_FONT_PX}px;
            padding: 2px 0;
        }}
        """
    )
    dest_link.setToolTip("Open in File Explorer")

    def _open_dest(_href: str = "") -> None:
        target = Path(dest_display)
        if target.is_dir() or target.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        elif target.parent.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    dest_link.linkActivated.connect(_open_dest)
    dest_body.addWidget(dest_link)
    lay.addWidget(dest_card)

    footer = QHBoxLayout()
    footer.setContentsMargins(0, 4, 0, 0)
    footer.setSpacing(10)
    footer.addStretch(1)
    close_btn = action_button(
        "Close", on_click=dlg.reject, accent=False, parent=shell
    )
    close_btn.setMinimumWidth(88)
    open_btn = action_button(
        "Open folder", on_click=dlg.accept, accent=True, parent=shell
    )
    open_btn.setMinimumWidth(110)
    footer.addWidget(close_btn)
    footer.addWidget(open_btn)
    lay.addLayout(footer)

    outer.addWidget(shell)
    shell.setMinimumWidth(520)
    dlg.adjustSize()
    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with dim_behind(parent if parent is not None else host):
        return dlg.exec() == QDialog.DialogCode.Accepted


def ask_balance_options(
    parent: QWidget,
    *,
    default_dest: str = "",
) -> Optional[BalanceDialogResult]:
    """Return mode + features + destination folder, or None if cancelled."""
    host = parent.window() if parent is not None else parent
    t = theme.DARK
    r = theme.DIALOG_CORNER_RADIUS

    dlg = QDialog(host)
    dlg.setWindowTitle("Balance library")
    dlg.setModal(True)
    dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
    dlg.setAttribute(Qt.WA_TranslucentBackground)
    dlg.setMinimumWidth(560)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(14, 14, 14, 14)

    shell = QFrame()
    shell.setObjectName("BalanceOptionsShell")
    shell.setStyleSheet(
        f"""
        QFrame#BalanceOptionsShell {{
            background-color: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
        }}
        """
    )
    lay = QVBoxLayout(shell)
    lay.setContentsMargins(24, 20, 24, 18)
    lay.setSpacing(14)

    title = StrongBodyLabel("Balance", shell)
    style_help_label(title, HELP_HEADING_PX, t["text"], bold=True)
    lay.addWidget(title)

    plan_card, plan_body = _section_card("Plan")
    plan_line1 = (
        "Roles (Instrumental / Vocal / Pairs / Samples) are always included."
    )
    plan_line2 = "Enable extra features to equalize joint categories."
    plan_line3 = (
        "Each combo keeps as many units as the rarest bucket (random pick)."
    )
    plan_lbl1 = _kv_label(plan_line1, dim=True)
    plan_lbl1.setWordWrap(True)
    plan_lbl1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    plan_lbl2 = _kv_label(plan_line2, dim=True)
    plan_lbl2.setWordWrap(True)
    plan_lbl2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    plan_lbl3 = _kv_label(plan_line3, dim=True)
    plan_lbl3.setWordWrap(True)
    plan_lbl3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    plan_body.addWidget(plan_lbl1)
    plan_body.addWidget(plan_lbl2)
    plan_body.addWidget(plan_lbl3)
    lay.addWidget(plan_card)

    feat_card, feat_body = _section_card("Also balance on")
    feat_grid = QGridLayout()
    feat_grid.setContentsMargins(0, 0, 0, 0)
    feat_grid.setHorizontalSpacing(16)
    feat_grid.setVerticalSpacing(6)
    feature_cbs: dict[str, CheckBox] = {}
    default_on = {"gender"}
    for i, (key, label) in enumerate(BALANCE_OPTIONAL_FEATURES):
        cb = CheckBox(label)
        cb.setCursor(Qt.PointingHandCursor)
        cb.setChecked(key in default_on)
        feature_cbs[key] = cb
        feat_grid.addWidget(cb, i // 2, i % 2)
    feat_body.addLayout(feat_grid)
    feat_hint = QLabel("Optional — leave all off to balance Roles only.")
    feat_hint.setStyleSheet(f"color: {t['fg_dim']}; font-size: 12px;")
    feat_body.addWidget(feat_hint)
    lay.addWidget(feat_card)

    mode_card, mode_body = _section_card("Mode")
    radios: dict[str, RadioButton] = {}
    for key, label in (
        ("copy", "Copy into {root}_BALANCED folders"),
        ("move", "Move into {root}_BALANCED folders"),
        ("csv", "CSV list only (no file copy/move)"),
    ):
        rb = RadioButton(label)
        rb.setCursor(Qt.PointingHandCursor)
        radios[key] = rb
        mode_body.addWidget(rb)
    radios["copy"].setChecked(True)
    lay.addWidget(mode_card)

    dest_card, dest_body = _section_card("Destination")
    dest_host = QWidget()
    dest_host_lay = QVBoxLayout(dest_host)
    dest_host_lay.setContentsMargins(0, 0, 0, 0)
    dest_host_lay.setSpacing(0)
    dest_row = PathRow(
        dest_host,
        "Folder",
        tip_text=(
            "Folder or drive where balanced roots are created "
            "(OriginalName_BALANCED). Browse opens File Explorer."
        ),
        label_width=56,
        caption="Choose folder or drive for balanced library",
    )
    if default_dest:
        dest_row.set_text(default_dest)
    dest_host_lay.addWidget(dest_row)
    dest_body.addWidget(dest_host)
    dest_note = QLabel(
        "Folder is created if missing. Roots are written under it as "
        "OriginalName_BALANCED."
    )
    dest_note.setWordWrap(True)
    dest_note.setStyleSheet(f"color: {t['fg_dim']}; font-size: 12px;")
    dest_body.addWidget(dest_note)
    lay.addWidget(dest_card)

    row = QHBoxLayout()
    row.addStretch(1)
    cancel_btn = action_button("Cancel", on_click=dlg.reject, parent=shell)
    ok_btn = action_button("Continue…", on_click=dlg.accept, accent=True, parent=shell)
    row.addWidget(cancel_btn)
    row.addWidget(ok_btn)
    lay.addLayout(row)

    def _dest_ok(text: str) -> bool:
        """Existing dir, or a path we can create (parent / drive exists)."""
        raw = (text or "").strip().strip('"')
        if not raw:
            return False
        path = Path(raw)
        if path.is_dir():
            return True
        if path.is_file():
            return False
        parent = path.parent
        try:
            if parent == path:
                return False
            return parent.is_dir() or (len(parent.parts) == 1 and parent.exists())
        except OSError:
            return False

    def _sync_ok() -> None:
        ok_btn.setEnabled(_dest_ok(dest_row.text()))

    for cb in feature_cbs.values():
        cb.toggled.connect(lambda _=False: _sync_ok())
    dest_row.entry.textChanged.connect(lambda _=None: _sync_ok())
    _sync_ok()

    outer.addWidget(shell)
    shell.setMinimumWidth(560)
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
            return None

    features = ("roles",) + tuple(
        k for k, _ in BALANCE_OPTIONAL_FEATURES if feature_cbs[k].isChecked()
    )
    dest = dest_row.text().strip().strip('"')
    if not _dest_ok(dest):
        return None

    mode = "copy"
    for key, rb in radios.items():
        if rb.isChecked():
            mode = key
            break
    return BalanceDialogResult(mode=mode, features=features, dest=dest)

"""Action button factory — Fluent PushButton / PrimaryPushButton variants."""
from __future__ import annotations

from typing import Callable, Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy, QWidget
from qfluentwidgets import PrimaryPushButton, PushButton, setCustomStyleSheet

from .. import theme

FluentButton = Union[PushButton, PrimaryPushButton]

# Soft CTk text on accent (Fluent paints PrimaryPushButton labels dark otherwise).
_SOFT = theme.COLORS["log_fg"]
_PRIMARY_LABEL_SHEET = f"""
PrimaryPushButton {{
    color: {_SOFT};
}}
PrimaryPushButton:hover {{
    color: {_SOFT};
}}
PrimaryPushButton:pressed {{
    color: {_SOFT};
}}
PrimaryPushButton:focus {{
    color: {_SOFT};
}}
"""


def action_button(
    text: str,
    *,
    on_click: Optional[Callable[[], None]] = None,
    parent: Optional[QWidget] = None,
    accent: bool = False,
    danger: bool = False,
    width: Optional[int] = None,
    height: int = theme.ACTION_BTN_HEIGHT,
    tip: str = "",
) -> FluentButton:
    """Build a Fluent action button (primary / normal / danger)."""
    if accent:
        btn: FluentButton = PrimaryPushButton(text, parent)
        # Dialogs (help Close, preset OK, …) are not walked by style_fluent_subtree.
        setCustomStyleSheet(btn, "", _PRIMARY_LABEL_SHEET)
    else:
        btn = PushButton(text, parent)
        if danger:
            btn.setStyleSheet(
                f"PushButton {{ color: {theme.DARK['danger']}; }}"
                f"PushButton:hover {{ background-color: {theme.DARK['danger']}; "
                f"color: {theme.COLORS['log_fg']}; }}"
            )
    btn.setCursor(Qt.PointingHandCursor)
    # Prevent Enter/Return after dialogs from activating Stop / other actions.
    if hasattr(btn, "setAutoDefault"):
        btn.setAutoDefault(False)
    if hasattr(btn, "setDefault"):
        btn.setDefault(False)
    # Content-sized like Classify action bar — Fixed so Minimum policy cannot stretch.
    btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    btn.setFixedHeight(height)
    btn.setFont(theme.F_ACTION)
    if width is not None:
        btn.setFixedWidth(width)
    if on_click is not None:
        btn.clicked.connect(on_click)
    if tip:
        btn.setToolTip(theme.format_tooltip(tip))
    return btn

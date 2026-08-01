"""Dim hint labels for option text in parentheses (Genre & Gender style).

'Batch (fast)' → radio 'Batch' + hint 'fast' (dimmer, same body size).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from .. import theme

_HINT_FONT_PX = theme.BODY_FONT_PX


def split_option_label(text: str) -> tuple[str, str | None]:
    """'Batch (fast)' → ('Batch', 'fast'); no paren → (text, None)."""
    paren = text.find("(")
    if paren > 0:
        hint = text[paren:].strip().removeprefix("(").removesuffix(")").strip()
        return text[:paren].rstrip(), hint or None
    return text.strip(), None


def style_option_hint(lbl: QLabel) -> None:
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


def style_how_it_works(lbl: QLabel) -> None:
    """Same dim text as RadioHint / AudioTester-style paren labels."""
    dim = theme.DARK["text_dim"]
    lbl.setObjectName("HowItWorks")
    if hasattr(lbl, "setTextColor"):
        lbl.setTextColor(dim, dim)
    lbl.setStyleSheet(
        f"""
        QLabel#HowItWorks {{
            color: {dim};
            font-family: "{theme.FONT_FAMILY}";
            font-size: {_HINT_FONT_PX}px;
            background: transparent;
        }}
        """
    )


def option_with_hint(
    control: QWidget,
    full_label: str,
    *,
    tip: str = "",
    on_hint_click=None,
) -> QWidget:
    """Wrap a RadioButton/CheckBox: main label on control, paren text as dim hint."""
    main, hint = split_option_label(full_label)
    if hasattr(control, "setText"):
        control.setText(main)
    if tip:
        control.setToolTip(tip)
    if hasattr(control, "setCursor"):
        control.setCursor(Qt.PointingHandCursor)

    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    control.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    lay.addWidget(control, 0, Qt.AlignVCenter)

    if hint:
        hint_lbl = QLabel(hint)
        hint_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        style_option_hint(hint_lbl)
        hint_lbl.setCursor(Qt.PointingHandCursor)
        if tip:
            hint_lbl.setToolTip(tip)
        if on_hint_click is not None:
            hint_lbl.mousePressEvent = (  # type: ignore[method-assign]
                lambda _e: on_hint_click()
            )
        lay.addWidget(hint_lbl, 0, Qt.AlignVCenter)

    lay.addStretch(1)
    return row

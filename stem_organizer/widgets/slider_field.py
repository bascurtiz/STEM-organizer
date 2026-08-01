"""Slider field — labeled Fluent Slider + readout / SpinBox."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget
from qfluentwidgets import BodyLabel, Slider, SpinBox

from .. import theme


class SliderField(QWidget):
    """Labeled horizontal slider with live readout.

    Value range is integer percent (0..100) — matching CTkSlider usage in the
    original. The readout label is updated via ``format_value`` callback.
    """

    valueChanged = Signal(int)

    def __init__(
        self,
        parent: QWidget,
        label: str,
        *,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        format_value: Optional[Callable[[int], str]] = None,
        spinbox: bool = False,
        label_width: int = 78,
        readout_width: int = 48,
    ) -> None:
        super().__init__(parent)
        self._format = format_value
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        self._lbl = BodyLabel(label)
        self._lbl.setFixedWidth(label_width)
        layout.addWidget(self._lbl)

        self.slider = Slider(Qt.Horizontal)
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setValue(value)
        self.slider.setFixedHeight(20)
        self.slider.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.slider, stretch=1)

        if spinbox:
            self._spin = SpinBox()
            self._spin.setRange(minimum, maximum)
            self._spin.setValue(value)
            self._spin.setFixedWidth(64)
            self._spin.setFixedHeight(26)
            self.slider.valueChanged.connect(self._spin.setValue)
            self._spin.valueChanged.connect(self.slider.setValue)
            layout.addWidget(self._spin)
            self._readout: Optional[BodyLabel] = None
        else:
            self._spin = None
            self._readout = BodyLabel("")
            self._readout.setFixedWidth(readout_width)
            self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(self._readout)
            self._update_readout(value)

        self.slider.valueChanged.connect(self._on_changed)

    def setToolTip(self, tip: str) -> None:  # noqa: N802 Qt naming
        """Propagate tip to label + slider (+ readout/spin) — hover hits children."""
        tip = theme.format_tooltip(tip)
        super().setToolTip(tip)
        self._lbl.setToolTip(tip)
        self.slider.setToolTip(tip)
        if self._spin is not None:
            self._spin.setToolTip(tip)
        if self._readout is not None:
            self._readout.setToolTip(tip)

    def _on_changed(self, value: int) -> None:
        if self._readout is not None:
            self._update_readout(value)
        self.valueChanged.emit(value)

    def _update_readout(self, value: int) -> None:
        if self._readout is None:
            return
        if self._format is not None:
            self._readout.setText(self._format(value))
        else:
            self._readout.setText(str(value))

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:  # noqa: N802 Qt naming
        self.slider.setValue(int(value))

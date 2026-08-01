"""Shared action bar — a QStackedWidget indexed by tab name.

Each tab calls ``attach_action_bar(name, page)`` to register its own row of
buttons. On tab change the main window calls ``show_action_bar(name)``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QWidget,
)

from .. import theme


class ActionBarPage(QWidget):
    """One button row inside the action bar."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ActionBarPage")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(theme.PAGE_CONTENT_INSET, 6, theme.PAGE_CONTENT_INSET, 6)
        self._layout.setSpacing(theme.ACTION_BTN_GAP)

    def add_button(self, button: QWidget, *, stretch_after: bool = False) -> None:
        self._layout.addWidget(button)
        if stretch_after:
            self._layout.addStretch(1)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)

    def add_widget(self, w: QWidget) -> None:
        self._layout.addWidget(w)


class ActionBarStack(QFrame):
    """Stacked action-bar pages keyed by tab name."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ActionBar")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._stack = QStackedWidget()
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)
        self._pages: dict[str, ActionBarPage] = {}
        # Placeholder so the stack is never empty
        self.add_page("__placeholder__", ActionBarPage())

    def add_page(self, name: str, page: ActionBarPage) -> None:
        self._pages[name] = page
        self._stack.addWidget(page)

    def make_page(self, name: str) -> ActionBarPage:
        page = ActionBarPage()
        self.add_page(name, page)
        return page

    def page(self, name: str) -> ActionBarPage:
        if name not in self._pages:
            return self.make_page(name)
        return self._pages[name]

    def show_page(self, name: str) -> None:
        page = self._pages.get(name)
        if page is not None:
            self._stack.setCurrentWidget(page)

    def has_page(self, name: str) -> bool:
        return name in self._pages

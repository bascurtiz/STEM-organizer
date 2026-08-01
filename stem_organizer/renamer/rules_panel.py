"""Rules panel — port of track_renamer.gui.rules_panel.

A scrollable list of rule cards. Each rule type renders differently:
  OpRule           → enable + op label + (optional inline entry) + delete
  OpRule(categoryBundle) → above + the category macro table
  ConditionGroup   → IF condition + THEN APPLY list of child OpRules + add-child
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    PushButton,
    StrongBodyLabel,
    ToggleButton,
)

from track_renamer.category_palette import (
    CATEGORY_PALETTE_COLORS,
    default_category_color,
    next_unused_category_color,
    sort_rule_category_keywords,
    sync_category_names_from_affix,
)
from track_renamer.engine.defaults import (
    DEFAULT_CATEGORY_SOURCE,
    RULE_CATALOG,
    make_category_bundle,
    make_category_rules,
)
from track_renamer.engine.models import CategoryRule, Condition, ConditionGroup, OpRule, Rule

from .. import theme
from .theme import TIPS

_COLOR_STRIP_PX = 8
_DELETE_BTN_SIZE = 24  # square ✕ button; tall enough to legibly render the glyph
_PREFIX_COL_W = 108  # prefix field + column header; Instrument source label matches
_CATEGORY_COL_GAP = 4


def _make_delete_button(tooltip: str) -> PushButton:
    """Small square ✕ button with a clearly visible muted glyph.

    Plain PushButton defaults render the ✕ too faint/small at 28×auto; pin a
    fixed square size, a readable glyph size, and explicit muted text color so
    the delete affordance reads as an X on every row. Object-name-scoped QSS
    (PushButton#RuleDelete) beats Fluent's cascade so the red hover fill wins —
    same trick as the Match-tab keyword × button (KeywordRemove).
    """
    btn = PushButton("✕")
    btn.setObjectName("RuleDelete")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(_DELETE_BTN_SIZE, _DELETE_BTN_SIZE)
    btn.setToolTip(tooltip)
    t = theme.DARK
    btn.setStyleSheet(
        f"""
        PushButton#RuleDelete {{
            color: {t['text_dim']};
            background-color: {theme.CONTROL_BG};
            border: 1px solid {t['border']};
            border-radius: 5px;
            font-size: 13px;
            font-weight: 600;
            padding: 0px;
        }}
        PushButton#RuleDelete:hover {{
            color: {t['text']};
            background-color: {t['danger']};
            border: 1px solid {t['danger']};
        }}
        """
    )
    return btn


def _make_add_button(tooltip: str) -> PushButton:
    """Compact dark rounded '+ Add' button matching the delete affordance's chrome."""
    btn = PushButton("+ Add")
    btn.setObjectName("RuleAdd")  # skip-from-polish, distinct from delete
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(_DELETE_BTN_SIZE)
    btn.setMinimumWidth(52)
    btn.setToolTip(tooltip)
    t = theme.DARK
    btn.setStyleSheet(
        f"""
        PushButton#RuleAdd {{
            color: {t['text_dim']};
            background-color: {theme.CONTROL_BG};
            border: 1px solid {t['border']};
            border-radius: 5px;
            font-size: 12px;
            font-weight: 600;
            padding: 0px 8px;
        }}
        PushButton#RuleAdd:hover {{
            color: {t['text']};
            background-color: {theme.COLORS['accent']};
            border: 1px solid {theme.COLORS['accent']};
        }}
        """
    )
    return btn


def _style_prefix_color(edit: LineEdit, color: str) -> None:
    """Show category color as a left strip on the prefix field."""
    from qfluentwidgets import setCustomStyleSheet

    t = theme.DARK
    focus = theme.COLORS["bg"]  # #1e1f26
    edit.setProperty("transparent", False)
    sheet = f"""
        LineEdit, LineEdit[transparent=false] {{
            background: {theme.CONTROL_BG};
            background-color: {theme.CONTROL_BG};
            border: 1px solid {t['border']};
            border-left: {_COLOR_STRIP_PX}px solid {color};
            border-radius: 6px;
            color: {t['text']};
            padding-left: 6px;
            selection-background-color: {theme.COLORS['accent']};
        }}
        LineEdit:hover, LineEdit[transparent=false]:hover {{
            background: {theme.CONTROL_BG_HOVER};
            background-color: {theme.CONTROL_BG_HOVER};
        }}
        LineEdit:focus, LineEdit:focus[transparent=false], LineEdit[transparent=false]:focus {{
            background: {focus};
            background-color: {focus};
            border: 1px solid {t['border']};
            border-left: {_COLOR_STRIP_PX}px solid {color};
        }}
        """
    edit.setStyleSheet(sheet)
    setCustomStyleSheet(edit, sheet, sheet)
    edit.style().unpolish(edit)
    edit.style().polish(edit)
    edit.update()


def _style_keywords_edit(edit: LineEdit) -> None:
    """Keyword fields — idle matches former hover (#262833); focus stays dark."""
    from qfluentwidgets import setCustomStyleSheet

    idle = theme.COLORS["panel"]  # #262833
    hover = theme.COLORS["panel2"]  # #2F3140 — a bit brighter than idle
    focus = theme.COLORS["bg"]  # #1e1f26
    t = theme.DARK
    edit.setObjectName("CategoryKeywords")
    edit.setProperty("hasKeywordsFill", True)
    edit.setProperty("transparent", False)
    sheet = f"""
        LineEdit#CategoryKeywords,
        LineEdit#CategoryKeywords[transparent=false] {{
            background: {idle};
            background-color: {idle};
            border: 1px solid {t['border']};
            border-radius: 5px;
            color: {t['text']};
            padding: 0px 8px;
            selection-background-color: {theme.COLORS['accent']};
        }}
        LineEdit#CategoryKeywords:hover,
        LineEdit#CategoryKeywords[transparent=false]:hover {{
            background: {hover};
            background-color: {hover};
        }}
        LineEdit#CategoryKeywords:focus,
        LineEdit#CategoryKeywords:focus[transparent=false],
        LineEdit#CategoryKeywords[transparent=false]:focus {{
            background: {focus};
            background-color: {focus};
            border: 1px solid {t['border']};
        }}
        """
    edit.setStyleSheet(sheet)
    setCustomStyleSheet(edit, sheet, sheet)
    edit.style().unpolish(edit)
    edit.style().polish(edit)
    edit.update()


class _PrefixColorFilter(QObject):
    """Click/hover the left color strip on a prefix LineEdit to open the color picker."""

    _STRIP_HIT = _COLOR_STRIP_PX + 4

    def __init__(self, edit: LineEdit, cat_dict: dict, panel: "RulesPanel") -> None:
        super().__init__(edit)
        self._edit = edit
        self._cat = cat_dict
        self._panel = panel
        edit.setMouseTracking(True)

    @staticmethod
    def _event_x(event: QEvent) -> float:
        pos = event.position() if hasattr(event, "position") else event.pos()
        return float(pos.x())

    def _sync_strip_cursor(self, x: float) -> None:
        hand = Qt.CursorShape.PointingHandCursor
        if x <= self._STRIP_HIT:
            if self._edit.cursor().shape() != hand:
                self._edit.setCursor(hand)
        elif self._edit.cursor().shape() != Qt.CursorShape.IBeamCursor:
            self._edit.setCursor(Qt.CursorShape.IBeamCursor)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is not self._edit:
            return False
        et = event.type()
        if et in (QEvent.Type.MouseMove, QEvent.Type.HoverMove, QEvent.Type.Enter):
            self._sync_strip_cursor(self._event_x(event))
        elif et == QEvent.Type.Leave:
            self._edit.unsetCursor()
        elif et == QEvent.Type.MouseButtonPress:
            x = self._event_x(event)
            if event.button() == Qt.MouseButton.LeftButton and x <= self._STRIP_HIT:
                self._panel._pick_category_color(self._cat, self._edit)
                return True
        return False


OP_LABELS = {
    "stripLeadingNumberPrefix": "Remove prefix numbers",
    "stripLeadingDashes": "Remove leading dashes",
    "collapseWhitespace": "Collapse whitespace",
    "trim": "Trim",
    "titleCase": "Title Case",
    "addTextAtBeginning": "Add text at the beginning",
    "addTextAtEnd": "Add text at the end",
    "replaceText": "Replace text",
    "removeText": "Remove text",
    "removeCharRange": "Remove a range of characters",
    "categoryBundle": "Category Macro",
    "padNumericSuffix": "Pad numeric suffix",
    "stripTrailingNumber": "Remove trailing number",
}

CONDITION_OPS = [
    ("contains", "contains"),
    ("equals", "equals"),
    ("matches", "matches"),
    ("notContains", "not contains"),
]
SOURCE_LABELS = [("filename", "Filename"), ("model", "Audio"), ("combo", "Combo")]


class RulesPanel(QWidget):
    """Left side: list of rules with Apply/Clear buttons."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_change: Optional[Callable[[], None]] = None,
        on_apply: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RulesPanel")
        self.on_change = on_change
        self.on_apply = on_apply
        self._rules: List[Rule] = []
        self._suspend_notify = False

        layout = QVBoxLayout(self)
        # Flush right — scrollbar is the rules|preview divider (no extra seam).
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header — RULES title only (Clear / Apply sit on the preset row above)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 8, 0)
        header.setSpacing(6)
        title = CaptionLabel("RULES")
        title.setObjectName("SectionTitle")
        title.setStyleSheet(
            f"color: {theme.DARK['text_dim']}; font-size: {theme.SECTION_TITLE_PX}px; "
            f'font-family: "{theme.FONT_FAMILY}"; font-weight: 600; background: transparent;'
        )
        title.setFixedHeight(theme.ACTION_BTN_HEIGHT)
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.section_title = title
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)

        from ..widgets.action_button import action_button

        self.clear_btn = action_button(
            "Clear", on_click=self._clear_rules, parent=self, tip=TIPS["clear_rules"]
        )
        self.apply_btn = action_button(
            "Apply",
            on_click=lambda: self.on_apply and self.on_apply(),
            parent=self,
            tip=TIPS["apply_preview"],
        )
        self.apply_btn.setObjectName("RenameApply")
        self.set_apply_pending(False)
        # Reparented onto the Rename preset row by TrackRenamerApp.
        # Size sync after polish (Clear keeps natural width; Apply clones it).
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self.match_apply_to_clear)

        # Add-rule dropdown — placeholder is item 0 so the combo rests on the
        # prompt and snaps back after a rule is added. Leading "+" is outside
        # the combo (not in the item text). Width matches rule cards: same left
        # gutter as the "+" column, same right inset as the AlwaysOn divider bar.
        _rules_bar_w = 12  # QScrollArea#RulesScroll vertical bar
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 8 + _rules_bar_w, 0)
        add_row.setSpacing(6)
        plus = BodyLabel("+")
        self.add_combo = ComboBox()
        self.add_combo.setToolTip(TIPS["add_rule"])
        self.add_combo.activated.connect(self._on_add_activated)
        self._rebuild_add_combo()
        add_row.addWidget(plus)
        add_row.addWidget(self.add_combo, stretch=1)
        layout.addLayout(add_row)
        _rules_left_gutter = plus.sizeHint().width() + add_row.spacing()

        # Qt ScrollArea (not Fluent overlay): AlwaysOn bar is the center divider.
        self.scroll = QScrollArea()
        self.scroll.setObjectName("RulesScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # Windows native style often ignores QScrollBar QSS — Fusion paints the thumb.
        from PySide6.QtWidgets import QStyleFactory

        _fusion = QStyleFactory.create("Fusion")
        if _fusion is not None:
            self.scroll.setStyle(_fusion)
            self.scroll.verticalScrollBar().setStyle(_fusion)
        self.stack_host = QWidget()
        self.stack_layout = QVBoxLayout(self.stack_host)
        # Left gutter matches combo (after "+"); right 8 matches add_row (bar is outside viewport)
        self.stack_layout.setContentsMargins(_rules_left_gutter, 0, 8, 0)
        self.stack_layout.setSpacing(6)
        self.stack_layout.addStretch(1)
        self.scroll.setWidget(self.stack_host)
        layout.addWidget(self.scroll, stretch=1)

    # ----- public API -----

    def set_rules(self, rules: List[Rule]) -> None:
        self._rules = list(rules)
        self._render()

    def get_rules(self) -> List[Rule]:
        return self._rules

    def match_apply_to_clear(self) -> None:
        """Pin Apply to Clear's outer size (Clear stays natural)."""
        h = theme.ACTION_BTN_HEIGHT
        # Prefer laid-out width once available; else sizeHint after Clear polish.
        cw = self.clear_btn.width()
        if cw < 8:
            cw = max(self.clear_btn.sizeHint().width(), 1)
        self.apply_btn.setFixedSize(cw, h)

    def set_apply_pending(self, pending: bool) -> None:
        self.apply_btn.setEnabled(True)
        h = theme.ACTION_BTN_HEIGHT
        if pending:
            # Active: accent fill — size comes from match_apply_to_clear (no QSS width)
            self.apply_btn.setStyleSheet(
                f"""
                PushButton#RenameApply {{
                    background-color: {theme.COLORS['accent']};
                    border: 1px solid {theme.COLORS['accent_hov']};
                    border-radius: 5px;
                    color: {theme.DARK['text']};
                    font-weight: 600;
                    padding: 0px;
                }}
                PushButton#RenameApply:hover {{
                    background-color: {theme.COLORS['accent_hov']};
                    color: {theme.DARK['text']};
                }}
                """
            )
        else:
            # Idle: muted vs Clear — same outer size, dimmer label
            self.apply_btn.setStyleSheet(
                f"""
                PushButton#RenameApply {{
                    background-color: {theme.CONTROL_BG};
                    border: 1px solid {theme.DARK['border']};
                    border-radius: 5px;
                    color: {theme.DARK['text_dim']};
                    font-weight: 400;
                    padding: 0px;
                }}
                PushButton#RenameApply:hover {{
                    background-color: {theme.CONTROL_BG_HOVER};
                    color: {theme.DARK['text']};
                }}
                """
            )
        self.apply_btn.setFixedHeight(h)
        self.match_apply_to_clear()

    # ----- render -----

    def _render(self) -> None:
        # Clear existing rows (keep trailing stretch)
        while self.stack_layout.count() > 1:
            item = self.stack_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for idx, rule in enumerate(self._rules):
            card = self._render_rule(rule, idx)
            self.stack_layout.insertWidget(self.stack_layout.count() - 1, card)
        self._rebuild_add_combo()

    def _top_level_ops(self) -> set:
        return {r.op for r in self._rules if isinstance(r, OpRule)}

    def _rebuild_add_combo(self) -> None:
        """Offer only rule types not already present (ops are unique by ``op``)."""
        used = self._top_level_ops()
        self.add_combo.blockSignals(True)
        self.add_combo.clear()
        self.add_combo.addItem("Add a rule…")
        for entry in RULE_CATALOG:
            if entry["kind"] == "op" and entry.get("op") in used:
                continue
            self.add_combo.addItem(entry["label"])
        self.add_combo.setCurrentIndex(0)
        self.add_combo.blockSignals(False)

    def _render_rule(self, rule: Rule, idx: int) -> QWidget:
        if isinstance(rule, ConditionGroup):
            return self._render_condition_group(rule, idx)
        if isinstance(rule, OpRule):
            return self._render_op_rule(rule, idx)
        # CategoryRule standalone — shouldn't appear at top level
        return QWidget()

    def _render_op_rule(self, rule: OpRule, idx: int, *, group: Optional[ConditionGroup] = None) -> QWidget:
        card = QFrame()
        card.setObjectName("Card")
        card_lay = QHBoxLayout(card)
        card_lay.setContentsMargins(8, 6, 8, 6)
        card_lay.setSpacing(8)

        enable = CheckBox()
        enable.setChecked(rule.enabled)
        enable.setToolTip(TIPS["rule_enable"])
        enable.toggled.connect(lambda v: self._on_op_enable(rule, v, group=group))
        card_lay.addWidget(enable)

        label = BodyLabel(OP_LABELS.get(rule.op, rule.op))
        card_lay.addWidget(label)

        # Inline entry for removeText / replaceText / addText*
        if rule.op in ("removeText", "replaceText", "addTextAtBeginning", "addTextAtEnd"):
            entry = LineEdit()
            entry.setText(rule.params.get("text", ""))
            entry.setPlaceholderText(TIPS.get("remove_text", "text"))
            entry.setToolTip(TIPS.get("rule_text", TIPS["remove_text"]))
            entry.textChanged.connect(lambda v, r=rule: self._on_op_text(r, v, group=group))
            theme.style_line_edit(entry)
            card_lay.addWidget(entry, stretch=1)
        elif rule.op == "replaceText":
            entry = LineEdit()
            entry.setText(rule.params.get("text", ""))
            entry.setToolTip(TIPS.get("rule_text", TIPS["remove_text"]))
            entry.textChanged.connect(lambda v, r=rule: self._on_op_text(r, v, group=group))
            theme.style_line_edit(entry)
            card_lay.addWidget(entry, stretch=1)
        else:
            card_lay.addStretch(1)

        delete = _make_delete_button(TIPS["remove_rule"])
        delete.clicked.connect(lambda _, r=rule, g=group: self._remove_rule(r, g))
        card_lay.addWidget(delete)

        if rule.op == "categoryBundle":
            outer = QFrame()
            outer_lay = QVBoxLayout(outer)
            outer_lay.setContentsMargins(0, 0, 0, 0)
            outer_lay.setSpacing(4)
            outer_lay.addWidget(card)
            outer_lay.addWidget(self._render_category_table(rule))
            return outer
        return card

    def _render_category_table(self, rule: OpRule) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("Section")
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(0, 10, 8, 6)
        wrap_lay.setSpacing(4)

        # Source row — label occupies PREFIX column so Filename lines up with KEYWORDS
        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 4, 0, 10)
        source_row.setSpacing(_CATEGORY_COL_GAP)
        src_lbl = BodyLabel("Instrument source")
        src_lbl.setFixedWidth(_PREFIX_COL_W)
        src_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        source_row.addWidget(src_lbl)
        source_group = QButtonGroup(self)
        cur_source = rule.params.get("source", DEFAULT_CATEGORY_SOURCE)
        for val, lbl in SOURCE_LABELS:
            rb = ToggleButton(lbl)
            rb.setCheckable(True)
            rb.setChecked(val == cur_source)
            theme.style_toggle_button(rb)  # after _render too (polish only runs once)
            rb.setToolTip(TIPS.get(f"instrument_source_{val}", TIPS["instrument_source"]))
            rb.clicked.connect(lambda _=False, v=val, r=rule: self._set_category_source(r, v))
            source_group.addButton(rb)
            source_row.addWidget(rb)
        source_row.addStretch(1)
        wrap_lay.addLayout(source_row)

        # Header
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(_CATEGORY_COL_GAP)
        for text, w in (("PREFIX", _PREFIX_COL_W), ("KEYWORDS (COMMA-SEPARATED)", 0)):
            lbl = CaptionLabel(text)
            lbl.setStyleSheet(f"color: {theme.DARK['text_dim']}; font-size: 8pt;")
            if w:
                lbl.setFixedWidth(w)
            header_row.addWidget(lbl, stretch=0 if w else 1)
        add_cat_btn = _make_add_button(TIPS["add_category_row"])
        add_cat_btn.clicked.connect(lambda _, r=rule: self._add_category_row(r))
        header_row.addWidget(add_cat_btn)
        wrap_lay.addLayout(header_row)

        # Category rows
        cats = rule.params.setdefault("categories", [])
        for cat_dict in cats:
            wrap_lay.addWidget(self._render_category_row(rule, cat_dict))
        return wrap

    def _render_category_row(self, rule: OpRule, cat_dict: dict) -> QWidget:
        row = QFrame()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(_CATEGORY_COL_GAP)

        color = cat_dict.get("color", "") or default_category_color(cat_dict.get("name", ""))

        prefix = LineEdit()
        prefix.setProperty("hasColorStrip", True)
        prefix.setText(cat_dict.get("affix", ""))
        prefix.setFixedWidth(_PREFIX_COL_W)
        prefix.setToolTip(TIPS["prefix_field"] + "\nClick the color strip to change category color.")
        prefix.textChanged.connect(lambda v, c=cat_dict: self._on_category_field(c, "affix", v))
        _style_prefix_color(prefix, color)
        # Click left color strip → color picker
        prefix.installEventFilter(_PrefixColorFilter(prefix, cat_dict, self))
        row_lay.addWidget(prefix)

        keywords = LineEdit()
        keywords.setText(cat_dict.get("keywords", ""))
        keywords.setToolTip(TIPS["keywords_field"])
        keywords.textChanged.connect(lambda v, c=cat_dict: self._on_category_field(c, "keywords", v))
        _style_keywords_edit(keywords)
        row_lay.addWidget(keywords, stretch=1)

        remove = _make_delete_button(TIPS["remove_category_row"])
        remove.clicked.connect(lambda _, r=rule, c=cat_dict: self._remove_category_row(r, c))
        row_lay.addWidget(remove)
        return row

    def _render_condition_group(self, group: ConditionGroup, idx: int) -> QWidget:
        card = QFrame()
        card.setObjectName("Section")
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(8, 6, 8, 6)
        card_lay.setSpacing(4)

        # Enable + title
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        enable = CheckBox()
        enable.setChecked(group.enabled)
        enable.setToolTip(TIPS["rule_enable"])
        enable.toggled.connect(lambda v, g=group: self._on_group_enable(g, v))
        head.addWidget(enable)
        cond = group.conditions[0] if group.conditions else Condition()
        title = StrongBodyLabel(f"If filename {cond.operator} '{cond.value}' ({len(group.children)})")
        title.setStyleSheet("font-weight: 600;")
        head.addWidget(title)
        head.addStretch(1)
        delete = _make_delete_button(TIPS["remove_rule"])
        delete.clicked.connect(lambda _, g=group: self._remove_rule(g, None))
        head.addWidget(delete)
        card_lay.addLayout(head)

        # IF row
        if_row = QHBoxLayout()
        if_row.setContentsMargins(20, 0, 0, 0)
        if_row.addWidget(BodyLabel("IF"))
        field_lbl = BodyLabel("filename")
        field_lbl.setToolTip(TIPS["condition_field"])
        if_row.addWidget(field_lbl)
        op_combo = ComboBox()
        for val, lbl in CONDITION_OPS:
            op_combo.addItem(lbl, userData=val)
        op_combo.setCurrentText(
            dict((v, l) for v, l in CONDITION_OPS).get(cond.operator, "contains")
        )
        op_combo.setToolTip(TIPS["condition_op"])
        op_combo.currentIndexChanged.connect(
            lambda i, c=cond, t=title, g=group: self._on_condition_op(c, op_combo.itemData(i), g, t)
        )
        if_row.addWidget(op_combo)
        value_entry = LineEdit()
        value_entry.setText(cond.value)
        value_entry.setToolTip(TIPS["condition_value"])
        value_entry.textChanged.connect(
            lambda v, c=cond, t=title: self._on_condition_value(c, v, t)
        )
        theme.style_line_edit(value_entry)
        if_row.addWidget(value_entry, stretch=1)
        card_lay.addLayout(if_row)

        # THEN APPLY
        then_lbl = CaptionLabel("THEN APPLY")
        then_lbl.setStyleSheet(f"color: {theme.DARK['text_dim']};")
        then_lbl.setContentsMargins(20, 4, 0, 0)
        card_lay.addWidget(then_lbl)

        for child in group.children:
            child_card = self._render_op_rule(child, idx, group=group)
            # Indent
            child_card.setContentsMargins(20, 0, 0, 0)
            card_lay.addWidget(child_card)

        add_child = ComboBox()
        add_child.setToolTip(TIPS["add_child_rule"])
        add_child.addItem("Add child rule…")
        used_child = {c.op for c in group.children if isinstance(c, OpRule)}
        for entry in RULE_CATALOG:
            if entry["kind"] != "op":
                continue
            if entry.get("op") in used_child:
                continue
            add_child.addItem(entry["label"])
        add_child.activated.connect(lambda i, g=group, c=add_child: self._on_add_child(g, c.itemText(i), c))
        card_lay.addWidget(add_child)
        return card

    # ----- mutators -----

    def _notify(self) -> None:
        if self._suspend_notify:
            return
        if self.on_change:
            self.on_change()

    def _on_add_activated(self, idx: int) -> None:
        label = self.add_combo.itemText(idx)
        # Always snap back to the "Add a rule…" placeholder (index 0).
        self.add_combo.setCurrentIndex(0)
        # Ignore the placeholder itself (and any label not in the catalog).
        entry = next((e for e in RULE_CATALOG if e["label"] == label), None)
        if entry is None:
            return
        if entry["kind"] == "op" and entry.get("op") in self._top_level_ops():
            return
        if entry["kind"] == "conditionGroup":
            self._rules.insert(
                0,
                ConditionGroup(conditions=[Condition(field="name", operator="contains", value="")]),
            )
        elif entry.get("op") == "categoryBundle":
            self._rules.insert(0, make_category_bundle())
        else:
            params = {"text": ""} if entry["op"] in ("removeText", "replaceText", "addTextAtBeginning", "addTextAtEnd") else {}
            self._rules.insert(0, OpRule(op=entry["op"], params=params))
        self._render()
        self._notify()

    def _on_add_child(self, group: ConditionGroup, label: str, combo: ComboBox) -> None:
        combo.setCurrentIndex(0)
        entry = next((e for e in RULE_CATALOG if e["label"] == label), None)
        if entry is None or entry["kind"] != "op":
            return
        if entry.get("op") in {c.op for c in group.children if isinstance(c, OpRule)}:
            return
        params = {"text": ""} if entry["op"] == "removeText" else {}
        group.children.append(OpRule(op=entry["op"], params=params))
        self._render()
        self._notify()

    def _clear_rules(self) -> None:
        self._rules = []
        self._render()
        self._notify()

    def _remove_rule(self, rule: Rule, group: Optional[ConditionGroup]) -> None:
        if group is not None:
            if rule in group.children:
                group.children.remove(rule)
        else:
            if rule in self._rules:
                self._rules.remove(rule)
        self._render()
        self._notify()

    def _on_op_enable(self, rule: OpRule, value: bool, *, group: Optional[ConditionGroup]) -> None:
        rule.enabled = value
        self._notify()

    def _on_group_enable(self, group: ConditionGroup, value: bool) -> None:
        group.enabled = value
        self._notify()

    def _on_op_text(self, rule: OpRule, value: str, *, group: Optional[ConditionGroup]) -> None:
        rule.params["text"] = value
        self._notify()

    def _on_condition_op(self, cond: Condition, op: str, group: ConditionGroup, title: StrongBodyLabel) -> None:
        cond.operator = op
        title.setText(f"If filename {cond.operator} '{cond.value}' ({len(group.children)})")
        self._notify()

    def _on_condition_value(self, cond: Condition, value: str, title: StrongBodyLabel) -> None:
        cond.value = value
        parent_text = title.text()
        # rebuild title — find operator from current text
        import re
        m = re.match(r"If filename (\w+) '.*' \(\d+\)", parent_text)
        op_str = m.group(1) if m else cond.operator
        title.setText(f"If filename {op_str} '{value}' ({parent_text.rsplit('(', 1)[-1]}")
        self._notify()

    def _set_category_source(self, rule: OpRule, value: str) -> None:
        rule.params["source"] = value
        rule.params.pop("mlConfidence", None)
        self._notify()

    def _on_category_field(self, cat_dict: dict, field: str, value: str) -> None:
        cat_dict[field] = value
        self._notify()

    def _add_category_row(self, rule: OpRule) -> None:
        cats = rule.params.setdefault("categories", [])
        existing_names = [c.get("name", "") for c in cats]
        candidate = "New"
        i = 1
        while candidate in existing_names:
            i += 1
            candidate = f"New {i}"
        color = next_unused_category_color(cats)
        cat = CategoryRule(
            name=candidate,
            keywords="",
            affix=f"{candidate.upper()} - ",
            color=color,
            color_override=True,
        )
        cats.insert(0, cat.to_dict())
        self._render()
        self._notify()

    def _remove_category_row(self, rule: OpRule, cat_dict: dict) -> None:
        cats = rule.params.get("categories", [])
        if cat_dict in cats:
            cats.remove(cat_dict)
        self._render()
        self._notify()

    def _pick_category_color(self, cat_dict: dict, prefix_edit: Optional[LineEdit] = None) -> None:
        from ..widgets.dialogs import dim_behind

        dlg = QColorDialog(self)
        dlg.setWindowTitle("Select Color")
        dlg.setOption(QColorDialog.ShowAlphaChannel, False)
        theme.style_color_dialog(dlg)
        current = cat_dict.get("color", "") or default_category_color(cat_dict.get("name", ""))
        dlg.setCurrentColor(QColor(current))
        for i, hex_color in enumerate(CATEGORY_PALETTE_COLORS[:16]):
            dlg.setCustomColor(i, QColor(hex_color))
        with dim_behind(self.window()):
            accepted = dlg.exec() == QColorDialog.Accepted
        if accepted:
            color = dlg.currentColor()
            cat_dict["color"] = color.name()
            # Must match CategoryRule.to_dict / from_dict key (camelCase).
            cat_dict["colorOverride"] = True
            cat_dict.pop("color_override", None)  # drop legacy snake_case if present
            if prefix_edit is not None:
                _style_prefix_color(prefix_edit, color.name())
            self._notify()

"""Condition evaluation."""

from __future__ import annotations

import re

from .models import Condition, Track
from .tokens import resolve_tokens


def _fold(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.casefold()


def eval_condition(
    condition: Condition,
    *,
    track: Track,
    current_name: str,
    original_name: str,
    index: int,
) -> bool:
    value = condition.value.strip()
    if not value and condition.field == "name":
        return False

    if condition.field == "name":
        left = current_name
        right = resolve_tokens(
            value,
            track=track,
            original_name=original_name,
            current_name=current_name,
            index=index,
            counter=index,
        )
        cs = condition.case_sensitive
        l = left if cs else left.casefold()
        r = right if cs else right.casefold()
        op = condition.operator
        if op == "contains":
            return r in l
        if op == "notContains":
            return r not in l
        if op == "equals":
            return l == r
        if op == "notEquals":
            return l != r
        if op == "matches":
            flags = 0 if cs else re.IGNORECASE
            return bool(re.search(right, left, flags))
        if op == "notMatches":
            flags = 0 if cs else re.IGNORECASE
            return not re.search(right, left, flags)
    return False


def eval_conditions(
    conditions: list[Condition],
    match_mode: str,
    **kwargs,
) -> bool:
    active = [c for c in conditions if c.value.strip() or c.field != "name"]
    if not active:
        return False
    results = [eval_condition(c, **kwargs) for c in active]
    if match_mode == "any":
        return any(results)
    return all(results)

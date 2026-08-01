"""Apply rule stacks to tracks and produce preview rows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .conditions import eval_conditions
from .models import CategoryRule, ConditionGroup, OpRule, PreviewRow, Rule, Track
from .ops import apply_op, compile_category_bundle


@dataclass(frozen=True, slots=True)
class PreparedRulePlan:
    """Immutable rules snapshot with expensive category data precompiled."""

    rules: tuple[Rule, ...]


def _prepare_rule(rule: Rule) -> Rule:
    prepared = deepcopy(rule)
    if isinstance(prepared, ConditionGroup):
        prepared.children = [_prepare_rule(child) for child in prepared.children]
        prepared.branches = [
            _prepare_rule(branch) for branch in prepared.branches  # type: ignore[list-item]
        ]
        return prepared
    if isinstance(prepared, CategoryRule):
        return OpRule(
            enabled=prepared.enabled,
            scope="both",
            op="categoryBundle",
            params={"categories": compile_category_bundle((prepared,))},
        )
    if isinstance(prepared, OpRule) and prepared.op in (
        "categoryBundle",
        "renameGroupsByCategory",
    ):
        prepared.params = dict(prepared.params)
        prepared.params["categories"] = compile_category_bundle(
            prepared.params.get("categories", ())
        )
    return prepared


def prepare_rules(rules: Sequence[Rule] | PreparedRulePlan) -> PreparedRulePlan:
    """Snapshot and compile a rule stack once before processing many tracks."""
    if isinstance(rules, PreparedRulePlan):
        return rules
    return PreparedRulePlan(tuple(_prepare_rule(rule) for rule in rules))


def _rule_sequence(
    rules: Sequence[Rule] | PreparedRulePlan,
) -> Sequence[Rule]:
    return rules.rules if isinstance(rules, PreparedRulePlan) else rules


def _apply_rules_to_name(
    name: str,
    rules: Sequence[Rule],
    *,
    track: Track,
    original_name: str,
    index: int,
    variables: dict[str, str] | None = None,
    match_info: dict[str, str] | None = None,
) -> str:
    current = name
    variables = variables or {}
    match_info = match_info if match_info is not None else {}

    for rule in rules:
        if not getattr(rule, "enabled", True):
            continue

        if isinstance(rule, ConditionGroup):
            matched = eval_conditions(
                rule.conditions,
                rule.match,
                track=track,
                current_name=current,
                original_name=original_name,
                index=index,
            )
            if matched:
                current = _apply_rules_to_name(
                    current,
                    rule.children,
                    track=track,
                    original_name=original_name,
                    index=index,
                    variables=variables,
                    match_info=match_info,
                )
            else:
                for branch in rule.branches:
                    if not branch.enabled:
                        continue
                    branch_matched = (
                        not branch.conditions
                        or eval_conditions(
                            branch.conditions,
                            branch.match,
                            track=track,
                            current_name=current,
                            original_name=original_name,
                            index=index,
                        )
                    )
                    if branch_matched:
                        current = _apply_rules_to_name(
                            current,
                            branch.children,
                            track=track,
                            original_name=original_name,
                            index=index,
                            variables=variables,
                            match_info=match_info,
                        )
                        break
            continue

        ctx: dict[str, Any] = {
            "track": track,
            "original_name": original_name,
            "current_name": current,
            "index": index,
            "counter": index,
            "variables": variables,
        }

        if isinstance(rule, CategoryRule):
            current = apply_op(
                current,
                "categoryBundle",
                {"categories": [rule.to_dict()]},
                ctx,
            )
            if kw := ctx.get("matched_keyword"):
                match_info["matched_keyword"] = kw
            continue

        if isinstance(rule, OpRule):
            current = apply_op(current, rule.op, rule.params, ctx)
            if kw := ctx.get("matched_keyword"):
                match_info["matched_keyword"] = kw

    return current


def compute_preview_row(
    track: Track,
    rules: Sequence[Rule] | PreparedRulePlan,
    *,
    index: int,
) -> PreviewRow:
    """Compute a single PreviewRow (used for viewport-priority lazy preview)."""
    original = track.name
    match_info: dict[str, str] = {}
    new_name = _apply_rules_to_name(
        original,
        _rule_sequence(rules),
        track=track,
        original_name=original,
        index=index,
        match_info=match_info,
    )
    return PreviewRow(
        track=track,
        original_name=original,
        new_name=new_name,
        changed=new_name != original,
        matched_keyword=match_info.get("matched_keyword", ""),
    )


def compute_preview(
    tracks: list[Track],
    rules: Sequence[Rule] | PreparedRulePlan,
    *,
    progress: Callable[[int, int], None] | None = None,
    on_batch: Callable[[list[PreviewRow], int, int], None] | None = None,
    batch_size: int = 200,
) -> list[PreviewRow]:
    prepared = prepare_rules(rules)
    rows: list[PreviewRow] = []
    total = len(tracks)
    for index, track in enumerate(tracks, start=1):
        rows.append(compute_preview_row(track, prepared, index=index))
        if on_batch and index % batch_size == 0:
            on_batch(rows, index, total)
        elif progress and index % 500 == 0:
            progress(index, total)
    if on_batch and total:
        on_batch(rows, total, total)
    elif progress and total:
        progress(total, total)
    return rows

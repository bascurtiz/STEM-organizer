"""Data models for tracks and rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import uuid


TrackType = Literal["audio", "midi", "group", "return", "master"]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class Track:
    id: str
    name: str
    track_type: TrackType = "audio"
    parent_id: str | None = None
    depth: int = 0
    selected: bool = True

    # File-backed items (folder scan mode)
    file_path: Path | None = None
    extension: str = ""
    relative_path: str = ""

    # metadata extracted from filename / context when available
    bpm: str = ""
    key: str = ""
    group: str = ""
    instrument: str = ""  # OpenMIC / instrument label (e.g. guitar)
    instrument_score: float = 0.0
    instrument_second: float = 0.0  # runner-up score (ambiguity check)
    category: str = ""  # mapped Category Macro name (e.g. Guitar)

    @property
    def display_name(self) -> str:
        return f"{self.name}{self.extension}" if self.extension else self.name

    @property
    def is_file(self) -> bool:
        return self.file_path is not None

    @property
    def is_group(self) -> bool:
        return self.track_type == "group"

    @property
    def is_audio(self) -> bool:
        return self.track_type == "audio"

    @property
    def is_midi(self) -> bool:
        return self.track_type == "midi"


@dataclass
class Condition:
    field: str = "name"
    operator: str = "contains"
    value: str = ""
    case_sensitive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "caseSensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(
            field=data.get("field", "name"),
            operator=data.get("operator", "contains"),
            value=data.get("value", ""),
            case_sensitive=bool(data.get("caseSensitive", False)),
        )


@dataclass
class CategoryRule:
    name: str = ""
    keywords: str = ""
    affix: str = ""
    affix_position: str = "prefix"
    match_mode: str = "wholeWord"
    enabled: bool = True
    existing_affix_policy: str = "skip"
    color: str = ""
    color_override: bool = False
    cat_id: str = field(default_factory=lambda: new_id("cat"))

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": "category",
            "id": self.cat_id,
            "enabled": self.enabled,
            "name": self.name,
            "keywords": self.keywords,
            "affix": self.affix,
            "affixPosition": self.affix_position,
            "matchMode": self.match_mode,
            "scope": "both",
            "existingAffixPolicy": self.existing_affix_policy,
        }
        if self.color:
            data["color"] = self.color
        if self.color_override:
            data["colorOverride"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryRule:
        name = (data.get("name") or "").strip()
        affix = data.get("affix")
        if not affix and name:
            affix = f"{name.upper()} - "
        return cls(
            name=name,
            keywords=data.get("keywords", ""),
            affix=affix or "",
            affix_position=data.get("affixPosition", "prefix"),
            match_mode=data.get("matchMode", "wholeWord"),
            enabled=bool(data.get("enabled", True)),
            existing_affix_policy=data.get("existingAffixPolicy", "skip"),
            color=data.get("color", ""),
            color_override=bool(
                data.get("colorOverride", data.get("color_override", False))
            ),
            cat_id=data.get("id") or new_id("cat"),
        )


@dataclass
class RuleBase:
    id: str = field(default_factory=lambda: new_id("rule"))
    enabled: bool = True
    scope: str = "both"

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class OpRule(RuleBase):
    kind: str = "op"
    op: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "enabled": self.enabled,
            "scope": self.scope,
            "op": self.op,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpRule:
        return cls(
            id=data.get("id", new_id("op")),
            enabled=bool(data.get("enabled", True)),
            scope=data.get("scope", "both"),
            op=data.get("op", ""),
            params=dict(data.get("params") or {}),
        )


@dataclass
class ConditionGroup(RuleBase):
    kind: str = "conditionGroup"
    match: str = "all"
    conditions: list[Condition] = field(default_factory=list)
    children: list[Rule] = field(default_factory=list)
    branches: list[ConditionGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "enabled": self.enabled,
            "scope": self.scope,
            "match": self.match,
            "conditions": [c.to_dict() for c in self.conditions],
            "children": [rule_to_dict(r) for r in self.children],
            "branches": [b.to_dict() for b in self.branches],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionGroup:
        return cls(
            id=data.get("id", new_id("cond")),
            enabled=bool(data.get("enabled", True)),
            scope=data.get("scope", "both"),
            match=data.get("match", "all"),
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            children=[rule_from_dict(r) for r in data.get("children", [])],
            branches=[cls.from_dict(b) for b in data.get("branches", [])],
        )


Rule = OpRule | ConditionGroup | CategoryRule


def rule_to_dict(rule: Rule) -> dict[str, Any]:
    if isinstance(rule, ConditionGroup):
        return rule.to_dict()
    if isinstance(rule, OpRule):
        return rule.to_dict()
    if isinstance(rule, CategoryRule):
        return rule.to_dict()
    raise TypeError(type(rule))


def rule_from_dict(data: dict[str, Any]) -> Rule:
    kind = data.get("kind")
    if kind == "conditionGroup":
        return ConditionGroup.from_dict(data)
    if kind == "category":
        return CategoryRule.from_dict(data)
    return OpRule.from_dict(data)


@dataclass
class PreviewRow:
    track: Track
    original_name: str
    new_name: str
    changed: bool
    # Category Macro keyword that selected the prefix (filename/combo), or
    # "<audio-determined>" when the prefix came from ML; empty when no category match.
    matched_keyword: str = ""

    @property
    def original_display(self) -> str:
        ext = self.track.extension
        return f"{self.original_name}{ext}" if ext else self.original_name

    @property
    def new_display(self) -> str:
        ext = self.track.extension
        return f"{self.new_name}{ext}" if ext else self.new_name

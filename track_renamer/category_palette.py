"""Category color palette — reds through grays to black (24 swatches)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_renamer.engine.models import Rule

# Ordered: red → warm → cool → green → yellow → orange → brown → gray → black
CATEGORY_PALETTE_COLORS: tuple[str, ...] = (
    "#EF4444",
    "#F59E0B",
    "#A855F7",
    "#10B981",
    "#8D95A4",
    "#C1090B",
    "#C41D63",
    "#8351A1",
    "#5C4EA0",
    "#485FAB",
    "#398BCB",
    "#00B8D3",
    "#25BAA2",
    "#3FB655",
    "#76C043",
    "#B0D236",
    "#FED600",
    "#A44F0D",
    "#F36E21",
    "#A45C7A",
    "#5E4138",
    "#626262",
    "#455A64",
    "#000000",
)

CATEGORY_BADGE_TEXT = "#ffffff"

CATEGORY_BADGE_LABELS: dict[str, str] = {
    "Percussion": "PERC",
    "Orchestra": "ORCHEST",
    "Orchestral": "ORCHEST",  # legacy name
}

# Legacy / alternate category names → canonical key in DEFAULT_CATEGORY_COLORS
CATEGORY_NAME_ALIASES: dict[str, str] = {
    "Synths": "Synth",
    "Orchestral": "Orchestra",
}

# Default color per category name (includes legacy names used in older presets)
DEFAULT_CATEGORY_COLORS: dict[str, str] = {
    "Bass": "#EF4444",
    "Drums": "#F59E0B",
    "Percussion": "#F36E21",
    "Synth": "#10B981",
    "Synths": "#10B981",
    "Pads": "#8351A1",
    "Wind": "#00B8D3",
    "Keys": "#485FAB",
    "Guitar": "#C1090B",
    "FX": "#000000",
    "Strings": "#76C043",
    "Vocals": "#A855F7",
    "Mallet": "#C41D63",
    "Orchestra": "#626262",
    "Orchestral": "#626262",  # legacy
}

_PALETTE_LOWER = {color.lower(): color for color in CATEGORY_PALETTE_COLORS}
_PALETTE_LOWER["#fbaa19"] = "#A44F0D"
_PALETTE_LOWER["#d41f26"] = "#C1090B"
_PALETTE_LOWER["#636b7a"] = "#8D95A4"
_PALETTE_LOWER["#dd3226"] = "#A45C7A"


def category_badge_label(name: str) -> str:
    return CATEGORY_BADGE_LABELS.get(name, name.upper())


def _prefix_token_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in DEFAULT_CATEGORY_COLORS:
        lookup[name.upper()] = name
        lookup[category_badge_label(name).upper()] = name
    for alias, canonical in CATEGORY_NAME_ALIASES.items():
        lookup[alias.upper()] = canonical
    return lookup


_PREFIX_TOKEN_TO_CATEGORY = _prefix_token_lookup()


def parse_category_prefix_display(
    display: str,
    *,
    known: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    """Split 'SYNTH - file.wav' into (canonical category name, remainder filename)."""
    if " - " not in display:
        return None
    head, tail = display.split(" - ", 1)
    token = head.strip().upper()
    category = _PREFIX_TOKEN_TO_CATEGORY.get(token)
    if not category and known:
        for name in known:
            stripped = (name or "").strip()
            if not stripped:
                continue
            if stripped.upper() == token or category_badge_label(stripped).upper() == token:
                category = resolve_category_name(stripped)
                break
    if not category:
        return None
    return (category, tail)


def resolve_category_name(name: str) -> str:
    stripped = (name or "").strip()
    return CATEGORY_NAME_ALIASES.get(stripped, stripped)


def default_category_color(name: str) -> str:
    stripped = (name or "").strip()
    if stripped in DEFAULT_CATEGORY_COLORS:
        return DEFAULT_CATEGORY_COLORS[stripped]
    canonical = resolve_category_name(stripped)
    return DEFAULT_CATEGORY_COLORS.get(canonical, CATEGORY_PALETTE_COLORS[0])


def next_unused_category_color(categories: list) -> str:
    """First palette swatch not already used by any category in the list."""
    used: set[str] = set()
    for cat in categories or []:
        if not isinstance(cat, dict):
            continue
        raw = (cat.get("color") or "").strip()
        if not raw:
            raw = default_category_color(cat.get("name", ""))
        used.add(raw.lower())
    for color in CATEGORY_PALETTE_COLORS:
        if color.lower() not in used:
            return color
    # All swatches taken — cycle by count
    n = len(categories or [])
    return CATEGORY_PALETTE_COLORS[n % len(CATEGORY_PALETTE_COLORS)]


def next_unused_palette_color(used_colors: list[str] | tuple[str, ...] | set[str]) -> str:
    """First palette swatch not already used; if all are taken, cycle by count."""
    used_lower = {(c or "").strip().lower() for c in used_colors if (c or "").strip()}
    for color in CATEGORY_PALETTE_COLORS:
        if color.lower() not in used_lower:
            return color
    n = len(CATEGORY_PALETTE_COLORS)
    return CATEGORY_PALETTE_COLORS[len(used_lower) % n] if n else "#EF4444"


def category_color(name: str, stored: str = "", *, override: bool = False) -> str:
    """Return the display color for a category — defaults are tied to the category name."""
    default = default_category_color(name)
    if override and stored:
        # Prefer canonical palette spelling; keep free-form picks as-is.
        return _PALETTE_LOWER.get(stored.lower(), stored)
    return default


def normalize_category_dict(cat: dict) -> None:
    name = (cat.get("name") or "").strip()
    # Accept legacy snake_case written by an older picker; canonicalize to camelCase.
    if cat.pop("color_override", None) and not cat.get("colorOverride"):
        cat["colorOverride"] = True
    if cat.get("colorOverride"):
        cat["color"] = category_color(name, cat.get("color", ""), override=True)
    else:
        cat["color"] = default_category_color(name)


def normalize_rules_category_colors(rules: list[Rule]) -> None:
    from track_renamer.engine.models import ConditionGroup, OpRule

    def walk(rule: Rule) -> None:
        if isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for cat in rule.params.get("categories", []):
                if isinstance(cat, dict):
                    normalize_category_dict(cat)
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)

    for rule in rules:
        walk(rule)


def applied_category_colors(rules: list[Rule]) -> dict[str, str]:
    """Resolve the effective badge palette from the currently applied rules."""
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    colors: dict[str, str] = {}

    def add_category(cat: CategoryRule) -> None:
        name = resolve_category_name(cat.name)
        colors[name] = category_color(
            name,
            cat.color,
            override=cat.color_override,
        )

    def walk(rule: Rule) -> None:
        if isinstance(rule, CategoryRule):
            add_category(rule)
        elif isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for raw in rule.params.get("categories", []):
                add_category(
                    CategoryRule.from_dict(raw) if isinstance(raw, dict) else raw
                )
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)
            for branch in rule.branches:
                walk(branch)

    for rule in rules:
        walk(rule)
    return colors


def list_category_rules(rules: list[Rule]) -> list:
    """Flatten Category Macro rows from the rule stack (first bundle wins).

    Returns ``CategoryRule`` instances in display order for preview menus.
    Disabled rows are included so callers can filter; empty-name rows are skipped.
    """
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    found: list[CategoryRule] = []

    def add_category(cat: CategoryRule) -> None:
        name = (cat.name or "").strip()
        if not name:
            return
        found.append(cat)

    def walk(rule: Rule) -> bool:
        """Return True once a categoryBundle (or CategoryRule) has been collected."""
        if isinstance(rule, CategoryRule):
            add_category(rule)
            return True
        if isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for raw in rule.params.get("categories", []):
                add_category(
                    CategoryRule.from_dict(raw) if isinstance(raw, dict) else raw
                )
            return True
        if isinstance(rule, ConditionGroup):
            for child in rule.children:
                if walk(child):
                    return True
            for branch in rule.branches:
                if walk(branch):
                    return True
        return False

    for rule in rules:
        if walk(rule):
            break
    return found


def affix_prefix_token(affix: str) -> str:
    """Extract the badge/prefix token from an affix like 'ELSE - '."""
    text = (affix or "").strip()
    if " - " in text:
        return text.split(" - ", 1)[0].strip()
    if text.endswith(" -"):
        return text[:-2].strip()
    if text.endswith("-"):
        return text[:-1].strip()
    return text


def category_name_from_affix(affix: str, current_name: str = "") -> str:
    """Derive a category name from PREFIX, preserving known names/casing when possible."""
    token = affix_prefix_token(affix)
    if not token:
        return (current_name or "").strip()
    current = (current_name or "").strip()
    if current:
        if current.upper() == token.upper():
            return current
        if category_badge_label(current).upper() == token.upper():
            return current
    known = _PREFIX_TOKEN_TO_CATEGORY.get(token.upper())
    if known:
        return known
    if token.isupper() and len(token) <= 3:
        return token
    return " ".join(
        part if part.isupper() and len(part) <= 3 else part.capitalize()
        for part in token.replace("_", " ").split()
    )


def sync_category_names_from_affix(rules: list[Rule]) -> bool:
    """Set each category name from its PREFIX token (e.g. ELSE - → Else / badge ELSE)."""
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    changed = False

    def update_dict(raw: dict) -> None:
        nonlocal changed
        new_name = category_name_from_affix(raw.get("affix", ""), raw.get("name", ""))
        if new_name and new_name != (raw.get("name") or ""):
            raw["name"] = new_name
            changed = True

    def update_category(cat: CategoryRule) -> None:
        nonlocal changed
        new_name = category_name_from_affix(cat.affix, cat.name)
        if new_name and new_name != cat.name:
            cat.name = new_name
            changed = True

    def walk(rule: Rule) -> None:
        if isinstance(rule, CategoryRule):
            update_category(rule)
        elif isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for raw in rule.params.get("categories", []):
                if isinstance(raw, dict):
                    update_dict(raw)
                else:
                    update_category(raw)
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)
            for branch in rule.branches:
                walk(branch)

    for rule in rules:
        walk(rule)
    return changed


def sort_rule_category_keywords(rules: list[Rule]) -> bool:
    """Alphabetize and de-duplicate every category keyword list in-place."""
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    changed = False

    def sorted_keywords(value: str) -> str:
        unique: dict[str, str] = {}
        for raw in value.split(","):
            keyword = raw.strip()
            if keyword:
                unique.setdefault(keyword.casefold(), keyword)
        return ", ".join(sorted(unique.values(), key=str.casefold))

    def update_category(cat: CategoryRule) -> None:
        nonlocal changed
        normalized = sorted_keywords(cat.keywords)
        if normalized != cat.keywords:
            cat.keywords = normalized
            changed = True

    def walk(rule: Rule) -> None:
        nonlocal changed
        if isinstance(rule, CategoryRule):
            update_category(rule)
        elif isinstance(rule, OpRule) and rule.op == "categoryBundle":
            categories = rule.params.get("categories", [])
            for index, raw in enumerate(categories):
                if isinstance(raw, dict):
                    normalized = sorted_keywords(raw.get("keywords", ""))
                    if normalized != raw.get("keywords", ""):
                        raw["keywords"] = normalized
                        changed = True
                else:
                    update_category(raw)
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)
            for branch in rule.branches:
                walk(branch)

    for rule in rules:
        walk(rule)
    return changed

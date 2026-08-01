"""Token resolution for rule text fields."""

from __future__ import annotations

import re
from typing import Any

from .models import Track

COUNTER_RE = re.compile(r"\{counter(?::(\d+))?\}", re.I)
INDEX_RE = re.compile(r"\{index\}", re.I)


def resolve_tokens(
    text: str,
    *,
    track: Track,
    original_name: str,
    current_name: str,
    index: int,
    counter: int,
    variables: dict[str, str] | None = None,
) -> str:
    if not text:
        return text

    variables = variables or {}
    mapping: dict[str, str] = {
        "original": original_name,
        "name": current_name,
        "bpm": track.bpm,
        "key": track.key,
        "group": track.group,
        "instrument": track.instrument,
        "category": track.category,
        "counter": str(counter),
        "index": str(index),
        "is_audio": "1" if track.is_audio else "0",
        "is_midi": "1" if track.is_midi else "0",
        "is_group": "1" if track.is_group else "0",
    }
    for key, value in variables.items():
        mapping[key.lower()] = value

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).lower()
        if key.startswith("counter:"):
            pad = int(key.split(":", 1)[1])
            return str(counter).zfill(pad)
        return mapping.get(key, match.group(0))

    return re.sub(r"\{([a-zA-Z0-9_:]+)\}", repl, text)

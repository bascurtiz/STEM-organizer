"""Short musical-key labels (model output) and display enharmonics."""

from __future__ import annotations

# Flat-preferring short keys — Camelot index 0–23 (matches MusicalKeyCNN).
CAMELOT_TO_SHORT_KEY: dict[int, str] = {
    0: "Abm",
    1: "Ebm",
    2: "Bbm",
    3: "Fm",
    4: "Cm",
    5: "Gm",
    6: "Dm",
    7: "Am",
    8: "Em",
    9: "Bm",
    10: "Gbm",
    11: "Dbm",
    12: "B",
    13: "Gb",
    14: "Db",
    15: "Ab",
    16: "Eb",
    17: "Bb",
    18: "F",
    19: "C",
    20: "G",
    21: "D",
    22: "A",
    23: "E",
}

# Display uses spaces around "/" for readable badges (Eb / D#).
SHORT_TO_DISPLAY: dict[str, str] = {
    "C": "C",
    "Am": "Am",
    "Db": "Db / C#",
    "Bbm": "Bbm / A#m",
    "D": "D",
    "Bm": "Bm",
    "Eb": "Eb / D#",
    "Cm": "Cm",
    "E": "E",
    "Dbm": "Dbm / C#m",
    "F": "F",
    "Dm": "Dm",
    "Gb": "Gb / F#",
    "Ebm": "Ebm / D#m",
    "G": "G",
    "Em": "Em",
    "Ab": "Ab / G#",
    "Fm": "Fm",
    "A": "A",
    "Gbm": "Gbm / F#m",
    "Bb": "Bb / A#",
    "Gm": "Gm",
    "B": "B",
    "Abm": "Abm / G#m",
}

KEY_DISPLAY_ORDER: tuple[str, ...] = tuple(SHORT_TO_DISPLAY.values())

SHORT_KEYS: tuple[str, ...] = tuple(SHORT_TO_DISPLAY.keys())


def short_to_display(short: str) -> str:
    s = (short or "").strip()
    return SHORT_TO_DISPLAY.get(s, s)


def display_to_short(label: str) -> str | None:
    """Map display or short (or sharp alias) → canonical short key."""
    text = (label or "").strip()
    if not text:
        return None
    compact = " / ".join(p.strip() for p in text.split("/")) if "/" in text else text
    if text in SHORT_TO_DISPLAY:
        return text
    for short, display in SHORT_TO_DISPLAY.items():
        if text == display or compact == display:
            return short
        if text == display.replace(" / ", "/"):
            return short
        parts = [p.strip() for p in display.split("/")]
        if text in parts:
            return short
    low = text.lower()
    for short, display in SHORT_TO_DISPLAY.items():
        if short.lower() == low or display.lower() == low:
            return short
        if any(p.strip().lower() == low for p in display.split("/")):
            return short
    return None

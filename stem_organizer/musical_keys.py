"""Musical key labels, chart order, and badge colors for STEM Organizer."""

from __future__ import annotations

# Short keys written to COMMENT / Initial key (TKEY / INITIALKEY).
# Display uses spaces around "/" for readable badges/charts (Eb / D#).
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

# Badge / chart fills (dark text on these backgrounds).
CHART_KEY_COLORS: dict[str, str] = {
    "C": "#EE83DA",
    "Am": "#F4B4E8",
    "Db / C#": "#88F24E",
    "Bbm / A#m": "#B8F794",
    "D": "#9FB7FF",
    "Bm": "#C5D3FF",
    "Eb / D#": "#FFA07C",
    "Cm": "#FEC6AF",
    "E": "#0BEBEB",
    "Dbm / C#m": "#6AF3F2",
    "F": "#FF80B4",
    "Dm": "#FFB3D2",
    "Gb / F#": "#3FED80",
    "Ebm / D#m": "#8BF4B2",
    "G": "#CC8FFF",
    "Em": "#E1BAFF",
    "Ab / G#": "#FFCB46",
    "Fm": "#FFDF90",
    "A": "#57D9F9",
    "Gbm / F#m": "#9AE8FB",
    "Bb / A#": "#FF8894",
    "Gm": "#FEB7BE",
    "B": "#0AEDCA",
    "Abm / G#m": "#6DF4DF",
}

# Also index short keys + compact "Eb/D#" aliases for lookups.
for _short, _disp in SHORT_TO_DISPLAY.items():
    CHART_KEY_COLORS.setdefault(_short, CHART_KEY_COLORS[_disp])
    compact = _disp.replace(" / ", "/")
    if compact != _disp:
        CHART_KEY_COLORS.setdefault(compact, CHART_KEY_COLORS[_disp])


def short_to_display(short: str) -> str:
    s = (short or "").strip()
    return SHORT_TO_DISPLAY.get(s, s)


def key_chart_label(display: str) -> str:
    """Chart/export label: stack enharmonics without slash (Abm\\nG#m)."""
    text = (display or "").strip()
    if " / " in text:
        return "\n".join(p.strip() for p in text.split(" / ") if p.strip())
    if "/" in text and " / " not in text:
        # Compact Db/C# → same stacking
        return "\n".join(p.strip() for p in text.split("/") if p.strip())
    return text


def normalize_key_short(raw: str) -> str:
    """Accept short, display, or sharp alias → canonical short, else ''."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Normalize spaced / compact slash forms.
    compact = " / ".join(p.strip() for p in text.split("/")) if "/" in text else text
    if text in SHORT_TO_DISPLAY:
        return text
    for short, display in SHORT_TO_DISPLAY.items():
        if text == display or compact == display:
            return short
        if text == display.replace(" / ", "/") or compact.replace(" / ", "/") == display.replace(
            " / ", "/"
        ):
            return short
        if text in [p.strip() for p in display.split("/")]:
            return short
    low = text.lower()
    compact_low = compact.lower()
    for short, display in SHORT_TO_DISPLAY.items():
        if short.lower() == low or display.lower() == low or display.lower() == compact_low:
            return short
        if any(p.strip().lower() == low for p in display.split("/")):
            return short
    return ""

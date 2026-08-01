"""Default category matrix and demo data ported from the original extension."""

from __future__ import annotations

from pathlib import Path

from track_renamer.category_palette import DEFAULT_CATEGORY_COLORS, default_category_color

from .models import CategoryRule, Condition, ConditionGroup, OpRule, Rule, Track

# Default macro keyword lists per category.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Bass": [
        "Sub",
        "Electric",
        "Acid",
        "Hoover",
        "Saw",
        "Wobble",
        "Pulse",
        "Bass Guitar",
        "Synth Bass",
        "Upright Bass",
        "bass",
        "reese",
        "growl",
        "bassline",
        "subbass",
        "sub-bass",
        "303",
        "Sine",
        "Square",
    ],
    "Drums": [
        "Kick",
        "Bassdrum",
        "Basedrum",
        "Snare",
        "Snareroll",
        "Hat",
        "ClosedHat",
        "Clap",
        "Tom",
        "Cymbal",
        "Break",
        "Fill",
        "Acoustic",
        "808",
        "909",
        "Brushes",
        "Drum Fill",
        "Full Drum Loop",
        "Rim Shot",
        "Stripped Drum Loop",
        "Top",
        "drum",
        "drums",
        "hi-hat",
        "hihat",
        "ohh",
        "chh",
        "ride",
        "crash",
        "bd",
        "sd",
        "kik",
        "handclap",
        "breakbeat",
        "drumloop",
        "drumtop",
        "linn",
        "rim",
        "rimshot",
    ],
    "Percussion": [
        "Shaker",
        "Groove",
        "Bongo",
        "Woodblock",
        "Djembe",
        "Conga",
        "Tambourine",
        "Cowbell",
        "Timbale",
        "Agogo",
        "Bata",
        "Berimbau",
        "Cabassa",
        "Cajon",
        "Caxixi",
        "Chimes",
        "Clave",
        "Click",
        "Cuica",
        "Darbuka",
        "Dholok",
        "Finger Cymbal",
        "Frame Drum",
        "Glass",
        "Gong",
        "Guiro",
        "Hang Drum",
        "Metal",
        "Mixed Percussion",
        "Percloop",
        "Pot",
        "Rainstick",
        "Rattle",
        "Snap",
        "Surdo",
        "Sweet Bell",
        "Talking Drum",
        "Tabla",
        "Taiko",
        "Tank Drum",
        "Timpani",
        "Triangle",
        "Udu",
        "Urumi",
        "Whistle",
        "Wood",
        "Percussion",
        "perc",
        "maracas",
    ],
    "Synth": [
        "Lead",
        "Pad",
        "Arp",
        "Stab",
        "Pluck",
        "Analog",
        "Synth Melody",
        "String Pad",
        "synth",
        "arpeggio",
        "square",
        "supersaw",
        "ambient",
        "wash",
        "swell",
        "Poly",
    ],
    "Wind": [
        "Wind",
        "Saxophone",
        "Trumpet",
        "Trombone",
        "Flute",
        "Harmonica",
        "Bassoon",
        "Brass",
        "Clarinet",
        "Didgeridoo",
        "Horn",
        "Oboe",
        "Panpipe",
        "Tuba",
    ],
    "Keys": [
        "Amapiano",
        "Piano",
        "Electric Piano",
        "Wurlitzer",
        "Organ",
        "Clavinet",
        "Chord",
        "Keys",
        "Keys Melody",
        "Classical",
        "Accordion",
        "Clavichord",
        "Harpsichord",
        "Melodica",
        "Thumb Piano",
        "Rhodes",
    ],
    "Guitar": [
        "Clean",
        "Distorted",
        "Lead",
        "Guitar",
        "GTR",
        "GTRS",
        "Guitar Melody",
        "Riff",
        "Rhythm",
        "Acoustic Guitar",
        "Banjo",
        "Bouzouki",
        "Classical Guitar",
        "Cumbus",
        "Cura",
        "Electric Guitar",
        "Harp",
    ],
    "FX": [
        "FX",
        "Noise",
        "Riser",
        "Downer",
        "Sweep",
        "Impact",
        "Atmosphere",
        "Texture",
        "Reverse",
        "Field Recording",
        "Ambience",
        "Boom",
        "Downshifter",
        "Drone",
        "Foley",
        "Found Sound",
        "Gate",
        "Glitched",
        "Material",
        "Mechanical",
        "Nature",
        "Sci-fi",
        "Scratch",
        "Siren",
        "Transition",
        "Vinyl",
        "Whoosh",
        "effect",
        "downlifter",
        "uplifter",
        "Sub Drop",
    ],
    "Strings": [
        "Violin",
        "Cello",
        "Viola",
        "Orchestral",
        "Staccato",
        "Strings",
        "Strings Melody",
        "Double Bass",
        "String Ensemble",
        "strings",
    ],
    "Vocals": [
        "Female Vocal",
        "Male Vocal",
        "Vocal FX",
        "Spoken Word",
        "Vocoder",
        "Vocal Phrase",
        "Scream",
        "Vocal Shout",
        "Whisper Vocal",
        "Whispers",
        "Dialogue",
        "Acapella",
        "Adlib",
        "Backing",
        "Beatbox",
        "Choir",
        "Chopped",
        "Female",
        "Hook",
        "MC",
        "Male",
        "Rap",
        "Robot",
        "Shout",
        "Spoken",
        "Verse",
        "vocal",
        "vox",
        "voice",
        "chant",
        "chorus",
        "lyric",
        "sing",
    ],
    "Mallet": [
        "Bell",
        "Glockenspiel",
        "Kalimba",
        "Mallet",
        "Marimba",
        "Vibraphone",
        "Xylophone",
    ],
    "Orchestra": [
        "Kopuz",
        "Koto",
        "Mandolin",
        "Oud",
        "Qanun",
        "Saz",
        "Shamisen",
        "Sitar",
        "Sitouki",
        "Swarmandal",
        "Tamboura",
        "Tar",
        "Ukelele",
        "Zither",
    ],
}

TITLE_CASE_ACRONYMS = "FX, KSHMR, DnB, EDM, ID, USA, UK, OG, BPM, DJ, LFO, MIDI, NYC, LA, DAW"

# OpenMIC-2018 (PaSST) instrument label → Category Macro row.
OPENMIC_TO_CATEGORY: dict[str, str] = {
    "accordion": "Keys",
    "banjo": "Guitar",
    "bass": "Bass",
    "cello": "Strings",
    "clarinet": "Wind",
    "cymbals": "Percussion",
    "drums": "Drums",
    "flute": "Wind",
    "guitar": "Guitar",
    "mallet_percussion": "Mallet",
    "mandolin": "Guitar",
    "organ": "Keys",
    "piano": "Keys",
    "saxophone": "Wind",
    "synthesizer": "Synth",
    "trombone": "Wind",
    "trumpet": "Wind",
    "ukulele": "Guitar",
    "violin": "Strings",
    "voice": "Vocals",
}

DEFAULT_CATEGORY_SOURCE = "filename"


def map_instrument_to_category(label: str) -> str:
    """OpenMIC class → Category Macro name, or ''."""
    return OPENMIC_TO_CATEGORY.get((label or "").strip().lower(), "")


def _merge_keywords(*groups: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw in group:
            keyword = raw.strip()
            if not keyword:
                continue
            key = keyword.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(keyword)
    return ", ".join(sorted(merged, key=str.casefold))


def _build_default_categories() -> list[dict[str, str]]:
    order = [
        "Bass",
        "Drums",
        "Percussion",
        "Synth",
        "Wind",
        "Keys",
        "Guitar",
        "FX",
        "Strings",
        "Vocals",
        "Mallet",
        "Orchestra",
    ]
    return [
        {
            "name": name,
            "keywords": _merge_keywords(_CATEGORY_KEYWORDS.get(name, [])),
            "color": DEFAULT_CATEGORY_COLORS.get(name, ""),
        }
        for name in order
    ]


DEFAULT_CATEGORIES: list[dict[str, str]] = _build_default_categories()


def make_category_rules() -> list[CategoryRule]:
    rules: list[CategoryRule] = []
    for item in DEFAULT_CATEGORIES:
        name = item["name"].strip()
        rules.append(
            CategoryRule(
                name=name,
                keywords=item["keywords"],
                affix=f"{name.upper()} - ",
                color=default_category_color(name),
            )
        )
    return rules


def make_category_bundle() -> OpRule:
    return OpRule(
        op="categoryBundle",
        params={
            "source": DEFAULT_CATEGORY_SOURCE,
            "categories": [c.to_dict() for c in make_category_rules()],
        },
    )


def make_default_rules() -> list[Rule]:
    """Default template."""
    return [
        OpRule(op="stripLeadingNumberPrefix"),
        OpRule(op="stripLeadingDashes"),
        OpRule(op="collapseWhitespace"),
        OpRule(op="trim"),
        make_category_bundle(),
    ]


def make_demo_tracks() -> list[Track]:
    """Sample filenames matching the original screenshot preview."""
    demo_names = [
        ("1-KSHMR Acoustic Drum Loop 120BPM D", ".wav", "audio"),
        ("2-KSHMR Bass Loop 128BPM F#", ".wav", "audio"),
        ("3-KSHMR Synth Loop 128BPM Am", ".wav", "audio"),
        ("4-KSHMR Pad Loop 90BPM C", ".wav", "audio"),
        ("5-KSHMR Vocal Loop 128BPM G", ".wav", "audio"),
        ("6-KSHMR FX Loop 128BPM", ".wav", "audio"),
        ("7-Audio", ".wav", "audio"),
        ("8-Audio", ".wav", "audio"),
        ("9-Audio", ".wav", "audio"),
        ("10-Audio", ".wav", "audio"),
        ("11-Audio", ".wav", "audio"),
        ("12-Audio", ".wav", "audio"),
        ("13-Audio", ".wav", "audio"),
        ("14-Audio", ".wav", "audio"),
        ("15-Audio", ".wav", "audio"),
        ("16-Audio", ".wav", "audio"),
        ("17-Group", ".wav", "group"),
        ("18-MIDI", ".mid", "midi"),
        ("19-MIDI", ".mid", "midi"),
        ("20-Audio", ".wav", "audio"),
        ("21-Audio", ".wav", "audio"),
        ("22-Audio", ".wav", "audio"),
        ("23-Audio", ".wav", "audio"),
        ("24-Audio", ".wav", "audio"),
        ("25-Audio", ".wav", "audio"),
        ("26-Audio", ".wav", "audio"),
        ("27-Audio", ".wav", "audio"),
        ("28-Audio", ".wav", "audio"),
        ("29-Audio", ".wav", "audio"),
    ]
    tracks: list[Track] = []
    group_id = "demo-folder/17-Group"
    for index, (stem, ext, kind) in enumerate(demo_names, start=1):
        parent = group_id if stem in ("18-MIDI", "19-MIDI", "20-Audio") else None
        depth = 1 if parent else 0
        fake_path = Path("demo-folder") / f"{stem}{ext}"
        tracks.append(
            Track(
                id=str(fake_path),
                name=stem,
                track_type=kind,  # type: ignore[arg-type]
                parent_id=parent,
                depth=depth,
                file_path=fake_path,
                extension=ext,
                relative_path=f"{stem}{ext}",
                group="17-Group" if parent else "",
            )
        )
    return tracks


RULE_CATALOG: list[dict] = [
    {"label": "If condition…", "kind": "conditionGroup"},
    {"label": "Add text at the beginning", "kind": "op", "op": "addTextAtBeginning"},
    {"label": "Add text at the end", "kind": "op", "op": "addTextAtEnd"},
    {"label": "Replace text", "kind": "op", "op": "replaceText"},
    {"label": "Remove text", "kind": "op", "op": "removeText"},
    {"label": "Remove prefix numbers", "kind": "op", "op": "stripLeadingNumberPrefix"},
    {"label": "Remove leading dashes", "kind": "op", "op": "stripLeadingDashes"},
    {"label": "Remove a range of characters", "kind": "op", "op": "removeCharRange"},
    {"label": "Category Macro", "kind": "op", "op": "categoryBundle"},
    {"label": "Trim", "kind": "op", "op": "trim"},
    {"label": "Title Case", "kind": "op", "op": "titleCase"},
    {"label": "Collapse whitespace", "kind": "op", "op": "collapseWhitespace"},
]

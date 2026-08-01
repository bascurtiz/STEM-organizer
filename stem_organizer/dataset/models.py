"""Dataset overview models + demo inventory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


VOCAL_TYPES = ("Singing", "Speech", "Rapping", "Humming", "Choir")
GENDERS = ("female", "male")
REVERBS = ("dry", "wet")
# Demo charts use the same 15 top-level genres as MAEST (Genre tab output).
DEMO_STYLES_BY_GENRE: dict[str, tuple[str, ...]] = {
    "Blues": ("Chicago Blues", "Delta Blues", "Electric Blues", "Texas Blues"),
    "Brass & Military": ("Marches", "Brass Band", "Military"),
    "Children's": ("Educational", "Lullaby", "Story"),
    "Classical": ("Baroque", "Romantic", "Modern Classical", "Opera"),
    "Electronic": ("House", "Techno", "Trance", "Drum and Bass", "Dubstep"),
    "Folk, World, & Country": (
        "Indie Folk",
        "Americana",
        "Country",
        "Afrobeat",
        "Celtic",
    ),
    "Funk / Soul": ("Contemporary R&B", "Neo Soul", "P-Funk", "Motown"),
    "Hip Hop": ("Trap", "Boom Bap", "UK Drill", "Lo-Fi"),
    "Jazz": ("Bebop", "Fusion", "Smooth Jazz", "Nu Jazz"),
    "Latin": ("Reggaeton", "Salsa", "Latin Pop", "Cumbia"),
    "Non-Music": ("Spoken Word", "Field Recording", "Interview"),
    "Pop": ("Dance Pop", "Synth Pop", "Indie Pop", "Electropop"),
    "Reggae": ("Roots", "Dancehall", "Dub", "Ska"),
    "Rock": ("Indie Rock", "Alternative", "Punk", "Progressive", "Hard Rock"),
    "Stage & Screen": ("Film Score", "Game Score", "Musical", "TV Soundtrack"),
}

# MAEST Discogs519 top-level genres (Genre tab output).
MAEST_GENRES = (
    "Blues",
    "Brass & Military",
    "Children's",
    "Classical",
    "Electronic",
    "Folk, World, & Country",
    "Funk / Soul",
    "Hip Hop",
    "Jazz",
    "Latin",
    "Non-Music",
    "Pop",
    "Reggae",
    "Rock",
    "Stage & Screen",
)

# Fixed colors for MAEST genres (Genre + Style charts + export).
# Saturated for dark UI — bumped from bright-bg palette.
# Saturated to match theme-colors.png / STEM_COLORS intensity (red→purple + neutrals).
MAEST_GENRE_COLORS: dict[str, str] = {
    "Blues": "#4281E8",
    "Brass & Military": "#497715",
    "Children's": "#F5CC29",
    "Classical": "#9AA0B2",
    "Electronic": "#21B3A3",
    "Folk, World, & Country": "#44BAEF",
    "Funk / Soul": "#EC9F1D",
    "Hip Hop": "#ED7623",
    "Jazz": "#FAEA33",
    "Latin": "#E04E97",
    "Non-Music": "#000000",
    "Pop": "#A45AEA",
    "Reggae": "#17BB72",
    "Rock": "#E14949",
    "Stage & Screen": "#636B78",
}


@dataclass
class ClassBucket:
    """One discrete class (e.g. Singing, female, Rock)."""

    name: str
    count: int = 0
    bytes: int = 0
    duration_sec: float = 0.0

    @property
    def avg_duration_sec(self) -> float:
        return self.duration_sec / self.count if self.count else 0.0


@dataclass
class ContinuousStats:
    """Continuous metric (SI-SDR, duration)."""

    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return float(sum(self.values)) if self.values else 0.0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def median(self) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        mid = len(s) // 2
        if len(s) % 2:
            return float(s[mid])
        return float(s[mid - 1] + s[mid]) / 2.0


@dataclass
class RoleCounts:
    instrumental: int = 0
    vocal: int = 0
    samples: int = 0
    pair_folders: int = 0

    @property
    def total_units(self) -> int:
        return self.instrumental + self.vocal + self.samples


@dataclass
class OverviewStats:
    """Aggregated library snapshot for Dataset overview charts."""

    demo: bool = True
    roles: RoleCounts = field(default_factory=RoleCounts)
    total_files: int = 0
    total_bytes: int = 0
    duration: ContinuousStats = field(default_factory=ContinuousStats)
    sdr: ContinuousStats = field(default_factory=ContinuousStats)
    vocal_type: dict[str, ClassBucket] = field(default_factory=dict)
    gender: dict[str, ClassBucket] = field(default_factory=dict)
    reverb: dict[str, ClassBucket] = field(default_factory=dict)
    genre: dict[str, ClassBucket] = field(default_factory=dict)
    # genre → style → bucket (styles as shares within that genre)
    styles_by_genre: dict[str, dict[str, ClassBucket]] = field(default_factory=dict)
    style_top: dict[str, ClassBucket] = field(default_factory=dict)
    style_other_count: int = 0
    compression: dict[str, ClassBucket] = field(default_factory=dict)
    key: dict[str, ClassBucket] = field(default_factory=dict)

    def class_pct(self, bucket: ClassBucket, *, of: Optional[int] = None) -> float:
        base = of if of is not None else self.total_files
        return (100.0 * bucket.count / base) if base else 0.0


def style_genres_from_stats(stats: OverviewStats) -> list[tuple[str, int]]:
    """Tagged genres ordered by file count desc (most at top)."""
    tagged = {name: b.count for name, b in stats.genre.items() if b.count > 0}
    if not tagged:
        return []
    return sorted(
        tagged.items(),
        key=lambda x: (-x[1], x[0].lower()),
    )


def make_demo_overview() -> OverviewStats:
    """Plausible demo inventory so users see charts before a real scan."""
    import random

    rng = random.Random(42)
    stats = OverviewStats(demo=True)

    # Role imbalance (pairs contribute equally to instrumental + vocal)
    pair_n = 12_400
    inst_extra = 28_000
    voc_extra = 41_200
    samples_n = 4_800
    stats.roles = RoleCounts(
        instrumental=inst_extra + pair_n,
        vocal=voc_extra + pair_n,
        samples=samples_n,
        pair_folders=pair_n,
    )
    stats.total_files = stats.roles.total_units

    # Duration / size (synthetic)
    n_files = stats.total_files
    for _ in range(2_000):  # subsample for continuous stats speed
        dur = max(5.0, rng.gauss(185.0, 55.0))
        stats.duration.values.append(dur)
    stats.total_bytes = int(1.85 * (1024**4))  # ~1.85 TB

    # SDR demo — histogram 30–55 dB (last bin is 50–55, exclusive upper → cap < 55)
    for _ in range(3_000):
        stats.sdr.values.append(max(30.0, min(54.99, rng.gauss(40.0, 5.0))))

    def _fill(names, weights, store: dict[str, ClassBucket], *, scale: int) -> None:
        total_w = sum(weights)
        for name, w in zip(names, weights):
            c = int(round(scale * w / total_w))
            b = ClassBucket(
                name=name,
                count=c,
                bytes=c * int(rng.randint(8, 22) * 1024 * 1024),
                duration_sec=c * max(30.0, rng.gauss(160.0, 40.0)),
            )
            store[name] = b

    _fill(
        VOCAL_TYPES,
        (58, 18, 12, 5, 7),
        stats.vocal_type,
        scale=voc_extra + pair_n,
    )
    _fill(GENDERS, (54, 46), stats.gender, scale=voc_extra + pair_n)
    _fill(REVERBS, (42, 58), stats.reverb, scale=voc_extra + pair_n)
    # Stem-library skew: Rock / Electronic / Hip Hop / Pop dominate;
    # niche genres stay present but much smaller.
    _fill(
        MAEST_GENRES,
        (
            4,   # Blues
            1,   # Brass & Military
            1,   # Children's
            5,   # Classical
            22,  # Electronic
            6,   # Folk, World, & Country
            5,   # Funk / Soul
            18,  # Hip Hop
            4,   # Jazz
            5,   # Latin
            1,   # Non-Music
            16,  # Pop
            3,   # Reggae
            24,  # Rock
            2,   # Stage & Screen
        ),
        stats.genre,
        scale=n_files,
    )

    # Nested styles under each genre (percentages sum ~100% within genre).
    flat_styles: dict[str, ClassBucket] = {}
    for genre_name, styles in DEMO_STYLES_BY_GENRE.items():
        g_bucket = stats.genre.get(genre_name)
        g_count = g_bucket.count if g_bucket else 0
        if g_count <= 0 or not styles:
            continue
        weights = [rng.randint(4, 18) for _ in styles]
        total_w = sum(weights) or 1
        nested: dict[str, ClassBucket] = {}
        assigned = 0
        for i, (style_name, w) in enumerate(zip(styles, weights)):
            if i == len(styles) - 1:
                c = max(0, g_count - assigned)
            else:
                c = int(round(g_count * w / total_w))
                assigned += c
            b = ClassBucket(
                name=style_name,
                count=c,
                bytes=c * int(rng.randint(8, 22) * 1024 * 1024),
                duration_sec=c * max(30.0, rng.gauss(160.0, 40.0)),
            )
            nested[style_name] = b
            prev = flat_styles.get(style_name)
            if prev is None:
                flat_styles[style_name] = ClassBucket(
                    name=style_name,
                    count=c,
                    bytes=b.bytes,
                    duration_sec=b.duration_sec,
                )
            else:
                prev.count += c
                prev.bytes += b.bytes
                prev.duration_sec += b.duration_sec
        stats.styles_by_genre[genre_name] = nested

    # Keep a flat top-N for any legacy callers.
    ranked = sorted(flat_styles.values(), key=lambda b: -b.count)
    stats.style_top = {b.name: b for b in ranked[:20]}
    stats.style_other_count = max(
        0, n_files - sum(b.count for b in flat_styles.values())
    )

    _fill(
        ("lossless", "lossy"),
        (78, 22),
        stats.compression,
        scale=n_files,
    )
    from ..musical_keys import KEY_DISPLAY_ORDER

    # Spread demo keys across the 24-key chart order.
    key_weights = [max(1, 12 - (i % 12)) for i in range(len(KEY_DISPLAY_ORDER))]
    _fill(KEY_DISPLAY_ORDER, key_weights, stats.key, scale=n_files)
    return stats

"""Build STEM-organizer-source.zip for manual repo upload.

Structure mirrors previous uploads: top-level ``STEM-organizer\\`` folder.
Excludes: model weights, venv/dist/build, bundled binaries, runtime state,
smoke scratch, log files, and the files the user asked to drop
(theme-colors.png, genre_colors.png, changelog.txt, AGENTS.md).
"""
from pathlib import Path
import zipfile

ROOT = Path(".")
OUT = Path("STEM-organizer-source.zip")
PREFIX = "STEM-organizer"

# --- directories never shipped -------------------------------------------------
SKIP_DIRS = {
    ".build-venv", "build", "dist", "__pycache__",
    "ffmpeg", "flac", "mp3val",           # bundled binaries
    ".freebuff", ".zcode",
}

# --- files never shipped --------------------------------------------------------
SKIP_FILES = {
    "settings.json",          # machine state
    "python-version.txt",
    "theme-colors.png",       # user-requested exclusion
    "genre_colors.png",       # user-requested exclusion
    "changelog.txt",          # user-requested exclusion
    "AGENTS.md",              # user-requested exclusion (agents.md)
}

# --- model weights (already in the models release) -----------------------------
WEIGHT_EXTS = {".onnx", ".pth", ".pt", ".pb", ".safetensors", ".bin"}

# keep these models/ sidecars (they belong to the source, not the weights)
MODEL_SIDECARS = {
    "models/class_labels_indices.csv",
    "models/maest_discogs519.id2label.json",
    "models/vocal_reverb.config.json",
    "models/htdemucs.batch.onnx.srcmeta",
}


def excluded(rel: str, path: Path) -> bool:
    parts = rel.split("/")
    if any(p in SKIP_DIRS for p in parts):
        return True
    if path.suffix.lower() == ".log":
        return True
    if path.suffix.lower() == ".pyc":
        return True
    if rel in SKIP_FILES:
        return True
    if rel in MODEL_SIDECARS:
        return False
    if rel.startswith("models/") and path.suffix.lower() in WEIGHT_EXTS:
        return True
    return False


count = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.as_posix()
        if rel.startswith("STEM-organizer-source.zip"):
            continue
        if excluded(rel, path):
            continue
        zf.write(path, f"{PREFIX}/{rel}")
        count += 1

print(f"wrote {OUT} — {count} files")

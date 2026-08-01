#!/usr/bin/env python
"""Regenerate the SHA-256 + size lines for the 8 downloadable model weights.

When a model weight changes (new export, re-trim, different source), the
`Hash:` and `ExternalSize:` values in stem_organizer.iss MUST be updated or
Inno's install-time verification will reject the download. This helper reads
the weights from the built onedir (or a source dir) and prints the exact
[Files]-entry fragments to paste, plus can patch the .iss in place.

Usage (from repo root, any venv with stdlib only):
  python _onnx_spike/hash_weights.py                    # print the 8 lines
  python _onnx_spike/hash_weights.py --src dist/STEM-organizer   # explicit dir
  python _onnx_spike/hash_weights.py --patch            # rewrite stem_organizer.iss
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (relative path under src dir, DestName, DestDir as written in the .iss)
WEIGHTS = [
    (r"models/htdemucs.onnx",                                  "htdemucs.onnx",                  r"{app}\models"),
    (r"panns_tagger/models/cnn14.onnx",                        "cnn14.onnx",                     r"{app}\panns_tagger\models"),
    (r"instrument_tagger/models/passt_openmic.onnx",           "passt_openmic.onnx",             r"{app}\instrument_tagger\models"),
    (r"genre_gender_tagger/models/maest_discogs519.onnx",      "maest_discogs519.onnx",          r"{app}\genre_gender_tagger\models"),
    (r"genre_gender_tagger/models/discogs-effnet-bsdynamic-1.onnx", "discogs-effnet-bsdynamic-1.onnx", r"{app}\genre_gender_tagger\models"),
    (r"genre_gender_tagger/models/gender-discogs-effnet-1.onnx","gender-discogs-effnet-1.onnx",  r"{app}\genre_gender_tagger\models"),
    (r"genre_gender_tagger/models/vocal_reverb.onnx",          "vocal_reverb.onnx",              r"{app}\genre_gender_tagger\models"),
    (r"key_tagger/checkpoints/nf50-q05-221125.onnx",           "nf50-q05-221125.onnx",           r"{app}\key_tagger\checkpoints"),
]


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def compute_all(src: Path) -> list[dict]:
    out = []
    for rel, name, destdir in WEIGHTS:
        p = src / rel
        if not p.is_file():
            print(f"  MISSING: {p}", file=sys.stderr)
            out.append(dict(name=name, missing=True))
            continue
        digest, size = sha256_and_size(p)
        out.append(dict(name=name, destname=name, destdir=destdir,
                        hash=digest, size=size, rel=rel))
    return out


def print_lines(items: list[dict]) -> None:
    base = "{#ModelsBaseUrl}"
    for it in items:
        if it.get("missing"):
            print(f"# MISSING: {it['name']}")
            continue
        print(f'Source: "{base}/{it["destname"]}"; Components: models; '
              f'DestName: "{it["destname"]}"; DestDir: "{it["destdir"]}"; '
              f'ExternalSize: {it["size"]}; Hash: "{it["hash"]}"; '
              f'Flags: external download ignoreversion uninsneveruninstall')


def patch_iss(items: list[dict]) -> None:
    """Rewrite the weight [Files] entries in stem_organizer.iss in place."""
    iss = ROOT / "stem_organizer.iss"
    text = iss.read_text(encoding="utf-8")
    # Match each weight line by DestName (stable per weight) and replace
    # ExternalSize + Hash in place, leaving the rest of the line intact.
    n = 0
    for it in items:
        if it.get("missing"):
            continue
        # Match the whole Source line for this DestName, then rebuild it.
        pat = re.compile(
            r'^Source: "\{#ModelsBaseUrl\}/' + re.escape(it["destname"]) +
            r'".*$',
            re.MULTILINE,
        )
        new = (f'Source: "{base}/{it["destname"]}"; Components: models; '
               f'DestName: "{it["destname"]}"; DestDir: "{it["destdir"]}"; '
               f'ExternalSize: {it["size"]}; Hash: "{it["hash"]}"; '
               f'Flags: external download ignoreversion uninsneveruninstall')
        text, k = pat.subn(new, text)
        if k != 1:
            print(f"  WARN: {it['destname']}: matched {k} lines (expected 1)", file=sys.stderr)
        n += k
    iss.write_text(text, encoding="utf-8")
    print(f"Patched {n} weight entries in {iss}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "dist" / "STEM-organizer"),
                    help="source dir containing the weight files (default: dist/STEM-organizer)")
    ap.add_argument("--patch", action="store_true",
                    help="rewrite the weight lines in stem_organizer.iss")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_dir():
        print(f"ERROR: source dir not found: {src}", file=sys.stderr)
        return 2
    print(f"# hashing weights under: {src}")
    items = compute_all(src)
    print()
    if a.patch:
        patch_iss(items)
    else:
        print_lines(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

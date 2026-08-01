"""One-shot: vendor HF transformers.audio_utils helpers needed by MAEST FE."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import transformers.audio_utils as mod

OUT = Path(__file__).resolve().parents[1] / "genre_gender_tagger" / "_maest_audio_utils.py"

PUBLIC = ("window_function", "mel_filter_bank", "spectrogram")


def main() -> None:
    src_path = Path(inspect.getsourcefile(mod))
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    class Refs(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: set[str] = set()

        def visit_Name(self, node: ast.Name) -> None:
            self.names.add(node.id)

    def deps_of(fname: str, seen: set[str] | None = None) -> set[str]:
        seen = seen if seen is not None else set()
        if fname in seen or fname not in funcs:
            return seen
        seen.add(fname)
        r = Refs()
        r.visit(funcs[fname])
        for n in r.names:
            if n in funcs:
                deps_of(n, seen)
        return seen

    needed: set[str] = set()
    for w in PUBLIC:
        needed |= deps_of(w)

    # Preserve source order
    ordered = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in needed]

    parts = [
        '"""Vendored subset of transformers.audio_utils (Apache-2.0).',
        "",
        "Only the helpers used by MAESTFeatureExtractor._extract_fbank_features.",
        f"Source: transformers {getattr(__import__('transformers'), '__version__', '?')}",
        '"""',
        "from __future__ import annotations",
        "",
        "import math",
        "from typing import Optional",
        "",
        "import numpy as np",
        "",
    ]
    for name in ordered:
        parts.append(inspect.getsource(getattr(mod, name)))
        parts.append("")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) funcs={ordered}")


if __name__ == "__main__":
    main()

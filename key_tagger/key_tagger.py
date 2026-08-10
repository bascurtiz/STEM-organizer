#!/usr/bin/env python3
"""CLI: detect musical key and write COMMENT or Initial key (TKEY / INITIALKEY).

Emits STEM Organizer LOG lines:
  === file.flac ===
  Db/C# 87%

Stdout markers:
  __progress__  <pct>  <eta|->  <n>  <total>
  JSON result lines (optional, for workers)
  DONE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as ``python key_tagger.py`` from this folder.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_AUDIO_EXTS = frozenset(
    {
        ".flac",
        ".wav",
        ".mp3",
        ".ogg",
        ".opus",
        ".m4a",
        ".mp4",
        ".aac",
        ".aif",
        ".aiff",
        ".ape",
    }
)


def _default_checkpoint() -> Path:
    """Prefer ONNX weight; fall back to legacy .pt name for resolve logic.
    Single model source: the root models/ folder."""
    root = _HERE.parent / "models"
    onnx = root / "nf50-q05-221125.onnx"
    if onnx.is_file():
        return onnx
    return root / "nf50-q05-221125.pt"


def _read_files_from(path: Path) -> list[Path]:
    out: list[Path] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS:
            out.append(p)
    return out


def _read_existing_key(path: Path, field: str) -> str:
    try:
        from stem_organizer.meta_tags import read_custom_tag, read_key_tag
        from stem_organizer.musical_keys import normalize_key_short

        if field == "key":
            raw = read_key_tag(path)
        else:
            raw = read_custom_tag(path, "comment")
        return normalize_key_short(raw) or ""
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return ""
    try:
        if field == "key":
            # EasyID3 often maps initialkey → TKEY when registered.
            audio = MutagenFile(str(path), easy=True)
            if audio is not None:
                for k in ("initialkey", "key"):
                    vals = audio.get(k)
                    if vals:
                        return str(vals[0]).strip()
            # Direct TKEY / Vorbis INITIALKEY
            audio = MutagenFile(str(path), easy=False)
            if audio is None:
                return ""
            tags = getattr(audio, "tags", None) or audio
            if tags is None:
                return ""
            if hasattr(tags, "get"):
                frame = tags.get("TKEY")
                if frame is not None:
                    text = getattr(frame, "text", None)
                    if text:
                        return str(text[0]).strip()
                for k in ("INITIALKEY", "initialkey", "KEY", "key"):
                    if k in tags:
                        v = tags.get(k)
                        if v:
                            return str(v[0] if isinstance(v, list) else v).strip()
            return ""
        audio = MutagenFile(str(path), easy=True)
        if audio is None:
            return ""
        vals = audio.get("comment")
        if vals:
            return str(vals[0]).strip()
    except Exception:
        return ""
    return ""


def _write_key_metadata(path: Path, short_key: str, *, field: str) -> None:
    """Write short key to COMMENT or Initial key (TKEY / INITIALKEY)."""
    try:
        from stem_organizer.meta_tags import write_comment_tag, write_key_tag

        ok = (
            write_key_tag(path, short_key)
            if field == "key"
            else write_comment_tag(path, short_key)
        )
        if not ok:
            raise OSError("tag write failed")
        return
    except ImportError:
        pass
    except OSError:
        raise
    except Exception as exc:
        raise OSError(str(exc)) from exc

    from mutagen.flac import FLAC
    from mutagen.id3 import COMM, ID3, TKEY, ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    value = str(short_key).strip()[:3]
    suffix = path.suffix.lower()

    if suffix == ".flac":
        audio = FLAC(str(path))
        if field == "key":
            audio["INITIALKEY"] = [value]
        else:
            audio["COMMENT"] = [value]
        audio.save()
        return

    if suffix in (".ogg",):
        audio = OggVorbis(str(path))
        if field == "key":
            audio["INITIALKEY"] = [value]
        else:
            audio["COMMENT"] = [value]
        audio.save()
        return

    if suffix == ".opus":
        audio = OggOpus(str(path))
        if field == "key":
            audio["INITIALKEY"] = [value]
        else:
            audio["COMMENT"] = [value]
        audio.save()
        return

    if suffix in (".mp3", ".wav", ".aif", ".aiff"):
        try:
            if suffix == ".mp3":
                audio = MP3(str(path))
            elif suffix in (".aif", ".aiff"):
                from mutagen.aiff import AIFF

                audio = AIFF(str(path))
            else:
                audio = WAVE(str(path))
            tags = audio.tags
            if tags is None:
                audio.add_tags()
                tags = audio.tags
        except ID3NoHeaderError:
            tags = ID3()
            if suffix == ".mp3":
                audio = MP3(str(path))
                audio.tags = tags
            else:
                audio = WAVE(str(path))
                audio.add_tags()
                tags = audio.tags
        if field == "key":
            tags.delall("TKEY")
            tags.add(TKEY(encoding=3, text=[value]))
        else:
            tags.delall("COMM")
            tags.add(COMM(encoding=3, lang="eng", desc="", text=[value]))
        audio.save()
        return

    if suffix in (".m4a", ".mp4", ".aac"):
        audio = MP4(str(path))
        if field == "key":
            audio["----:com.apple.iTunes:initialkey"] = [
                MP4FreeForm(value.encode("utf-8"))
            ]
        else:
            audio["\xa9cmt"] = [value]
        audio.save()
        return

    raise OSError(f"unsupported format: {suffix}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Musical key detect (KeyNet nf50)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--folder", type=str, help="Input folder")
    src.add_argument("--files-from", type=str, help="Text file of audio paths")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument(
        "--mode",
        choices=("batch", "per_file"),
        default="batch",
        help="Batch = chunk GPU batches; per_file = live one-by-one",
    )
    ap.add_argument("--write-meta", action="store_true")
    ap.add_argument(
        "--tag-field",
        choices=("comment", "key"),
        default="key",
        help="Write short key to COMMENT or Initial key (TKEY / INITIALKEY)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--checkpoint", type=str, default="", help="Path to .pt")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu")
    ap.add_argument("--json", action="store_true", help="Emit JSON result lines")
    ap.add_argument(
        "--log-pace",
        type=float,
        default=-1.0,
        help="Seconds between LOG file results (batch). 0=off. Default 0.5.",
    )
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint) if args.checkpoint else _default_checkpoint()
    if not ckpt.is_file():
        # Allow passing a .pt path when only the .onnx sibling ships.
        alt = ckpt.with_suffix(".onnx")
        if alt.is_file():
            ckpt = alt
        else:
            print(f"ERROR: checkpoint not found: {ckpt}", flush=True)
            return 2

    # Heavy imports — print first so LOG isn't blank while librosa loads.
    print("  Loading libraries (librosa / onnxruntime)…", flush=True)
    from keys import short_to_display  # noqa: E402
    from inference import (  # noqa: E402
        collect_audio_files,
        default_cqt_workers,
        load_model,
        process_batched,
        process_per_file,
        resolve_device,
    )

    if args.files_from:
        print("  Reading file list…", flush=True)
        paths = _read_files_from(Path(args.files_from))
    else:
        folder = Path(args.folder).expanduser().resolve()
        if not folder.is_dir():
            print(f"ERROR: folder not found: {folder}", flush=True)
            return 2
        print("  Scanning folder for audio…", flush=True)
        paths = collect_audio_files(folder, recursive=not args.no_recursive)

    if not paths:
        print("No audio files found.", flush=True)
        print("DONE", flush=True)
        return 0

    device = resolve_device(args.device)
    print(
        f"  Key Detect · {len(paths):,} file(s) · device={device} · mode={args.mode}",
        flush=True,
    )
    if args.mode == "batch":
        print(
            f"  CQT workers: {default_cqt_workers()} (parallel preprocess)",
            flush=True,
        )
    print(f"  Model: {ckpt.name}", flush=True)
    print("  Loading KeyNet weights…", flush=True)
    print("", flush=True)

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    model = load_model(ckpt, device)
    backend = getattr(model, "backend", "torch")
    actual = getattr(model, "device", device)
    print(f"  Model ready ({backend}, {actual}).", flush=True)
    print("", flush=True)
    t0 = time.monotonic()
    written = skipped = errors = 0
    stop = {"flag": False}

    # Skip already-tagged files BEFORE inference (not after).
    to_run: list[Path] = []
    live_log = args.mode == "per_file"
    if args.write_meta and not args.overwrite:
        print("  Checking existing Initial key / Comment tags…", flush=True)
        for path in paths:
            if _read_existing_key(path, args.tag_field):
                skipped += 1
                # Per-file: list each skip. Batch: summary only (Genre-style).
                if live_log:
                    print(f"  [skip existing] {path.name}", flush=True)
                if args.json:
                    print(
                        json.dumps(
                            {
                                "path": str(path),
                                "skipped": True,
                                "reason": "already tagged",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            else:
                to_run.append(path)
        if skipped:
            print(
                f"  Skipping {skipped:,} already-tagged · {len(to_run):,} to analyze",
                flush=True,
            )
            print("", flush=True)
    else:
        to_run = list(paths)

    def stop_check() -> bool:
        return stop["flag"]

    def on_progress(n: int, total: int) -> None:
        pct = 100.0 * n / max(1, total)
        print(f"__progress__\t{pct:.1f}\t-\t{n}\t{total}", flush=True)

    def on_file(
        path_s: str,
        key: str | None,
        conf: float | None,
        err: str | None,
    ) -> None:
        nonlocal written, errors
        done_i[0] += 1
        name = Path(path_s).name
        n = max(1, total_run)
        header = f"=== [{done_i[0]}/{n}] {name} ==="
        if err:
            errors += 1
            print(header, flush=True)
            print(f"  ERROR: {err}", flush=True)
            if args.json:
                print(
                    json.dumps(
                        {"path": path_s, "error": err},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            return

        assert key is not None and conf is not None
        display = short_to_display(key)
        pct = conf * 100.0

        tag_error = None
        if args.write_meta:
            try:
                _write_key_metadata(Path(path_s), key, field=args.tag_field)
                written += 1
            except Exception as exc:
                tag_error = str(exc)
                errors += 1

        # Batch: progress counter only (no per-file === / badges).
        # Per-file: live === [i/n] + key badge like Genre Per-file.
        if live_log:
            print(header, flush=True)
            print(f"  {display} {pct:.0f}%", flush=True)
            if tag_error:
                print(f"  Tag write failed: {tag_error}", flush=True)
        elif tag_error:
            print(header, flush=True)
            print(f"  Tag write failed: {tag_error}", flush=True)
        if args.json:
            print(
                json.dumps(
                    {
                        "path": path_s,
                        "label": key,
                        "display": display,
                        "score": conf,
                        "tag_error": tag_error,
                        "index": done_i[0],
                        "total": n,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    done_i = [0]
    total_run = len(to_run)

    if to_run:
        runner = process_batched if args.mode == "batch" else process_per_file
        kwargs = dict(
            on_file=on_file,
            stop_check=stop_check,
            on_progress=on_progress,
        )
        if args.mode == "batch":
            # No file LOG in batch — pacing unused; keep CLI flag harmless.
            pace = 0.0 if args.log_pace < 0 else float(args.log_pace)
            kwargs["log_pace_s"] = pace
        runner(model, to_run, device, **kwargs)
    elif not paths:
        pass
    else:
        print("  Nothing left to analyze.", flush=True)

    elapsed = time.monotonic() - t0
    # Summary (emit_run_summary-compatible shape for the worker).
    print("", flush=True)
    print("=== Key Detect Summary ===", flush=True)
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"  Total time: {mins}:{secs:02d}", flush=True)
    print(f"  Files: {len(paths):,}", flush=True)
    if paths and elapsed > 0:
        print(f"  Sec/file: {elapsed / len(paths):.2f}", flush=True)
        print(f"  Files/min: {60.0 * len(paths) / elapsed:.1f}", flush=True)
    print(f"  Tagged: {written:,} | Skipped: {skipped:,}", flush=True)
    if errors:
        print(f"  Errors: {errors:,}", flush=True)
    print("", flush=True)
    print("DONE", flush=True)
    return 0 if errors == 0 or written > 0 or skipped > 0 else 1


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())

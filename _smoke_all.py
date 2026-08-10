"""Pre-ship smoke gate — one runner for all codec + tagger smoke tests.

Targets the FRESHLY BUILT dist (dist\\STEM-organizer), not the installed app:
  - every tagger decode path on m4a / opus / ogg (librosa fallback + sf paths)
  - genre/gender tag writes (m4a) through the fallback decode
  - pair-matcher scan + matching logic (MoisesDB multitracks)
  - static scan-set coverage (all taggers list ogg/opus/m4a)

Fixture files that don't exist are SKIPped (never FAIL) so the gate still runs
on machines without the local sample folders; genuine failures abort.

Usage:
    python _smoke_all.py [--app dist\\STEM-organizer] [--python <host-python>]

Exit 0 = all PASS/SKIP, 1 = any FAIL.

Runs against the *source* tagger copies by default? NO — it drives the dist
copies (the exact .py files build.bat ships), so the gate validates what
actually gets packaged.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fixtures (skip when absent)
# ---------------------------------------------------------------------------
M4A = Path(r"E:\Audio\!Unsorted\01 BAM BAM.m4a")                      # sf-rejected -> fallback
OPUS_MATROSKA = Path(r"F:\youtube-dl\Other\Ed Sheeran x Spotify-dTz3-NzSx8E.opus")  # sf-rejected -> fallback
OPUS_OGG = Path(r"F:\Downloads_temp\sample.opus\Billy Joel - Say Goodbye to Hollywood.opus")  # sf-readable
OGG_DIR = Path(r"E:\Audio\!OGG")
MOISES = Path(r"E:\Audio\Datasets\moisesdb_rigo")

CODED_FILES = [p for p in (M4A, OPUS_MATROSKA, OPUS_OGG) if p.is_file()]
SCRATCH = REPO / "_smoke_all_scratch"

results: list[tuple[str, bool, float]] = []  # (label, ok, seconds)


def out(label: str, ok: bool, dt: float, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    if detail:
        print(f"  [{mark}] {label} — {detail}")
    else:
        print(f"  [{mark}] {label} ({dt:.1f}s)")
    results.append((label, ok, dt))


def skip(label: str, reason: str) -> None:
    print(f"  [SKIP] {label} — {reason}")
    results.append((label, True, 0.0))


def run_py(python: Path, script: Path, *args: str, env: dict | None = None,
           timeout: int = 900, cwd: Path | None = None) -> subprocess.CompletedProcess:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        [str(python), "-X", "utf8", str(script), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged,
        cwd=str(cwd or REPO),
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------------------
# Gate 1: static scan-set coverage
# ---------------------------------------------------------------------------
def gate_scan_sets() -> None:
    print("\n=== Gate 1: tagger scan-set coverage (ogg/opus/m4a) ===")
    t0 = time.monotonic()
    checks = {
        "instrument_tagger/instrument_tagger.py": (".ogg", ".opus", ".m4a"),
        "panns_tagger/panns_tagger.py": (".ogg", ".opus", ".m4a"),
        "key_tagger/key_tagger.py": (".ogg", ".opus", ".m4a"),
        "genre_gender_tagger/genre_gender_tagger.py": (".ogg", ".opus", ".m4a"),
    }
    for rel, exts in checks.items():
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        missing = [e for e in exts if e not in src]
        ok = not missing
        out(f"scan set {Path(rel).parent}", ok, time.monotonic() - t0,
            f"missing={missing}" if missing else "all formats present")


# ---------------------------------------------------------------------------
# Gate 2: decode smoke — every tagger on m4a + opus (both decode paths)
# ---------------------------------------------------------------------------
def gate_decode(app: Path, python: Path) -> None:
    print("\n=== Gate 2: tagger decode (m4a + opus, fallback + sf paths) ===")
    if not CODED_FILES:
        skip("decode smoke", f"no fixture files found ({[str(p) for p in (M4A, OPUS_MATROSKA, OPUS_OGG) if not p.is_file()]})")
        return

    files_list = SCRATCH / "codec_files.txt"
    files_list.write_text("\n".join(str(p) for p in CODED_FILES) + "\n", encoding="utf-8")
    app_env = {"PYTHONPATH": str(app)}  # ort_util resolves (key_worker does the same)

    def run_tagger(label, rel_script, *extra):
        script = app / rel_script
        t0 = time.monotonic()
        proc = run_py(python, script, "--files-from", str(files_list), *extra, env=app_env)
        dt = time.monotonic() - t0
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        # For instrument: expect one JSON result line per input file
        if "instrument_tagger" in rel_script and ok:
            n_json = sum(1 for ln in output.splitlines() if ln.strip().startswith("{"))
            ok = n_json == len(CODED_FILES)
        # For key: expect JSON result lines per file
        if "key_tagger" in rel_script and ok:
            n_json = sum(1 for ln in output.splitlines() if ln.strip().startswith("{"))
            ok = n_json == len(CODED_FILES)
        # For panns: expect "Files: N" summary
        if "panns_tagger" in rel_script and ok:
            ok = f"Files: {len(CODED_FILES)}" in output
        detail = f"rc={proc.returncode}" if not ok else f"{len(CODED_FILES)} file(s)"
        out(f"{label} decode", ok, dt, detail)
        if not ok:
            print("   " + "\n   ".join(output.splitlines()[-8:]))

    run_tagger("instrument (Stem CNN6)", "instrument_tagger/instrument_tagger.py", "--top", "2")
    run_tagger("panns (Cnn14)", "panns_tagger/panns_tagger.py")
    run_tagger("key (KeyNet)", "key_tagger/key_tagger.py", "--json")


# ---------------------------------------------------------------------------
# Gate 3: genre/gender decode + real tag write on an m4a copy
# ---------------------------------------------------------------------------
def gate_genre_gender(app: Path, python: Path) -> None:
    print("\n=== Gate 3: genre/gender decode + tag write (fallback path) ===")
    if not M4A.is_file():
        skip("genre/gender tag write", f"fixture missing: {M4A}")
        return

    copy = SCRATCH / "gg_test.m4a"
    shutil.copy2(M4A, copy)
    files_list = SCRATCH / "gg_files.txt"
    files_list.write_text(f'"{copy}"\n', encoding="utf-8")
    gg = app / "genre_gender_tagger" / "genre_gender_tagger.py"

    base_env = {
        "GG_INPUT": str(SCRATCH),
        "GG_FILES_FROM": str(files_list),
        "GG_WRITE_META": "1",
        "GG_OVERWRITE": "1",
        "GG_BATCH": "1",
        "GG_RECURSIVE": "0",
    }

    for mode in ("genre", "gender"):
        env = dict(base_env, GG_MODE=mode)
        t0 = time.monotonic()
        proc = run_py(python, gg, env=env, timeout=1200)
        dt = time.monotonic() - t0
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and "Tagged: 1" in output
        out(f"{mode} decode + write", ok, dt, f"rc={proc.returncode}")
        if not ok:
            print("   " + "\n   ".join(output.splitlines()[-8:]))

    # Verify the tags actually landed via mutagen
    try:
        from mutagen.mp4 import MP4
        audio = MP4(str(copy))
        genre = audio.get("\xa9gen", [""])
        genre_val = bytes(genre[0]).decode("utf-8", errors="replace") if genre and isinstance(genre[0], (bytes, bytearray)) else str(genre[0] if genre else "")
        comment = audio.get("\xa9cmt", [""])
        comment_val = bytes(comment[0]).decode("utf-8", errors="replace") if comment and isinstance(comment[0], (bytes, bytearray)) else str(comment[0] if comment else "")
        t0 = time.monotonic()
        out("tag round-trip via mutagen", bool(genre_val) and bool(comment_val), 0,
            f"genre={genre_val!r} comment={comment_val!r}")
    except Exception as exc:
        out("tag round-trip via mutagen", False, 0, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Gate 4: pair-matcher scan + matching on MoisesDB multitracks
# ---------------------------------------------------------------------------
def gate_pair() -> None:
    print("\n=== Gate 4: pair-matcher scan + matching (MoisesDB) ===")
    if not MOISES.is_dir():
        skip("pair matching", f"fixture missing: {MOISES}")
        return
    t0 = time.monotonic()
    sys.path.insert(0, str(REPO))
    import pair_matcher  # noqa: E402

    # Scan: ogg folder must resolve files (was 0 pre-widening fix)
    if OGG_DIR.is_dir():
        n = len(pair_matcher.iter_audio_files(OGG_DIR, recursive=True))
        ok = n > 0
        out("pair scan on ogg folder", ok, time.monotonic() - t0, f"{n} files")
    else:
        skip("pair scan on ogg folder", f"fixture missing: {OGG_DIR}")

    # Matching: 2 songs, acapella vs instrumental, strict=100
    songs = [
        "Andy Bennett - Baby Let Me Hold You Tonight",
        "Battlestar - Distant Eyes",
    ]
    acap, inst = SCRATCH / "acap", SCRATCH / "inst"
    shutil.rmtree(acap, ignore_errors=True)
    shutil.rmtree(inst, ignore_errors=True)
    acap.mkdir(parents=True)
    inst.mkdir(parents=True)
    placed = 0
    for song in songs:
        tracks = MOISES / song / "tracks"
        vocals = next(tracks.glob("vocals_*"), None)
        inst_stem = next((p for p in sorted(tracks.iterdir())
                          if p.is_file() and p.suffix.lower() in (".flac", ".wav")
                          and not p.name.startswith("vocals_")), None)
        if vocals is not None:
            shutil.copy2(vocals, acap / f"{song} (Acapella).flac")
            placed += 1
        if inst_stem is not None:
            shutil.copy2(inst_stem, inst / f"{song} (Instrumental).flac")
            placed += 1

    res = pair_matcher.find_pairs(
        acap, inst, reference_is_acapella=True, strictness=100.0,
        use_filename_fallback=True, include_subfolders=False,
    )
    ok = len(res.pairs) == len(songs) and all(abs(m.score - 1.0) < 1e-9 for m in res.pairs)
    out("pair matching (2 songs, strict=100)", ok, time.monotonic() - t0,
        f"{len(res.pairs)} pair(s), placed={placed}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default=str(REPO / "dist" / "STEM-organizer"),
                    help="Dist folder holding the tagger copies (default dist\\STEM-organizer)")
    ap.add_argument("--python", default=str(REPO / ".build-venv" / "Scripts" / "python.exe"),
                    help="Host python for tagger children (default .build-venv python)")
    args = ap.parse_args()

    app = Path(args.app)
    python = Path(args.python)
    if not app.is_dir():
        print(f"ERROR: --app folder missing: {app}")
        return 1
    if not python.is_file():
        print(f"ERROR: --python missing: {python}")
        return 1

    print(f"Pre-ship smoke gate\n  app:    {app}\n  python: {python}\n  fixtures: {len(CODED_FILES)} codec file(s), ogg dir={'yes' if OGG_DIR.is_dir() else 'no'}, moises={'yes' if MOISES.is_dir() else 'no'}")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)

    gate_scan_sets()
    gate_decode(app, python)
    gate_genre_gender(app, python)
    gate_pair()

    shutil.rmtree(SCRATCH, ignore_errors=True)

    print("\n==================== SUMMARY ====================")
    failed = 0
    for label, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}  ({dt:.1f}s)")
        if not ok:
            failed += 1
    print("================================================")
    if failed:
        print(f"GATE FAILED: {failed} check(s)")
        return 1
    print("GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Repair helpers — MP3 stream rebuild (re-encode) + mp3val polish; ffmpeg for others."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .deep import verify_deep
from .fast_mp3 import MP3_EXTS, sync_xing_to_decoded, verify_mp3_fast
from .tools import find_ffmpeg, find_ffprobe, find_mp3val

try:
    from ffmpeg_bootstrap import subprocess_kwargs
except Exception:  # pragma: no cover
    def subprocess_kwargs() -> dict:
        return {}


@dataclass
class FixResult:
    ok: bool
    out_path: str = ""
    detail: str = ""


def _fixed_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_FIXED{src.suffix}")


def remove_fixed_sibling(path: Path | str) -> None:
    """Delete leftover ``*_FIXED`` next to ``path`` (or the path itself if it is one)."""
    p = Path(path)
    candidates = [p]
    if not p.stem.endswith("_FIXED"):
        candidates.append(_fixed_path(p))
    for dest in candidates:
        if not dest.is_file():
            continue
        if not dest.stem.endswith("_FIXED"):
            continue
        try:
            dest.unlink()
        except OSError:
            pass


def _short_ffmpeg_err(err: str) -> str:
    """First useful ffmpeg error line (drop dump noise)."""
    if not err:
        return ""
    for line in err.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "nothing was written" in low or "invalid argument" in low or "error" in low:
            # Trim leading junk like "0:0/png @ 0x..."
            if "] " in s:
                s = s.split("] ", 1)[-1]
            return s[:200]
    return err.strip()[-200:]


def wav_has_leading_id3(path: Path | str) -> bool:
    """True when a .wav starts with raw ID3 instead of RIFF (corrupt tag write)."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(12)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return False
    return head.startswith(b"ID3")


def salvage_wav_leading_id3(path: Path | str) -> bool:
    """Strip a prepended MP3-style ID3 blob so RIFF/WAVE is at offset 0.

    Older writers called plain ``ID3.save()`` on WAV, which prepends ID3 and
    makes ffmpeg report \"Invalid data found when processing input\".
    Returns True if the file was rewritten.
    """
    p = Path(path)
    if p.suffix.lower() != ".wav" or not wav_has_leading_id3(p):
        return False
    try:
        data = p.read_bytes()
    except OSError:
        return False
    idx = data.find(b"RIFF")
    if idx <= 0 or data[idx + 8 : idx + 12] != b"WAVE":
        return False
    try:
        p.write_bytes(data[idx:])
    except OSError:
        return False
    return True


def _run(cmd: list[str], timeout: float = 900.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **subprocess_kwargs(),
        )
        return int(proc.returncode), ((proc.stdout or "") + (proc.stderr or "")).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _mp3val_issues(path: Path, mp3val_bin: str) -> str:
    code, out = _run([mp3val_bin, "-si", str(path)], timeout=120)
    low = out.lower()
    if "error:" in low:
        for line in out.splitlines():
            if "error" in line.lower():
                return line.strip()[-160:]
        return "mp3val error"
    if "warning:" in low:
        for line in out.splitlines():
            if "warning" not in line.lower():
                continue
            # Conflicts with post-Fix Xing sync to ffprobe decode count
            # (mp3val wants structural frames, decoders report N−1).
            if "xing header" in line.lower() and "mpeg frames" in line.lower():
                continue
            return line.strip()[-160:]
        # Only Xing frame-count warnings → treat as clean
        if "xing header" in low and "mpeg frames" in low:
            return ""
        return "mp3val warning"
    if code not in (0, 1):
        return f"mp3val exit {code}"
    return ""


def remaining_issues(
    path: Path | str,
    *,
    mp3val_bin: str | None = None,
    ffmpeg_bin: str | None = None,
) -> str:
    """Return a short reason if the file is still not clean, else \"\"."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MP3_EXTS:
        fr = verify_mp3_fast(path)
        if not fr.ok:
            return fr.error or "structural fail"
    dr = verify_deep(path, ffmpeg_bin=ffmpeg_bin)
    if dr.status == "failed":
        return dr.detail or "decode failed"
    if dr.status == "minor":
        return dr.detail or "minor problems"
    if suffix in MP3_EXTS and mp3val_bin:
        issue = _mp3val_issues(path, mp3val_bin)
        if issue:
            return issue
    return ""


def _mp3val_fix_inplace(path: Path, mp3val_bin: str) -> tuple[bool, str]:
    code, out = _run([mp3val_bin, "-f", "-nb", str(path)], timeout=600)
    bak = Path(str(path) + ".bak")
    if bak.is_file():
        try:
            bak.unlink()
        except OSError:
            pass
    if code not in (0, 1):
        return False, out[-300:] or f"exit {code}"
    return True, out[-200:] if out else "mp3val"


def fix_with_mp3val(
    path: Path | str,
    *,
    mp3val_bin: str | None = None,
) -> FixResult:
    """Copy to ``*_FIXED.mp3`` then run ``mp3val -f`` on the copy."""
    src = Path(path)
    mp3val = mp3val_bin or find_mp3val()
    if not mp3val:
        return FixResult(False, detail="mp3val not found")
    dest = _fixed_path(src)
    try:
        shutil.copy2(str(src), str(dest))
    except OSError as exc:
        return FixResult(False, detail=str(exc))
    ok, detail = _mp3val_fix_inplace(dest, mp3val)
    if not ok:
        return FixResult(False, out_path=str(dest), detail=detail)
    return FixResult(True, out_path=str(dest), detail=detail or "mp3val")


def fix_with_ffmpeg(
    path: Path | str,
    *,
    ffmpeg_bin: str | None = None,
    force_reencode: bool = False,
) -> FixResult:
    """Remux ``-c copy`` then re-encode fallback → ``*_FIXED``.

    When ``force_reencode``, skip remux (needed for true MPEG stream rebuild).
    """
    src = Path(path)
    ffmpeg = ffmpeg_bin or find_ffmpeg()
    if not ffmpeg:
        return FixResult(False, detail="ffmpeg not found")
    dest = _fixed_path(src)
    if dest.exists():
        try:
            dest.unlink()
        except OSError:
            pass

    suffix = src.suffix.lower()
    input_path = src
    salvage_tmp: Path | None = None
    salvage_note = ""

    # WAV previously tagged with plain ID3.save() — strip prefix before ffmpeg.
    if suffix == ".wav" and wav_has_leading_id3(src):
        salvage_tmp = src.with_name(f"{src.stem}.__id3salvage__{src.suffix}")
        try:
            if salvage_tmp.exists():
                salvage_tmp.unlink()
            shutil.copy2(str(src), str(salvage_tmp))
            if salvage_wav_leading_id3(salvage_tmp):
                input_path = salvage_tmp
                salvage_note = "stripped leading ID3; "
            else:
                salvage_tmp.unlink(missing_ok=True)
                salvage_tmp = None
        except OSError:
            if salvage_tmp is not None:
                try:
                    salvage_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                salvage_tmp = None

    def _encode_args() -> list[str]:
        if suffix == ".flac":
            return ["-c:a", "flac"]
        if suffix in (".mp3", ".mp2", ".m2a"):
            return ["-c:a", "libmp3lame", "-q:a", "2"]
        if suffix in (".ogg", ".opus"):
            return ["-c:a", "libopus" if suffix == ".opus" else "libvorbis"]
        if suffix in (".m4a", ".aac", ".mp4"):
            return ["-c:a", "aac", "-b:a", "192k"]
        if suffix == ".wav":
            return ["-c:a", "pcm_s16le"]
        return ["-c:a", "copy"]

    # Audio only — embedded cover (mjpeg/png "video") breaks MP3 encode and can
    # leave a destroyed *_FIXED with no packets.
    def _cleanup_dest() -> None:
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass

    def _cleanup_salvage() -> None:
        if salvage_tmp is None:
            return
        try:
            salvage_tmp.unlink(missing_ok=True)
        except OSError:
            pass

    # Salvage alone may already be enough for WAV
    if (
        salvage_tmp is not None
        and input_path == salvage_tmp
        and not remaining_issues(input_path, ffmpeg_bin=ffmpeg)
    ):
        try:
            shutil.move(str(input_path), str(dest))
            salvage_tmp = None
            return FixResult(
                True, out_path=str(dest), detail="stripped leading ID3"
            )
        except OSError:
            pass

    try:
        if not force_reencode:
            code, err = _run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-map_metadata",
                    "0",
                    "-c",
                    "copy",
                    str(dest),
                ]
            )
            if code == 0 and dest.is_file() and dest.stat().st_size > 0:
                return FixResult(
                    True,
                    out_path=str(dest),
                    detail=f"{salvage_note}remux".strip(),
                )
            _cleanup_dest()

        code, err = _run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-map_metadata",
                "0",
                *_encode_args(),
                str(dest),
            ]
        )
        if code == 0 and dest.is_file() and dest.stat().st_size > 0:
            return FixResult(
                True,
                out_path=str(dest),
                detail=f"{salvage_note}re-encode".strip(),
            )
        _cleanup_dest()
        return FixResult(False, detail=_short_ffmpeg_err(err) or "ffmpeg fix failed")
    finally:
        _cleanup_salvage()


def fix_file(
    path: Path | str,
    *,
    mp3val_bin: str | None = None,
    ffmpeg_bin: str | None = None,
) -> FixResult:
    """Repair into ``*_FIXED``.

    MP3 path (foobar Rebuild–like):
      1) mp3val -f (header / light stream cleanup)
      2) if still dirty → ffmpeg re-encode (true stream rewrite)
      3) mp3val polish on the result
      4) only ``ok=True`` when post-verify is clean

    MPEG content with a ``.wav`` extension: rename to ``.mp3`` (foobar wrong-ext).

    Other formats: ffmpeg remux, then re-encode; verified the same way.
    """
    src = Path(path)
    ffmpeg = ffmpeg_bin or find_ffmpeg()
    mp3val = mp3val_bin or find_mp3val()
    suffix = src.suffix.lower()

    from .sniff import is_mpeg_audio

    # foobar: "wrong extension: wav; Correct handler: MPEG Decoder"
    if suffix == ".wav" and is_mpeg_audio(src):
        dest = src.with_suffix(".mp3")
        if dest.exists() and dest.resolve() != src.resolve():
            dest = src.with_name(f"{src.stem}_FIXED.mp3")
        try:
            if dest.exists():
                dest.unlink()
            shutil.copy2(str(src), str(dest))
        except OSError as exc:
            return FixResult(False, detail=str(exc))
        leftover = remaining_issues(dest, mp3val_bin=mp3val, ffmpeg_bin=ffmpeg)
        if leftover:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return FixResult(False, detail=f"renamed but still: {leftover}")
        return FixResult(
            True, out_path=str(dest), detail="renamed .wav→.mp3 (MPEG content)"
        )

    if suffix in MP3_EXTS:
        # mp3val alone leaves many foobar "Expected N / decoded N-1" and
        # AudioTester lost-sync cases looking "fixed". True rebuild = re-encode.
        dest = _fixed_path(src)
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        fr = fix_with_ffmpeg(src, ffmpeg_bin=ffmpeg, force_reencode=True)
        if not fr.ok or not fr.out_path:
            # Completely undecodable — drop any partial *_FIXED
            remove_fixed_sibling(src)
            return FixResult(False, detail=fr.detail or "ffmpeg fix failed")
        if mp3val:
            _mp3val_fix_inplace(Path(fr.out_path), mp3val)
        # LAME Xing often claims N while ffprobe decodes N−1 — patch so Deep/Fix
        # don't treat a successful rebuild as still broken.
        sync_xing_to_decoded(fr.out_path, ffprobe_bin=find_ffprobe())
        leftover = remaining_issues(
            fr.out_path, mp3val_bin=mp3val, ffmpeg_bin=ffmpeg
        )
        if leftover:
            detail = f"wrote {Path(fr.out_path).name} but still: {leftover}"
            remove_fixed_sibling(fr.out_path)
            return FixResult(False, detail=detail)
        return FixResult(True, out_path=fr.out_path, detail="re-encode")

    # Non-MP3
    fr = fix_with_ffmpeg(src, ffmpeg_bin=ffmpeg, force_reencode=False)
    if not fr.ok or not fr.out_path:
        # Try forced re-encode
        fr = fix_with_ffmpeg(src, ffmpeg_bin=ffmpeg, force_reencode=True)
    if not fr.ok or not fr.out_path:
        remove_fixed_sibling(src)
        return FixResult(False, detail=fr.detail or "ffmpeg fix failed")
    leftover = remaining_issues(fr.out_path, ffmpeg_bin=ffmpeg)
    if leftover:
        # Trailing garbage etc. — try forced re-encode once more if we only remuxed
        if fr.detail == "remux":
            try:
                Path(fr.out_path).unlink(missing_ok=True)
            except OSError:
                pass
            fr2 = fix_with_ffmpeg(src, ffmpeg_bin=ffmpeg, force_reencode=True)
            if fr2.ok and fr2.out_path:
                leftover2 = remaining_issues(fr2.out_path, ffmpeg_bin=ffmpeg)
                if leftover2:
                    remove_fixed_sibling(fr2.out_path)
                    return FixResult(
                        False,
                        detail=f"wrote {Path(fr2.out_path).name} but still: {leftover2}",
                    )
                return FixResult(True, out_path=fr2.out_path, detail="re-encode")
        remove_fixed_sibling(fr.out_path)
        return FixResult(
            False,
            detail=f"wrote {Path(fr.out_path).name} but still: {leftover}",
        )
    return fr


def quarantine_file(path: Path | str, dest_root: Path | str) -> tuple[bool, str]:
    """Move ``path`` under ``dest_root``, preserving relative name when possible."""
    src = Path(path)
    root = Path(dest_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        dest = root / src.name
        n = 1
        while dest.exists():
            dest = root / f"{src.stem}_{n}{src.suffix}"
            n += 1
        shutil.move(str(src), str(dest))
        return True, str(dest)
    except OSError as exc:
        return False, str(exc)

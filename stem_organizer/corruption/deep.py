"""Deep integrity verify — ffmpeg decode-null + optional flac -t."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .tools import find_ffmpeg, find_flac

try:
    from ffmpeg_bootstrap import subprocess_kwargs
except Exception:  # pragma: no cover
    def subprocess_kwargs() -> dict:
        return {}


@dataclass
class DeepResult:
    """ok / minor / failed."""

    status: str  # ok | minor | failed
    detail: str = ""


# Length mismatch slack before Deep marks minor (triggers Fix).
# Sub-frame / ~26 ms skew is common VBR rounding — not worth a lossy re-encode.
_LENGTH_SLACK_SEC = 0.5

_MINOR_HINTS = (
    "inaccurate",
    "duration",
    "truncat",
    "overread",
    "underread",
    "invalid discarding",
    "header missing",
    "estimating duration",
)


def _run(cmd: list[str], timeout: float = 600.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            **subprocess_kwargs(),
        )
        err = (proc.stderr or "") + (proc.stdout or "")
        return int(proc.returncode), err
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except OSError as exc:
        return 1, str(exc)


def _tagged_duration_sec(path: Path) -> Optional[float]:
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path))
        if audio is None:
            return None
        info = getattr(audio, "info", None)
        length = getattr(info, "length", None)
        if length is None:
            return None
        return float(length)
    except Exception:
        return None


def _flac_streaminfo(path: Path) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (bits_per_sample, sample_rate, channels) from STREAMINFO, else Nones."""
    try:
        from mutagen.flac import FLAC

        audio = FLAC(str(path))
        info = getattr(audio, "info", None)
        if info is None:
            return None, None, None
        bits = getattr(info, "bits_per_sample", None)
        sr = getattr(info, "sample_rate", None)
        ch = getattr(info, "channels", None)
        return (
            int(bits) if bits is not None else None,
            int(sr) if sr is not None else None,
            int(ch) if ch is not None else None,
        )
    except Exception:
        return None, None, None


def verify_deep(
    path: Path | str,
    *,
    flac_bin: str | None = None,
    ffmpeg_bin: str | None = None,
) -> DeepResult:
    """Full decode verify. Prefer ``flac -t`` for FLAC when available."""
    path = Path(path)
    suffix = path.suffix.lower()

    from .sniff import wrong_extension_note

    wrong_ext = wrong_extension_note(path)

    if suffix in (".flac", ".fla"):
        bits, _sr, _ch = _flac_streaminfo(path)
        # Spec allows up to 32 in STREAMINFO, but common tools (libsndfile,
        # many DAWs) only implement ≤24 — treat >24 as corrupt/unsupported.
        if bits is not None and bits > 24:
            return DeepResult(
                "failed",
                f"FLAC bit depth {bits} unsupported (max 24 for decode/tag tools)",
            )

        flac = flac_bin or find_flac()
        if flac:
            code, out = _run([flac, "-t", "-s", str(path)])
            low = out.lower()
            if code != 0:
                if "md5" in low:
                    return DeepResult("failed", "FLAC MD5 mismatch")
                detail = out.strip().splitlines()[-1] if out.strip() else "flac -t failed"
                return DeepResult("failed", detail[:200])
            return DeepResult("ok", "")
        # No reference flac CLI — ffmpeg may decode junk that foobar/AudioTester
        # reject (MD5 / sync). Prefer failing closed only on known-bad headers;
        # continue to ffmpeg but note MD5 was not verified in detail if ok.

    ffmpeg = ffmpeg_bin or find_ffmpeg()
    if not ffmpeg:
        return DeepResult("failed", "ffmpeg not found")

    # -xerror: treat decode errors as fatal (exit non-zero).
    # -stats: emit time= progress so we can compare decoded length.
    code, out = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-stats",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ]
    )
    low = out.lower()
    if code != 0:
        detail = ""
        for line in out.strip().splitlines():
            s = line.strip()
            if not s:
                continue
            # Prefer real error lines over -stats progress (time=… bitrate=…)
            if s.lower().startswith("size=") or "bitrate=" in s.lower() and "time=" in s.lower():
                continue
            if "error" in s.lower() or "invalid" in s.lower() or "corrupt" in s.lower():
                detail = s
                break
            detail = s
        # Misnamed MPEG-as-WAV often still decodes if we don't force wav demux —
        # but when ffmpeg fails, keep the wrong-ext hint visible.
        if wrong_ext:
            return DeepResult("failed", f"{wrong_ext}; {detail or 'decode failed'}"[:200])
        return DeepResult("failed", (detail or "decode failed")[:200])

    # Soft warnings even when exit 0
    if wrong_ext:
        # foobar: "Decoded with minor problems" + wrong extension
        return DeepResult("minor", wrong_ext)

    if any(h in low for h in _MINOR_HINTS):
        # Prefer foobar-style length wording when present
        for line in out.splitlines():
            if "inaccurate" in line.lower() or "duration" in line.lower():
                return DeepResult("minor", line.strip()[:200])
        return DeepResult("minor", "decode warning")

    # Duration vs mutagen — only flag larger skew (re-encode costs quality).
    # Foobar's Expected N / decoded N−1 is intentionally not a Fix trigger:
    # ffmpeg still decodes cleanly and a rebuild is a lossy generation.
    tagged = _tagged_duration_sec(path)
    decoded = _parsed_ffmpeg_time_sec(out)
    if (
        tagged is not None
        and tagged > 0.5
        and decoded is not None
        and decoded > 0.25
    ):
        if abs(decoded - tagged) > _LENGTH_SLACK_SEC:
            return DeepResult(
                "minor",
                f"Reported length inaccurate: {tagged:.6f}s vs {decoded:.6f}s decoded",
            )

    return DeepResult("ok", "")


def _parsed_ffmpeg_time_sec(ffmpeg_output: str) -> Optional[float]:
    """Parse last ``time=HH:MM:SS.xx`` from ffmpeg -stats output."""
    last = None
    for m in re.finditer(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", ffmpeg_output):
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        last = h * 3600 + mi * 60 + s
    return last

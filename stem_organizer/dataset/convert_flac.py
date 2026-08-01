"""Convert audio libraries to consistent FLAC (44.1 kHz stereo, ≤24-bit).

Decode via ffmpeg/ffprobe (install-deps). soundfile is only used to write FLAC
and to probe already-optimal FLAC for the skip/copy path.

- FLAC max bit depth is 24; 32-bit float/int is reduced safely.
- Float peaks above 0 dBFS get uniform gain reduction (not a limiter).
- Already-optimal FLAC (target rate, 16/24-bit, stereo) is copied unchanged.
- Other rates are resampled to ``target_samplerate`` (default 44100).
"""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import threading
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..integrity_pool import integrity_pool
from ..run_summary import emit_run_summary

LogFn = Callable[[str, str], None]
ProgressFn = Callable[[int, int, str], None]

# MSST wants WAV/FLAC; everything else is still accepted and written as FLAC.
AUDIO_EXTS = {
    ".wav",
    ".aiff",
    ".aif",
    ".w64",
    ".caf",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
}
LOSSY_EXTS = {".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}
# Common CBR steps — anything else is labeled vbr in the filename suffix.
_LOSSY_CBR_KBPS = frozenset({64, 96, 128, 160, 192, 256, 320})
_LOSSY_ORIGIN_STEM_RE = re.compile(
    r"_(?:mp3|ogg|opus|m4a|aac|wma)(?:-\d+|-vbr)?$",
    re.IGNORECASE,
)
FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}
SUBTYPE_BITS = {
    "PCM_U8": 8,
    "PCM_S8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}
FLAC_MAX_BITS = 24

_SAMPLE_FMT_TO_SUBTYPE = {
    "u8": "PCM_U8",
    "u8p": "PCM_U8",
    "s16": "PCM_16",
    "s16p": "PCM_16",
    "s24": "PCM_24",
    "s24p": "PCM_24",
    "s32": "PCM_32",
    "s32p": "PCM_32",
    "flt": "FLOAT",
    "fltp": "FLOAT",
    "dbl": "DOUBLE",
    "dblp": "DOUBLE",
}


def _ffmpeg_tools() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) from install-deps / PATH. Raises if missing."""
    from ffmpeg_bootstrap import ffmpeg_path

    from ..corruption.tools import find_ffprobe

    ffmpeg = ffmpeg_path()
    ffprobe = find_ffprobe()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Re-run install-deps.bat (or put ffmpeg on PATH)."
        )
    if not ffprobe:
        raise RuntimeError(
            "ffprobe not found next to ffmpeg. Re-run install-deps.bat."
        )
    return str(ffmpeg), str(ffprobe)


def _probe_audio(src: Path, ffprobe: str) -> tuple[int, int, str]:
    """Return (sample_rate, channels, subtype label) via ffprobe."""
    import json
    import subprocess

    from ffmpeg_bootstrap import subprocess_kwargs

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,sample_fmt,bits_per_raw_sample,codec_name",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=False,
        **subprocess_kwargs(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"ffprobe failed ({src.name}): {err}")
    try:
        streams = json.loads(proc.stdout or "{}").get("streams") or []
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned bad JSON ({src.name})") from exc
    if not streams:
        raise RuntimeError(f"No audio stream ({src.name})")
    st = streams[0]
    try:
        sr = int(st.get("sample_rate") or 0)
        ch = int(st.get("channels") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Bad stream metadata ({src.name})") from exc
    if sr <= 0 or ch <= 0:
        raise RuntimeError(f"Missing sample rate/channels ({src.name})")

    fmt = str(st.get("sample_fmt") or "").lower()
    subtype = _SAMPLE_FMT_TO_SUBTYPE.get(fmt, "FLOAT")
    raw_bits = st.get("bits_per_raw_sample")
    try:
        bits = int(raw_bits) if raw_bits not in (None, "N/A", "") else 0
    except (TypeError, ValueError):
        bits = 0
    if bits in (8, 16, 24, 32) and subtype.startswith("PCM_"):
        subtype = f"PCM_{bits}"
    codec = str(st.get("codec_name") or "").lower()
    if codec in ("mp3", "aac", "opus", "vorbis", "wmav2", "wmav1", "ac3"):
        subtype = "FLOAT"
    return sr, ch, subtype


def load_audio(src: Path) -> tuple[np.ndarray, int, str]:
    """Decode with ffmpeg (install-deps) to float64 (N, C); subtype from ffprobe."""
    import subprocess

    from ffmpeg_bootstrap import subprocess_kwargs

    ffmpeg, ffprobe = _ffmpeg_tools()
    sr, channels, subtype = _probe_audio(src, ffprobe)

    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=False,
        **subprocess_kwargs(),
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg decode failed ({src.name}): {err or f'exit {proc.returncode}'}"
        )

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size % channels != 0:
        raise RuntimeError(
            f"ffmpeg decode size mismatch ({src.name}): "
            f"{samples.size} samples, {channels} ch"
        )
    data = samples.reshape(-1, channels).astype(np.float64, copy=False)
    return data, sr, subtype


def probe_bitrate_kbps(src: Path) -> Optional[int]:
    """Average bitrate in kbps from ffprobe, or None."""
    import json
    import subprocess

    from ffmpeg_bootstrap import subprocess_kwargs

    try:
        _, ffprobe = _ffmpeg_tools()
    except RuntimeError:
        return None
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=bit_rate:format=bit_rate",
            "-of",
            "json",
            str(src),
        ],
        capture_output=True,
        text=True,
        check=False,
        **subprocess_kwargs(),
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    raw = None
    streams = data.get("streams") or []
    if streams:
        raw = streams[0].get("bit_rate")
    if raw in (None, "N/A", "", "0"):
        raw = (data.get("format") or {}).get("bit_rate")
    try:
        bps = int(raw)
    except (TypeError, ValueError):
        return None
    if bps <= 0:
        return None
    return max(1, int(round(bps / 1000.0)))


def lossy_origin_suffix(src: Path) -> str:
    """e.g. ``_mp3-320`` / ``_ogg-vbr`` / ``_opus`` for lossy sources; else ``\"\"``.

    Exact CBR steps (64/96/128/160/192/256/320) keep the kbps number; other
    probed rates become ``vbr``.
    """
    ext = src.suffix.lower()
    if ext not in LOSSY_EXTS:
        return ""
    label = ext.lstrip(".")
    kbps = probe_bitrate_kbps(src)
    if not kbps:
        return f"_{label}"
    if kbps in _LOSSY_CBR_KBPS:
        return f"_{label}-{kbps}"
    return f"_{label}-vbr"


def with_lossy_origin_name(src: Path, dst: Path) -> Path:
    """Append source-origin suffix to FLAC stem when converting from lossy."""
    tag = lossy_origin_suffix(src)
    if not tag:
        return dst
    stem = dst.stem
    if _LOSSY_ORIGIN_STEM_RE.search(stem):
        return dst
    return dst.with_name(f"{stem}{tag}{dst.suffix}")


def target_bit_depth(subtype: str) -> int:
    bits = SUBTYPE_BITS.get(subtype, 16)
    return min(bits, FLAC_MAX_BITS)


def find_audio_files(input_dir: Path, *, recursive: bool = True) -> list[Path]:
    input_dir = Path(input_dir)
    out: list[Path] = []
    if recursive:
        for root, dirs, files in os_walk_skip_backup(input_dir):
            for name in files:
                if Path(name).suffix.lower() in AUDIO_EXTS:
                    out.append(Path(root) / name)
    else:
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                out.append(p)
    return out


def os_walk_skip_backup(root: Path):
    """``os.walk`` that skips ``_backup_before_align`` folders."""
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "_backup_before_align"]
        yield dirpath, dirnames, filenames


def tpdf_dither(data: np.ndarray, bit_depth: int) -> np.ndarray:
    lsb = 1.0 / (2 ** (bit_depth - 1))
    noise = (
        np.random.uniform(-1, 1, data.shape) + np.random.uniform(-1, 1, data.shape)
    ) * 0.5
    return data + noise * lsb


def is_already_optimal(
    src: Path, target_samplerate: int, *, target_channels: int = 2
) -> bool:
    import soundfile as sf

    if src.suffix.lower() != ".flac":
        return False
    ch = 1 if int(target_channels) == 1 else 2
    info = sf.info(str(src))
    return (
        info.samplerate == target_samplerate
        and info.subtype in ("PCM_16", "PCM_24")
        and info.channels == ch
    )


def ensure_stereo(data: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    channels = data.shape[1]
    if channels == 2:
        return data, None
    if channels == 1:
        return np.repeat(data, 2, axis=1), "mono → stereo (duplicated)"
    mono = data.mean(axis=1, keepdims=True)
    return (
        np.repeat(mono, 2, axis=1),
        f"{channels}ch → stereo (downmixed, then duplicated)",
    )


def ensure_channels(
    data: np.ndarray, target_channels: int
) -> tuple[np.ndarray, Optional[str]]:
    """Force stereo (2) or mono (1)."""
    ch = 1 if int(target_channels) == 1 else 2
    if ch == 2:
        return ensure_stereo(data)
    n = data.shape[1]
    if n == 1:
        return data, None
    mono = data.mean(axis=1, keepdims=True)
    return mono, f"{n}ch → mono (downmixed)"


def resample_audio(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or data.size == 0:
        return data
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    cols = [
        resample_poly(data[:, c], up, down).astype(np.float64, copy=False)
        for c in range(data.shape[1])
    ]
    return np.stack(cols, axis=1)


def process_file(
    src: Path,
    dst: Path,
    *,
    headroom_db: float,
    dither: bool,
    target_samplerate: int,
    target_channels: int = 2,
) -> dict:
    import soundfile as sf

    data, sr, subtype = load_audio(src)
    is_float = subtype in FLOAT_SUBTYPES
    source_bits = SUBTYPE_BITS.get(subtype, 16)
    bit_depth = target_bit_depth(subtype)
    is_reduction = source_bits > bit_depth
    lossy = src.suffix.lower() in LOSSY_EXTS

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    peak_dbfs = 20 * math.log10(peak) if peak > 0 else float("-inf")

    # UI may show signed dB (e.g. -1.0); engine uses magnitude below 0 dBFS.
    headroom = abs(float(headroom_db))
    target_ceiling = 10 ** (-headroom / 20.0)
    gain = 1.0
    if lossy:
        action = (
            f"decoded {src.suffix.lower()} → {bit_depth}-bit FLAC "
            f"(peak {peak_dbfs:.2f} dBFS)"
        )
    else:
        action = f"lossless re-encode ({subtype} → {bit_depth}-bit)"

    if is_float and peak > 1.0:
        gain = target_ceiling / peak
        action = (
            f"gain-reduced by {20 * math.log10(gain):.2f} dB "
            f"(peak was {peak_dbfs:.2f} dBFS), then quantized to {bit_depth}-bit"
        )
    elif is_reduction and not lossy:
        action = (
            f"{subtype} truncated to {bit_depth}-bit (FLAC ceiling), "
            f"no gain change (peak {peak_dbfs:.2f} dBFS)"
        )

    out_data = data * gain

    if int(sr) != int(target_samplerate):
        out_data = resample_audio(out_data, int(sr), int(target_samplerate))
        action += f"; resampled {int(sr)} → {int(target_samplerate)} Hz"
        sr = int(target_samplerate)

    out_data, channel_note = ensure_channels(out_data, target_channels)
    if channel_note:
        action += f"; {channel_note}"

    if dither and is_reduction:
        out_data = tpdf_dither(out_data, bit_depth)

    out_data = np.clip(
        out_data, -1.0, target_ceiling if gain != 1.0 else 1.0
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(dst),
        out_data,
        int(sr),
        subtype=f"PCM_{bit_depth}",
        format="FLAC",
    )

    try:
        from ..meta_tags import copy_tags_to_flac

        if copy_tags_to_flac(src, dst):
            action += "; metadata copied"
        else:
            action += "; metadata not copied"
    except Exception as exc:
        action += f"; metadata copy failed ({exc})"

    return {
        "file": str(src),
        "source_subtype": subtype,
        "peak_dbfs": f"{peak_dbfs:.2f}",
        "gain_db": f"{20 * math.log10(gain):.2f}" if gain != 1.0 else "0.00",
        "action": action,
        "output_bit_depth": bit_depth,
        "status": "ok",
    }


def _log_file_result(
    log: LogFn,
    path: Path,
    status: str,
    action: str,
    *,
    index: int | None = None,
    total: int | None = None,
) -> None:
    """=== [i/n] file === + Classify-style ``[skip]/detail`` (yellow) / fail / ok."""
    from ..run_summary import file_progress_header

    log(file_progress_header(path.name, index, total), "info")
    detail = (action or "").strip()
    label = str(status or "").strip().lower() or "ok"
    if label == "skip":
        log(f"  [skip] {detail}" if detail else "  [skip]", "warn")
        return
    if label == "fail":
        log(f"  [fail] {detail}" if detail else "  [fail]", "err")
        return
    # ok — green badge + optional description (single chip → no blank after)
    log(f"  {label}", "info")
    if detail:
        log(f"  {detail}", "ok")


def _cpu_core_count() -> int:
    return max(1, int(os.cpu_count() or 4))


def _convert_one_worker(payload: dict) -> dict:
    """Process-pool worker: decode → FLAC write → optional source delete.

    Returns a picklable result dict (no Path objects in the response).
    """
    src = Path(payload["src"])
    dst = Path(payload["dst"])
    write_path = Path(payload["write_path"])
    tmp_s = payload.get("tmp_path") or ""
    tmp_path = Path(tmp_s) if tmp_s else None
    same_path = bool(payload.get("same_path"))
    inplace = bool(payload.get("inplace"))
    origin_tag = str(payload.get("origin_tag") or "")
    try:
        row = process_file(
            src,
            write_path,
            headroom_db=float(payload["headroom_db"]),
            dither=bool(payload["dither"]),
            target_samplerate=int(payload["target_samplerate"]),
            target_channels=int(payload["target_channels"]),
        )
        if tmp_path is not None:
            tmp_path.replace(dst)
        elif inplace and not same_path and src.exists():
            try:
                src.unlink()
                row["action"] = (
                    str(row.get("action") or "")
                    + f"; removed source {src.suffix.lower()}"
                )
            except OSError as exc:
                row["action"] = (
                    str(row.get("action") or "")
                    + f"; could not remove source ({exc})"
                )
        if origin_tag:
            row["action"] = (
                str(row.get("action") or "") + f"; named …{origin_tag}.flac"
            )
        gained = row.get("gain_db") not in ("0.00", "", None)
        return {
            "ok": True,
            "src_name": src.name,
            "row": row,
            "gained": bool(gained),
        }
    except Exception as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {
            "ok": False,
            "src_name": src.name,
            "row": {
                "file": str(src),
                "source_subtype": "ERROR",
                "peak_dbfs": "",
                "gain_db": "",
                "action": f"FAILED: {exc}",
                "output_bit_depth": "",
                "status": "fail",
            },
            "gained": False,
            "error": str(exc),
        }


def run_convert_to_flac(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    inplace: bool = False,
    recursive: bool = True,
    skip_optimal: bool = True,
    tag_lossy_filename: bool = True,
    max_workers: Optional[int] = None,
    headroom_db: float = 1.0,
    dither: bool = True,
    target_samplerate: int = 44100,
    target_channels: int = 2,
    write_report: bool = True,
    on_log: Optional[LogFn] = None,
    on_progress: Optional[ProgressFn] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Batch-convert a folder tree to FLAC. Returns counts dict.

    ``inplace=True`` writes beside sources (``.wav`` → ``.flac``, then deletes
    the original when different). Already-optimal FLAC is left untouched.

    ``tag_lossy_filename`` appends e.g. ``_mp3-320`` to the FLAC stem when the
    source was lossy (then deletes the original on inplace).

    ``max_workers`` is how many files to convert in parallel (CPU cores).
    """

    def log(msg: str, tag: str = "info") -> None:
        if on_log:
            on_log(msg, tag)

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    try:
        _ffmpeg_tools()
    except RuntimeError:
        raise
    try:
        import soundfile as sf  # noqa: F401 — FLAC write + already-optimal probe
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for Convert FLAC output. Install with:\n"
            "  pip install soundfile"
        ) from exc

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found:\n{input_dir}")

    if inplace:
        output_dir = input_dir
    else:
        if output_dir is None or not str(output_dir).strip():
            raise ValueError("Output folder is required unless overwrite in place is on.")
        output_dir = Path(output_dir)
        if output_dir.resolve() == input_dir.resolve():
            raise ValueError(
                "Output must differ from Input unless overwrite in place is on."
            )

    ch = 1 if int(target_channels) == 1 else 2
    ch_label = "mono" if ch == 1 else "stereo"

    t0 = time.monotonic()
    files = find_audio_files(input_dir, recursive=recursive)
    total = max(1, len(files))
    converted = copied = skipped = errors = gained_n = 0
    error_files: list[str] = []
    report_rows: list[dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(max_workers or _cpu_core_count()), _cpu_core_count()))
    log(f"{len(files):,} file(s) under {input_dir}", "info")
    log(
        f"Target: FLAC · {int(target_samplerate)} Hz · {ch_label} · ≤24-bit"
        + (" · dither on" if dither else "")
        + (" · lossy name suffix" if tag_lossy_filename else "")
        + f" · {workers} core(s)"
        + (" · overwrite in place" if inplace else f" · → {output_dir}"),
        "info",
    )

    def live(n: int, count: int, label: str) -> None:
        log(f"__live_progress__\t{int(n)}\t{int(count)}\t{label}", "")

    live(0, total, "files")

    # Resolve destinations; finish skip/copy on this thread; queue re-encodes.
    jobs: list[dict] = []
    done = 0

    def bump_progress() -> None:
        nonlocal done
        done += 1
        if on_progress:
            on_progress(done, total, "converting")
        if done == 1 or done == total or done % 5 == 0:
            live(done, total, "files")

    for src in files:
        if stopped():
            break

        if inplace:
            dst = src.with_suffix(".flac")
        else:
            dst = output_dir / src.relative_to(input_dir).with_suffix(".flac")
        origin_tag = ""
        if tag_lossy_filename and src.suffix.lower() in LOSSY_EXTS:
            origin_tag = lossy_origin_suffix(src)
            if origin_tag and not _LOSSY_ORIGIN_STEM_RE.search(dst.stem):
                dst = dst.with_name(f"{dst.stem}{origin_tag}{dst.suffix}")
        same_path = src.resolve() == dst.resolve()

        try:
            if skip_optimal and is_already_optimal(
                src, int(target_samplerate), target_channels=ch
            ):
                import soundfile as sf

                info = sf.info(str(src))
                if same_path:
                    action = (
                        f"already FLAC {info.samplerate} Hz {info.subtype} "
                        f"{info.channels}ch — left unchanged"
                    )
                    skipped += 1
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    action = (
                        f"already FLAC {info.samplerate} Hz {info.subtype} "
                        f"{info.channels}ch — copied unchanged"
                    )
                    copied += 1
                    skipped += 1
                row = {
                    "file": str(src),
                    "source_subtype": info.subtype,
                    "peak_dbfs": "",
                    "gain_db": "0.00",
                    "action": action,
                    "output_bit_depth": SUBTYPE_BITS.get(info.subtype, ""),
                    "status": "skip",
                }
                report_rows.append(row)
                _log_file_result(
                    log, src, "skip", action, index=done + 1, total=total,
                )
                bump_progress()
                continue
        except Exception as exc:
            errors += 1
            error_files.append(src.name)
            row = {
                "file": str(src),
                "source_subtype": "ERROR",
                "peak_dbfs": "",
                "gain_db": "",
                "action": f"FAILED: {exc}",
                "output_bit_depth": "",
                "status": "fail",
            }
            report_rows.append(row)
            _log_file_result(
                log, src, "fail", str(exc), index=done + 1, total=total,
            )
            bump_progress()
            continue

        tmp_path: Optional[Path] = None
        write_path = dst
        if same_path:
            tmp_path = dst.with_name(dst.stem + ".__stem_cvt__.flac")
            write_path = tmp_path
        jobs.append(
            {
                "src": str(src),
                "dst": str(dst),
                "write_path": str(write_path),
                "tmp_path": str(tmp_path) if tmp_path is not None else "",
                "same_path": same_path,
                "inplace": bool(inplace),
                "origin_tag": origin_tag,
                "headroom_db": float(headroom_db),
                "dither": bool(dither),
                "target_samplerate": int(target_samplerate),
                "target_channels": int(ch),
            }
        )

    if not stopped() and jobs:
        log(f"Converting {len(jobs):,} file(s) · {workers} core(s)", "info")
        if workers == 1:
            for payload in jobs:
                if stopped():
                    break
                result = _convert_one_worker(payload)
                row = result["row"]
                report_rows.append(row)
                if result.get("ok"):
                    converted += 1
                    if result.get("gained"):
                        gained_n += 1
                    _log_file_result(
                        log,
                        Path(payload["src"]),
                        "ok",
                        str(row.get("action") or ""),
                        index=done + 1,
                        total=total,
                    )
                else:
                    errors += 1
                    error_files.append(str(result.get("src_name") or ""))
                    _log_file_result(
                        log,
                        Path(payload["src"]),
                        "fail",
                        str(result.get("error") or row.get("action") or ""),
                        index=done + 1,
                        total=total,
                    )
                bump_progress()
        else:
            with integrity_pool(max_workers=workers) as pool:
                futures = {
                    pool.submit(_convert_one_worker, payload): payload
                    for payload in jobs
                }
                for future in as_completed(futures):
                    if stopped():
                        for f in futures:
                            f.cancel()
                        break
                    payload = futures[future]
                    src = Path(payload["src"])
                    try:
                        result = future.result()
                    except Exception as exc:
                        errors += 1
                        error_files.append(src.name)
                        row = {
                            "file": str(src),
                            "source_subtype": "ERROR",
                            "peak_dbfs": "",
                            "gain_db": "",
                            "action": f"FAILED: {exc}",
                            "output_bit_depth": "",
                            "status": "fail",
                        }
                        report_rows.append(row)
                        _log_file_result(
                            log, src, "fail", str(exc),
                            index=done + 1, total=total,
                        )
                        bump_progress()
                        continue
                    row = result["row"]
                    report_rows.append(row)
                    if result.get("ok"):
                        converted += 1
                        if result.get("gained"):
                            gained_n += 1
                        _log_file_result(
                            log, src, "ok", str(row.get("action") or ""),
                            index=done + 1, total=total,
                        )
                    else:
                        errors += 1
                        error_files.append(str(result.get("src_name") or src.name))
                        _log_file_result(
                            log,
                            src,
                            "fail",
                            str(result.get("error") or row.get("action") or ""),
                            index=done + 1,
                            total=total,
                        )
                    bump_progress()

    log("__live_progress_end__", "")

    report_path = ""
    if write_report and report_rows:
        report_path = str(output_dir / "conversion_report.csv")
        with open(report_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "file",
                    "source_subtype",
                    "peak_dbfs",
                    "gain_db",
                    "action",
                    "output_bit_depth",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(report_rows)

    if stopped():
        log("Convert stopped.", "warn")

    stat_lines: list[tuple[str, str]] = [
        (
            f"Converted: {converted:,} | Copied (optimal): {copied:,} | "
            f"Skipped: {skipped:,}",
            "info",
        ),
        (f"Gain-reduced (float overs): {gained_n:,}", "info"),
        (f"Cores used: {workers}", "info"),
    ]
    if report_path:
        stat_lines.append((f"Report: {report_path}", "info"))
    if errors:
        stat_lines.append((f"Errors: {errors:,}", "err"))
        for name in error_files:
            stat_lines.append((f"  {name}", "err"))

    emit_run_summary(
        log,
        "Convert",
        elapsed=time.monotonic() - t0,
        files=len(files),
        stat_lines=stat_lines,
    )

    return {
        "files": len(files),
        "converted": converted,
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
        "gained": gained_n,
        "workers": workers,
        "report": report_path,
    }

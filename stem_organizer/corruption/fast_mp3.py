"""Fast MP3 structural frame-walk verifier.

Ported from AudioTester ``MP3Decoder.cpp`` (James Chapman / VUPlayer, MIT).
Does **not** Huffman-decode — header sync + frame length geometry only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# MIT: Copyright (c) 2015 James Chapman — https://www.vuplayer.com

MP3_EXTS = frozenset({".mp3", ".mp2", ".m2a"})

MP3_BITRATES = (
    (0, 0, 0, 0, 0),
    (32, 32, 32, 32, 8),
    (64, 48, 40, 48, 16),
    (96, 56, 48, 56, 24),
    (128, 64, 56, 64, 32),
    (160, 80, 64, 80, 40),
    (192, 96, 80, 96, 48),
    (224, 112, 96, 112, 56),
    (256, 128, 112, 128, 64),
    (288, 160, 128, 144, 80),
    (320, 192, 160, 160, 96),
    (352, 224, 192, 176, 112),
    (384, 256, 224, 192, 128),
    (416, 320, 256, 224, 144),
    (448, 384, 320, 256, 160),
    (0, 0, 0, 0, 0),
)
MP3_SAMPRATES = (
    (44100, 22050, 11025),
    (48000, 24000, 12000),
    (32000, 16000, 8000),
    (0, 0, 0),
)
MP3_CRCBYTES = ((32, 17), (17, 9))  # [mono][mpeg2?]

MPEG1, MPEG2, MPEG2_5 = 3, 2, 0
LAYER1, LAYER2, LAYER3 = 3, 2, 1
MAX_RESYNC = 65536
APE_TAG_FOOTER_BYTES = 32
APE_TAG_FOOTER_ID = b"APETAGEX"
APE_TAG_FLAG_CONTAINS_HEADER = 1 << 31


@dataclass
class FastResult:
    ok: bool
    error: str = ""


@dataclass
class FrameAudit:
    """Foobar-style Expected N / decoded M frame check."""

    actual: int = 0
    expected: int = 0  # from Xing/Info; 0 = unknown
    sample_rate: int = 0
    detail: str = ""

    @property
    def mismatch(self) -> bool:
        return bool(self.expected and self.actual and self.expected != self.actual)


def _crc16(crc: int, buf: bytes) -> int:
    for b in buf:
        k = b << 8
        for _ in range(8):
            k <<= 1
            crc <<= 1
            if (crc ^ k) & 0x10000:
                crc ^= 0x8005
    return crc & 0xFFFF


def _id3v2_len(data: bytes) -> tuple[int, str]:
    if len(data) < 10:
        return 0, ""
    if data[0:3] != b"ID3":
        return 0, ""
    if data[3] >= 0xFF or data[4] >= 0xFF:
        return 0, "BAD ID3v2 TAG"
    if any(data[i] >= 0x80 for i in (6, 7, 8, 9)):
        return 0, "BAD ID3v2 TAG"
    size = (
        (data[6] << 21)
        | (data[7] << 14)
        | (data[8] << 7)
        | data[9]
    )
    extra = 20 if (data[3] == 4 and (data[5] & 0x10)) else 10
    total = size + extra
    if total <= 0:
        return 0, "BAD ID3v2 TAG"
    return total, ""


def _lyrics_len_at(data: bytes, end: int) -> tuple[int, str]:
    """Return lyrics tag length ending at ``end`` (exclusive), or 0."""
    if end < 9:
        return 0, ""
    start = end - 9
    tag = data[start:end]
    if tag == b"LYRICSEND":
        window = max(0, end - 5100)
        chunk = data[window:end]
        idx = chunk.find(b"LYRICSBEGIN")
        if idx < 0:
            return 0, "BAD LYRICS3v1 TAG"
        return len(chunk) - idx, ""
    if tag == b"LYRICS200" and end >= 15:
        # length digits sit 6 bytes before LYRICS200
        dig = data[end - 15 : end - 9]
        try:
            length = int(dig.decode("ascii", errors="ignore") or "0")
        except ValueError:
            length = 0
        if length and end >= 15 + length:
            begin_at = end - 15 - length
            if data[begin_at : begin_at + 11] == b"LYRICSBEGIN":
                return length + 15, ""
            return 0, "BAD LYRICS3v2 TAG"
    return 0, ""


def _footer_len(data: bytes) -> tuple[int, str]:
    n = len(data)
    offset = 0
    err = ""
    if n >= 128 and data[n - 128 : n - 125] == b"TAG":
        offset -= 128
        lyr, e = _lyrics_len_at(data, n + offset)
        if e:
            err = e
        offset -= lyr
    ape_end = n + offset
    if ape_end >= APE_TAG_FOOTER_BYTES:
        footer = data[ape_end - APE_TAG_FOOTER_BYTES : ape_end]
        if footer[:8] == APE_TAG_FOOTER_ID:
            size = struct.unpack_from("<i", footer, 12)[0]
            flags = struct.unpack_from("<i", footer, 20)[0]
            if size < APE_TAG_FOOTER_BYTES or size > n:
                err = err or "BAD APE TAG"
            else:
                offset -= size
                if flags & APE_TAG_FLAG_CONTAINS_HEADER:
                    offset -= APE_TAG_FOOTER_BYTES
                lyr, e = _lyrics_len_at(data, n + offset)
                if e:
                    err = e
                offset -= lyr
    return -offset, err


def _frame_length(header: int, state: dict) -> int:
    if header <= 0xFFE00000:
        return 0
    version = (header >> 19) & 0x03
    layer = (header >> 17) & 0x03
    bitrate_i = (header >> 12) & 0x0F
    samplerate_i = (header >> 10) & 0x03
    padding = (header >> 9) & 0x01
    mono = ((header >> 6) & 0x03) == 0x03
    crc = not ((header >> 16) & 0x01)

    column = 5
    if version == MPEG1:
        column = {LAYER1: 0, LAYER2: 1, LAYER3: 2}.get(layer, 5)
    elif version in (MPEG2, MPEG2_5):
        column = {LAYER1: 3, LAYER2: 4, LAYER3: 4}.get(layer, 5)
    if column >= 5:
        return 0
    bitrate = 1000 * MP3_BITRATES[bitrate_i][column]
    if version == MPEG1:
        sr_col = 0
    elif version == MPEG2:
        sr_col = 1
    elif version == MPEG2_5:
        sr_col = 2
    else:
        return 0
    samplerate = MP3_SAMPRATES[samplerate_i][sr_col]
    if not samplerate or not bitrate:
        return 0

    state.update(
        version=version,
        layer=layer,
        bitrate=bitrate,
        samplerate=samplerate,
        padding=padding,
        mono=mono,
        crc=crc,
        rate=samplerate,
    )
    if layer == LAYER1:
        state["samplepos"] = state.get("samplepos", 0) + 384
        return (12 * bitrate // samplerate + padding) * 4
    if layer == LAYER2:
        state["samplepos"] = state.get("samplepos", 0) + 1152
        return 144 * bitrate // samplerate + padding
    if layer == LAYER3:
        if version == MPEG1:
            state["samplepos"] = state.get("samplepos", 0) + 1152
            return 144 * bitrate // samplerate + padding
        state["samplepos"] = state.get("samplepos", 0) + 576
        return 72 * bitrate // samplerate + padding
    return 0


def _check_crc(data: bytes, offset: int, framelen: int, state: dict) -> str:
    if not state.get("crc") or state.get("layer") != LAYER3:
        return ""
    mono = 1 if state.get("mono") else 0
    mpeg2 = 0 if state.get("version") == MPEG1 else 1
    crc_bytes = MP3_CRCBYTES[mono][mpeg2]
    # After 4-byte header: 2-byte CRC then side info
    start = offset + 4
    need = 2 + crc_bytes
    if start + need > len(data) or start + need > offset + framelen:
        return ""
    stored = (data[start] << 8) | data[start + 1]
    # AudioTester feeds header last 2 bytes? It seeks -2 from after reading header
    # Simplification: CRC over side-info only with seed — match AudioTester buf layout:
    # crcbuf[0:2] = first 2 of side? Actually Read after Seek(-2) from post-header position
    # which is at offset+4, Seek(-2) → offset+2 (last 2 of header), then crc, then side.
    hdr_tail = data[offset + 2 : offset + 4]
    side = data[offset + 6 : offset + 6 + crc_bytes]
    if len(side) < crc_bytes:
        return ""
    calc = _crc16(0xFFFF, hdr_tail + side)
    if calc != stored:
        pos = state.get("samplepos", 0)
        rate = max(1, int(state.get("rate") or 44100))
        sec = pos / rate
        return f"CRC ERROR @ {int(sec) // 60}m {int(sec) % 60:02d}s"
    return ""


def _resync(data: bytes, pos: int, end: int, last_header: int, state: dict) -> tuple[int, int, bool]:
    """Return (new_pos, new_header, ok). Iterative (AudioTester used nested calls)."""
    header = 0
    resync = MAX_RESYNC
    candidate = last_header
    while resync > 0 and pos < end:
        resync -= 1
        header = ((header << 8) | data[pos]) & 0xFFFFFFFF
        pos += 1
        fl = _frame_length(header, state)
        if not fl:
            continue
        if candidate:
            if (header & 0xFFFE0C00) == (candidate & 0xFFFE0C00):
                return pos - 4, header, True
            # Mismatch after a candidate — keep scanning with same candidate
            continue
        # First plausible frame: require a second matching header
        candidate = header
        header = 0
    return pos, header, False

def _xing_expected_frames(data: bytes, frame_off: int, framelen: int) -> int:
    """Read Xing/Info frame count from the first MPEG frame (0 if absent)."""
    if framelen < 16 or frame_off < 0:
        return 0
    end = min(len(data), frame_off + framelen)
    window = data[frame_off:end]
    for magic in (b"Xing", b"Info"):
        idx = window.find(magic)
        if idx < 0 or idx + 8 > len(window):
            continue
        flags = struct.unpack_from(">I", window, idx + 4)[0]
        if not (flags & 0x0001):
            return 0
        if idx + 12 > len(window):
            return 0
        return int(struct.unpack_from(">I", window, idx + 8)[0])
    return 0


def _xing_frames_field_offset(data: bytes, frame_off: int, framelen: int) -> int:
    """Absolute file offset of Xing/Info frames field, or -1."""
    if framelen < 16 or frame_off < 0:
        return -1
    end = min(len(data), frame_off + framelen)
    window = data[frame_off:end]
    for magic in (b"Xing", b"Info"):
        idx = window.find(magic)
        if idx < 0 or idx + 12 > len(window):
            continue
        flags = struct.unpack_from(">I", window, idx + 4)[0]
        if not (flags & 0x0001):
            return -1
        return frame_off + idx + 8
    return -1


def sync_xing_to_decoded(
    path: Path | str,
    *,
    ffprobe_bin: str | None = None,
) -> bool:
    """Rewrite Xing/Info frame count to match ffprobe decode count.

    libmp3lame often writes Expected N while decoders report N−1; that residue
    made every post-Fix verify fail even when the stream itself was clean.
    Returns True when matched, patched, or nothing to sync.
    """
    path = Path(path)
    audit = audit_mp3_frames(path, ffprobe_bin=ffprobe_bin)
    if not audit.expected or not audit.actual:
        return True
    if audit.expected == audit.actual:
        return True
    try:
        data = bytearray(path.read_bytes())
    except OSError:
        return False

    header_len, _ = _id3v2_len(data[:10])
    footer_len, _ = _footer_len(data)
    audio_end = len(data) - footer_len
    pos = header_len
    state: dict = {"samplepos": 0}
    if pos + 4 > audio_end:
        return False
    header = struct.unpack_from(">I", data, pos)[0]
    fl = _frame_length(header, state)
    if not fl:
        new_pos, new_h, ok = _resync(data, pos, audio_end, 0, state)
        if not ok:
            return False
        pos = new_pos
        header = new_h
        fl = _frame_length(header, state)
        if not fl:
            return False

    off = _xing_frames_field_offset(data, pos, fl)
    if off < 0:
        return False
    struct.pack_into(">I", data, off, int(audit.actual))
    try:
        path.write_bytes(data)
    except OSError:
        return False
    return True


def audit_mp3_frames(
    path: Path | str,
    *,
    ffprobe_bin: str | None = None,
) -> FrameAudit:
    """Compare Xing/Info expected frames to ffprobe-decoded count (foobar N vs N−1)."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return FrameAudit(detail=str(exc))
    if len(data) < 4:
        return FrameAudit(detail="UNRECOGNISED FORMAT")

    header_len, _ = _id3v2_len(data[:10])
    footer_len, _ = _footer_len(data)
    audio_end = len(data) - footer_len
    if audio_end <= header_len + 4:
        return FrameAudit(detail="UNRECOGNISED FORMAT")

    # Find first frame + Xing expected count; also count structural frames as fallback
    pos = header_len
    state: dict = {"samplepos": 0}
    header = struct.unpack_from(">I", data, pos)[0]
    fl = _frame_length(header, state)
    if not fl:
        # resync once
        new_pos, new_h, ok = _resync(data, pos, audio_end, 0, state)
        if not ok:
            return FrameAudit(detail="UNRECOGNISED FORMAT")
        pos = new_pos
        header = new_h
        fl = _frame_length(header, state)
        if not fl:
            return FrameAudit(detail="UNRECOGNISED FORMAT")

    rate = int(state.get("rate") or 0)
    expected = _xing_expected_frames(data, pos, fl)

    # Walk remaining frames for a structural count (fallback if no ffprobe)
    structural = 1
    p2 = pos + fl
    last_header = header
    while p2 + 4 <= audio_end:
        h = struct.unpack_from(">I", data, p2)[0]
        fl2 = _frame_length(h, state)
        if not fl2 or (h & 0xFFFE0C00) != (last_header & 0xFFFE0C00):
            break
        structural += 1
        p2 += fl2

    decoded = 0
    probe = ffprobe_bin
    if not probe:
        try:
            from .tools import find_ffprobe

            probe = find_ffprobe()
        except Exception:
            probe = None
    if probe and expected:
        try:
            import subprocess

            try:
                from ffmpeg_bootstrap import subprocess_kwargs as _skw
            except Exception:  # pragma: no cover
                def _skw() -> dict:
                    return {}

            proc = subprocess.run(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=600,
                **_skw(),
            )
            out = (proc.stdout or "").strip().splitlines()
            if out and out[0].isdigit():
                decoded = int(out[0])
        except Exception:
            decoded = 0

    actual = decoded or structural
    audit = FrameAudit(actual=actual, expected=expected, sample_rate=rate)
    if expected and actual and expected != actual:
        audit.detail = (
            f"Expected {expected} MP3 frames, decoded only {actual}, bad header?"
        )
    return audit


def verify_mp3_fast(path: Path | str) -> FastResult:
    """Structural MP3 check. Returns ok=False with error string on failure."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return FastResult(False, str(exc))
    if len(data) < 4:
        return FastResult(False, "UNRECOGNISED FORMAT")

    header_len, herr = _id3v2_len(data[:10])
    footer_len, ferr = _footer_len(data)
    err = herr or ferr
    audio_end = len(data) - footer_len
    if audio_end <= header_len:
        return FastResult(False, err or "UNRECOGNISED FORMAT")

    pos = header_len
    last_header = 0
    state: dict = {"samplepos": 0}
    rate = 44100

    while pos < audio_end:
        if pos + 4 > audio_end:
            return FastResult(False, "TRUNCATED")
        header = struct.unpack_from(">I", data, pos)[0]
        fl = _frame_length(header, state)
        if fl and (
            (header & 0xFFFE0C00) == (last_header & 0xFFFE0C00) or not last_header
        ):
            if not last_header:
                last_header = header
                rate = int(state.get("rate") or rate)
            crc_err = _check_crc(data, pos, fl, state)
            if crc_err:
                return FastResult(False, crc_err)
            pos += fl
            if pos < audio_end:
                continue
            if pos == audio_end:
                return FastResult(True, "")
            return FastResult(False, "TRUNCATED")

        # Lost sync
        if state.get("samplepos"):
            new_pos, new_h, ok = _resync(data, pos + 1, audio_end, last_header, state)
            if ok:
                sec = state["samplepos"] / max(1, rate)
                return FastResult(
                    False, f"LOST SYNC @ {int(sec) // 60}m {int(sec) % 60:02d}s"
                )
            return FastResult(False, "LOST SYNC @ END OF FILE")
        new_pos, new_h, ok = _resync(data, pos, audio_end, 0, state)
        if not ok:
            return FastResult(False, err or "UNRECOGNISED FORMAT")
        if header_len:
            return FastResult(False, "BAD ID3v2 TAG")
        return FastResult(False, "BAD STARTING SYNC")

    return FastResult(True if last_header else False, "" if last_header else "UNRECOGNISED FORMAT")

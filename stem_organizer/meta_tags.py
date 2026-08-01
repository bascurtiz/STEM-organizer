"""Shared audio metadata writers (SDR, COMPRESSION, …).

Logical field names map to Vorbis / ID3 TXXX / MP4 freeform the same way
Genre & Gender and PANNs vocal tags do.
"""

from __future__ import annotations

from pathlib import Path

from .file_writable import ensure_writable


def _wav_has_leading_id3(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(12)
    except OSError:
        return False
    if len(head) < 12:
        return False
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return False
    return head.startswith(b"ID3")


def _salvage_wav_leading_id3(path: Path) -> bool:
    """Strip prepended MP3-style ID3 so RIFF/WAVE starts at offset 0."""
    if not _wav_has_leading_id3(path):
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    idx = data.find(b"RIFF")
    if idx <= 0 or data[idx + 8 : idx + 12] != b"WAVE":
        return False
    try:
        path.write_bytes(data[idx:])
    except OSError:
        return False
    return True


def _write_mp3_txxx(path: Path, key: str, text: str) -> bool:
    """Write ID3v2.4 TXXX onto MPEG content (also misnamed .wav)."""
    from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
    from mutagen.mp3 import MP3

    try:
        audio = MP3(str(path))
        if audio.tags is None:
            audio.add_tags()
        tags = audio.tags
        assert tags is not None
        if not isinstance(tags, ID3):
            try:
                tags = ID3(str(path))
            except ID3NoHeaderError:
                tags = ID3()
        tags.delall(f"TXXX:{key}")
        tags.add(TXXX(encoding=3, desc=key, text=[text]))
        tags.save(str(path), v2_version=4)
        return True
    except Exception:
        try:
            tags = ID3(str(path))
        except Exception:
            tags = ID3()
        tags.delall(f"TXXX:{key}")
        tags.add(TXXX(encoding=3, desc=key, text=[text]))
        tags.save(str(path), v2_version=4)
        return True


def write_custom_tag(path: Path | str, field: str, value: str) -> bool:
    """Write a single custom text tag. Returns True on success.

    FLAC: Vorbis comment ``field`` (uppercased for SDR/COMPRESSION/VOCAL_TYPE).
    MP3: ID3v2 ``TXXX:FIELD``.
    OGG / Opus: Vorbis comment ``FIELD``.
    WAV / AIFF: ID3v2.4 ``TXXX:FIELD`` in the container (WAV uses RIFF ``id3`` chunk).
    Misnamed MPEG-in-.wav: tagged as MP3 (foobar \"wrong extension\" case).
    M4A: iTunes freeform ``----:com.apple.iTunes:FIELD``.
    """
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import TXXX
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    from .corruption.sniff import is_mpeg_audio, is_riff_wave

    path = Path(path)
    text = str(value).strip()
    if not text:
        return False
    key = str(field).strip().upper() or "TAG"
    suffix = path.suffix.lower()
    ensure_writable(path)

    try:
        if suffix == ".flac":
            audio = FLAC(str(path))
            audio[key] = [text]
            audio.save()
            return True

        if suffix == ".wav":
            # Older writers used plain ID3.save() and prepended ID3 — mutagen.wave
            # then refuses to open (Invalid chunk ID 'ID3…'). Repair first.
            _salvage_wav_leading_id3(path)

            if is_riff_wave(path):
                # Must use mutagen.wave._WaveID3 so the tag lands in a RIFF "id3"
                # chunk. Plain ID3.save() prepends an MP3-style header and corrupts WAV.
                audio = WAVE(str(path))
                if audio.tags is None:
                    audio.add_tags()
                tags = audio.tags
                assert tags is not None
                tags.delall(f"TXXX:{key}")
                tags.add(TXXX(encoding=3, desc=key, text=[text]))
                audio.save(v2_version=4)
                return True

            # foobar: "wrong extension: wav; Correct handler: MPEG Decoder"
            if is_mpeg_audio(path):
                return _write_mp3_txxx(path, key, text)
            return False

        if suffix in (".aif", ".aiff"):
            audio = AIFF(str(path))
            if audio.tags is None:
                audio.add_tags()
            tags = audio.tags
            assert tags is not None
            tags.delall(f"TXXX:{key}")
            tags.add(TXXX(encoding=3, desc=key, text=[text]))
            audio.save()
            return True

        if suffix == ".mp3":
            return _write_mp3_txxx(path, key, text)

        if suffix == ".ogg":
            audio = OggVorbis(str(path))
            audio[key] = [text]
            audio.save()
            return True

        if suffix == ".opus":
            audio = OggOpus(str(path))
            audio[key] = [text]
            audio.save()
            return True

        if suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(path))
            ff = f"----:com.apple.iTunes:{key}"
            audio[ff] = [MP4FreeForm(text.encode("utf-8"))]
            audio.save()
            return True
    except Exception:
        return False
    return False


def write_sdr_tag(path: Path | str, score: float) -> bool:
    """Write SI-SDR dB value to the ``SDR`` tag (one decimal)."""
    return write_custom_tag(path, "SDR", f"{float(score):.1f}")


def write_compression_tag(path: Path | str, value: str) -> bool:
    """Write ``COMPRESSION`` as ``lossless`` or ``lossy``."""
    v = str(value).strip().lower()
    if v not in ("lossless", "lossy"):
        return False
    return write_custom_tag(path, "COMPRESSION", v)


def write_key_tag(path: Path | str, value: str) -> bool:
    """Write musical key to the DJ-standard Initial key field.

    ID3: ``TKEY`` · Vorbis/FLAC/Ogg: ``INITIALKEY`` · MP4: iTunes ``initialkey``.
    """
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, ID3NoHeaderError, TKEY
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    from .corruption.sniff import is_mpeg_audio, is_riff_wave
    from .musical_keys import normalize_key_short

    path = Path(path)
    short = normalize_key_short(str(value))
    if not short:
        short = str(value).strip()
    if not short:
        return False
    # ID3 TKEY is max 3 characters (e.g. Dbm, A, Am).
    text = short[:3]
    suffix = path.suffix.lower()
    ensure_writable(path)

    try:
        if suffix == ".flac":
            audio = FLAC(str(path))
            audio["INITIALKEY"] = [text]
            audio.save()
            return True

        if suffix == ".ogg":
            audio = OggVorbis(str(path))
            audio["INITIALKEY"] = [text]
            audio.save()
            return True

        if suffix == ".opus":
            audio = OggOpus(str(path))
            audio["INITIALKEY"] = [text]
            audio.save()
            return True

        if suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(path))
            audio["----:com.apple.iTunes:initialkey"] = [
                MP4FreeForm(text.encode("utf-8"))
            ]
            audio.save()
            return True

        if suffix in (".mp3", ".wav", ".aif", ".aiff"):
            use_mp3 = suffix == ".mp3" or (
                suffix == ".wav" and not is_riff_wave(path) and is_mpeg_audio(path)
            )
            if suffix == ".wav" and is_riff_wave(path):
                _salvage_wav_leading_id3(path)
            audio = None
            tags = None
            try:
                if use_mp3:
                    audio = MP3(str(path))
                elif suffix in (".aif", ".aiff"):
                    audio = AIFF(str(path))
                else:
                    audio = WAVE(str(path))
                if audio.tags is None:
                    audio.add_tags()
                tags = audio.tags
            except Exception:
                try:
                    tags = ID3(str(path))
                except ID3NoHeaderError:
                    tags = ID3()
                audio = None
            assert tags is not None
            tags.delall("TKEY")
            tags.add(TKEY(encoding=3, text=[text]))
            if audio is not None:
                audio.save()
            else:
                tags.save(str(path), v2_version=4)
            return True
    except Exception:
        return False
    return False


def read_key_tag(path: Path | str) -> str:
    """Read Initial key (TKEY / INITIALKEY), with legacy KEY / TXXX:KEY fallback."""
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    path = Path(path)
    suffix = path.suffix.lower()

    def _first(val) -> str:
        if val is None:
            return ""
        if isinstance(val, (list, tuple)):
            return str(val[0]).strip() if val else ""
        return str(val).strip()

    try:
        if suffix == ".flac":
            audio = FLAC(str(path))
            for k in ("INITIALKEY", "initialkey", "KEY", "key"):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix == ".ogg":
            audio = OggVorbis(str(path))
            for k in ("INITIALKEY", "initialkey", "KEY", "key"):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix == ".opus":
            audio = OggOpus(str(path))
            for k in ("INITIALKEY", "initialkey", "KEY", "key"):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(path))
            for desc in ("initialkey", "INITIALKEY", "KEY", "key"):
                raw = audio.get(f"----:com.apple.iTunes:{desc}")
                if not raw:
                    continue
                item = raw[0]
                if isinstance(item, (bytes, bytearray, MP4FreeForm)):
                    return bytes(item).decode("utf-8", errors="replace").strip()
                return _first(item)
            return ""

        if suffix in (".mp3", ".wav", ".aif", ".aiff"):
            from .corruption.sniff import is_mpeg_audio, is_riff_wave

            use_mp3 = suffix == ".mp3" or (
                suffix == ".wav" and not is_riff_wave(path) and is_mpeg_audio(path)
            )
            tags = None
            try:
                if use_mp3:
                    audio = MP3(str(path))
                elif suffix in (".aif", ".aiff"):
                    audio = AIFF(str(path))
                else:
                    if suffix == ".wav":
                        _salvage_wav_leading_id3(path)
                    audio = WAVE(str(path))
                tags = audio.tags
            except Exception:
                try:
                    from mutagen.id3 import ID3 as _ID3

                    tags = _ID3(str(path))
                except Exception:
                    return ""
            if tags is None:
                return ""
            frame = tags.get("TKEY")
            if frame is not None:
                return _first(getattr(frame, "text", None))
            for desc in ("KEY", "key", "INITIALKEY", "initialkey"):
                frames = tags.getall(f"TXXX:{desc}")
                if frames:
                    return _first(frames[0].text)
            return ""
    except Exception:
        return ""
    return ""


def write_comment_tag(path: Path | str, value: str) -> bool:
    """Write standard COMMENT / COMM / ©cmt (not TXXX:COMMENT)."""
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import COMM, ID3, ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    from .corruption.sniff import is_mpeg_audio, is_riff_wave

    path = Path(path)
    text = str(value).strip()
    if not text:
        return False
    suffix = path.suffix.lower()
    ensure_writable(path)

    try:
        if suffix == ".flac":
            audio = FLAC(str(path))
            audio["COMMENT"] = [text]
            audio.save()
            return True

        if suffix == ".ogg":
            audio = OggVorbis(str(path))
            audio["COMMENT"] = [text]
            audio.save()
            return True

        if suffix == ".opus":
            audio = OggOpus(str(path))
            audio["COMMENT"] = [text]
            audio.save()
            return True

        if suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(path))
            audio["\xa9cmt"] = [text]
            audio.save()
            return True

        if suffix in (".mp3", ".wav", ".aif", ".aiff"):
            use_mp3 = suffix == ".mp3" or (
                suffix == ".wav" and not is_riff_wave(path) and is_mpeg_audio(path)
            )
            if suffix == ".wav" and is_riff_wave(path):
                _salvage_wav_leading_id3(path)
            try:
                if use_mp3:
                    audio = MP3(str(path))
                elif suffix in (".aif", ".aiff"):
                    audio = AIFF(str(path))
                else:
                    audio = WAVE(str(path))
                if audio.tags is None:
                    audio.add_tags()
                tags = audio.tags
            except ID3NoHeaderError:
                tags = ID3()
                if use_mp3:
                    audio = MP3(str(path))
                    audio.tags = tags
                elif suffix in (".aif", ".aiff"):
                    audio = AIFF(str(path))
                    audio.add_tags()
                    tags = audio.tags
                else:
                    audio = WAVE(str(path))
                    audio.add_tags()
                    tags = audio.tags
            assert tags is not None
            tags.delall("COMM")
            tags.add(COMM(encoding=3, lang="eng", desc="", text=[text]))
            audio.save()
            return True
    except Exception:
        return False
    return False



CORRUPTION_VALUES = frozenset({"ok", "minor", "failed", "suspect"})


def write_corruption_tag(path: Path | str, value: str) -> bool:
    """Write ``CORRUPTION`` as ok | minor | failed | suspect."""
    v = str(value).strip().lower()
    if v not in CORRUPTION_VALUES:
        return False
    return write_custom_tag(path, "CORRUPTION", v)


def read_corruption_tag(path: Path | str) -> str:
    """Read ``CORRUPTION`` tag (lowercase), or empty string."""
    return read_custom_tag(path, "CORRUPTION").strip().lower()


def read_custom_tag(path: Path | str, field: str) -> str:
    """Read a custom text tag (SDR / COMPRESSION / VOCAL_TYPE / …).

    Tries uppercase and lowercase Vorbis keys; ID3 ``TXXX:FIELD``;
    iTunes freeform. Also checks standard genre/comment where relevant.
    """
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3NoHeaderError
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    path = Path(path)
    key = str(field).strip()
    if not key:
        return ""
    key_u = key.upper()
    key_l = key.lower()
    suffix = path.suffix.lower()

    def _first(val) -> str:
        if val is None:
            return ""
        if isinstance(val, (list, tuple)):
            return str(val[0]).strip() if val else ""
        return str(val).strip()

    try:
        if suffix == ".flac":
            audio = FLAC(str(path))
            for k in (key_u, key_l, key):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix in (".aif", ".aiff"):
            try:
                audio = AIFF(str(path))
            except Exception:
                return ""
            tags = audio.tags
            if tags is None:
                return ""
            for desc in (key_u, key_l, key):
                frames = tags.getall(f"TXXX:{desc}")
                if frames:
                    return _first(frames[0].text)
            return ""

        if suffix in (".mp3", ".wav"):
            from .corruption.sniff import is_mpeg_audio, is_riff_wave

            use_mp3 = suffix == ".mp3" or (
                suffix == ".wav" and not is_riff_wave(path) and is_mpeg_audio(path)
            )
            try:
                if use_mp3:
                    audio = MP3(str(path))
                else:
                    if suffix == ".wav":
                        _salvage_wav_leading_id3(path)
                    audio = WAVE(str(path))
            except ID3NoHeaderError:
                return ""
            except Exception:
                # Unreadable container — still try raw ID3 (MPEG / misnamed wav)
                if not use_mp3 and suffix == ".wav":
                    return ""
                try:
                    from mutagen.id3 import ID3 as _ID3

                    tags = _ID3(str(path))
                except Exception:
                    return ""
                for desc in (key_u, key_l, key):
                    frames = tags.getall(f"TXXX:{desc}")
                    if frames:
                        return _first(frames[0].text)
                return ""
            tags = audio.tags
            if tags is None:
                return ""
            if key_l == "genre":
                frame = tags.get("TCON")
                return _first(getattr(frame, "text", None))
            if key_l == "comment":
                frames = tags.getall("COMM")
                return _first(frames[0].text if frames else "")
            for desc in (key_u, key_l, key):
                frames = tags.getall(f"TXXX:{desc}")
                if frames:
                    return _first(frames[0].text)
            return ""

        if suffix == ".ogg":
            audio = OggVorbis(str(path))
            for k in (key_u, key_l, key):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix == ".opus":
            audio = OggOpus(str(path))
            for k in (key_u, key_l, key):
                if k in audio:
                    return _first(audio.get(k))
            return ""

        if suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(path))
            if key_l == "genre":
                return _first(audio.get("\xa9gen"))
            if key_l == "comment":
                return _first(audio.get("\xa9cmt"))
            for desc in (key_u, key_l, key):
                ff = f"----:com.apple.iTunes:{desc}"
                raw = audio.get(ff)
                if not raw:
                    continue
                item = raw[0]
                if isinstance(item, (bytes, bytearray, MP4FreeForm)):
                    return bytes(item).decode("utf-8", errors="replace").strip()
                return _first(item)
            return ""
    except Exception:
        return ""
    return ""


# ID3 text frames → Vorbis comment keys (FLAC destination).
_ID3_TO_VORBIS = {
    "TIT2": "TITLE",
    "TPE1": "ARTIST",
    "TPE2": "ALBUMARTIST",
    "TALB": "ALBUM",
    "TCON": "GENRE",
    "TCOM": "COMPOSER",
    "TDRC": "DATE",
    "TDOR": "DATE",
    "TYER": "DATE",
    "TRCK": "TRACKNUMBER",
    "TPOS": "DISCNUMBER",
    "TBPM": "BPM",
    "TKEY": "INITIALKEY",
    "TSRC": "ISRC",
    "TPUB": "PUBLISHER",
    "TIT1": "CONTENTGROUP",
    "TIT3": "SUBTITLE",
    "TPE3": "CONDUCTOR",
    "TPE4": "REMIXER",
}

# MP4 atoms → Vorbis
_MP4_TO_VORBIS = {
    "\xa9nam": "TITLE",
    "\xa9ART": "ARTIST",
    "aART": "ALBUMARTIST",
    "\xa9alb": "ALBUM",
    "\xa9gen": "GENRE",
    "\xa9wrt": "COMPOSER",
    "\xa9day": "DATE",
    "\xa9cmt": "COMMENT",
    "\xa9too": "ENCODER",
    "trkn": "TRACKNUMBER",
    "disk": "DISCNUMBER",
    "tmpo": "BPM",
    "----:com.apple.iTunes:initialkey": "INITIALKEY",
    "----:com.apple.iTunes:INITIALKEY": "INITIALKEY",
}


def _vorbis_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_vorbis_values(item))
        return out
    if isinstance(value, (bytes, bytearray)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            text = bytes(value).decode("latin-1", errors="replace")
        text = text.strip()
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def _set_vorbis(out, key: str, values) -> int:
    vals = _vorbis_values(values)
    if not vals:
        return 0
    # Vorbis comment keys: ASCII printable except '=' (RFC 3533).
    raw = str(key).strip()
    if not raw:
        return 0
    clean = "".join(
        ch if 0x20 <= ord(ch) <= 0x7D and ch != "=" else "_"
        for ch in raw
    ).strip("_")
    if not clean:
        return 0
    try:
        out[clean] = vals
    except Exception:
        return 0
    return 1


def _copy_id3_to_flac(tags, out) -> int:
    """Map ID3 frames onto a FLAC Vorbis comment block. Returns tag count."""
    from mutagen.flac import Picture

    n = 0
    if tags is None:
        return 0
    for frame in tags.values():
        try:
            fid = getattr(frame, "FrameID", None) or getattr(frame, "ID", None)
            if not fid:
                continue
            fid = str(fid)
            if fid == "APIC":
                try:
                    pic = Picture()
                    pic.type = int(getattr(frame, "type", 3) or 3)
                    pic.mime = str(getattr(frame, "mime", "image/jpeg") or "image/jpeg")
                    pic.desc = str(getattr(frame, "desc", "") or "")
                    pic.data = bytes(frame.data)
                    out.add_picture(pic)
                    n += 1
                except Exception:
                    pass
                continue
            if fid == "TXXX":
                desc = str(getattr(frame, "desc", "") or "").strip() or "TXXX"
                n += _set_vorbis(out, desc.upper(), getattr(frame, "text", []))
                continue
            if fid == "COMM":
                n += _set_vorbis(out, "COMMENT", getattr(frame, "text", []))
                continue
            if fid == "UFID":
                try:
                    owner = str(getattr(frame, "owner", "") or "").strip()
                    data = bytes(getattr(frame, "data", b"") or b"")
                    if owner and data:
                        n += _set_vorbis(
                            out,
                            f"UFID:{owner}",
                            data.decode("utf-8", errors="replace"),
                        )
                except Exception:
                    pass
                continue
            vorbis_key = _ID3_TO_VORBIS.get(fid)
            if vorbis_key and hasattr(frame, "text"):
                n += _set_vorbis(out, vorbis_key, frame.text)
        except Exception:
            continue
    return n


def _copy_riff_info_to_flac(src: Path, out) -> int:
    """Best-effort RIFF LIST/INFO → Vorbis (WAVs without ID3)."""
    info_map = {
        b"INAM": "TITLE",
        b"IART": "ARTIST",
        b"IPRD": "ALBUM",
        b"IGNR": "GENRE",
        b"ICMT": "COMMENT",
        b"ICRD": "DATE",
        b"ITRK": "TRACKNUMBER",
        b"ISFT": "ENCODER",
    }
    n = 0
    try:
        data = src.read_bytes()
    except OSError:
        return 0
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0
    pos = 12
    end = len(data)
    while pos + 8 <= end:
        chunk_id = data[pos : pos + 4]
        chunk_size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        pos += 8
        payload = data[pos : pos + max(0, chunk_size)]
        pos += chunk_size + (chunk_size & 1)
        if chunk_id != b"LIST" or len(payload) < 4 or payload[0:4] != b"INFO":
            continue
        ip = 4
        while ip + 8 <= len(payload):
            key = payload[ip : ip + 4]
            size = int.from_bytes(payload[ip + 4 : ip + 8], "little")
            ip += 8
            raw = payload[ip : ip + max(0, size)]
            ip += size + (size & 1)
            vorbis_key = info_map.get(key)
            if not vorbis_key:
                continue
            text = raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            if text:
                n += _set_vorbis(out, vorbis_key, text)
    return n


def copy_tags_to_flac(src: Path | str, dst: Path | str) -> bool:
    """Copy text tags and cover art from ``src`` onto destination FLAC ``dst``.

    Preserves STEM custom fields (COMPRESSION, CORRUPTION, SDR, INITIALKEY, …)
    as Vorbis comments. Returns True if any tag or picture was written.
    """
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC, Picture
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    from mutagen.wave import WAVE

    from .corruption.sniff import is_mpeg_audio, is_riff_wave

    src = Path(src)
    dst = Path(dst)
    if not src.is_file() or not dst.is_file():
        return False
    if dst.suffix.lower() != ".flac":
        return False
    if src.resolve() == dst.resolve():
        return False

    ensure_writable(dst)
    try:
        out = FLAC(str(dst))
    except Exception:
        return False
    # Clear in memory only — do NOT call out.delete() (that writes an empty
    # tag block to disk first; any later copy failure would leave tags gone).
    if out.tags is None:
        out.add_tags()
    else:
        out.tags.clear()
    out.clear_pictures()

    suffix = src.suffix.lower()
    n = 0

    try:
        if suffix == ".flac":
            audio = FLAC(str(src))
            if audio.tags:
                for key, values in list(audio.tags.items()):
                    n += _set_vorbis(out, key, values)
            for pic in list(audio.pictures):
                try:
                    out.add_picture(pic)
                    n += 1
                except Exception:
                    pass

        elif suffix == ".ogg":
            audio = OggVorbis(str(src))
            for key, values in list(audio.items()):
                n += _set_vorbis(out, key, values)

        elif suffix == ".opus":
            audio = OggOpus(str(src))
            for key, values in list(audio.items()):
                n += _set_vorbis(out, key, values)

        elif suffix == ".mp3":
            audio = MP3(str(src))
            n += _copy_id3_to_flac(audio.tags, out)

        elif suffix == ".wav":
            _salvage_wav_leading_id3(src)
            if is_riff_wave(src):
                audio = WAVE(str(src))
                n += _copy_id3_to_flac(audio.tags, out)
                if n <= 0:
                    n += _copy_riff_info_to_flac(src, out)
            elif is_mpeg_audio(src):
                audio = MP3(str(src))
                n += _copy_id3_to_flac(audio.tags, out)

        elif suffix in (".aif", ".aiff"):
            audio = AIFF(str(src))
            n += _copy_id3_to_flac(audio.tags, out)

        elif suffix in (".m4a", ".mp4", ".aac"):
            audio = MP4(str(src))
            for key, values in list(audio.items()):
                try:
                    key_s = str(key)
                    if key_s == "covr":
                        for cover in values:
                            try:
                                pic = Picture()
                                pic.type = 3
                                fmt = getattr(cover, "imageformat", None)
                                if fmt == MP4Cover.FORMAT_PNG:
                                    pic.mime = "image/png"
                                else:
                                    pic.mime = "image/jpeg"
                                pic.data = bytes(cover)
                                out.add_picture(pic)
                                n += 1
                            except Exception:
                                pass
                        continue
                    if key_s.startswith("----:"):
                        parts = key_s.split(":", 2)
                        field = parts[2] if len(parts) >= 3 else key_s
                        decoded: list[str] = []
                        for item in values:
                            if isinstance(item, (bytes, bytearray, MP4FreeForm)):
                                decoded.append(
                                    bytes(item).decode("utf-8", errors="replace")
                                )
                            else:
                                decoded.extend(_vorbis_values(item))
                        n += _set_vorbis(out, field.upper(), decoded)
                        continue
                    vorbis_key = _MP4_TO_VORBIS.get(key_s)
                    if not vorbis_key:
                        continue
                    if key_s in ("trkn", "disk") and values:
                        try:
                            num = int(values[0][0])
                            total = int(values[0][1]) if len(values[0]) > 1 else 0
                            text = f"{num}/{total}" if total else str(num)
                            n += _set_vorbis(out, vorbis_key, text)
                        except Exception:
                            n += _set_vorbis(out, vorbis_key, values)
                    else:
                        n += _set_vorbis(out, vorbis_key, values)
                except Exception:
                    continue
        else:
            from mutagen import File as MutagenFile

            audio = MutagenFile(str(src), easy=True)
            if audio is not None and audio.tags:
                for key, values in list(audio.tags.items()):
                    n += _set_vorbis(out, str(key).upper(), values)
    except Exception:
        # Keep whatever we managed to stage; still try to save below.
        pass

    if n <= 0:
        return False
    try:
        out.save()
        return True
    except Exception:
        return False

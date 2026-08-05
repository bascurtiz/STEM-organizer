"""Locate / download mp3val next to the app (same pattern as ffmpeg)."""
from __future__ import annotations

import hashlib
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

_MP3VAL: Optional[str] = None
_INITIALIZED = False
_ENSURE_LOCK = threading.Lock()
_ENSURE_ATTEMPTED = False

# Official SourceForge Windows binary (CLI + optional frontend).
_MP3VAL_URL = (
    "https://downloads.sourceforge.net/project/mp3val/mp3val-bundle/"
    "MP3val%200.1.8%20with%20MP3val-frontend%200.1.1%20included/"
    "mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip"
)
_MP3VAL_URL_FALLBACK = (
    "https://sourceforge.net/projects/mp3val/files/mp3val-bundle/"
    "MP3val%200.1.8%20with%20MP3val-frontend%200.1.1%20included/"
    "mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip/download"
)
# Pin the known SF 0.1.8 win32 zip so Schannel/unverified fallbacks cannot
# accept an HTML interstitial or a swapped archive.
_MP3VAL_ZIP_SHA256 = (
    "ad087fcecd98f6a61b0fc028320dc3059f26b18d931b37af9c9291ea75b62451"
)
_MP3VAL_ZIP_MIN_BYTES = 50_000
_UA = "STEM-organizer/mp3val-bootstrap"


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _exe_name() -> str:
    return "mp3val.exe" if sys.platform == "win32" else "mp3val"


def _bundled_candidates() -> list[Path]:
    name = _exe_name()
    out: list[Path] = []
    for base in (_app_dir(), _resource_dir()):
        out.append(base / "mp3val" / name)
        out.append(base / name)
    return out


def _find_bundled() -> Optional[str]:
    for candidate in _bundled_candidates():
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _find_on_path() -> Optional[str]:
    found = shutil.which("mp3val") or shutil.which("mp3val.exe")
    if found and Path(found).is_file() and "WindowsApps" not in found:
        return found
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_zip(dest: Path) -> None:
    if dest.stat().st_size < _MP3VAL_ZIP_MIN_BYTES:
        raise RuntimeError("download too small")
    digest = _sha256_file(dest)
    if digest != _MP3VAL_ZIP_SHA256:
        raise RuntimeError(
            f"mp3val zip sha256 mismatch (got {digest}, want {_MP3VAL_ZIP_SHA256})"
        )


def _download_urllib(url: str, dest: Path, *, unverified: bool = False) -> None:
    req = Request(url, headers={"User-Agent": _UA})
    ctx = ssl._create_unverified_context() if unverified else None
    with urlopen(req, timeout=120, context=ctx) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def _download_curl(url: str, dest: Path) -> None:
    """Windows curl uses Schannel; SF mirrors often fail Python/OpenSSL verify."""
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        raise FileNotFoundError("curl not found")
    # -L follow redirects; -f fail on HTTP errors; -S show errors with -s.
    cmd = [
        curl,
        "-fsSL",
        "-A",
        _UA,
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        "-o",
        str(dest),
        url,
    ]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"curl download failed: {err}")


def _download(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest``, tolerating SourceForge mirror SSL breakage.

    Order: strict urllib → Windows curl (Schannel) → urllib with verify off.
    Every successful path is SHA-256 pinned to the known 0.1.8 win32 zip.
    """
    last_err: Exception | None = None
    attempts: list[tuple[str, object]] = [
        ("urllib", lambda: _download_urllib(url, dest, unverified=False)),
    ]
    if sys.platform == "win32":
        attempts.append(("curl", lambda: _download_curl(url, dest)))
    attempts.append(
        ("urllib-insecure", lambda: _download_urllib(url, dest, unverified=True))
    )

    for label, fn in attempts:
        try:
            if dest.exists():
                dest.unlink()
            fn()
            _validate_zip(dest)
            return
        except Exception as exc:  # noqa: BLE001 — try next transport
            last_err = exc
            continue
    assert last_err is not None
    raise last_err


def _extract_mp3val(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target = None
        for n in names:
            if Path(n).name.lower() in ("mp3val.exe", "mp3val"):
                target = n
                break
        if not target:
            raise FileNotFoundError("mp3val.exe not found in archive")
        # Extract only the CLI binary (skip GUI)
        data = zf.read(target)
    out = dest_dir / _exe_name()
    out.write_bytes(data)
    try:
        out.chmod(out.stat().st_mode | 0o111)
    except OSError:
        pass
    return out


def ensure_mp3val(*, force_download: bool = False) -> Optional[str]:
    """Return path to mp3val, downloading into ``<app>/mp3val/`` if missing."""
    global _MP3VAL, _ENSURE_ATTEMPTED

    existing = _find_bundled() or _find_on_path()
    if existing and not force_download:
        _MP3VAL = existing
        return existing

    with _ENSURE_LOCK:
        if _ENSURE_ATTEMPTED and not force_download:
            return _find_bundled() or _find_on_path()
        _ENSURE_ATTEMPTED = True

        again = _find_bundled() or _find_on_path()
        if again and not force_download:
            _MP3VAL = again
            return again

        dest_dir = _app_dir() / "mp3val"
        zip_path = Path(tempfile.gettempdir()) / "stem-organizer-mp3val.zip"
        last_err: Exception | None = None
        for url in (_MP3VAL_URL, _MP3VAL_URL_FALLBACK):
            try:
                _download(url, zip_path)
                exe = _extract_mp3val(zip_path, dest_dir)
                _MP3VAL = str(exe.resolve())
                return _MP3VAL
            except Exception as exc:  # noqa: BLE001 — try next URL
                last_err = exc
                continue
        if last_err:
            # Leave a breadcrumb for logs; callers treat None as missing
            sys.stderr.write(f"[mp3val_bootstrap] download failed: {last_err}\n")
        return None


def setup_mp3val() -> Optional[str]:
    """Resolve once (no download). Prefer ``ensure_mp3val`` when Fix needs it."""
    global _MP3VAL, _INITIALIZED
    if _INITIALIZED and _MP3VAL:
        return _MP3VAL
    _INITIALIZED = True
    found = _find_bundled() or _find_on_path()
    _MP3VAL = found
    return found


def mp3val_path(*, ensure: bool = True) -> Optional[str]:
    """Return mp3val path. When ``ensure``, download on first miss."""
    found = setup_mp3val()
    if found:
        return found
    if ensure:
        return ensure_mp3val()
    return None

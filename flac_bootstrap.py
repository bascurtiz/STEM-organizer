"""Locate / download reference flac.exe next to the app (same pattern as mp3val).

``flac -t`` verifies STREAMINFO MD5 (foobar-style). ffmpeg decode-null does not.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

_FLAC: Optional[str] = None
_INITIALIZED = False
_ENSURE_LOCK = threading.Lock()
_ENSURE_ATTEMPTED = False

# Official Xiph Windows binaries (includes win64/flac.exe + DLLs).
_FLAC_URL = "https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.5.0-win.zip"
_FLAC_URL_FALLBACK = (
    "https://downloads.xiph.org/releases/flac/flac-1.5.0-win.zip"
)


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _exe_name() -> str:
    return "flac.exe" if sys.platform == "win32" else "flac"


def _bundled_candidates() -> list[Path]:
    name = _exe_name()
    out: list[Path] = []
    for base in (_app_dir(), _resource_dir()):
        out.append(base / "flac" / name)
        out.append(base / "flac" / "win64" / name)
        out.append(base / "flac" / "win32" / name)
        out.append(base / name)
    return out


def _find_bundled() -> Optional[str]:
    for candidate in _bundled_candidates():
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _find_on_path() -> Optional[str]:
    found = shutil.which("flac") or shutil.which("flac.exe")
    if found and Path(found).is_file() and "WindowsApps" not in found:
        return found
    return None


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "STEM-organizer/flac-bootstrap"})
    with urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def _pick_member(names: list[str]) -> str:
    """Prefer win64/flac.exe, then any flac.exe / flac."""
    lowered = [(n, n.replace("\\", "/").lower()) for n in names]
    for n, low in lowered:
        if low.endswith("win64/flac.exe") or low.endswith("win64/flac"):
            return n
    for n, low in lowered:
        if Path(low).name in ("flac.exe", "flac") and "win32" not in low:
            return n
    for n, low in lowered:
        if Path(low).name in ("flac.exe", "flac"):
            return n
    raise FileNotFoundError("flac.exe not found in archive")


def _extract_flac(zip_path: Path, dest_dir: Path) -> Path:
    """Extract flac.exe (+ sibling DLLs from same folder) into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        target = _pick_member(names)
        folder = str(Path(target).parent).replace("\\", "/")
        # Copy exe + DLLs from the same archive folder (libFLAC etc.).
        for n in names:
            norm = n.replace("\\", "/")
            if Path(norm).parent.as_posix() != Path(folder).as_posix():
                continue
            name = Path(norm).name
            if not name or name.endswith("/"):
                continue
            low = name.lower()
            if low == "flac.exe" or low == "flac" or low.endswith(".dll"):
                data = zf.read(n)
                out = dest_dir / name
                if low in ("flac",) and sys.platform == "win32":
                    out = dest_dir / "flac.exe"
                out.write_bytes(data)
                try:
                    out.chmod(out.stat().st_mode | 0o111)
                except OSError:
                    pass
    exe = dest_dir / _exe_name()
    if not exe.is_file():
        raise FileNotFoundError(f"extract failed — missing {exe}")
    return exe


def ensure_flac(*, force_download: bool = False) -> Optional[str]:
    """Return path to flac, downloading into ``<app>/flac/`` if missing."""
    global _FLAC, _ENSURE_ATTEMPTED

    existing = _find_bundled() or _find_on_path()
    if existing and not force_download:
        _FLAC = existing
        return existing

    with _ENSURE_LOCK:
        if _ENSURE_ATTEMPTED and not force_download:
            return _find_bundled() or _find_on_path()
        _ENSURE_ATTEMPTED = True

        again = _find_bundled() or _find_on_path()
        if again and not force_download:
            _FLAC = again
            return again

        dest_dir = _app_dir() / "flac"
        zip_path = Path(tempfile.gettempdir()) / "stem-organizer-flac.zip"
        last_err: Exception | None = None
        for url in (_FLAC_URL, _FLAC_URL_FALLBACK):
            try:
                _download(url, zip_path)
                if zip_path.stat().st_size < 10_000:
                    raise RuntimeError("download too small")
                exe = _extract_flac(zip_path, dest_dir)
                _FLAC = str(exe.resolve())
                return _FLAC
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if last_err:
            sys.stderr.write(f"[flac_bootstrap] download failed: {last_err}\n")
        return None


def setup_flac() -> Optional[str]:
    """Resolve once (no download)."""
    global _FLAC, _INITIALIZED
    if _INITIALIZED and _FLAC:
        return _FLAC
    _INITIALIZED = True
    found = _find_bundled() or _find_on_path()
    _FLAC = found
    return found


def flac_path(*, ensure: bool = True) -> Optional[str]:
    """Return flac path. When ``ensure``, download on first miss."""
    found = setup_flac()
    if found:
        return found
    if ensure:
        return ensure_flac()
    return None

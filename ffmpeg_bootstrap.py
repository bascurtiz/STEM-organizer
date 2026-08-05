"""Locate FFmpeg tools next to the app or on PATH and expose them to subprocess/demucs."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

_FFMPEG: str | None = None
_FFPROBE: str | None = None
_FFPLAY: str | None = None
_INITIALIZED = False
_SUBPROCESS_PATCHED = False
_ENSURE_LOCK = threading.Lock()
_ENSURE_ATTEMPTED = False
_FFMPEG_EXE_NAMES = frozenset({
    'ffmpeg', 'ffmpeg.exe',
    'ffprobe', 'ffprobe.exe',
    'ffplay', 'ffplay.exe',
})

# Gyan.dev Windows essentials build — same source as install-deps.bat.
_FFMPEG_URL = (
    'https://github.com/GyanD/codexffmpeg/releases/download/8.1/'
    'ffmpeg-8.1-essentials_build.zip'
)
_FFMPEG_URL_FALLBACK = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'


def subprocess_kwargs() -> dict:
    """Extra kwargs for subprocess calls that must not flash a console on Windows."""
    if sys.platform != 'win32':
        return {}
    return {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)}


def _command_exe(args: tuple | list) -> str | None:
    if not args:
        return None
    cmd = args[0]
    if isinstance(cmd, (list, tuple)):
        if not cmd:
            return None
        return Path(cmd[0]).name.lower()
    if isinstance(cmd, str):
        return Path(cmd.strip().split()[0]).name.lower()
    return None


def _is_ffmpeg_invocation(args: tuple, kwargs: dict) -> bool:
    exe = _command_exe(args)
    if exe in _FFMPEG_EXE_NAMES:
        return True
    nested = kwargs.get('args')
    if nested is not None:
        return _command_exe([nested]) in _FFMPEG_EXE_NAMES
    return False


def _with_hidden_console(kwargs: dict) -> dict:
    flags = kwargs.get('creationflags', 0)
    kwargs['creationflags'] = flags | getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
    return kwargs


def _patch_subprocess_hide_console() -> None:
    """Hide ffmpeg/ffprobe console windows when spawned from a windowed .exe."""
    global _SUBPROCESS_PATCHED
    if _SUBPROCESS_PATCHED or sys.platform != 'win32':
        return
    _SUBPROCESS_PATCHED = True

    orig_run = subprocess.run
    orig_check_output = subprocess.check_output
    orig_popen = subprocess.Popen

    def run(*args, **kwargs):
        if _is_ffmpeg_invocation(args, kwargs):
            kwargs = _with_hidden_console(dict(kwargs))
        return orig_run(*args, **kwargs)

    def check_output(*args, **kwargs):
        if _is_ffmpeg_invocation(args, kwargs):
            kwargs = _with_hidden_console(dict(kwargs))
        return orig_check_output(*args, **kwargs)

    class _HiddenConsolePopen(orig_popen):
        def __init__(self, *args, **kwargs):
            if _is_ffmpeg_invocation(args, kwargs):
                kwargs = _with_hidden_console(dict(kwargs))
            super().__init__(*args, **kwargs)

    subprocess.run = run
    subprocess.check_output = check_output
    subprocess.Popen = _HiddenConsolePopen


def _app_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _prepend_path(directory: Path) -> None:
    entry = str(directory)
    if not directory.is_dir():
        return
    current = os.environ.get('PATH', '')
    if entry not in current.split(os.pathsep):
        os.environ['PATH'] = entry + (os.pathsep + current if current else '')


def _is_usable_executable(path: str | None) -> bool:
    return bool(path) and Path(path).is_file() and 'WindowsApps' not in path


def _find_bundled(exe_name: str) -> str | None:
    for base in (_app_dir(), _resource_dir()):
        candidate = base / 'ffmpeg' / exe_name
        if candidate.is_file():
            return str(candidate)
    return None


def _find_on_path(name: str) -> str | None:
    """First usable match, scanning every PATH entry.

    On Windows, skip the fake Microsoft Store aliases in ``WindowsApps`` — a
    crippled ffmpeg 7.0 build with most encoders/decoders/network disabled
    (no ``pcm_f32le``) that makes ``-f f32le`` fail with
    "Error opening output files: Encoder not found". ``shutil.which`` returns
    only the first PATH match, so a real ffmpeg later in PATH was shadowed.
    """
    if sys.platform == 'win32':
        names = (name, f'{name}.exe') if not name.lower().endswith('.exe') else (name,)
        for directory in os.environ.get('PATH', '').split(os.pathsep):
            directory = (directory or '').strip().strip('"')
            if not directory or 'WindowsApps' in directory:
                continue
            for exe_name in names:
                candidate = Path(directory) / exe_name
                if candidate.is_file():
                    return str(candidate)
    found = shutil.which(name)
    if _is_usable_executable(found):
        return found
    return None


def _extra_windows_candidates(exe_name: str) -> list[Path]:
    if sys.platform != 'win32':
        return []
    env_roots = [
        os.environ.get('ProgramFiles', r'C:\Program Files'),
        os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
        os.environ.get('LOCALAPPDATA', ''),
    ]
    rel_paths = (
        Path('ffmpeg') / 'bin' / exe_name,
        Path('FFmpeg') / 'bin' / exe_name,
        Path('scoop') / 'apps' / 'ffmpeg' / 'current' / 'bin' / exe_name,
        Path('chocolatey') / 'bin' / exe_name,
    )
    candidates: list[Path] = []
    for root in env_roots:
        if not root:
            continue
        root_path = Path(root)
        for rel in rel_paths:
            candidates.append(root_path / rel)
    return candidates


def _resolve_tool(exe_name: str, path_name: str, sibling_of: str | None) -> str | None:
    if sibling_of:
        sibling = Path(sibling_of).parent / exe_name
        if sibling.is_file():
            return str(sibling)

    bundled = _find_bundled(exe_name)
    if bundled:
        return bundled

    for candidate in _extra_windows_candidates(exe_name):
        if candidate.is_file():
            return str(candidate)

    return _find_on_path(path_name)


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={'User-Agent': 'STEM-organizer/ffmpeg-bootstrap'})
    with urlopen(req, timeout=180) as resp, open(dest, 'wb') as fh:
        shutil.copyfileobj(resp, fh)


def _extract_ffmpeg(zip_path: Path, dest_dir: Path) -> tuple[Path, Path]:
    """Extract ffmpeg.exe + ffprobe.exe from a Gyan essentials zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = '.exe' if sys.platform == 'win32' else ''
    wanted = {'ffmpeg', 'ffprobe'}
    found: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            base = Path(member).name.lower().removesuffix('.exe')
            if base not in wanted or base in found:
                continue
            data = zf.read(member)
            out = dest_dir / f'{base}{suffix}'
            out.write_bytes(data)
            try:
                out.chmod(out.stat().st_mode | 0o111)
            except OSError:
                pass
            found[base] = out
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f'{sorted(missing)[0]}{suffix} not found in archive')
    return found['ffmpeg'], found['ffprobe']


def ensure_ffmpeg(*, force_download: bool = False) -> str | None:
    """Return ffmpeg path, downloading a real build into ``<app>/ffmpeg/`` if none.

    Mirrors ``ensure_mp3val`` / ``ensure_flac``. The download is synchronous
    (~80 MB) — call from worker / background contexts, not the UI thread.
    """
    global _FFMPEG, _FFPROBE, _ENSURE_ATTEMPTED

    setup_ffmpeg()
    if _FFMPEG and _FFPROBE and not force_download:
        return _FFMPEG

    with _ENSURE_LOCK:
        if _ENSURE_ATTEMPTED and not force_download:
            return _FFMPEG
        _ENSURE_ATTEMPTED = True

        setup_ffmpeg()
        if _FFMPEG and _FFPROBE and not force_download:
            return _FFMPEG

        dest_dir = _app_dir() / 'ffmpeg'
        zip_path = Path(tempfile.gettempdir()) / 'stem-organizer-ffmpeg.zip'
        last_err: Exception | None = None
        for url in (_FFMPEG_URL, _FFMPEG_URL_FALLBACK):
            try:
                _download(url, zip_path)
                if zip_path.stat().st_size < 1_000_000:
                    raise RuntimeError('download too small')
                ffmpeg, ffprobe = _extract_ffmpeg(zip_path, dest_dir)
                _FFMPEG = str(ffmpeg.resolve())
                _FFPROBE = str(ffprobe.resolve())
                # Note: _FFPLAY stays unset — nothing in the app needs ffplay
                # (playback uses sounddevice; waveform/duration use ffmpeg/ffprobe).
                _prepend_path(dest_dir)
                return _FFMPEG
            except Exception as exc:  # noqa: BLE001 — try next URL
                last_err = exc
                continue
        if last_err:
            sys.stderr.write(f'[ffmpeg_bootstrap] download failed: {last_err}\n')
        return _FFMPEG


def setup_ffmpeg() -> tuple[str | None, str | None]:
    """Resolve FFmpeg tools once and prepend their directory to PATH."""
    global _FFMPEG, _FFPROBE, _FFPLAY, _INITIALIZED
    _patch_subprocess_hide_console()
    if _INITIALIZED:
        return _FFMPEG, _FFPROBE
    _INITIALIZED = True

    exe = 'ffmpeg.exe' if sys.platform == 'win32' else 'ffmpeg'
    probe = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
    play = 'ffplay.exe' if sys.platform == 'win32' else 'ffplay'

    ffmpeg = _resolve_tool(exe, 'ffmpeg', None)
    ffprobe = _resolve_tool(probe, 'ffprobe', ffmpeg)
    ffplay = _resolve_tool(play, 'ffplay', ffmpeg)

    if ffmpeg:
        _prepend_path(Path(ffmpeg).parent)

    _FFMPEG = ffmpeg
    _FFPROBE = ffprobe
    _FFPLAY = ffplay
    return ffmpeg, ffprobe


def ffmpeg_path() -> str | None:
    setup_ffmpeg()
    return _FFMPEG


def ffmpeg_folder_path() -> str | None:
    """Parent folder of ffmpeg.exe, for log display."""
    setup_ffmpeg()
    if not _FFMPEG:
        return None
    return str(Path(_FFMPEG).resolve().parent)


def ffprobe_path() -> str | None:
    setup_ffmpeg()
    return _FFPROBE


def ffplay_path() -> str | None:
    setup_ffmpeg()
    return _FFPLAY


def ffmpeg_missing_message() -> str:
    return (
        'ffmpeg not found — some stems may fail to decode. '
        'Put ffmpeg.exe and ffprobe.exe in an ffmpeg\\ folder next to the app, '
        'or re-run install-deps.bat / add ffmpeg to PATH. '
        '(Rename audition uses sounddevice like STEM Player — ffplay not required.)'
    )

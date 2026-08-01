"""Locate FFmpeg tools next to the app or on PATH and expose them to subprocess/demucs."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_FFMPEG: str | None = None
_FFPROBE: str | None = None
_FFPLAY: str | None = None
_INITIALIZED = False
_SUBPROCESS_PATCHED = False
_FFMPEG_EXE_NAMES = frozenset({
    'ffmpeg', 'ffmpeg.exe',
    'ffprobe', 'ffprobe.exe',
    'ffplay', 'ffplay.exe',
})


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
        'Put ffmpeg.exe, ffprobe.exe, and ffplay.exe in an ffmpeg\\ folder next to the app, '
        'or re-run install-deps.bat / add ffmpeg to PATH.'
    )

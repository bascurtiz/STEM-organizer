"""Load PyTorch / Demucs from beside the exe (or normal pip env when running from source)."""
from __future__ import annotations

import importlib.util
import os
import sys
import webbrowser
from pathlib import Path

TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
TORCH_CUDA_INDEX = 'https://download.pytorch.org/whl/cu124'  # RTX 20/30/40 series
TORCH_CUDA_NEW_INDEX = 'https://download.pytorch.org/whl/cu128'  # RTX 50-series (Blackwell)
REPO_URL = 'https://github.com/bascurtiz/STEM-organizer'

SUPPORTED_PYTHON = ((3, 10), (3, 11))
SUPPORTED_LABEL = '3.10.x or 3.11.x'
LEGACY_PREBUILT_PYTHON = (3, 11)
PYTHON_INSTALLERS = {
    (3, 10): 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe',
    (3, 11): 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe',
}
APP_VERSION_FILE = 'python-version.txt'
SITE_PACKAGES_MARKER = '.python-version-used'


def _version_label(ver: tuple[int, int]) -> str:
    return f'{ver[0]}.{ver[1]}.x'


def _parse_version_tag(text: str) -> tuple[int, int] | None:
    text = text.strip()
    parts = text.replace(',', '.').split('.')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def _embedded_python(app_dir: Path) -> tuple[int, int] | None:
    marker = app_dir / APP_VERSION_FILE
    if marker.is_file():
        parsed = _parse_version_tag(marker.read_text(encoding='utf-8'))
        if parsed in SUPPORTED_PYTHON:
            return parsed
    if any((app_dir / name).is_file() for name in ('STEM-organizer.exe', 'stem-organizer.exe')):
        return LEGACY_PREBUILT_PYTHON
    return None


def _mismatch_hint(app_dir: Path) -> str | None:
    if is_frozen():
        expected = sys.version_info[:2]
    else:
        expected = _embedded_python(app_dir)
        if expected is None:
            return None
    markers = [app_dir / 'site-packages' / SITE_PACKAGES_MARKER]
    if is_frozen():
        for dest in external_site_dirs():
            markers.append(dest / SITE_PACKAGES_MARKER)
    seen: set[str] = set()
    for marker in markers:
        key = str(marker)
        if key in seen:
            continue
        seen.add(key)
        if not marker.is_file():
            continue
        installed = _parse_version_tag(marker.read_text(encoding='utf-8'))
        if installed is None or installed == expected:
            return None
        return (
            f'site-packages were installed with Python {_version_label(installed)}, '
            f'but this app needs {_version_label(expected)}.\n'
            f'Delete site-packages\\ and run install-deps.bat with py -{expected[0]}.{expected[1]}.'
        )
    return None


def is_frozen() -> bool:
    return getattr(sys, 'frozen', False)


def _fix_frozen_stdio() -> None:
    """PyInstaller windowed builds set stdout/stderr to None; torch hub needs them."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w', encoding='utf-8')  # noqa: SIM115


if is_frozen():
    _fix_frozen_stdio()


_EXE_NAMES = frozenset({'stem-organizer.exe', 'stem_organizer.exe'})


def _is_app_exe(path: Path) -> bool:
    return path.is_file() and path.name.lower() in _EXE_NAMES


def _climb_out_of_internal(path: Path) -> Path:
    """Never treat ``_internal`` (PyInstaller MEIPASS) as the app root."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved.name.lower() == '_internal':
        return resolved.parent
    return resolved


def frozen_exe_dir() -> Path:
    """Directory containing ``STEM-organizer.exe`` (not ``_internal``).

    install-deps.bat puts ``site-packages\\`` beside the .exe. Prefer that
    folder over anything derived from bundled script ``__file__`` paths.
    """
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        path = _climb_out_of_internal(path)
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    add(Path(sys.executable).resolve().parent)
    add(Path(sys.executable).parent)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        add(Path(meipass).resolve().parent)
        add(Path(meipass).parent)

    # Prefer a folder that actually contains the app exe.
    for base in candidates:
        if any(_is_app_exe(base / name) for name in (
            'STEM-organizer.exe', 'stem-organizer.exe', 'STEM_organizer.exe',
        )):
            return base
    return candidates[0] if candidates else Path(sys.executable).resolve().parent


def app_dir() -> Path:
    """Folder containing the .exe (frozen) or this repo root (source).

    install-deps.bat installs into ``<this>/site-packages`` when
    ``STEM-organizer.exe`` sits beside the bat — must stay aligned.
    """
    if is_frozen():
        return frozen_exe_dir()
    return Path(__file__).resolve().parent


def _frozen_app_bases() -> list[Path]:
    """Candidate app folders for frozen site-packages discovery.

    PyInstaller 6 onedir: exe next to ``_internal`` (``sys._MEIPASS``).
    Prefer the exe directory; never use ``_internal`` itself.
    """
    bases: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        path = _climb_out_of_internal(path)
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            return
        seen.add(key)
        bases.append(path)

    add(frozen_exe_dir())
    add(Path(sys.executable).resolve().parent)
    add(Path(sys.executable).parent)
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        add(Path(meipass).resolve().parent)
    return bases


def external_site_dirs() -> list[Path]:
    """Paths where install-deps.bat (or a project .venv) may put wheels."""
    bases = _frozen_app_bases() if is_frozen() else [app_dir()]
    dirs: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        for candidate in (
            base / 'site-packages',
            base / '.venv' / 'Lib' / 'site-packages',
        ):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            dirs.append(candidate)
    return dirs


def _prepend_path(path: Path) -> None:
    entry = str(path)
    if not path.is_dir():
        return
    current = os.environ.get('PATH', '')
    if entry not in current.split(os.pathsep):
        os.environ['PATH'] = entry + (os.pathsep + current if current else '')


def _register_native_libs(site_packages: Path) -> None:
    """Make torch/numpy/soundfile DLLs visible to the frozen exe on Windows."""
    candidates = [
        site_packages,
        site_packages / 'torch' / 'lib',
        site_packages / 'numpy.libs',
        site_packages / '_soundfile_data',
        site_packages / 'bin',
    ]
    for path in candidates:
        if not path.is_dir():
            continue
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
        _prepend_path(path)


def _prepare_soundfile_dll(site_packages: Path) -> None:
    data = site_packages / '_soundfile_data'
    if not data.is_dir():
        return
    dlls = sorted(data.glob('libsndfile*.dll'))
    if not dlls:
        return
    primary = dlls[0]
    alias = data / 'libsndfile.dll'
    if not alias.exists() and primary.name.lower() != 'libsndfile.dll':
        try:
            import shutil
            shutil.copy2(primary, alias)
        except OSError:
            pass


def _preload_site_extensions(site_packages: Path) -> None:
    """Load top-level extension modules pip drops directly into site-packages."""
    if not is_frozen():
        return
    for pyd in site_packages.glob('*.pyd'):
        stem = pyd.name.partition('.')[0]
        if not stem or stem in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(stem, pyd)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[stem] = module
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(stem, None)


def init_external_deps(set_status=None) -> None:
    def status(msg: str) -> None:
        if set_status is not None:
            set_status(msg)

    status('Setting up FFmpeg…')
    from ffmpeg_bootstrap import setup_ffmpeg

    setup_ffmpeg()
    status('Preparing native libraries…')
    # Insert last→first so the first candidate (exe-side site-packages) ends
    # up at sys.path[0]. install-deps.bat frozen mode uses that folder.
    existing = [p for p in external_site_dirs() if p.is_dir()]
    for path in reversed(existing):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        _register_native_libs(path)
        _prepare_soundfile_dll(path)
        _preload_site_extensions(path)

    torch_home = app_dir() / 'torch_home'
    checkpoints = torch_home / 'hub' / 'checkpoints'
    if checkpoints.is_dir() and any(checkpoints.glob('*.th')):
        os.environ['TORCH_HOME'] = str(torch_home)


def load_ml_deps(set_status=None):
    """Import ML stack.

    Default (STEM_ONNX): numpy + soundfile + onnxruntime only — no torch/demucs
    (avoids co-loading torch+ORT which froze the dev machine). Returns
    ``(np, sf, None, None, None, None)`` on the ONNX path.

    STEM_ONNX=0 or missing HTDemucs/Vocal CNN6 weights: full torch + demucs stack (legacy).
    """
    def status(msg: str) -> None:
        if set_status is not None:
            set_status(msg)

    if is_frozen():
        from frozen_stdlib_imports import ensure_stdlib_for_external_ml

        ensure_stdlib_for_external_ml()

    status('Loading numpy…')
    import numpy as np
    status('Loading soundfile…')
    import soundfile as sf

    # Prefer HTDemucs / Stem CNN6 ONNX — skip torch entirely when any
    # separator/classifier weight is present (avoids co-loading torch+ORT).
    try:
        from demucs_onnx import resolve_htdemucs_onnx

        if resolve_htdemucs_onnx() is not None:
            status('HTDemucs ONNX ready (torch not loaded)…')
            import onnxruntime  # noqa: F401 — fail early if missing

            return np, sf, None, None, None, None
    except ImportError:
        pass
    try:
        from stem_cnn6_onnx import stem_cnn6_installed

        if stem_cnn6_installed():
            status('Stem CNN6 ONNX ready (torch not loaded)…')
            import onnxruntime  # noqa: F401

            return np, sf, None, None, None, None
    except ImportError:
        pass

    status('Loading PyTorch…')
    import torch
    status('Loading Demucs…')
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    return np, sf, torch, get_model, apply_model, AudioFile


def demucs_onnx_present() -> bool:
    """True when any Classify separator ONNX weight is bundled."""
    try:
        from demucs_onnx import resolve_htdemucs_onnx

        if resolve_htdemucs_onnx() is not None:
            return True
    except Exception:
        pass
    try:
        from stem_cnn6_onnx import stem_cnn6_installed

        return bool(stem_cnn6_installed())
    except Exception:
        return False


def htdemucs_onnx_present() -> bool:
    try:
        from demucs_onnx import resolve_htdemucs_onnx

        return resolve_htdemucs_onnx() is not None
    except Exception:
        return False


def _is_missing_stdlib(exc: ImportError) -> bool:
    """True when the missing name is stdlib (rebuild exe), not a pip package."""
    name = (exc.name or '').split('.')[0]
    if not name:
        return False
    try:
        from frozen_stdlib_imports import _ML_STDLIB_MODULES

        roots = {m.split('.')[0] for m in _ML_STDLIB_MODULES}
    except Exception:
        roots = {'timeit', 'unittest', 'platform', 'sysconfig'}
    roots |= {'unittest', 'timeit'}
    return name in roots


def _pkg_present(dest: Path, name: str) -> bool:
    return (dest / name).is_dir() or (dest / f'{name}.py').is_file()


def _import_probe(name: str) -> str:
    try:
        mod = __import__(name)
        ver = getattr(mod, '__version__', '')
        return f'{name}: OK' + (f' ({ver})' if ver else '')
    except Exception as exc:
        return f'{name}: failed ({exc})'


def _site_packages_hint() -> str:
    candidates = external_site_dirs()
    dest = next((p for p in candidates if p.is_dir()), candidates[0] if candidates else app_dir() / 'site-packages')
    if not dest.is_dir():
        tried = ', '.join(str(p) for p in candidates) or str(dest)
        return f'site-packages\\ folder not found next to this app (tried: {tried}).'

    lines: list[str] = [f'Using: {dest}']
    torch_dir = dest / 'torch'
    if torch_dir.is_dir():
        lib_dir = torch_dir / 'lib'
        dlls = len(list(lib_dir.glob('*.dll'))) if lib_dir.is_dir() else 0
        lines.append(f'torch\\ found ({dlls} DLLs in torch\\lib).')
    else:
        lines.append('torch\\ not found in site-packages.')

    for name in ('numpy', 'demucs', 'soundfile', 'cffi'):
        lines.append(f'{name}: on disk' if _pkg_present(dest, name) else f'{name}: missing on disk')

    lines.append('')
    lines.append('Import check:')
    for name in ('numpy', 'torch', '_cffi_backend', 'soundfile', 'demucs'):
        lines.append(f'  {_import_probe(name)}')

    return '\n'.join(lines)


def _required_python_label() -> str:
    if is_frozen():
        return _version_label(sys.version_info[:2])
    needed = _embedded_python(app_dir())
    return _version_label(needed) if needed else SUPPORTED_LABEL


def missing_deps_message(exc: ImportError) -> str:
    detail = str(exc).strip()
    if detail and detail != str(exc.name or ''):
        missing_line = f'Missing: {exc.name or "package"} ({detail})'
    else:
        missing_line = f'Missing: {exc.name or exc}'

    if is_frozen() and _is_missing_stdlib(exc):
        return '\n'.join([
            'This .exe is missing a Python standard-library module needed by PyTorch.',
            '',
            missing_line,
            '',
            'install-deps.bat / site-packages cannot fix this.',
            'Rebuild STEM-organizer.exe (run build.bat) so PyInstaller includes',
            'stdlib hiddenimports (e.g. timeit) from stem_organizer_py6.spec.',
        ])

    hint = _mismatch_hint(app_dir())
    version_line = _required_python_label()
    lines = [
        'STEM organizer needs Python packages that are not bundled in this .exe.',
        '',
        missing_line,
        '',
        _site_packages_hint(),
    ]
    if hint:
        lines.extend(['', hint])
    lines.extend([
        '',
        'One-time setup:',
        f'1. Install Python {version_line} from python.org',
        f'   (supported: {SUPPORTED_LABEL}; not 3.12+)',
        '2. Double-click install-deps.bat and choose CPU or the matching NVIDIA option',
        '   (RTX 50-series / 5090 needs the cu128 option, not cu124)',
        '3. Restart STEM organizer',
        '',
        'install-deps.bat must sit in the same folder as this .exe.',
        'It installs matching wheels into site-packages\\ beside the app.',
        'Use the same Python major.minor as the .exe (e.g. py -3.10 install-deps.bat).',
        '',
        'If packages are listed above but import still fails, reinstall with',
        'install-deps.bat (delete site-packages\\ first).',
        '',
        'Demucs models (~450 MB) download on first run into torch_home\\',
        '(internet required once).',
    ])
    return '\n'.join(lines)


def _show_missing_deps_dialog(exc: ImportError) -> None:
    # Legacy pre-Qt dialog; tkinter is excluded from the frozen build, so this
    # path must degrade to a message instead of crashing.
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        print(f'Missing dependencies dialog unavailable: {exc}')
        return

    root = tk.Tk()
    root.withdraw()
    root.title('Missing dependencies')

    version_line = _required_python_label()
    msg = missing_deps_message(exc)
    folder = str(app_dir())
    messagebox.showinfo('Missing dependencies', msg, parent=root)

    if messagebox.askyesno(
        'Download Python?',
        f'Download a Python {version_line} installer (64-bit)?\n\n'
        'Tick "Add python.exe to PATH" during setup, then run install-deps.bat.',
        parent=root,
    ):
        if is_frozen():
            installer = PYTHON_INSTALLERS.get(sys.version_info[:2])
        else:
            needed = _embedded_python(app_dir())
            installer = PYTHON_INSTALLERS.get(needed or sys.version_info[:2])
        if installer:
            webbrowser.open(installer)

    if messagebox.askyesno(
        'Open app folder?',
        'Open the folder containing install-deps.bat?',
        parent=root,
    ):
        if sys.platform == 'win32':
            os.startfile(folder)  # noqa: S606
        else:
            webbrowser.open(folder)

    root.destroy()


def demucs_models_present() -> bool:
    """True when Demucs ONNX is bundled, or a torch hub checkpoint exists."""
    if demucs_onnx_present():
        return True
    torch_home = Path(os.environ.get('TORCH_HOME', str(app_dir() / 'torch_home')))
    checkpoints = torch_home / 'hub' / 'checkpoints'
    return checkpoints.is_dir() and any(checkpoints.glob('*.th'))


def demucs_model_cached(model_id: str) -> bool:
    """Best-effort check whether the named model's weights are already on disk.

    Retired separator ids (umxl/xumxl/scnet_tran/bsroformer/demucs) route to
    the HTDemucs check; they all map to HTDemucs at load time.
    """
    mid = (model_id or "").strip().lower()
    if mid == "vocal_cnn6":
        # Only stem_cnn6.onnx — legacy vocal_classifier.onnx retired.
        # Resolver searches app_dir (exe folder) when frozen.
        # Use the resolver (searches app_dir = exe folder when frozen) so the
        # check finds the installer-placed weight, not just paths beside this file.
        try:
            from stem_cnn6_onnx import stem_cnn6_installed

            return bool(stem_cnn6_installed())
        except Exception:
            return False
    # htdemucs + all legacy/retired ids resolve to HTDemucs.
    return htdemucs_onnx_present()


def ensure_ml_deps(*, show_dialog: bool = True, set_status=None) -> bool:
    init_external_deps(set_status)
    hint = _mismatch_hint(app_dir())
    if hint:
        if show_dialog:
            try:
                import tkinter as tk
                from tkinter import messagebox

                root = tk.Tk()
                root.withdraw()
                messagebox.showerror('Python version mismatch', hint, parent=root)
                root.destroy()
            except Exception:
                pass
            return False
        raise RuntimeError(hint)
    try:
        load_ml_deps(set_status)
        return True
    except ImportError as exc:
        if show_dialog:
            try:
                _show_missing_deps_dialog(exc)
            except Exception:
                pass
            return False
        raise RuntimeError(missing_deps_message(exc)) from exc
    except OSError as exc:
        wrapped = ImportError(f'DLL load failed: {exc}')
        wrapped.__cause__ = exc
        if show_dialog:
            try:
                _show_missing_deps_dialog(wrapped)
            except Exception:
                pass
            return False
        raise RuntimeError(missing_deps_message(wrapped)) from exc


def python_for_install() -> str:
    """Python executable used by install-deps.bat (best effort)."""
    if is_frozen():
        return 'python'
    return sys.executable

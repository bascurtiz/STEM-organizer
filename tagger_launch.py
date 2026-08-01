"""Resolve a real Python for genre/gender + instrument + PANNs tagger subprocesses.

Frozen builds ship tagger scripts beside the exe (no nested venv). ML wheels
live in ``site-packages\\`` from root ``install-deps.bat``. ``sys.executable``
is the .exe, so we spawn a matching system / ``py`` launcher interpreter with
``PYTHONPATH`` pointing at that folder.

Source / legacy: prefer ``genre_gender_tagger\\venv\\Scripts\\python.exe``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from deps_bootstrap import (
    APP_VERSION_FILE,
    SITE_PACKAGES_MARKER,
    SUPPORTED_PYTHON,
    app_dir,
    external_site_dirs,
    frozen_exe_dir,
    is_frozen,
)
from ffmpeg_bootstrap import subprocess_kwargs

# Child taggers also read this (belt-and-suspenders if PYTHONPATH is dropped).
STEM_SITE_PACKAGES_ENV = "STEM_SITE_PACKAGES"


def _parse_version_tag(text: str) -> tuple[int, int] | None:
    text = text.strip()
    parts = text.replace(",", ".").split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def tagger_app_root() -> Path:
    """Folder that holds ``genre_gender_tagger\\`` / ``instrument_tagger\\`` / ``panns_tagger\\``."""
    return app_dir()


def genre_gender_dir() -> Path:
    """``genre_gender_tagger`` beside the exe, or under ``_internal`` when frozen."""
    root = tagger_app_root()
    candidates = [root / "genre_gender_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "genre_gender_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "genre_gender_tagger")
    for path in candidates:
        if (path / "genre_gender_tagger.py").is_file():
            return path
    return candidates[0]


def genre_gender_script() -> Path:
    return genre_gender_dir() / "genre_gender_tagger.py"


def instrument_tagger_dir() -> Path:
    root = tagger_app_root()
    candidates = [root / "instrument_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "instrument_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "instrument_tagger")
    for path in candidates:
        if (path / "instrument_tagger.py").is_file():
            return path
    return candidates[0]


def instrument_tagger_script() -> Path:
    return instrument_tagger_dir() / "instrument_tagger.py"


def panns_tagger_dir() -> Path:
    root = tagger_app_root()
    candidates = [root / "panns_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "panns_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "panns_tagger")
    for path in candidates:
        if (path / "panns_tagger.py").is_file():
            return path
    return candidates[0]


def panns_tagger_script() -> Path:
    return panns_tagger_dir() / "panns_tagger.py"


def key_tagger_dir() -> Path:
    root = tagger_app_root()
    candidates = [root / "key_tagger"]
    if is_frozen():
        candidates.append(root / "_internal" / "key_tagger")
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "key_tagger")
    for path in candidates:
        if (path / "key_tagger.py").is_file():
            return path
    return candidates[0]


def key_tagger_script() -> Path:
    return key_tagger_dir() / "key_tagger.py"


def _site_packages() -> Path | None:
    """Prefer ``<exe_dir>\\site-packages`` that holds hear21passt when frozen."""
    dirs = [p for p in external_site_dirs() if p.is_dir()]
    if is_frozen():
        exe_site = frozen_exe_dir() / "site-packages"
        if exe_site.is_dir():
            try:
                exe_key = str(exe_site.resolve())
            except OSError:
                exe_key = str(exe_site)

            def _same(p: Path) -> bool:
                try:
                    return str(p.resolve()) == exe_key
                except OSError:
                    return str(p) == exe_key

            # Put exe-dir first so Auto-detect matches install-deps.bat layout.
            dirs = [exe_site] + [p for p in dirs if not _same(p)]
        with_passt = [p for p in dirs if (p / "hear21passt").is_dir()]
        if with_passt:
            return with_passt[0]
    return dirs[0] if dirs else None


def _expected_python() -> tuple[int, int]:
    root = tagger_app_root()
    for marker in (
        root / "site-packages" / SITE_PACKAGES_MARKER,
        root / APP_VERSION_FILE,
    ):
        if not marker.is_file():
            continue
        parsed = _parse_version_tag(marker.read_text(encoding="utf-8"))
        if parsed in SUPPORTED_PYTHON:
            return parsed
    if is_frozen():
        return sys.version_info[:2]
    return sys.version_info[:2]


def _python_version(exe: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            [str(exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
            **subprocess_kwargs(),
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_version_tag(out)


def resolve_host_python() -> Path | None:
    """System / py-launcher Python matching site-packages / build version."""
    major, minor = _expected_python()
    candidates: list[Path] = []

    try:
        out = subprocess.check_output(
            ["py", f"-{major}.{minor}", "-c", "import sys; print(sys.executable)"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
            **subprocess_kwargs(),
        ).strip()
        if out:
            candidates.append(Path(out))
    except (OSError, subprocess.SubprocessError):
        pass

    for name in ("python", "python.exe"):
        try:
            out = subprocess.check_output(
                ["where" if sys.platform == "win32" else "which", name],
                text=True,
                timeout=10,
                stderr=subprocess.DEVNULL,
                **subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            line = line.strip()
            if line:
                candidates.append(Path(line))

    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        ver = _python_version(path)
        if ver == (major, minor):
            return path
    return None


def _venv_python_candidates(root: Path) -> tuple[Path, ...]:
    gg = root / "genre_gender_tagger"
    inst = root / "instrument_tagger"
    panns = root / "panns_tagger"
    return (
        gg / "venv" / "Scripts" / "python.exe",
        gg / "venv" / "bin" / "python",
        inst / "venv" / "Scripts" / "python.exe",
        inst / "venv" / "bin" / "python",
        panns / "venv" / "Scripts" / "python.exe",
        panns / "venv" / "bin" / "python",
    )


def _site_has_hear21passt(site: Path) -> bool:
    return (site / "hear21passt").is_dir()


def resolve_tagger_python() -> Path | None:
    """Interpreter for tagger subprocesses.

    Frozen + STEM_ONNX: the .exe itself (argv self-dispatch via ``--run-tagger``).
    Frozen + legacy: host Python + site-packages.
    Source: genre_gender_tagger\\venv when present.
    """
    root = tagger_app_root()
    if is_frozen():
        # Self-dispatch: embedded CPython runs the bundled .py taggers.
        if os.environ.get("STEM_ONNX", "1").strip() != "0":
            exe = Path(sys.executable)
            if exe.is_file():
                return exe
        site = _site_packages()
        host = resolve_host_python()
        if host is not None and site is not None and _site_has_hear21passt(site):
            return host
        for path in _venv_python_candidates(root):
            if path.is_file():
                return path
        if host is not None and site is not None:
            return host
        # Last resort: still try self-dispatch
        exe = Path(sys.executable)
        return exe if exe.is_file() else None
    for path in _venv_python_candidates(root):
        if path.is_file():
            return path
    # Source / no venv: current interpreter (dev machines with system torch/ORT).
    return Path(sys.executable)


TAGGER_NAMES = frozenset({"genre_gender", "instrument", "panns", "key"})


def tagger_name_for_script(script: Path) -> str | None:
    name = script.name.lower()
    if name == "genre_gender_tagger.py":
        return "genre_gender"
    if name == "instrument_tagger.py":
        return "instrument"
    if name == "panns_tagger.py":
        return "panns"
    if name == "key_tagger.py":
        return "key"
    return None


def tagger_script_for_name(name: str) -> Path:
    if name == "genre_gender":
        return genre_gender_script()
    if name == "instrument":
        return instrument_tagger_script()
    if name == "panns":
        return panns_tagger_script()
    if name == "key":
        return key_tagger_script()
    raise KeyError(name)


def build_tagger_command(script: Path, *script_args: str) -> list[str]:
    """Argv to spawn a tagger.

    Frozen ONNX builds: ``STEM-organizer.exe --run-tagger <name> …``
    (no host Python). Source / legacy: ``python -u script.py …``.
    """
    python = resolve_tagger_python()
    if python is None:
        raise RuntimeError("tagger Python not resolved")
    name = tagger_name_for_script(script)
    frozen_self = (
        is_frozen()
        and name is not None
        and os.environ.get("STEM_ONNX", "1").strip() != "0"
        and Path(sys.executable).resolve() == Path(python).resolve()
    )
    if frozen_self:
        return [str(python), "--run-tagger", name, *script_args]
    return [str(python), "-u", str(script), *script_args]


def maybe_run_tagger_dispatch(argv: list[str] | None = None) -> int | None:
    """If argv is ``--run-tagger <name> …``, run that tagger and return exit code.

    Returns None when this process should continue as the GUI.
    """
    args = list(sys.argv if argv is None else argv)
    # argv[0] is exe; look for --run-tagger
    try:
        idx = args.index("--run-tagger")
    except ValueError:
        return None
    if idx + 1 >= len(args):
        print("ERROR: --run-tagger requires a name", file=sys.stderr)
        return 2
    name = args[idx + 1].strip().lower()
    if name not in TAGGER_NAMES:
        print(f"ERROR: unknown tagger {name!r}; expected one of {sorted(TAGGER_NAMES)}", file=sys.stderr)
        return 2
    extra = args[idx + 2 :]
    script = tagger_script_for_name(name)
    if not script.is_file():
        print(f"ERROR: tagger script missing: {script}", file=sys.stderr)
        return 2

    # Child must not grab the GUI single-instance mutex / splash.
    os.environ["STEM_TAGGER_CHILD"] = "1"
    os.environ.setdefault("STEM_ALLOW_MULTI", "1")

    import runpy

    # Put script dir first so local imports (passt_mel_np, etc.) resolve.
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    # Rebuild argv as the tagger expects: script + extras (no --run-tagger).
    sys.argv = [str(script), *extra]
    try:
        os.chdir(script.parent)
    except OSError:
        pass
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return 0


def _env_get_ci(env: dict[str, str], name: str) -> str:
    """Case-insensitive env lookup (Windows may store ``PythonPath`` etc.)."""
    if name in env:
        return env[name]
    want = name.lower()
    for key, val in env.items():
        if key.lower() == want:
            return val
    return ""


def _env_set_ci(env: dict[str, str], name: str, value: str) -> None:
    """Set env var, dropping other-case duplicates on Windows."""
    drop = [k for k in env if k.lower() == name.lower()]
    for key in drop:
        del env[key]
    env[name] = value


def tagger_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for tagger spawn; sets PYTHONPATH to site-packages when frozen.

    Host Python (not the .exe) must see hear21passt / onnxruntime wheels
    installed by root install-deps.bat into ``site-packages\\`` beside the
    .exe — same shape as: ``set PYTHONPATH=%CD%\\site-packages``.
    """
    env = dict(base if base is not None else os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if not is_frozen():
        return env
    site = _site_packages()
    if site is None:
        # Last resort: exe-dir site-packages even if not yet created (clearer errors).
        site = frozen_exe_dir() / "site-packages"
        if not site.is_dir():
            return env
    try:
        entry = str(site.resolve())
    except OSError:
        entry = str(site)
    existing = _env_get_ci(env, "PYTHONPATH")
    parts = [p for p in existing.split(os.pathsep) if p]
    # Always put site-packages first so Auto-detect finds hear21passt.
    parts = [p for p in parts if os.path.normcase(p) != os.path.normcase(entry)]
    _env_set_ci(
        env,
        "PYTHONPATH",
        entry + (os.pathsep + os.pathsep.join(parts) if parts else ""),
    )
    _env_set_ci(env, STEM_SITE_PACKAGES_ENV, entry)
    return env


def missing_tagger_python_hint() -> str:
    if is_frozen():
        if os.environ.get("STEM_ONNX", "1").strip() != "0":
            return (
                "Could not start tagger via STEM-organizer.exe self-dispatch.\n"
                "Rebuild the app (build.bat) so tagger scripts sit beside the exe."
            )
        return (
            "No Python found for Genre & Gender / Rename Auto-detect / PANNs.\n"
            "Run install-deps.bat beside STEM-organizer.exe once "
            "(installs into site-packages\\).\n"
            "Need Python 3.10 or 3.11 on PATH (or: py -3.11)."
        )
    return (
        "Genre & Gender venv not found.\n"
        "Run genre_gender_tagger\\install-deps.bat once "
        "(or root install-deps.bat)."
    )

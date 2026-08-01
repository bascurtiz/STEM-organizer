"""Stdlib modules needed by external numpy/torch/soundfile/demucs in the frozen exe.

PyInstaller onedir apps omit unused stdlib. External torch (package_exporter,
strobelight, etc.) then fails with ModuleNotFoundError. Prefer a slightly
larger bundle over rebuild whack-a-mole.

- ``_ML_STDLIB_MODULES``: curated set force-imported at runtime.
- ``iter_ml_stdlib_module_names()``: nearly all stdlib names for .spec
  hiddenimports (same strategy as STEM-organizer CTk ``stem_organizer.spec``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Keep in sync with hiddenimports in stem_organizer_py6.spec (imports this list /
# iter_ml_stdlib_module_names).
_ML_STDLIB_SKIP = frozenset({
    'test',
    'tests',
    'idlelib',
    'turtledemo',
    'lib2to3',
    'ensurepip',
    'venv',
    'tkinter',
    '__pycache__',
    # CPython frozen/demo stubs — not real packaging targets (PyInstaller
    # hard-fails on missing hiddenimports like __phello__.foo).
    '__phello__',
    '__hello__',
    'antigravity',
    'this',
})


def _skip_stdlib_part(name: str) -> bool:
    """True if this path segment should be omitted from hiddenimports."""
    if name in _ML_STDLIB_SKIP or name.startswith('test'):
        return True
    # Other __dunder__ demo/frozen packages (keep __future__).
    if name.startswith('__') and name.endswith('__') and name != '__future__':
        return True
    return False


def is_excluded_stdlib_hiddenimport(name: str) -> bool:
    """True if a dotted module name must never be a PyInstaller hiddenimport.

    Catches ``__phello__``, ``__phello__.foo`` (intentional ImportError stub),
    ``__hello__``, and other skip-list tops (``antigravity``, ``this``, …).
    """
    if not name:
        return True
    if '__phello__' in name:
        return True
    if name == '__hello__' or name.startswith('__hello__.'):
        return True
    return _skip_stdlib_part(name.split('.', 1)[0])


def is_available_stdlib_hiddenimport(name: str) -> bool:
    """True if *name* may be a PyInstaller hiddenimport on this build Python.

    Combines the hard skip list with ``importlib.util.find_spec`` so removed
    stdlib modules (``binhex`` on 3.11+, ``imp``/``formatter``/… on older
    cuts) are omitted automatically for the interpreter running the .spec.
    """
    if is_excluded_stdlib_hiddenimport(name):
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

# Generous curated set for runtime ensure_* (and Analysis when traced from entry).
_ML_STDLIB_MODULES = (
    # --- already required / previously missing ---
    'platform',
    'sysconfig',
    'timeit',
    'ctypes',
    'ctypes.util',
    'pickle',
    'pickletools',
    'gzip',
    'bz2',
    'lzma',
    'zlib',
    'zipfile',
    'tarfile',
    'logging',
    'logging.config',
    'logging.handlers',
    'mmap',
    'multiprocessing',
    'multiprocessing.spawn',
    'multiprocessing.pool',
    'multiprocessing.shared_memory',
    'socket',
    'ssl',
    'struct',
    'select',
    'selectors',
    # --- inspect / codegen (torch, numpy) ---
    'inspect',
    'dis',
    'ast',
    'token',
    'tokenize',
    'keyword',
    'linecache',
    'pydoc',
    'doctest',
    'difflib',
    'pprint',
    'code',
    'codeop',
    'py_compile',
    'compileall',
    # --- core helpers ---
    'copy',
    'copyreg',
    'functools',
    'itertools',
    'operator',
    'collections',
    'collections.abc',
    'typing',
    'enum',
    'traceback',
    'warnings',
    'contextlib',
    'contextvars',
    'threading',
    'types',
    'dataclasses',
    'abc',
    'io',
    'codecs',
    'weakref',
    'gc',
    'atexit',
    # --- importlib ---
    'importlib',
    'importlib.metadata',
    'importlib.resources',
    'importlib.machinery',
    'importlib.util',
    'importlib.abc',
    'pkgutil',
    'modulefinder',
    # --- crypto / encoding ---
    'hashlib',
    'hmac',
    'secrets',
    'base64',
    'binascii',
    'quopri',
    'uu',
    # --- numbers / text ---
    'numbers',
    'decimal',
    'fractions',
    'statistics',
    'math',
    'cmath',
    'random',
    'textwrap',
    'string',
    'stringprep',
    're',
    'json',
    'csv',
    'configparser',
    'netrc',
    # --- argparse / CLI ---
    'argparse',
    'getopt',
    'shlex',
    'cmd',
    # --- pathlib / fs ---
    'pathlib',
    'os',
    'os.path',
    'ntpath',
    'posixpath',
    'genericpath',
    'tempfile',
    'glob',
    'fnmatch',
    'stat',
    'fileinput',
    'filecmp',
    'shutil',
    'errno',
    # --- concurrency ---
    'concurrent',
    'concurrent.futures',
    'concurrent.futures.thread',
    'concurrent.futures.process',
    'asyncio',
    'asyncio.events',
    'asyncio.base_events',
    'asyncio.coroutines',
    'asyncio.futures',
    'asyncio.tasks',
    'asyncio.locks',
    'asyncio.queues',
    'asyncio.subprocess',
    'asyncio.streams',
    'queue',
    'sched',
    'signal',
    'subprocess',
    # --- net / email / http / urllib ---
    'email',
    'email.message',
    'email.parser',
    'email.policy',
    'email.header',
    'email.utils',
    'email.mime',
    'email.mime.text',
    'email.mime.multipart',
    'email.mime.base',
    'http',
    'http.client',
    'http.server',
    'http.cookiejar',
    'http.cookies',
    'urllib',
    'urllib.parse',
    'urllib.request',
    'urllib.error',
    'urllib.response',
    'urllib.robotparser',
    'ipaddress',
    'ftplib',
    'smtplib',
    'poplib',
    'imaplib',
    # --- xml / html / markup ---
    'html',
    'html.parser',
    'html.entities',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'xml.parsers',
    'xml.parsers.expat',
    'xml.sax',
    'xml.dom',
    'xml.dom.minidom',
    'xmlrpc',
    'xmlrpc.client',
    # --- db / persistence ---
    'sqlite3',
    'dbm',
    'shelve',
    # --- misc commonly pulled by ML stacks ---
    'uuid',
    'locale',
    'gettext',
    'calendar',
    'datetime',
    'time',
    'zoneinfo',
    'graphlib',
    'heapq',
    'bisect',
    'array',
    'unicodedata',
    'encodings',
    'mimetypes',
    'wsgiref',
    'wsgiref.simple_server',
    'unittest',
    'unittest.mock',
)


def iter_ml_stdlib_module_names(lib_root: Path | None = None) -> list[str]:
    """Nearly all stdlib module names under Lib/ for PyInstaller hiddenimports.

    Skips test suites, idle, venv, tkinter, and CPython demo/frozen stubs
    (``__phello__``, ``__hello__``, etc.). Call from the .spec at build time.
    """
    root = lib_root if lib_root is not None else Path(sys.base_prefix) / 'Lib'
    if not root.is_dir():
        return list(_ML_STDLIB_MODULES)

    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if _skip_stdlib_part(entry.name):
            continue
        if entry.is_dir():
            if not (entry / '__init__.py').is_file():
                continue
            names.append(entry.name)
            for py in sorted(entry.rglob('*.py')):
                if py.name == '__init__.py':
                    continue
                if any(_skip_stdlib_part(part)
                       for part in py.relative_to(root).parts):
                    continue
                rel = py.relative_to(root).with_suffix('')
                names.append('.'.join(rel.parts))
        elif entry.suffix == '.py' and entry.name != '__init__.py':
            if not _skip_stdlib_part(entry.stem):
                names.append(entry.stem)
    return [n for n in names if is_available_stdlib_hiddenimport(n)]


def ensure_stdlib_for_external_ml() -> None:
    for name in _ML_STDLIB_MODULES:
        try:
            __import__(name)
        except ImportError:
            pass

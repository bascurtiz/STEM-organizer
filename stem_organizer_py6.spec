# PyInstaller spec for STEM organizer (PySide6)
# Build: pyinstaller stem_organizer_py6.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

from frozen_stdlib_imports import (
    _ML_STDLIB_MODULES,
    is_available_stdlib_hiddenimport,
    is_excluded_stdlib_hiddenimport,
    iter_ml_stdlib_module_names,
)

block_cipher = None

# Minimal PySide6 data (plugins/translations). Heavy modules stripped after Analysis.
datas = []
datas += collect_data_files(
    'PySide6',
    include_py_files=False,
    excludes=[
        'qml',
        'qml/**',
        'resources/qtwebengine*',
        'resources/qtwebengine*/**',
        'translations/qtwebengine*',
        'translations/qtwebengine*/**',
        'metatypes/**',
        'Qt*/**',  # avoid shipping unused Qt* helper trees when present
    ],
)
binaries = []

# Logo + tagger Python helpers. ONNX weights are copied by build.bat to the
# top-level dist folder only (avoids duplicating ~1.3 GB under _internal/).
datas += [('logo.png', '.')]
datas += [('logo.ico', '.')]
datas += [('genre_gender_tagger/genre_gender_tagger.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/vocal_reverb.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/file_writable.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/maest_onnx.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/maest_fe_np.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/_maest_audio_utils.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/requirements.txt', 'genre_gender_tagger')]
datas += [('instrument_tagger/instrument_tagger.py', 'instrument_tagger')]
datas += [('panns_tagger/panns_tagger.py', 'panns_tagger')]
datas += [('panns_tagger/file_writable.py', 'panns_tagger')]
datas += [('panns_tagger/readme.md', 'panns_tagger')]
datas += [('key_tagger/key_tagger.py', 'key_tagger')]
datas += [('key_tagger/inference.py', 'key_tagger')]
datas += [('key_tagger/keys.py', 'key_tagger')]
datas += [('key_tagger/model.py', 'key_tagger')]
datas += [('key_tagger/log_pace.py', 'key_tagger')]
datas += [('key_tagger/requirements.txt', 'key_tagger')]
datas += [('demucs_onnx.py', '.')]
datas += [('stem_cnn6_onnx.py', '.')]
datas += [('ort_util.py', '.')]

try:
    from PyInstaller.utils.hooks import collect_all as _collect_all
    _ort_d, _ort_b, _ort_h = _collect_all('onnxruntime')
    _ort_d = [
        (src, dest)
        for src, dest in _ort_d
        if 'datasets' not in str(dest).replace('\\', '/')
        and not str(src).lower().endswith('.onnx')
    ]
    datas += _ort_d
    binaries += _ort_b
    _ort_hidden = list(_ort_h)
    # CUDA EP deps (cudnn/cublas/runtime/nvrtc) live under nvidia/*/bin.
    for _nv_pkg in (
        'nvidia.cudnn',
        'nvidia.cublas',
        'nvidia.cuda_runtime',
        'nvidia.cuda_nvrtc',
        'nvidia.cufft',
    ):
        try:
            _nd, _nb, _nh = _collect_all(_nv_pkg)
            datas += _nd
            binaries += _nb
            _ort_hidden += list(_nh)
        except Exception:
            pass
except Exception:
    _ort_hidden = ['onnxruntime']

_PYSIDE6_KEEP = (
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtCharts',
    'PySide6.QtPrintSupport',
    'PySide6.QtNetwork',
    'PySide6.QtSvg',
    'PySide6.QtSvgWidgets',
    'PySide6.QtOpenGL',  # Charts / Fluent may need GL helpers
    'PySide6.QtOpenGLWidgets',
)

hiddenimports = []
hiddenimports += list(_PYSIDE6_KEEP)
hiddenimports += ['shiboken6']
hiddenimports += ['classify_backend', 'pair_matcher', 'stem_align',
                  'ffmpeg_bootstrap', 'mp3val_bootstrap', 'flac_bootstrap', 'deps_bootstrap', 'tagger_launch',
                  'demucs_onnx', 'stem_cnn6_onnx', 'ort_util',
                  'resource_monitor',
                  'update_checker', 'single_instance', 'done_sound',
                  'audio_resample',
                  'sounddevice', 'soundfile', 'numpy', 'soxr', 'mutagen',
                  'librosa', 'audioread', 'flac_detective',
                  'track_renamer.engine', 'track_renamer.folder_scanner',
                  'track_renamer.audio_preview', 'track_renamer.instrument_enrich',
                  'track_renamer.category_palette',
                  'stem_organizer.player.audio_io',
                  'stem_organizer.player.audio_engine',
                  'stem_organizer.player.track_state']
hiddenimports += _ort_hidden
_seen = set()
for _name in list(iter_ml_stdlib_module_names()) + list(_ML_STDLIB_MODULES):
    if _name not in _seen and is_available_stdlib_hiddenimport(_name):
        _seen.add(_name)
        hiddenimports.append(_name)
hiddenimports += ['frozen_stdlib_imports']
hiddenimports = [m for m in hiddenimports if not is_excluded_stdlib_hiddenimport(m)]
assert not any(
    m == '__hello__' or m.startswith('__hello__.') or '__phello__' in m
    for m in hiddenimports
), 'stdlib demo stubs leaked into hiddenimports'
import sys as _sys
if _sys.version_info >= (3, 11):
    assert 'binhex' not in hiddenimports, (
        'binhex must not be a hiddenimport on Python 3.11+ (removed from stdlib)'
    )

_PYSIDE6_EXCLUDE = [
    'PySide6.scripts',
    'qfluentwidgets.multimedia',
    'PySide6.QtWebEngine',
    'PySide6.QtWebEngineCore',
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebEngineQuick',
    'PySide6.QtWebChannel',
    'PySide6.QtWebSockets',
    'PySide6.QtWebView',
    'PySide6.QtQuick',
    'PySide6.QtQuick3D',
    'PySide6.QtQuickControls2',
    'PySide6.QtQuickWidgets',
    'PySide6.QtQuickTest',
    'PySide6.QtQml',
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic',
    'PySide6.Qt3DAnimation',
    'PySide6.Qt3DExtras',
    'PySide6.QtBluetooth',
    'PySide6.QtNfc',
    'PySide6.QtSensors',
    'PySide6.QtPositioning',
    'PySide6.QtLocation',
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    'PySide6.QtDataVisualization',
    'PySide6.QtGraphs',
    'PySide6.QtGraphsWidgets',
    'PySide6.QtRemoteObjects',
    'PySide6.QtScxml',
    'PySide6.QtTextToSpeech',
    'PySide6.QtSerialPort',
    'PySide6.QtSerialBus',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'PySide6.QtSpatialAudio',
    'PySide6.QtHttpServer',
    'PySide6.QtDesigner',
    'PySide6.QtUiTools',
    'PySide6.QtHelp',
    'PySide6.QtTest',
    'PySide6.QtCanvasPainter',
]

# Substrings matched against TOC dest/src paths (case-insensitive).
# Keep Core/Gui/Widgets/Charts/PrintSupport/Network/Svg/OpenGL + plugins.
_QT_STRIP_SUBSTR = (
    'webengine',
    'qtwebengine',
    'quick3d',
    'qt6quick',
    'qtquick',
    'quickcontrols',
    'quickwidgets',
    'quicktest',
    'qt6qml',
    'qtqml',
    '/qml/',
    '\\qml\\',
    'qmlls',
    'qt6pdf',
    'qtpdf',
    'multimedia',
    'spatialaudio',
    'qt63d',
    'qt3d',
    'designer',
    'uitools',
    'assistant',
    'linguist',
    'lupdate',
    'lrelease',
    'bluetooth',
    'qtnfc',
    'sensors',
    'positioning',
    'location',
    'datavisualization',
    'qt6graphs',
    'qtgraphs',
    'remoteobjects',
    'scxml',
    'texttospeech',
    'serialport',
    'serialbus',
    'httpserver',
    'webview',
    'webchannel',
    'websockets',
    'avcodec',
    'avformat',
    'avutil',
    'swresample',
    'swscale',
    'libva',
    'metatypes',
)


def _keep_qt_toc_entry(src, dest=None):
    """Drop unused Qt/WebEngine/QML/Multimedia TOC entries after Analysis."""
    blob = f'{src}::{dest or ""}'.replace('\\', '/').lower()
    # Non-Qt assets stay.
    if (
        'pyside6' not in blob
        and 'shiboken' not in blob
        and 'qt6' not in blob
        and 'qt5' not in blob
        and 'opengl32sw' not in blob
    ):
        return True
    for bad in _QT_STRIP_SUBSTR:
        if bad in blob:
            return False
    return True


a = Analysis(
    ['run_stem_organizer.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'tkinter', 'customtkinter',
        '__phello__', '__phello__.foo', '__phello__.spam', '__hello__',
    ] + _PYSIDE6_EXCLUDE,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Hooks still pull WebEngine/Quick via dependency walks — strip hard.
_before_b, _before_d = len(a.binaries), len(a.datas)
a.binaries = [b for b in a.binaries if _keep_qt_toc_entry(b[0], b[1])]
a.datas = [d for d in a.datas if _keep_qt_toc_entry(d[0], d[1])]
print(
    f'[spec] Qt strip: binaries {_before_b}->{len(a.binaries)}, '
    f'datas {_before_d}->{len(a.datas)}'
)

# --- Installer slim-down: drop unused / duplicate binaries after Analysis ---
# Each pattern below is a guaranteed- or near-zero-risk size win. Combined they
# remove ~0.8 GB from the onedir. See AGENTS.md for the rationale.
import os as _os
def _binary_dest_name(b):
    # b is (src, dest); normalise to forward slashes for substring checks.
    return (b[1] or b[0]).replace('\\', '/').lower()

_SIZE_DROP_SUBSTR = (
    # Duplicate CUDA-13 cuBLAS DLLs (~507 MB). PyInstaller hoovered these off a
    # system CUDA 13 toolkit install; the cu12 wheels already supply the loaded
    # libs (nvidia/cublas/bin/cublas*64_12.dll). Pure duplicates.
    'cublaslt64_13.dll',
    'cublas64_13.dll',
    # Alternate-build NVRTC blob (~86 MB). nvrtc64_120_0.dll (the canonical one)
    # is retained; the .alt variant is only for rare kernel-compile paths.
    'nvrtc64_120_0.alt.dll',
    # Stray system-CUDA companions never loaded by ORT CUDA EP (~290 MB).
    # Analysis PATH-picks system toolkit copies when CUDA DLLs are scanned.
    'cusparse',
    'curand',
    'nvjitlink',
    'cufftw',
)
# Path/data drops (matched against dest/src, case-insensitive).
_SIZE_DROP_DATA_SUBSTR = (
    'nvidia/nvjitlink',
    'nvidia\\nvjitlink',
)

def _drop_size_toc(entry):
    # TOC entry is (name/dest, path/src, typecode) after Analysis, or (src, dest)
    # for our pre-Analysis lists. Handle both.
    if len(entry) >= 2:
        blob = f'{entry[0]}::{entry[1]}'.replace('\\', '/').lower()
    else:
        blob = str(entry[0]).replace('\\', '/').lower()
    if any(s in blob for s in _SIZE_DROP_SUBSTR):
        return True
    if any(s.replace('\\', '/') in blob for s in _SIZE_DROP_DATA_SUBSTR):
        return True
    return False

_b_b, _d_b = len(a.binaries), len(a.datas)
a.binaries = [b for b in a.binaries if not _drop_size_toc(b)]
a.datas = [d for d in a.datas if not _drop_size_toc(d)]
# ffplay.exe (~98 MB) ships in ffmpeg/ but the app only ever spawns ffmpeg.exe.
if not str(_os.environ.get('STEM_KEEP_FFPLAY', '')).strip():
    a.binaries = [b for b in a.binaries if 'ffmpeg/ffplay.exe' not in _binary_dest_name(b)
                  and not _binary_dest_name(b).endswith('ffplay.exe')]
# Optional A/B drop (Phase A2): cuDNN precompiled-engines blob (~522 MB). ORT's
# cuDNN frontend may need it for some conv configs, so it stays by default. Set
# STEM_DROP_CUDNN_ENGINES=1 to exclude it, then smoke-test Classify GPU before
# shipping. Disabled by default to preserve the working CUDA path.
if str(_os.environ.get('STEM_DROP_CUDNN_ENGINES', '')).strip():
    a.binaries = [b for b in a.binaries if 'cudnn_engines_precompiled' not in _binary_dest_name(b)]
print(
    f'[spec] size strip: binaries {_b_b}->{len(a.binaries)}, '
    f'datas {_d_b}->{len(a.datas)}'
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='STEM-organizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='STEM-organizer',
)

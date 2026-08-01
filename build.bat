@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM =============================================================================
REM STEM organizer (PySide6) - Windows build (PyInstaller)
REM
REM Entry: run_stem_organizer.py  /  package stem_organizer
REM Spec:  stem_organizer_py6.spec  (onedir under dist\STEM-organizer\)
REM
REM After a successful build:
REM   1. Open dist\STEM-organizer\
REM   2. (Optional) install-deps.bat = ffmpeg / mp3val / flac if missing
REM   3. Run STEM-organizer.exe
REM   4. Optional installer: compile stem_organizer.iss with Inno Setup 6
REM      (models download from GitHub; not baked into the setup EXE)
REM =============================================================================

echo.
echo ========================================
echo   STEM organizer (PySide6) - Windows build
echo ========================================
echo.

set "VENV=.build-venv"
set "PY=%VENV%\Scripts\python.exe"

echo [1/4] Checking Python ...
REM Windows Store alias makes "where python" succeed with a stub that is not a real interpreter.
python -c "import sys" >nul 2>&1
if errorlevel 1 goto no_python

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Wrong Python version:
    python --version
    echo Download 3.10 or 3.11 from here:
    echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo NOTE: During install, Add python.exe to PATH
    echo After installed, run build.bat again
    pause
    exit /b 1
)
python --version
echo.
goto python_ok

:no_python
echo Python was not found; Download and install it from here:
echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo NOTE: During install, Add python.exe to PATH
echo After installed, run build.bat again
pause
exit /b 1

:python_ok

echo [2/4] Preparing build environment ...
if not exist "%PY%" (
    echo   Creating %VENV% ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo   Installing PyInstaller and packager deps ...
"%PY%" -m pip install -q -U pip
"%PY%" -m pip install -q pyinstaller packaging
if errorlevel 1 (
    echo ERROR: Failed to install build dependencies.
    pause
    exit /b 1
)
REM Freeze deps = requirements.txt (pinned onnxruntime-gpu + librosa + flac-detective).
if not exist "requirements.txt" (
    echo ERROR: requirements.txt missing.
    pause
    exit /b 1
)
echo   Installing requirements.txt into build venv ...
"%PY%" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.txt.
    pause
    exit /b 1
)
"%PY%" -c "import onnxruntime as o, librosa, flac_detective; print('  OK ort', o.__version__, 'librosa', librosa.__version__, 'flac_detective OK')"
if errorlevel 1 (
    echo ERROR: Freeze deps failed import check.
    pause
    exit /b 1
)
"%PY%" --version
echo   Ready.
echo.

echo [3/4] Running PyInstaller ...
echo   Bundling UI into dist\STEM-organizer\STEM-organizer.exe
echo   This usually takes a few minutes - output below:
echo.

REM ERROR keeps real failures visible; hides WARN (e.g. optional ctypes/AppKit/nvml).
"%PY%" -m PyInstaller --noconfirm --clean --log-level=ERROR stem_organizer_py6.spec 2>&1
if errorlevel 1 goto failed

echo.
echo [4/4] Finishing dist folder ...
"%PY%" -c "import sys; open('python-version.txt','w',encoding='utf-8').write(f'{sys.version_info[0]}.{sys.version_info[1]}\n')"

set "OUT=dist\STEM-organizer"
if not exist "%OUT%" (
    echo ERROR: expected output folder missing: %OUT%
    goto failed
)

copy /Y install-deps.bat "%OUT%\install-deps.bat" >nul
copy /Y python-version.txt "%OUT%\python-version.txt" >nul
if exist requirements.txt copy /Y requirements.txt "%OUT%\requirements.txt" >nul
if exist demucs_onnx.py copy /Y demucs_onnx.py "%OUT%\demucs_onnx.py" >nul
if exist ort_util.py copy /Y ort_util.py "%OUT%\ort_util.py" >nul
REM Phase 3 Demucs ONNX weight (also next to spike for dev).
if not exist "%OUT%\models" mkdir "%OUT%\models" >nul
if exist "models\htdemucs.onnx" copy /Y "models\htdemucs.onnx" "%OUT%\models\htdemucs.onnx" >nul
if exist "_onnx_spike\onnx_out\htdemucs.onnx" if not exist "%OUT%\models\htdemucs.onnx" (
    copy /Y "_onnx_spike\onnx_out\htdemucs.onnx" "%OUT%\models\htdemucs.onnx" >nul
)

echo   Copying genre_gender_tagger\ ^(bundled tagger, no venv^) ...
if exist "%OUT%\genre_gender_tagger" rmdir /S /Q "%OUT%\genre_gender_tagger"
mkdir "%OUT%\genre_gender_tagger" >nul
mkdir "%OUT%\genre_gender_tagger\models" >nul
copy /Y "genre_gender_tagger\genre_gender_tagger.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\vocal_reverb.py" copy /Y "genre_gender_tagger\vocal_reverb.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\file_writable.py" copy /Y "genre_gender_tagger\file_writable.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\maest_onnx.py" copy /Y "genre_gender_tagger\maest_onnx.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\maest_fe_np.py" copy /Y "genre_gender_tagger\maest_fe_np.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\_maest_audio_utils.py" copy /Y "genre_gender_tagger\_maest_audio_utils.py" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\install-deps.bat" copy /Y "genre_gender_tagger\install-deps.bat" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\run.bat" copy /Y "genre_gender_tagger\run.bat" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\requirements.txt" copy /Y "genre_gender_tagger\requirements.txt" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\readme.md" copy /Y "genre_gender_tagger\readme.md" "%OUT%\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\models\*.onnx" copy /Y "genre_gender_tagger\models\*.onnx" "%OUT%\genre_gender_tagger\models\" >nul
REM ONNX weights + config sidecars (vocal_reverb .pt / MAEST HF weights not shipped).
if exist "genre_gender_tagger\models\vocal_reverb.config.json" copy /Y "genre_gender_tagger\models\vocal_reverb.config.json" "%OUT%\genre_gender_tagger\models\" >nul
if exist "genre_gender_tagger\models\maest_discogs519.id2label.json" copy /Y "genre_gender_tagger\models\maest_discogs519.id2label.json" "%OUT%\genre_gender_tagger\models\" >nul

echo   Copying instrument_tagger\ ^(Rename Auto-detect, no venv^) ...
if exist "%OUT%\instrument_tagger" rmdir /S /Q "%OUT%\instrument_tagger"
mkdir "%OUT%\instrument_tagger" >nul
mkdir "%OUT%\instrument_tagger\models" >nul
copy /Y "instrument_tagger\instrument_tagger.py" "%OUT%\instrument_tagger\" >nul
if exist "instrument_tagger\passt_mel.py" copy /Y "instrument_tagger\passt_mel.py" "%OUT%\instrument_tagger\" >nul
if exist "instrument_tagger\passt_mel_np.py" copy /Y "instrument_tagger\passt_mel_np.py" "%OUT%\instrument_tagger\" >nul
if exist "instrument_tagger\install-deps.bat" copy /Y "instrument_tagger\install-deps.bat" "%OUT%\instrument_tagger\" >nul
REM Phase 2 ONNX weight (PaSST OpenMIC — hear21passt .pt is not shipped).
if exist "instrument_tagger\models\*.onnx" copy /Y "instrument_tagger\models\*.onnx" "%OUT%\instrument_tagger\models\" >nul

echo   Copying panns_tagger\ ^(AudioSet Cnn14, no venv^) ...
if exist "%OUT%\panns_tagger" rmdir /S /Q "%OUT%\panns_tagger"
mkdir "%OUT%\panns_tagger" >nul
mkdir "%OUT%\panns_tagger\models" >nul
copy /Y "panns_tagger\panns_tagger.py" "%OUT%\panns_tagger\" >nul
if exist "panns_tagger\file_writable.py" copy /Y "panns_tagger\file_writable.py" "%OUT%\panns_tagger\" >nul
if exist "panns_tagger\install-deps.bat" copy /Y "panns_tagger\install-deps.bat" "%OUT%\panns_tagger\" >nul
if exist "panns_tagger\readme.md" copy /Y "panns_tagger\readme.md" "%OUT%\panns_tagger\" >nul
REM ONNX weight + AudioSet labels CSV (Cnn14 .pth is gone — ONNX is the only weight).
if exist "panns_tagger\models\*.onnx" copy /Y "panns_tagger\models\*.onnx" "%OUT%\panns_tagger\models\" >nul
if exist "panns_tagger\models\class_labels_indices.csv" copy /Y "panns_tagger\models\class_labels_indices.csv" "%OUT%\panns_tagger\models\" >nul

echo   Copying key_tagger\ ^(Key Detect, no venv^) ...
if exist "%OUT%\key_tagger" rmdir /S /Q "%OUT%\key_tagger"
mkdir "%OUT%\key_tagger" >nul
mkdir "%OUT%\key_tagger\checkpoints" >nul
copy /Y "key_tagger\key_tagger.py" "%OUT%\key_tagger\" >nul
if exist "key_tagger\inference.py" copy /Y "key_tagger\inference.py" "%OUT%\key_tagger\" >nul
if exist "key_tagger\keys.py" copy /Y "key_tagger\keys.py" "%OUT%\key_tagger\" >nul
if exist "key_tagger\model.py" copy /Y "key_tagger\model.py" "%OUT%\key_tagger\" >nul
if exist "key_tagger\log_pace.py" copy /Y "key_tagger\log_pace.py" "%OUT%\key_tagger\" >nul
if exist "key_tagger\install-deps.bat" copy /Y "key_tagger\install-deps.bat" "%OUT%\key_tagger\" >nul
if exist "key_tagger\requirements.txt" copy /Y "key_tagger\requirements.txt" "%OUT%\key_tagger\" >nul
REM ONNX weight (nf50 .pt is gone — ONNX is the only weight).
if exist "key_tagger\checkpoints\*.onnx" copy /Y "key_tagger\checkpoints\*.onnx" "%OUT%\key_tagger\checkpoints\" >nul

if not exist "%OUT%\key_tagger\key_tagger.py" (
    echo ERROR: key_tagger\key_tagger.py missing from dist — Key Detect will not run.
    goto failed
)
if not exist "%OUT%\panns_tagger\panns_tagger.py" (
    echo ERROR: panns_tagger\panns_tagger.py missing from dist — Vocal type will not run.
    goto failed
)
if not exist "%OUT%\genre_gender_tagger\genre_gender_tagger.py" (
    echo ERROR: genre_gender_tagger\genre_gender_tagger.py missing from dist.
    goto failed
)

REM Phase 1 ONNX weights must ship with the build (.pth deleted; STEM_ONNX default).
if not exist "%OUT%\genre_gender_tagger\models\vocal_reverb.onnx" (
    echo ERROR: vocal_reverb.onnx missing from dist — Gender reverb ONNX path will fail.
    echo   Expected at: genre_gender_tagger\models\vocal_reverb.onnx
    goto failed
)
if not exist "%OUT%\genre_gender_tagger\models\vocal_reverb.config.json" (
    echo ERROR: vocal_reverb.config.json missing from dist — reverb ONNX needs the sidecar.
    goto failed
)
if not exist "%OUT%\key_tagger\checkpoints\nf50-q05-221125.onnx" (
    echo ERROR: nf50-q05-221125.onnx missing from dist — Key Detect ONNX path will fail.
    echo   Expected at: key_tagger\checkpoints\nf50-q05-221125.onnx
    goto failed
)
if not exist "%OUT%\panns_tagger\models\cnn14.onnx" (
    echo ERROR: cnn14.onnx missing from dist — Vocal type ONNX path will fail.
    echo   Expected at: panns_tagger\models\cnn14.onnx
    goto failed
)
if not exist "%OUT%\panns_tagger\models\class_labels_indices.csv" (
    echo ERROR: class_labels_indices.csv missing from dist — PANNs labels required.
    goto failed
)
if not exist "%OUT%\instrument_tagger\passt_mel_np.py" (
    echo ERROR: passt_mel_np.py missing from dist — PaSST ONNX mel frontend required.
    goto failed
)
if not exist "%OUT%\instrument_tagger\models\passt_openmic.onnx" (
    echo ERROR: passt_openmic.onnx missing from dist — Rename Auto-detect ONNX path will fail.
    echo   Expected at: instrument_tagger\models\passt_openmic.onnx
    goto failed
)
if not exist "%OUT%\genre_gender_tagger\models\maest_discogs519.onnx" (
    echo ERROR: maest_discogs519.onnx missing from dist — Genre MAEST ONNX path will fail.
    echo   Expected at: genre_gender_tagger\models\maest_discogs519.onnx
    goto failed
)
if not exist "%OUT%\genre_gender_tagger\models\maest_discogs519.id2label.json" (
    echo ERROR: maest_discogs519.id2label.json missing from dist — MAEST labels required.
    goto failed
)
if not exist "%OUT%\genre_gender_tagger\maest_onnx.py" (
    echo ERROR: maest_onnx.py missing from dist — Genre MAEST ONNX runner required.
    goto failed
)
if not exist "%OUT%\demucs_onnx.py" (
    echo ERROR: demucs_onnx.py missing from dist — Classify Demucs ONNX runner required.
    goto failed
)
if not exist "%OUT%\models\htdemucs.onnx" (
    echo ERROR: htdemucs.onnx missing from dist — Classify ONNX path will fail.
    echo   Expected at: models\htdemucs.onnx
    goto failed
)

REM ffmpeg is NOT bundled by the .spec - install-deps.bat downloads it next to the exe after build.
REM If a local ffmpeg\ already exists (dev machine), copy it for convenience.
REM Copy ffmpeg.exe + ffprobe.exe only (ffplay.exe ~98 MB optional — Rename audition).
if exist "ffmpeg\ffmpeg.exe" if not exist "%OUT%\ffmpeg\ffmpeg.exe" (
    echo   Copying ffmpeg\ ^(ffmpeg.exe + ffprobe.exe only^) ...
    if not exist "%OUT%\ffmpeg" mkdir "%OUT%\ffmpeg" >nul
    copy /Y "ffmpeg\ffmpeg.exe" "%OUT%\ffmpeg\ffmpeg.exe" >nul
    if exist "ffmpeg\ffprobe.exe" copy /Y "ffmpeg\ffprobe.exe" "%OUT%\ffmpeg\ffprobe.exe" >nul
)

echo.
echo ========================================
echo   SUCCESS
echo ========================================
echo   Exe:  dist\STEM-organizer\STEM-organizer.exe
echo   Next: start STEM-organizer.exe
echo         ^(ORT + librosa + flac-detective are in the freeze^)
echo         Optional: install-deps.bat for missing ffmpeg/mp3val/flac
echo         Installer: compile stem_organizer.iss ^(models download from GitHub^)
echo.
pause
exit /b 0

:failed
echo.
echo ========================================
echo   BUILD FAILED
echo ========================================
echo   See messages above.
echo.
pause
exit /b 1

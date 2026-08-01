@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Key Detect tagger - install dependencies
echo.
echo You need:
echo   - Python 3.10 or 3.11 on PATH
echo   - Internet ^(torch + librosa + more-itertools^)
echo.
echo Installs into shared genre_gender_tagger\venv
echo ^(run Genre ^& Gender install first if that venv is empty^).
echo.
echo Frozen STEM-organizer.exe: use root install-deps.bat beside the .exe
echo instead ^(puts wheels into site-packages\^).
echo.
if /I not "%STEM_KEY_BUNDLED%"=="1" if /I not "%STEM_GG_BUNDLED%"=="1" (
    if exist "%~dp0..\STEM-organizer.exe" (
        echo ERROR: STEM-organizer.exe detected in parent folder.
        echo Do NOT run this nested bat for a frozen build.
        echo Run instead:
        echo   %~dp0..\install-deps.bat
        pause
        exit /b 1
    )
)

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not on PATH.
    echo Install 3.10/3.11 - tick "Add to PATH".
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo ERROR: need Python 3.10 or 3.11. Got:
    python --version
    echo Tip: py -3.11 "%~f0"
    pause
    exit /b 1
)

echo Python:
python --version
echo.

set "VENV=%~dp0..\genre_gender_tagger\venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating genre_gender_tagger\venv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: could not create shared venv.
        pause
        exit /b 1
    )
)

set "PY=%VENV%\Scripts\python.exe"

set "KEY_TORCH_CHOICE=%~1"
if "%KEY_TORCH_CHOICE%"=="" set "KEY_TORCH_CHOICE=%STEM_GG_TORCH%"
if "%KEY_TORCH_CHOICE%"=="1" goto cuda_torch
if "%KEY_TORCH_CHOICE%"=="2" goto cpu_torch
if "%KEY_TORCH_CHOICE%"=="3" goto cuda_new_torch
goto ask_gpu

:ask_gpu
echo PyTorch build ^(shared venv^):
echo   1 = NVIDIA RTX 20/30/40  ^(CUDA 12.4^)
echo   2 = CPU only
echo   3 = NVIDIA RTX 50-series ^(CUDA 12.8^)
choice /C 123 /N /M "Enter 1, 2, or 3: "
if errorlevel 3 goto cuda_new_torch
if errorlevel 2 goto cpu_torch
if errorlevel 1 goto cuda_torch
goto ask_gpu

:cuda_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "TORCH_LABEL=CUDA 12.4"
goto install_pkgs

:cuda_new_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_LABEL=CUDA 12.8"
goto install_pkgs

:cpu_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU"
goto install_pkgs

:install_pkgs
echo.
echo [1/3] audio helpers ...
REM Base audio deps (numpy/soundfile/librosa/soxr/mutagen/tqdm) are installed
REM once by the shared genre_gender_tagger venv. Only install here if standalone.
"%PY%" -c "import numpy, soundfile, librosa, soxr, mutagen, tqdm" 2>nul
if errorlevel 1 (
    "%PY%" -m pip install numpy soundfile librosa soxr mutagen tqdm
    if errorlevel 1 goto fail
)

echo [2/3] ensure torch + torchaudio (%TORCH_LABEL%) ...
"%PY%" -c "import torch, torchaudio; print('OK torch', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo torch missing - installing (%TORCH_LABEL%) ...
    "%PY%" -m pip install torch torchaudio --index-url "%TORCH_INDEX%"
    if errorlevel 1 (
        echo WARNING: torch stack install failed - leaving existing packages.
    )
)

echo [3/3] verify KeyNet deps ...
"%PY%" -c "import torch, librosa, numpy; print('OK key_tagger deps')"
if errorlevel 1 goto fail

if not exist "%~dp0checkpoints\nf50-q05-221125.pt" (
    echo WARNING: checkpoint missing:
    echo   %~dp0checkpoints\nf50-q05-221125.pt
)

echo.
echo Done (%TORCH_LABEL%). Key Detect uses checkpoints\nf50-q05-221125.pt
if /I not "%STEM_KEY_BUNDLED%"=="1" if /I not "%STEM_GG_BUNDLED%"=="1" pause
exit /b 0

:fail
echo.
echo ERROR: install failed. Check messages above.
pause
exit /b 1

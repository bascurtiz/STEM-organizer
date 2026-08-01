@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo Genre ^& Gender - install dependencies
echo.
echo You need:
echo   - Python 3.10 or 3.11 on PATH
echo   - Internet ^(PyTorch, onnxruntime, transformers^)
echo.
echo Creates venv\ here. Shared with Rename Auto-detect.
echo.

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

set "VENV=%~dp0venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating venv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: could not create venv.
        pause
        exit /b 1
    )
)

set "PY=%VENV%\Scripts\python.exe"

set "GG_TORCH_CHOICE=%~1"
if "%GG_TORCH_CHOICE%"=="" set "GG_TORCH_CHOICE=%STEM_GG_TORCH%"
if "%GG_TORCH_CHOICE%"=="1" goto cuda_torch
if "%GG_TORCH_CHOICE%"=="2" goto cpu_torch
if "%GG_TORCH_CHOICE%"=="3" goto cuda_new_torch
goto ask_gpu

:ask_gpu
echo PyTorch build:
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
goto install_torch

:cuda_new_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_LABEL=CUDA 12.8"
goto install_torch

:cpu_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU"
goto install_torch

:install_torch
echo.
echo [1/2] PyTorch (%TORCH_LABEL%) ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
"%PY%" -m pip install torch --index-url "%TORCH_INDEX%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo [2/2] requirements.txt ...
"%PY%" -m pip install -r "%~dp0requirements.txt" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo.
if /I "%TORCH_LABEL%"=="CPU" (
    "%PY%" -c "import torch, soundfile, tqdm, mutagen, transformers, librosa, onnxruntime; v=torch.__version__; assert '+cpu' in v or 'cpu' in v.lower(), f'expected CPU torch, got {v}'; print('OK torch', v)"
) else (
    "%PY%" -c "import torch, soundfile, tqdm, mutagen, transformers, librosa, onnxruntime; print('OK torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    "%PY%" -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>nul
    if errorlevel 1 echo WARNING: no CUDA GPU seen - tagger falls back to CPU. Pick 2 if this is a CPU-only VM.
)

echo.
echo Done (%TORCH_LABEL%). Use Genre ^& Gender in STEM organizer, or run.bat.
if /I not "%STEM_GG_BUNDLED%"=="1" pause
exit /b 0

:failed
echo.
echo ERROR: install failed. Check messages above.
pause
exit /b 1

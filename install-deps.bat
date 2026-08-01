@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM =============================================================================
REM STEM organizer — install optional deps / tools (v1.0.8+)
REM
REM Frozen build (STEM-organizer.exe beside this script):
REM   - ML stack is already inside the freeze (onnxruntime-gpu + librosa +
REM     flac-detective, etc.). This script only fetches ffmpeg / mp3val / flac
REM     if missing.
REM
REM Source / .venv (no STEM-organizer.exe here):
REM   - Creates/uses .venv and installs pinned deps from requirements.txt
REM     (onnxruntime-gpu==1.28.0 + NVIDIA CUDA wheels + audio helpers).
REM   - Then fetches ffmpeg / mp3val / flac.
REM
REM Demucs GPU = CUDA EP only. DirectML is not used for HTDemucs.
REM ONNX model weights are NOT installed here — installer downloads them from
REM GitHub (bascurtiz/STEM-organizer-models tag "models"), or place them under
REM models\ / tagger folders for a local freeze.
REM =============================================================================

echo.
echo STEM organizer - install dependencies / tools
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not on PATH.
    echo Install 3.10/3.11 from https://www.python.org/downloads/ - tick "Add to PATH".
    pause
    exit /b 1
)

set "HOST_PY="
for /f "delims=" %%P in ('where python 2^>nul') do (
    set "HOST_PY=%%P"
    goto host_py_found
)
:host_py_found
if not defined HOST_PY (
    echo ERROR: python not on PATH.
    pause
    exit /b 1
)
for /f "tokens=1,2" %%A in ('python -c "import sys; print(sys.version_info[0], sys.version_info[1])"') do set "HOST_VER=%%A.%%B"
if not defined HOST_VER (
    echo ERROR: could not read Python version from:
    echo   %HOST_PY%
    pause
    exit /b 1
)

"%HOST_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo ERROR: need Python 3.10 or 3.11. Got:
    "%HOST_PY%" --version
    echo Install from https://www.python.org/downloads/ - tick Add to PATH.
    pause
    exit /b 1
)

set "REQ_VER="
if exist "%~dp0python-version.txt" (
    for /f "usebackq delims=" %%R in ("%~dp0python-version.txt") do set "REQ_VER=%%R"
)
if defined REQ_VER (
    if /I not "%HOST_VER%"=="%REQ_VER%" (
        echo ERROR: Python mismatch. This folder expects Python %REQ_VER%.
        echo You have: %HOST_VER%  ^(%HOST_PY%^)
        echo Install Python %REQ_VER%, then re-run: py -%REQ_VER% "%~f0"
        pause
        exit /b 1
    )
)

set "USE_SITE=0"
if exist "%~dp0STEM-organizer.exe" set "USE_SITE=1"

if "%USE_SITE%"=="1" (
    echo Mode: frozen EXE present — skipping pip ML install ^(already in freeze^).
    echo Will only ensure ffmpeg / mp3val / flac beside the app.
    echo.
    goto tools_section
)

REM --- Source mode: .venv + requirements.txt ---
echo Mode: source — install pinned ONNX/CUDA stack into .venv
echo Runtime: onnxruntime-gpu ^(CUDA EP^) — DirectML not used for Demucs.
echo.
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -c "import sys; v='%HOST_VER%'.split('.'); raise SystemExit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" 1>nul 2>nul
    if errorlevel 1 (
        echo Existing .venv is broken or wrong Python - recreating with %HOST_VER% ...
        rmdir /S /Q "%~dp0.venv" 2>nul
    )
)
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Creating .venv ...
    "%HOST_PY%" -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo ERROR: failed to create .venv
        pause
        exit /b 1
    )
)
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%~dp0requirements.txt" (
    echo ERROR: requirements.txt not found beside this script.
    pause
    exit /b 1
)

echo Installing from requirements.txt ...
"%PY%" -m pip install -q -U pip
if errorlevel 1 goto failed
"%PY%" -m pip install -r "%~dp0requirements.txt" --upgrade
if errorlevel 1 goto failed
"%PY%" -c "import onnxruntime as o, librosa, flac_detective, numpy, soundfile; print('OK ort', o.__version__, 'librosa', librosa.__version__, 'flac_detective OK'); print(' providers', o.get_available_providers()[:4])"
if errorlevel 1 goto failed
echo.
echo Python deps OK. Continuing with tools ...
echo.

:tools_section
REM --- ffmpeg ---
set "FFMPEG_DIR=%~dp0ffmpeg"
if exist "%FFMPEG_DIR%\ffmpeg.exe" goto ffmpeg_done

set "FFMPEG_ZIP=%TEMP%\stem-organizer-ffmpeg.zip"
set "FFMPEG_URL=https://github.com/GyanD/codexffmpeg/releases/download/8.1/ffmpeg-8.1-essentials_build.zip"
set "FFMPEG_URL_FALLBACK=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

echo Downloading ffmpeg ...
where curl >nul 2>&1
if errorlevel 1 goto ffmpeg_download_ps

curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL%"
if errorlevel 1 curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL_FALLBACK%"
if not errorlevel 1 goto ffmpeg_download_ok
goto ffmpeg_download_failed

:ffmpeg_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$urls=@('%FFMPEG_URL%','%FFMPEG_URL_FALLBACK%');" ^
  "$out='%FFMPEG_ZIP%';" ^
  "$ok=$false;" ^
  "foreach ($url in $urls) {" ^
  "  try {" ^
  "    Write-Host ('Downloading ' + $url);" ^
  "    $wc = New-Object System.Net.WebClient;" ^
  "    $wc.DownloadFile($url, $out);" ^
  "    $ok=$true; break" ^
  "  } catch { Write-Host $_.Exception.Message }" ^
  "};" ^
  "if (-not $ok) { exit 1 }"
if errorlevel 1 goto ffmpeg_download_failed

:ffmpeg_download_ok
if not exist "%FFMPEG_ZIP%" goto ffmpeg_download_failed
goto ffmpeg_extract

:ffmpeg_download_failed
echo WARNING: ffmpeg download failed - some stems may not decode.
goto ffmpeg_done

:ffmpeg_extract
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-ffmpeg-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath $tmp -Force;" ^
  "$bin = (Get-ChildItem -Path $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1).Directory.FullName;" ^
  "if (-not $bin) { throw 'ffmpeg.exe not found in archive' };" ^
  "Copy-Item -Path (Join-Path $bin '*') -Destination '%FFMPEG_DIR%' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: ffmpeg extract failed - some stems may not decode.
    goto ffmpeg_done
)
del /Q "%FFMPEG_ZIP%" 2>nul
if exist "%FFMPEG_DIR%\ffmpeg.exe" echo OK ffmpeg -^> %FFMPEG_DIR%\ffmpeg.exe

:ffmpeg_done

REM --- mp3val ---
set "MP3VAL_DIR=%~dp0mp3val"
if exist "%MP3VAL_DIR%\mp3val.exe" goto mp3val_done

set "MP3VAL_ZIP=%TEMP%\stem-organizer-mp3val.zip"
set "MP3VAL_URL=https://downloads.sourceforge.net/project/mp3val/mp3val-bundle/MP3val%%200.1.8%%20with%%20MP3val-frontend%%200.1.1%%20included/mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip"
set "MP3VAL_URL_FALLBACK=https://sourceforge.net/projects/mp3val/files/mp3val-bundle/MP3val%%200.1.8%%20with%%20MP3val-frontend%%200.1.1%%20included/mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip/download"

echo Downloading mp3val ...
where curl >nul 2>&1
if errorlevel 1 goto mp3val_download_ps

curl -L --fail --progress-bar -o "%MP3VAL_ZIP%" "%MP3VAL_URL%"
if errorlevel 1 curl -L --fail --progress-bar -o "%MP3VAL_ZIP%" "%MP3VAL_URL_FALLBACK%"
if not errorlevel 1 goto mp3val_download_ok
goto mp3val_download_failed

:mp3val_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$urls=@('%MP3VAL_URL%','%MP3VAL_URL_FALLBACK%');" ^
  "$out='%MP3VAL_ZIP%';" ^
  "$ok=$false;" ^
  "foreach ($url in $urls) {" ^
  "  try {" ^
  "    Write-Host ('Downloading ' + $url);" ^
  "    $wc = New-Object System.Net.WebClient;" ^
  "    $wc.DownloadFile($url, $out);" ^
  "    $ok=$true; break" ^
  "  } catch { Write-Host $_.Exception.Message }" ^
  "};" ^
  "if (-not $ok) { exit 1 }"
if errorlevel 1 goto mp3val_download_failed

:mp3val_download_ok
if not exist "%MP3VAL_ZIP%" goto mp3val_download_failed
goto mp3val_extract

:mp3val_download_failed
echo WARNING: mp3val download failed - Detect corruption Fix will use ffmpeg fallback.
goto mp3val_done

:mp3val_extract
if not exist "%MP3VAL_DIR%" mkdir "%MP3VAL_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-mp3val-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -Path '%MP3VAL_ZIP%' -DestinationPath $tmp -Force;" ^
  "$exe = Get-ChildItem -Path $tmp -Recurse -Filter mp3val.exe | Select-Object -First 1;" ^
  "if (-not $exe) { throw 'mp3val.exe not found in archive' };" ^
  "Copy-Item -Path $exe.FullName -Destination '%MP3VAL_DIR%\mp3val.exe' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: mp3val extract failed - Detect corruption Fix will use ffmpeg fallback.
    goto mp3val_done
)
del /Q "%MP3VAL_ZIP%" 2>nul
if exist "%MP3VAL_DIR%\mp3val.exe" echo OK mp3val -^> %MP3VAL_DIR%\mp3val.exe

:mp3val_done

REM --- flac ---
set "FLAC_DIR=%~dp0flac"
if exist "%FLAC_DIR%\flac.exe" goto flac_done

set "FLAC_ZIP=%TEMP%\stem-organizer-flac.zip"
set "FLAC_URL=https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.5.0-win.zip"
set "FLAC_URL_FALLBACK=https://downloads.xiph.org/releases/flac/flac-1.5.0-win.zip"

echo Downloading flac ...
curl -L --fail --retry 2 -o "%FLAC_ZIP%" "%FLAC_URL%" 2>nul
if errorlevel 1 goto flac_download_ps
if exist "%FLAC_ZIP%" goto flac_download_ok
curl -L --fail --retry 2 -o "%FLAC_ZIP%" "%FLAC_URL_FALLBACK%" 2>nul
if not errorlevel 1 goto flac_download_ok
goto flac_download_failed

:flac_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try {" ^
  "  $ProgressPreference = 'SilentlyContinue';" ^
  "  Invoke-WebRequest -Uri '%FLAC_URL%' -OutFile '%FLAC_ZIP%' -UseBasicParsing;" ^
  "} catch {" ^
  "  try {" ^
  "    Invoke-WebRequest -Uri '%FLAC_URL_FALLBACK%' -OutFile '%FLAC_ZIP%' -UseBasicParsing;" ^
  "  } catch { exit 1 }" ^
  "}"
if errorlevel 1 goto flac_download_failed

:flac_download_ok
if not exist "%FLAC_ZIP%" goto flac_download_failed
goto flac_extract

:flac_download_failed
echo WARNING: flac download failed - Deep FLAC verify will use ffmpeg only ^(no MD5^).
goto flac_done

:flac_extract
mkdir "%FLAC_DIR%" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-flac-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -LiteralPath '%FLAC_ZIP%' -DestinationPath $tmp -Force;" ^
  "$exe = Get-ChildItem -Path $tmp -Recurse -Filter flac.exe | Where-Object { $_.FullName -match 'win64' } | Select-Object -First 1;" ^
  "if (-not $exe) { $exe = Get-ChildItem -Path $tmp -Recurse -Filter flac.exe | Select-Object -First 1 };" ^
  "if (-not $exe) { throw 'flac.exe not found in archive' };" ^
  "$srcDir = $exe.Directory.FullName;" ^
  "Copy-Item -Path (Join-Path $srcDir 'flac.exe') -Destination '%FLAC_DIR%\flac.exe' -Force;" ^
  "Get-ChildItem -Path $srcDir -Filter '*.dll' | Copy-Item -Destination '%FLAC_DIR%\' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: flac extract failed - Deep FLAC verify will use ffmpeg only ^(no MD5^).
    goto flac_done
)
del /Q "%FLAC_ZIP%" 2>nul
if exist "%FLAC_DIR%\flac.exe" echo OK flac -^> %FLAC_DIR%\flac.exe

:flac_done

echo.
echo ========================================
echo   Done
echo ========================================
if "%USE_SITE%"=="1" (
    echo Frozen app: tools checked/installed beside the EXE.
    echo Models: downloaded by the installer, or keep existing models\ folders.
) else (
    echo Source .venv: requirements.txt installed ^(onnxruntime-gpu + audio^).
    echo Demucs GPU needs an NVIDIA driver; AMD/Intel Classify uses CPU.
)
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: dependency install failed. See messages above.
pause
exit /b 1

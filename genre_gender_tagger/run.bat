@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0genre_gender_tagger.py"

if not exist "%PY%" (
    echo ERROR: venv not found.
    echo Run install-deps.bat first.
    echo.
    pause
    exit /b 1
)

"%PY%" "%SCRIPT%" %*
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
    echo Exited with error code %ERR%.
    pause
)
exit /b %ERR%

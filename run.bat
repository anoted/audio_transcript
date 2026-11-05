@echo off
rem Run the transcriber from source using .build-venv (shared with build.bat).
rem The first run creates .build-venv from the active Python. Each run (re)installs a requirements
rem file only if it changed since the last install, so later starts skip pip entirely.
rem   run.bat            install if needed, then start
rem   run.bat --gpu      also install the CUDA libraries for the "Use GPU" box (Whisper only)
setlocal
cd /d "%~dp0"
set "PY=.build-venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating .build-venv ...
    python -m venv .build-venv || goto :error
)
call :sync requirements.txt || goto :error
if /i "%~1"=="--gpu" call :sync requirements-gpu.txt || goto :error

"%PY%" transcriber.py
exit /b %errorlevel%

:sync
rem Install %1 unless .build-venv already holds an identical copy from the last install.
fc /b "%~1" ".build-venv\%~n1.installed" >nul 2>&1 && exit /b 0
echo Installing %~1 ...
"%PY%" -m pip install --disable-pip-version-check -r "%~1" || exit /b 1
copy /y "%~1" ".build-venv\%~n1.installed" >nul
exit /b 0

:error
echo Setup failed.
exit /b 1

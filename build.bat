@echo off
rem Build dist\transcriber.exe with PyInstaller.
rem PyInstaller scans every import it can reach, so the build runs in a dedicated .build-venv
rem (created once from the active Python) - keeps the scan small and fast.
rem The exe never uses the GPU unless "Use GPU" is ticked, and then only if CUDA 12 cuBLAS/cuDNN 9
rem DLLs are on PATH. PyTorch is excluded in case it is installed: optional imports would drag it in.
rem Speech models are downloaded on first use into %%USERPROFILE%%\.cache\huggingface.
setlocal
cd /d "%~dp0"
set "PY=.build-venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating .build-venv ...
    python -m venv .build-venv || goto :error
)
"%PY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt pyinstaller || goto :error

rem PyInstaller finds DLLs through PATH. Conda keeps tcl86t.dll, OpenSSL etc. in Library\bin, which
rem a venv does not put on PATH, so list the base Python first or they get skipped or come from
rem whichever other Python is on PATH.
for /f "delims=" %%B in ('%PY% -c "import sys; print(sys.base_prefix)"') do set "BASE=%%B"
set "PATH=%BASE%;%BASE%\Library\bin;%PATH%"

"%PY%" -m PyInstaller --noconfirm --onefile --windowed ^
    --name transcriber ^
    --icon "%~dp0icon.ico" ^
    --add-data "%~dp0icon.ico;." ^
    --distpath dist --workpath build --specpath build ^
    --collect-data faster_whisper ^
    --collect-binaries ctranslate2 ^
    --collect-data onnx_asr ^
    --copy-metadata onnx-asr ^
    --collect-data soundcard ^
    --exclude-module torch ^
    transcriber.py || goto :error

echo.
echo Built: %~dp0dist\transcriber.exe
exit /b 0

:error
echo Build failed.
exit /b 1

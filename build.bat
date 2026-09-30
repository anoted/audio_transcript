@echo off
rem Build dist\transcriber.exe with PyInstaller.
rem PyInstaller scans every import it can reach, so the build runs in a dedicated .build-venv
rem (created once from the active Python) holding only requirements.txt - keeps the scan small and fast.
rem The exe uses the GPU if CUDA 12 cuBLAS/cuDNN 9 DLLs are on PATH, otherwise the CPU.
rem Whisper models are downloaded on first use into %%USERPROFILE%%\.cache\huggingface.
setlocal
cd /d "%~dp0"
set "PY=.build-venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating .build-venv ...
    python -m venv .build-venv || goto :error
)
"%PY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt pyinstaller || goto :error

"%PY%" -m PyInstaller --noconfirm --onefile --windowed ^
    --name transcriber ^
    --icon "%~dp0icon.ico" ^
    --add-data "%~dp0icon.ico;." ^
    --distpath dist --workpath build --specpath build ^
    --collect-data faster_whisper ^
    --collect-binaries ctranslate2 ^
    --collect-data soundcard ^
    transcriber.py || goto :error

echo.
echo Built: %~dp0dist\transcriber.exe
exit /b 0

:error
echo Build failed.
exit /b 1

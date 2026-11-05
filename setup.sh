#!/usr/bin/env bash
# Create .build-venv (once) and install the dependencies into it. Linux; on Windows use run.bat.
# Usage: ./setup.sh          CPU only (battery friendly)
#        ./setup.sh --gpu    also install the CUDA libraries for the "Use GPU" box (Whisper only)
set -euo pipefail
cd "$(dirname "$0")"
PY=.build-venv/bin/python

if [[ ! -x "$PY" ]]; then
    echo "Creating .build-venv ..."
    python3 -m venv .build-venv
fi

"$PY" -m pip install --disable-pip-version-check -r requirements.txt

if [[ "${1:-}" == "--gpu" ]]; then
    "$PY" -m pip install --disable-pip-version-check -r requirements-gpu.txt
fi

echo "Setup done. Start the app with ./run.sh"

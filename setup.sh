#!/usr/bin/env bash
# Create .build-venv (once) and install the dependencies into it. Linux; on Windows use build.bat.
# Usage: ./setup.sh          CPU, or GPU if the CUDA libraries are already on the system
#        ./setup.sh --gpu    also install the CUDA libraries faster-whisper needs for GPU
set -euo pipefail
cd "$(dirname "$0")"
PY=.build-venv/bin/python

if [[ ! -x "$PY" ]]; then
    echo "Creating .build-venv ..."
    python3 -m venv .build-venv
fi

"$PY" -m pip install --disable-pip-version-check -r requirements.txt

if [[ "${1:-}" == "--gpu" ]]; then
    "$PY" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
fi

echo "Setup done. Start the app with ./run.sh"

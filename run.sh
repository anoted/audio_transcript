#!/usr/bin/env bash
# Run the transcriber from the .build-venv created by ./setup.sh.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .build-venv/bin/python ]]; then
    echo "No .build-venv found - run ./setup.sh first." >&2
    exit 1
fi
exec .build-venv/bin/python transcriber.py "$@"

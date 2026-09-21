#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: nvcc was not found. Install or load the CUDA Toolkit first." >&2
    exit 1
fi

make


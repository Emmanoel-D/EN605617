#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if (( $# != 0 )); then
    echo "Usage: bash build.sh (optional environment variable: CUDA_ARCH)" >&2
    exit 1
fi

rm -f assignment assignment.exe *.o *.obj assignment.lib assignment.exp
compiler="${NVCC:-nvcc}"
if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "Error: nvcc was not found. Install or load the CUDA Toolkit." >&2
    exit 1
fi
flags=(-O2 -std=c++17)
if [[ -n "${CUDA_ARCH:-}" ]]; then
    flags+=("-arch=${CUDA_ARCH}")
fi
target=assignment
if [[ "${OS:-}" == "Windows_NT" ]]; then
    target=assignment.exe
fi
"$compiler" "${flags[@]}" main.cu streams.cu data.cpp -o "$target"

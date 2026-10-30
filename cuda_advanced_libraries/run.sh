#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
binary="./build/audio_denoiser"
if [[ -f ./build/audio_denoiser.exe ]]; then
    binary="./build/audio_denoiser.exe"
fi
if [[ ! -f "$binary" ]]; then
    printf 'Executable missing. Run bash build.sh first.\n' >&2
    exit 1
fi

mkdir -p results
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum src/* build.sh run.sh > results/source_sha256.txt
fi
{
    printf 'Execution started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Command: %s' "$binary"
    if [[ $# -gt 0 ]]; then
        printf ' %q' "$@"
    fi
    printf '\n'
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version --format=csv || true
    fi
    "$binary" "$@"
} 2>&1 | tee results/execution.log

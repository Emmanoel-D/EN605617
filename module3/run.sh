#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x ./assignment.exe ]]; then
    ./build.sh
fi

: > run_output.txt

echo "./assignment.exe 512 256" | tee -a run_output.txt
./assignment.exe 512 256 | tee -a run_output.txt

echo | tee -a run_output.txt
echo "./assignment.exe 1024 128" | tee -a run_output.txt
./assignment.exe 1024 128 | tee -a run_output.txt

echo | tee -a run_output.txt
echo "./assignment.exe 2048 64" | tee -a run_output.txt
./assignment.exe 2048 64 | tee -a run_output.txt

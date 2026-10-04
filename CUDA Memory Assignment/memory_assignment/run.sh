#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x ./assignment.exe ]]; then
    ./build.sh
fi

: > run_output.txt

run_case() {
    local blocks="$1"
    local threads="$2"

    echo "$ ./assignment.exe $blocks $threads" | tee -a run_output.txt
    ./assignment.exe "$blocks" "$threads" | tee -a run_output.txt
    echo | tee -a run_output.txt
}

run_case 8 64
run_case 32 128
run_case 128 256

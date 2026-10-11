#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x ./assignment ]]; then
    executable=./assignment
elif [[ -x ./assignment.exe ]]; then
    executable=./assignment.exe
else
    echo "Error: no executable. Run bash build.sh first." >&2
    exit 1
fi

if (( $# > 0 )); then
    exec "$executable" "$@"
fi

# Each case executes the kernel on two arrays in separate streams.
for elements in 50000 250000; do
    for launch in "8 64" "32 128" "128 256"; do
        read -r blocks threads <<< "$launch"
        echo "Running ${elements} elements per array, ${blocks} blocks, ${threads} threads per block"
        "$executable" "$blocks" "$threads" "$elements" 10
        echo
    done
done

#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
binary=build/motor_monitor
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) binary+=.exe ;;
esac
if [[ ! -f "$binary" ]]; then
    printf 'Build first: bash opencl_motor_monitor/build.sh\n' >&2
    exit 1
fi
mkdir -p results
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum src/* kernels/* third_party/CL/* build.sh run.sh \
        scripts/*.sh tests/*.py tests/*.sh > results/source_sha256.txt
    if [[ -f ../assignment_config.yaml ]]; then
        sha256sum ../assignment_config.yaml >> results/source_sha256.txt
    fi
fi
{
    printf 'Execution started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Command: '
    printf '%q ' "$binary" "$@"
    printf '\n'
    if [[ -n "${POCL_MAX_PTHREAD_COUNT:-}" ]]; then
        printf 'PoCL thread limit: %s\n' "$POCL_MAX_PTHREAD_COUNT"
    fi
    "$binary" "$@"
    printf 'Execution finished successfully.\n'
} 2>&1 | tee results/execution.log

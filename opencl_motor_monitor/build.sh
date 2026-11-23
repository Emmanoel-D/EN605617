#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ $# -ne 0 ]]; then
    printf 'build.sh takes no arguments. Set CXX to select a compiler.\n' >&2
    exit 2
fi
mkdir -p build
rm -f build/motor_monitor build/motor_monitor.exe

build_project() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) bash scripts/build_windows.sh ;;
        *)
            compiler="${CXX:-g++}"
            "$compiler" --version
            library_flags=(-l:libOpenCL.so.1)
            if [[ "$(uname -s)" == Darwin ]]; then
                library_flags=(-framework OpenCL)
            elif [[ -n "${OPENCL_LIB_DIR:-}" ]]; then
                library_flags=(-L"$OPENCL_LIB_DIR" -lOpenCL)
            fi
            "$compiler" -std=c++17 -O2 -Wall -Wextra -Wpedantic \
                -Ithird_party src/*.cpp "${library_flags[@]}" \
                -o build/motor_monitor
            ;;
    esac
}

{
    printf 'Build started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    build_project
    printf 'Build succeeded.\n'
} 2>&1 | tee build/build.log

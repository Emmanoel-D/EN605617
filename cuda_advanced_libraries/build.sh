#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ $# -ne 0 ]]; then
    printf 'build.sh takes no arguments; use CUDA_ARCH or NVCC.\n' >&2
    exit 2
fi

compiler="${NVCC:-nvcc}"
if ! command -v "$compiler" >/dev/null 2>&1; then
    if [[ -n "${CUDA_PATH:-}" ]] && command -v cygpath >/dev/null 2>&1; then
        cuda_path="$(cygpath -u "$CUDA_PATH")"
        compiler="$cuda_path/bin/nvcc.exe"
    fi
fi
if ! command -v "$compiler" >/dev/null 2>&1; then
    printf 'nvcc was not found. Install CUDA and add its bin to PATH.\n' >&2
    exit 1
fi

suffix=""
host_flags=()
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        suffix=".exe"
        host_flags=(-Xcompiler=/Zc:preprocessor)
        ;;
    *) host_flags=(-Xcompiler=-Wall,-Wextra) ;;
esac
architecture="${CUDA_ARCH:-75}"
if [[ ! "$architecture" =~ ^[0-9]+$ ]]; then
    printf 'CUDA_ARCH must contain digits, for example 75 or 120.\n' >&2
    exit 2
fi

mkdir -p build
rm -f build/audio_denoiser build/audio_denoiser.exe build/*.o build/*.obj
{
    printf 'Build started: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    "$compiler" --version
    MSYS_NO_PATHCONV=1 "$compiler" -std=c++17 -O2 --expt-relaxed-constexpr \
        "-gencode=arch=compute_$architecture,code=compute_$architecture" \
        "${host_flags[@]}" src/app.cpp src/signal.cpp src/reference.cpp \
        src/wav.cpp src/gpu.cu -lcufft -o "build/audio_denoiser$suffix"
    printf 'Build succeeded: build/audio_denoiser%s\n' "$suffix"
} 2>&1 | tee build/build.log

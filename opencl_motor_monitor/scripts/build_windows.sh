#!/usr/bin/env bash
set -euo pipefail

# Called by build.sh after changing to the assignment directory.
locate_msvc() {
    if command -v cl.exe >/dev/null 2>&1; then
        vcvars=""
        return
    fi
    vswhere='/c/Program Files (x86)/Microsoft Visual Studio/Installer/'
    vswhere+='vswhere.exe'
    if [[ ! -f "$vswhere" ]]; then
        printf 'Install Visual Studio C++ tools, or use its x64 prompt.\n' >&2
        exit 1
    fi
    install_path="$("$vswhere" -latest -products '*' -requires \
        Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationPath | tr -d '\r')"
    if [[ -z "$install_path" ]]; then
        printf 'Visual Studio x64 C++ build tools were not found.\n' >&2
        exit 1
    fi
    vcvars="$install_path\\VC\\Auxiliary\\Build\\vcvars64.bat"
}

locate_library() {
    library_dir="${OPENCL_LIB_DIR:-}"
    if [[ -z "$library_dir" && -n "${CUDA_PATH:-}" ]]; then
        library_dir="$(cygpath -u "$CUDA_PATH")/lib/x64"
    fi
    if [[ -z "$library_dir" ]]; then
        printf 'Set CUDA_PATH or OPENCL_LIB_DIR for OpenCL.lib.\n' >&2
        exit 1
    fi
    library_dir="$(cygpath -u "$library_dir")"
    if [[ ! -f "$library_dir/OpenCL.lib" ]]; then
        printf 'OpenCL.lib was not found in %s\n' "$library_dir" >&2
        exit 1
    fi
    library_native="$(cygpath -w "$library_dir")"
}

write_command_file() {
    printf '@echo off\r\nsetlocal\r\n'
    if [[ -n "$vcvars" ]]; then
        printf 'call "%s" >nul\r\n' "$vcvars"
        printf 'if errorlevel 1 exit /b 1\r\n'
    fi
    printf 'cd /d "%s"\r\n' "$(cygpath -w "$PWD")"
    printf 'cl.exe /nologo /std:c++17 /EHsc /O2 /W4 '
    printf '/Ithird_party /Fobuild\\ /Febuild\\motor_monitor.exe '
    printf 'src\\*.cpp /link /LIBPATH:"%s" OpenCL.lib\r\n' "$library_native"
    printf 'exit /b %%errorlevel%%\r\n'
}

locate_msvc
locate_library
write_command_file > build/windows_build.cmd
# Prevent Git Bash from rewriting cmd switches or native quoted paths.
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
    cmd.exe /d /c "call \"$(cygpath -w "$PWD/build/windows_build.cmd")\""

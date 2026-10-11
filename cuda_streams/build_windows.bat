@echo off
setlocal
cd /d "%~dp0"

if exist assignment.exe del /q assignment.exe
if exist assignment del /q assignment
if exist *.obj del /q *.obj
if exist assignment.lib del /q assignment.lib
if exist assignment.exp del /q assignment.exp
if exist assignment.exe (
    echo Error: could not remove the previous executable. Close it and try again.
    exit /b 1
)

where cl.exe >nul 2>nul
if not errorlevel 1 goto compiler_ready
if not exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo Error: open a Visual Studio x64 developer command prompt first.
    exit /b 1
)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

:compiler_ready
where nvcc >nul 2>nul
if errorlevel 1 (
    echo Error: nvcc was not found. Install or load the CUDA Toolkit.
    exit /b 1
)
nvcc -O2 -std=c++17 -arch=native main.cu streams.cu data.cpp -o assignment.exe
if errorlevel 1 exit /b 1
echo Build completed.
exit /b 0

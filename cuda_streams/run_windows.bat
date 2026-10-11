@echo off
setlocal
cd /d "%~dp0"

if not exist assignment.exe (
    echo Error: no executable. Run build_windows.bat first.
    exit /b 1
)
if not "%~1"=="" (
    assignment.exe %*
    exit /b
)

for %%N in (50000 250000) do (
    for %%L in (a b c) do (
        call :run_case %%N %%L
        if errorlevel 1 exit /b 1
    )
)
exit /b 0

:run_case
set blocks=8
set threads=64
if "%~2"=="b" set blocks=32
if "%~2"=="b" set threads=128
if "%~2"=="c" set blocks=128
if "%~2"=="c" set threads=256
echo Running %~1 elements per array, %blocks% blocks, %threads% threads per block
assignment.exe %blocks% %threads% %~1 10
exit /b %errorlevel%

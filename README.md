assignment_config.yaml - root-level configuration pointing to the required build and run scripts
assignment.cu - All CPU and CUDA source code.
Makefile - creates assignment.exe when make is run without arguments.
build.sh, run.sh - Linux scripts
build_windows.bat, run_windows.bat - Windows equivalent
thoughts.txt - Required text file. The run script replaces it with observations based on the last run.


build the executable with build_windows.bat

Test with assignment.exe 512 256

run_windows.bat runs it with all required configuration and creates the chart and report.
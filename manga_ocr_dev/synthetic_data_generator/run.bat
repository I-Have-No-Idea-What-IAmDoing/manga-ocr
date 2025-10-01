@echo off
for /L %%i in (0,20,99) do (
	echo Running iteration %%i...
    uv run run_generate.py --package %%i
)
echo All runs completed.
pause
@echo off
SETLOCAL
SET PYTHONPATH="J:/Applications/manga-ocr/;"
for /L %%i in (21,1,99) do (
	echo Running iteration %%i...
    uv run run_generate.py --package %%i
)
ENDLOCAL
echo All runs completed.
pause
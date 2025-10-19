@echo off
SETLOCAL
SET PYTHONPATH="J:/Applications/manga-ocr/;"
for /L %%i in (64,1,99) do (
	echo Running iteration %%i...
    uv run manga_ocr_dev\synthetic_data_generator\run_generate.py --renderer pictex --package %%i --n_random 12500 --max_workers 15 --min_font_size 10 --max_font_size 95
	echo !errorlevel!
)
ENDLOCAL
echo All runs completed.
pause
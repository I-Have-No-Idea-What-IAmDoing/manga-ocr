"""Tests for the main command-line entry point.

This module contains tests to ensure that the `manga_ocr` command-line
interface is correctly set up. It verifies that executing the `__main__` module
properly invokes the `fire` library with the main `run` function.
"""

import runpy
from unittest.mock import patch

from manga_ocr.__main__ import main
from manga_ocr.run import run


@patch("fire.Fire")
def test_main(mock_fire):
    """Tests that the main function correctly calls `fire.Fire`.

    This test ensures that when the `main()` function from `manga_ocr.__main__`
    is called, it passes the `run` function to `fire.Fire`, which is the
    expected behavior for setting up the command-line interface.
    """
    main()
    mock_fire.assert_called_once_with(run)


@patch("fire.Fire")
def test_main_entry_point(mock_fire):
    """Tests that running the package as a script invokes the entry point.

    This test uses `runpy` to simulate executing the `manga_ocr` package as a
    script (e.g., `python -m manga_ocr`). It verifies that this action correctly
    triggers the `main` function and, consequently, `fire.Fire` with the `run`
    function.
    """
    runpy.run_module("manga_ocr.__main__", run_name="__main__")
    mock_fire.assert_called_with(run)
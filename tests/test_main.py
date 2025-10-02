import runpy
from unittest.mock import patch

from manga_ocr.__main__ import main
from manga_ocr.run import run


@patch("fire.Fire")
def test_main(mock_fire):
    """
    Tests that the main function calls fire.Fire with the run function.
    """
    main()
    mock_fire.assert_called_once_with(run)


@patch("fire.Fire")
def test_main_entry_point(mock_fire):
    """
    Tests that the script's entry point calls the main function.
    """
    runpy.run_module("manga_ocr.__main__", run_name="__main__")
    mock_fire.assert_called_with(run)
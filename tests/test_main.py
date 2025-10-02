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
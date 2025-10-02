import json
from pathlib import Path

import pytest

from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process

TEST_DATA_ROOT = Path(__file__).parent / "data"


def test_ocr():
    """
    Tests the MangaOcr model by comparing its output against a set of expected results.

    This test loads the pre-generated expected results from a JSON file, runs the
    OCR model on the corresponding test images, and asserts that the output
    matches the expected text.
    """
    mocr = MangaOcr()

    expected_results = json.loads((TEST_DATA_ROOT / "expected_results.json").read_text(encoding="utf-8"))

    for item in expected_results:
        result = mocr(TEST_DATA_ROOT / "images" / item["filename"])
        assert result == item["result"]


def test_ocr_invalid_input():
    """
    Tests that MangaOcr raises ValueError for invalid input types.
    """
    mocr = MangaOcr()
    with pytest.raises(ValueError, match="img_or_path must be a path or PIL.Image"):
        mocr(123)


def test_post_process():
    """
    Tests the text post-processing function.

    This test verifies that the `post_process` function correctly handles
    specific text replacements, such as converting standard ellipsis and dots
    to full-width Japanese characters.
    """
    assert post_process("…") == "．．．"
    assert post_process("・・") == "．．"
    assert post_process("a b c") == "ａｂｃ"
    assert post_process("a　b　c") == "ａｂｃ"
    assert post_process("a.b") == "ａ．ｂ"
    assert post_process("a..b") == "ａ．．ｂ"
    assert post_process("a・b") == "ａ・ｂ"
    assert post_process("a・・b") == "ａ．．ｂ"
    assert post_process("a.・b") == "ａ．．ｂ"

import json
from pathlib import Path
from unittest.mock import patch

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


@patch("pathlib.Path.is_file", return_value=False)
def test_manga_ocr_missing_example_image(mock_is_file):
    """
    Tests that MangaOcr raises FileNotFoundError if the example image is missing.
    """
    with pytest.raises(FileNotFoundError, match="Missing example image"):
        MangaOcr()


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcrModel.cuda")
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_cuda_available(mock_logger_info, mock_call, mock_model_cuda, mock_mps, mock_cuda):
    """
    Tests that MangaOcr uses CUDA when available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using CUDA")
    mock_model_cuda.assert_called_once()


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
@patch("manga_ocr.ocr.MangaOcrModel.to")
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_mps_available(mock_logger_info, mock_call, mock_model_to, mock_mps, mock_cuda):
    """
    Tests that MangaOcr uses MPS when available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using MPS")
    mock_model_to.assert_called_once_with("mps")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_cpu_fallback(mock_logger_info, mock_call, mock_mps, mock_cuda):
    """
    Tests that MangaOcr falls back to CPU when no GPU is available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using CPU")


@patch("torch.cuda.is_available", return_value=True)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_force_cpu(mock_logger_info, mock_call, mock_cuda):
    """
    Tests that MangaOcr uses CPU when force_cpu is True, even if a GPU is available.
    """
    MangaOcr(force_cpu=True)
    mock_logger_info.assert_any_call("Using CPU")

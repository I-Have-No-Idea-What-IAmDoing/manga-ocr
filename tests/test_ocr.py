"""Tests for the core OCR functionality.

This module contains tests for the `MangaOcr` class and its associated
functions. It includes integration tests that verify the OCR performance
against a baseline, as well as unit tests for text post-processing, device
selection logic, and input validation.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process

TEST_DATA_ROOT = Path(__file__).parent / "data"


def test_ocr_integration():
    """
    Performs an integration test of the MangaOcr model.

    This test compares the model's output against a set of pre-generated,
    expected results from a JSON file. It runs the OCR on each test image
    and asserts that the output matches the expected text, ensuring that
    model changes do not cause performance regressions.
    """
    mocr = MangaOcr()

    expected_results = json.loads(
        (Path(TEST_DATA_ROOT) / "expected_results.json").read_text(encoding="utf-8")
    )

    for item in expected_results:
        result = mocr(Path(TEST_DATA_ROOT) / "images" / item["filename"])
        assert result == item["result"]


def test_post_process():
    """
    Tests the text post-processing function for correct normalization.

    This test verifies that the `post_process` function correctly handles
    various text cleaning and normalization tasks, such as removing whitespace,
    standardizing punctuation, and converting half-width characters to
    full-width.
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


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcrModel.cuda")
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_uses_cuda_when_available(mock_logger_info, mock_call, mock_model_cuda, mock_mps, mock_cuda):
    """
    Tests that MangaOcr correctly selects the CUDA device when it is available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using CUDA")
    mock_model_cuda.assert_called_once()


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
@patch("manga_ocr.ocr.MangaOcrModel.to")
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_uses_mps_when_available(mock_logger_info, mock_call, mock_model_to, mock_mps, mock_cuda):
    """
    Tests that MangaOcr correctly selects the MPS device when it is available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using MPS")
    mock_model_to.assert_called_once_with("mps")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_falls_back_to_cpu(mock_logger_info, mock_call, mock_mps, mock_cuda):
    """
    Tests that MangaOcr correctly falls back to CPU when no GPU is available.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using CPU")


@patch("torch.cuda.is_available", return_value=True)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_force_cpu_option(mock_logger_info, mock_call, mock_cuda):
    """
    Tests that MangaOcr uses CPU when `force_cpu=True`, even if a GPU is available.
    """
    MangaOcr(force_cpu=True)
    mock_logger_info.assert_any_call("Using CPU")


def test_ocr_raises_value_error_for_invalid_input_type():
    """
    Tests that MangaOcr raises a ValueError for unsupported input types.

    This test ensures that the `__call__` method validates its input and
    raises a `ValueError` if the provided argument is not a file path or a
    PIL Image object.
    """
    mocr = MangaOcr()
    with pytest.raises(ValueError, match="img_or_path must be a path or PIL.Image"):
        mocr(123)
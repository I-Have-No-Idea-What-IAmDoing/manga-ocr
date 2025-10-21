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
    """Performs an integration test of the `MangaOcr` model.

    This test provides a baseline check for the OCR performance. It loads a
    set of test images and their expected text transcriptions from a JSON
    file. The test then runs the OCR on each image and asserts that the output
    matches the expected result.

    This is crucial for ensuring that changes to the model, its weights, or
    the preprocessing logic do not cause unintended regressions in performance
    on a known set of examples.
    """
    mocr = MangaOcr()

    expected_results = json.loads(
        (Path(TEST_DATA_ROOT) / "expected_results.json").read_text(encoding="utf-8")
    )

    for item in expected_results:
        result = mocr(Path(TEST_DATA_ROOT) / "images" / item["filename"])
        assert result == item["result"]


def test_post_process():
    """Tests the text post-processing function for correct normalization.

    This test covers various text cleaning and normalization scenarios handled
    by the `post_process` function. It asserts that the function correctly:
    - Removes whitespace.
    - Standardizes sequences of dots and other punctuation.
    - Converts half-width (hankaku) characters to full-width (zenkaku).

    This ensures that the raw output from the OCR model is consistently
    cleaned and normalized before being returned to the user.
    """
    assert post_process("…") == "．．．"
    assert post_process("・・") == "．．．"
    assert post_process("a b c") == "ａｂｃ"
    assert post_process("a　b　c") == "ａｂｃ"
    assert post_process("a.b") == "ａ．ｂ"
    assert post_process("a..b") == "ａ．．．ｂ"
    assert post_process("a・b") == "ａ・ｂ"
    assert post_process("a・・b") == "ａ．．．ｂ"
    assert post_process("a.・b") == "ａ．．．ｂ"
    assert post_process("a...b") == "ａ．．．ｂ"
    assert post_process("a....b") == "ａ．．．ｂ"


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcrModel.cuda")
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_uses_cuda_when_available(mock_logger_info, mock_call, mock_model_cuda, mock_mps, mock_cuda):
    """Tests that MangaOcr correctly selects the CUDA device when available.

    This test simulates an environment where a CUDA-enabled GPU is present.
    It verifies that the `MangaOcr` class correctly detects the CUDA device,
    logs that it is being used, and calls the `.cuda()` method to move the
    model to the GPU.

    Args:
        mock_logger_info: Mock for the logger to check for correct messages.
        mock_call: Mock for the `__call__` method to prevent execution.
        mock_model_cuda: Mock for the model's `.cuda()` method.
        mock_mps: Mock for `torch.backends.mps.is_available`.
        mock_cuda: Mock for `torch.cuda.is_available`.
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
    """Tests that MangaOcr correctly selects the MPS device when available.

    This test simulates an environment where a Metal Performance Shaders (MPS)
    device is available (e.g., on an Apple Silicon Mac), but CUDA is not. It
    verifies that the `MangaOcr` class correctly identifies the MPS backend,
    logs its use, and calls the `.to('mps')` method to move the model to the
    device.

    Args:
        mock_logger_info: Mock for the logger to check for correct messages.
        mock_call: Mock for the `__call__` method to prevent execution.
        mock_model_to: Mock for the model's `.to()` method.
        mock_mps: Mock for `torch.backends.mps.is_available`.
        mock_cuda: Mock for `torch.cuda.is_available`.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using MPS")
    mock_model_to.assert_called_once_with("mps")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_falls_back_to_cpu(mock_logger_info, mock_call, mock_mps, mock_cuda):
    """Tests that MangaOcr correctly falls back to CPU when no GPU is available.

    This test simulates an environment where neither CUDA nor MPS is available.
    It verifies that the `MangaOcr` class correctly defaults to using the CPU
    for computation and logs the appropriate message.

    Args:
        mock_logger_info: Mock for the logger to check for correct messages.
        mock_call: Mock for the `__call__` method to prevent execution.
        mock_mps: Mock for `torch.backends.mps.is_available`.
        mock_cuda: Mock for `torch.cuda.is_available`.
    """
    MangaOcr()
    mock_logger_info.assert_any_call("Using CPU")


@patch("torch.cuda.is_available", return_value=True)
@patch("manga_ocr.ocr.MangaOcr.__call__")
@patch("loguru.logger.info")
def test_manga_ocr_force_cpu_option(mock_logger_info, mock_call, mock_cuda):
    """Tests that MangaOcr uses CPU when `force_cpu=True`.

    This test verifies that the `force_cpu` flag in the `MangaOcr`
    constructor correctly overrides the automatic device selection. Even when
    a CUDA device is available, setting `force_cpu=True` should ensure that
    the model runs on the CPU.

    Args:
        mock_logger_info: Mock for the logger to check for correct messages.
        mock_call: Mock for the `__call__` method to prevent execution.
        mock_cuda: Mock for `torch.cuda.is_available`.
    """
    MangaOcr(force_cpu=True)
    mock_logger_info.assert_any_call("Using CPU")


def test_ocr_raises_value_error_for_invalid_input_type():
    """Tests that MangaOcr raises a ValueError for unsupported input types.

    This test ensures that the `__call__` method validates its input and
    raises a `ValueError` if the provided argument is not a file path or a
    PIL Image object. This is important for providing clear feedback to the
    user when they provide an invalid input type.
    """
    mocr = MangaOcr()
    with pytest.raises(ValueError, match="img_or_path must be a path or PIL.Image"):
        mocr(123)
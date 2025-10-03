"""Tests for the main synthetic data generation script.

This module contains tests for the `run_generate.py` script, which orchestrates
the entire synthetic data generation process. The tests cover the parallel
worker function and the main `run` function, ensuring that data is processed
correctly and that exceptions are handled properly.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from manga_ocr_dev.synthetic_data_generator.run_generate import worker_fn, run
from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
def test_worker_fn(mock_imwrite):
    """
    Tests the `worker_fn` for correct data processing.

    This test verifies that the `worker_fn`, which is executed in parallel,
    correctly processes a single data sample. It ensures that the function
    calls the data generator, saves the resulting image, and returns the
    correct metadata for the generated sample.
    """
    mock_generator = MagicMock()
    mock_generator.process.return_value = (
        MagicMock(),
        'test_text',
            {'font_path': 'dummy.ttf', 'vertical': True}
    )

    # Set the global OUT_DIR for the test
    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        result = worker_fn(args, mock_generator)

    mock_generator.process.assert_called_once_with('text')
    mock_imwrite.assert_called_once()
    assert result == ('source', 'id_001', 'test_text', True, 'dummy.ttf')

@patch('pathlib.Path.exists', return_value=True)
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.DataFrame.to_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.Path.mkdir')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.SyntheticDataGenerator')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.os.environ.get')
def test_run(mock_os_get, mock_generator, mock_renderer, mock_thread_map, mock_mkdir, mock_to_csv, mock_read_csv, mock_exists):
    """
    Tests the main `run` function for orchestrating data generation.

    This test provides an end-to-end check of the `run` function, verifying
    that it correctly reads input data, sets up the necessary directories,
    invokes the parallel processing map, and saves the final metadata CSV
    file. It uses extensive mocking to isolate the function's logic.
    """
    # Mock input data
    mock_read_csv.return_value = pd.DataFrame({
        'source': ['test_source'],
        'id': ['test_id'],
        'line': ['test_line']
    })

    # Mock worker function return value
    mock_thread_map.return_value = [('test_source', 'test_id', 'test_line', True, 'dummy.ttf')]

    # Run the function
    run(package=0, n_random=0, n_limit=1, max_workers=1)

    mock_read_csv.assert_called_once()
    mock_thread_map.assert_called_once()
    mock_to_csv.assert_called_once()

    # Verify that the output directory is created
    assert (DATA_SYNTHETIC_ROOT / "img" / "0000").mkdir.called
    assert (DATA_SYNTHETIC_ROOT / "meta").mkdir.called

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
@patch('builtins.print')
def test_worker_fn_exception_handling(mock_print, mock_imwrite):
    """
    Tests that the `worker_fn` correctly handles exceptions.

    This test ensures that if an exception occurs within the `worker_fn`
    (e.g., during image generation), the exception is caught, the traceback
    is printed, and the exception is re-raised to be handled by the main
    process.
    """
    mock_generator = MagicMock()
    mock_generator.process.side_effect = Exception("Test exception")

    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        with pytest.raises(Exception, match="Test exception"):
            worker_fn(args, mock_generator)

    mock_print.assert_called()
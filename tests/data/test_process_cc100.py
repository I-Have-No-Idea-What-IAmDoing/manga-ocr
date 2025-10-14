"""Tests for the CC-100 corpus processing script.

This module contains unit tests for the `process_cc100.py` script, which is
responsible for reading the raw CC-100 text corpus, filtering it, and packaging
it into smaller CSV files for use in synthetic data generation.
"""

import pandas as pd
from unittest.mock import patch, mock_open

from manga_ocr_dev.data.process_cc100 import export_lines
from manga_ocr_dev.env import ASSETS_PATH


@patch('pandas.DataFrame')
@patch('builtins.open', new_callable=mock_open, read_data='line1\nline2\nab\nline4')
def test_export_lines(mock_file, MockDataFrame):
    """Tests the `export_lines` function for correct filtering and packaging.

    This test verifies that `export_lines` correctly reads lines from a mock
    text file, filters out lines that are too short, and packages the valid
    lines into a pandas DataFrame, which is then saved to a CSV file.

    Args:
        mock_file: Mock for the `open` function to provide mock file content.
        MockDataFrame: Mock for the `pandas.DataFrame` class.
    """
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    (ASSETS_PATH / "lines").mkdir(exist_ok=True)
    mock_df_instance = MockDataFrame.return_value

    export_lines(num_lines_in_each_package=3, num_packages=1)

    assert MockDataFrame.called
    call_args, _ = MockDataFrame.call_args
    data_list = call_args[0]

    assert len(data_list) == 3
    lines = [d['line'] for d in data_list]
    assert 'line1' in lines
    assert 'line2' in lines
    assert 'line4' in lines
    assert 'ab' not in lines

    mock_df_instance.to_csv.assert_called_once()
    _, kwargs = mock_df_instance.to_csv.call_args
    assert not kwargs['index']


@patch('pandas.DataFrame')
@patch('builtins.open', new_callable=mock_open, read_data='line1\nline2\nline3\nline4\nline5\nline6')
def test_export_lines_multiple_packages(mock_file, MockDataFrame):
    """Tests that `export_lines` can create multiple data packages correctly.

    This test ensures that when configured to create multiple packages, the
    `export_lines` function correctly iterates through the input file, creates
    the specified number of DataFrames, and saves each one as a separate CSV
    file.

    Args:
        mock_file: Mock for the `open` function.
        MockDataFrame: Mock for the `pandas.DataFrame` class.
    """
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)
    (ASSETS_PATH / "lines").mkdir(exist_ok=True)
    mock_df_instance = MockDataFrame.return_value

    export_lines(num_lines_in_each_package=2, num_packages=3)

    assert MockDataFrame.call_count == 3
    assert mock_df_instance.to_csv.call_count == 3

    first_call_args, _ = MockDataFrame.call_args_list[0]
    assert len(first_call_args[0]) == 2
    second_call_args, _ = MockDataFrame.call_args_list[1]
    assert len(second_call_args[0]) == 2
    third_call_args, _ = MockDataFrame.call_args_list[2]
    assert len(third_call_args[0]) == 2
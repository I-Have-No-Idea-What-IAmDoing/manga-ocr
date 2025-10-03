import pandas as pd
from unittest.mock import patch, mock_open

from manga_ocr_dev.data.process_cc100 import export_lines
from manga_ocr_dev.env import ASSETS_PATH


@patch('pandas.DataFrame')
@patch('builtins.open', new_callable=mock_open, read_data='line1\nline2\nab\nline4')
def test_export_lines(mock_file, MockDataFrame):
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
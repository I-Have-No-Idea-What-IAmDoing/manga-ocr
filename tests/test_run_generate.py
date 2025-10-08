import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

# The script to be tested
from manga_ocr_dev.synthetic_data_generator.run_generate import worker_fn, run

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
def test_worker_fn(mock_imwrite):
    """
    Tests the `worker_fn` for correct data processing.
    """
    mock_generator = MagicMock()
    mock_generator.process.return_value = (MagicMock(), 'test_text', {'font_path': 'dummy.ttf', 'vertical': True})
    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        result = worker_fn(args, mock_generator, renderer_type='html')
    mock_generator.process.assert_called_once_with('text')
    mock_imwrite.assert_called_once()
    assert result == ('source', 'id_001', 'test_text', True, 'dummy.ttf')

@patch('pathlib.Path.exists', return_value=True)
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.DataFrame.to_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.Path.mkdir')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.SyntheticDataGenerator')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.os.environ.get')
def test_run(mock_os_get, mock_generator_class, mock_renderer_class, mock_thread_map, mock_mkdir, mock_to_csv, mock_read_csv, mock_exists, tmp_path):
    """
    Tests the main `run` function for orchestrating data generation.
    """
    # Setup temporary directories and files
    background_dir = tmp_path / "backgrounds"
    # Use os.makedirs because pathlib.Path.mkdir is patched
    os.makedirs(background_dir, exist_ok=True)
    (background_dir / "dummy_bg.png").touch()

    # Mock pd.read_csv to handle multiple calls
    mock_read_csv.side_effect = [
        pd.DataFrame({'source': ['test_source'], 'id': ['test_id'], 'line': ['test_line']}),
        pd.DataFrame({'char': ['a', 'b', 'c']}),
        pd.DataFrame({'len': [10], 'p': [1.0]}),
        pd.DataFrame({'font_path': ['dummy.ttf'], 'supported_chars': ['abc'], 'label': ['regular'], 'num_chars': [3]})
    ]
    mock_thread_map.return_value = [('test_source', 'test_id', 'test_line', True, 'dummy.ttf')]

    # Patch the BACKGROUND_DIR to point to our temp dir
    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.BACKGROUND_DIR', background_dir):
        run(renderer='html', package=0, n_random=0, n_limit=1, max_workers=1)

    assert mock_read_csv.call_count == 4
    mock_thread_map.assert_called_once()
    mock_to_csv.assert_called_once()
    mock_mkdir.assert_called()

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
@patch('builtins.print')
def test_worker_fn_exception_handling(mock_print, mock_imwrite):
    """
    Tests that the `worker_fn` correctly handles exceptions.
    """
    mock_generator = MagicMock()
    mock_generator.process.side_effect = Exception("Test exception")
    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        with pytest.raises(Exception, match="Test exception"):
            worker_fn(args, mock_generator, renderer_type='html')
    mock_print.assert_called()
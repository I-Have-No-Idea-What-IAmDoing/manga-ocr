import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from manga_ocr_dev.synthetic_data_generator.run_generate import f, run
from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
def test_f(mock_imwrite):
    """
    Tests the f worker function.
    """
    mock_generator = MagicMock()
    mock_generator.process.return_value = (
        MagicMock(),
        'test_text',
        {'font_path': str(FONTS_ROOT / 'dummy.ttf'), 'vertical': True}
    )

    # Set the global OUT_DIR for the test
    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        result = f(args, mock_generator)

    mock_generator.process.assert_called_once_with('text')
    mock_imwrite.assert_called_once()
    assert result == ('source', 'id_001', 'test_text', True, 'dummy.ttf')

@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.DataFrame.to_csv')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.Path.mkdir')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.SyntheticDataGenerator')
@patch('manga_ocr_dev.synthetic_data_generator.run_generate.os.environ.get')
def test_run(mock_os_get, mock_generator, mock_renderer, mock_thread_map, mock_mkdir, mock_to_csv, mock_read_csv):
    """
    Tests the run function.
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
def test_f_exception(mock_print, mock_imwrite):
    """
    Tests that the f worker function catches and prints exceptions.
    """
    mock_generator = MagicMock()
    mock_generator.process.side_effect = Exception("Test exception")

    with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
        args = (0, 'source', 'id_001', 'text')
        with pytest.raises(Exception, match="Test exception"):
            f(args, mock_generator)

    mock_print.assert_called()
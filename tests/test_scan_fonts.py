import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import numpy as np

# Since scan_fonts.py loads vocab at import time, we need to patch it here
with patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.pd.read_csv') as mock_read_csv:
    mock_read_csv.return_value = pd.DataFrame({'char': ['a', 'b', 'c']})
    from manga_ocr_dev.synthetic_data_generator.scan_fonts import has_glyph, process, main

def test_has_glyph():
    mock_font = MagicMock()
    mock_table = MagicMock()
    mock_table.cmap.keys.return_value = [ord('a')]
    mock_font["cmap"].tables = [mock_table]

    assert has_glyph(mock_font, 'a')
    assert not has_glyph(mock_font, 'b')

def test_has_glyph_exception():
    mock_font = MagicMock()
    mock_table = MagicMock()
    mock_table.cmap.keys.side_effect = Exception("Test exception")
    mock_font["cmap"].tables = [mock_table]
    assert not has_glyph(mock_font, 'a')

@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.TTFont')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.ImageFont.truetype')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.PIL.Image.new')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.ImageDraw.Draw')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.has_glyph')
def test_process(mock_has_glyph, mock_draw, mock_new, mock_truetype, mock_ttfont, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'char': ['a', 'b', 'c']})
    mock_has_glyph.side_effect = lambda font, char: char in ['a', 'b']

    # Mock the image and draw objects
    mock_image = MagicMock()
    mock_draw_instance = MagicMock()
    mock_new.return_value = mock_image
    mock_draw.return_value = mock_draw_instance

    def draw_side_effect(pos, char, fill, font):
        # Simulate 'b' being a blank glyph
        if char == 'a':
            # Non-blank image
            mock_image.__array__ = lambda copy=True: np.zeros((40, 40))
        else:
            # Blank image
            mock_image.__array__ = lambda copy=True: np.full((40, 40), 255)

    mock_draw_instance.text.side_effect = draw_side_effect

    result = process('dummy_font.ttf')
    assert result == 'a'

@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.TTFont', side_effect=Exception('Test Error'))
@patch('builtins.print')
def test_process_handles_exception(mock_print, mock_ttfont, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'char': ['a', 'b', 'c']})
    result = process('dummy_font.ttf')
    assert result == ''
    mock_print.assert_called_once()

@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.pd.DataFrame')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.process_map')
@patch('pathlib.Path.glob')
@patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.FONTS_ROOT', new=Path('/dummy/fonts'))
def test_main(mock_glob, mock_process_map, mock_dataframe_class):
    mock_glob.return_value = [Path('/dummy/fonts/font1.ttf')]
    mock_process_map.return_value = ['abc']
    mock_df_instance = mock_dataframe_class.return_value

    with patch('manga_ocr_dev.synthetic_data_generator.scan_fonts.ASSETS_PATH', new=Path('/dummy/assets')):
        main()

    mock_process_map.assert_called_once()
    mock_dataframe_class.assert_called_with({"font_path": ['font1.ttf'], "supported_chars": ['abc']})
    assert mock_df_instance.__setitem__.call_count == 2
    mock_df_instance.to_csv.assert_called_once_with(Path('/dummy/assets/fonts.csv'), index=False)
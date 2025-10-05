import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
from pathlib import Path
import pandas as pd
from fontTools.ttLib import TTFont

# Add project root to path to allow sibling imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.env import FONTS_ROOT, ASSETS_PATH


@patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('pandas.read_csv')
def test_synthetic_data_generator_with_given_text(
    mock_read_csv, mock_get_font_meta, mock_get_charsets, MockRenderer
):
    """
    Tests the data generator's ability to produce an image from a given text,
    ensuring that spaces are preserved.
    """
    # Arrange
    mock_renderer_instance = MockRenderer.return_value
    mock_renderer_instance.render.return_value = (np.zeros((10, 10)), {'font_path': 'dummy.ttf'})

    valid_chars = set('test ')
    mock_get_charsets.return_value = (valid_chars, set('t'), set('e'))
    mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [10]})

    font_path_rel = 'NotoSansJP-Regular.ttf'
    mock_fonts_df = pd.DataFrame([{'font_path': font_path_rel, 'label': 'regular', 'num_chars': len(valid_chars)}])
    mock_font_map = {font_path_rel: valid_chars}
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    generator = SyntheticDataGenerator(renderer=mock_renderer_instance)

    input_text = 'test text'
    # The ground truth text should preserve the space.
    expected_text = 'test text'

    # Act
    img, text, params = generator.process(
        input_text,
        override_css_params={'font_path': font_path_rel}
    )

    # Assert
    assert img is not None
    assert text == expected_text
    mock_renderer_instance.render.assert_called_once()
    rendered_lines = mock_renderer_instance.render.call_args[0][0]
    # The `words_to_lines` function with a long max_line_len will produce a single line
    assert "".join(rendered_lines) == expected_text


@patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('pandas.read_csv')
def test_synthetic_data_generator_with_random_text(
    mock_read_csv, mock_get_font_meta, mock_get_charsets, MockRenderer
):
    """
    Tests the data generator's ability to produce an image from random text.
    """
    # Arrange
    mock_renderer_instance = MockRenderer.return_value
    mock_renderer_instance.render.return_value = (np.zeros((10, 10)), {'font_path': 'dummy.ttf'})

    valid_chars = set('test ')
    mock_get_charsets.return_value = (valid_chars, set('t'), set('e'))
    mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [10]})

    rel_font_path = 'NotoSansJP-Regular.ttf'
    mock_fonts_df = pd.DataFrame([{'font_path': rel_font_path, 'label': 'regular', 'num_chars': len(valid_chars)}])
    mock_font_map = {rel_font_path: valid_chars}
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    generator = SyntheticDataGenerator(renderer=mock_renderer_instance)

    # Act
    img, text, params = generator.process(text=None)

    # Assert
    assert img is not None
    assert isinstance(text, str)
    assert all(c in valid_chars for c in text if c != '\n')
    mock_renderer_instance.render.assert_called_once()
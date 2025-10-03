"""Tests for the synthetic data generator.

This module contains tests for the `SyntheticDataGenerator` class, which is
responsible for creating synthetic training data. These tests verify that the
generator can handle various conditions, such as malformed font metadata,
and that it correctly processes text and styles.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.env import FONTS_ROOT

@patch('manga_ocr_dev.synthetic_data_generator.utils.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
def test_generator_handles_missing_font_data_after_fix(
    mock_budoux, mock_gen_read_csv, mock_get_charsets, mock_renderer, mock_utils_read_csv
):
    """
    Test that SyntheticDataGenerator initializes and runs correctly after fixing `get_font_meta`
    to handle fonts with missing data.
    """
    # Mock the return of `pd.read_csv` in `utils.py` to simulate reading a fonts.csv
    # with a row that contains NaN.
    mock_fonts_df_with_nan = pd.DataFrame({
        'font_path': ['good_font.ttf', 'bad_font.ttf'],
        'supported_chars': ['abc', float('nan')],
        'label': ['regular', 'regular'],
        'num_chars': [3, 0]
    })
    mock_utils_read_csv.return_value = mock_fonts_df_with_nan

    # Mock other dependencies for SyntheticDataGenerator initialization
    mock_get_charsets.return_value = (set('abc'), set('a'), set('b'))
    mock_gen_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})

    # Configure the mock renderer to return a tuple
    mock_renderer.return_value.render.return_value = (MagicMock(), {})

    # Initialize the generator. This will call the *real*, fixed `get_font_meta`,
    # which in turn calls our mocked `pd.read_csv`.
    generator = SyntheticDataGenerator()

    # Assert that the bad font was dropped and is not in the font map or dataframe
    bad_font_path = str(FONTS_ROOT / 'bad_font.ttf')
    assert bad_font_path not in generator.font_map
    assert not any(generator.fonts_df['font_path'] == bad_font_path)

    # The `process` method should now run without raising an exception.
    # We patch `get_random_words` to avoid a separate bug.
    with patch.object(generator, 'get_random_words', return_value=['a']):
        generator.process()
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

@patch('manga_ocr_dev.synthetic_data_generator.common.utils.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
def test_generator_handles_missing_font_data_after_fix(
    mock_budoux, mock_gen_read_csv, mock_get_charsets, mock_renderer, mock_utils_read_csv
):
    """Tests that the generator can handle malformed font metadata.

    This test verifies that the `SyntheticDataGenerator` can initialize and
    operate correctly even when the `fonts.csv` file contains rows with
    missing data (e.g., NaN values). It specifically tests the fix in the
    `get_font_meta` function, which should now gracefully handle such malformed
    entries by dropping them.

    The test mocks the `pd.read_csv` call within the `utils` module to simulate
    loading a defective `fonts.csv`, then asserts that the generator
    initializes without error and that the invalid font is excluded from its
    internal font map.

    Args:
        mock_budoux: Mock for the BudouX parser.
        mock_gen_read_csv: Mock for `pd.read_csv` in the generator module.
        mock_get_charsets: Mock for the `get_charsets` utility.
        mock_renderer: Mock for the `Renderer` class.
        mock_utils_read_csv: Mock for `pd.read_csv` in the `utils` module,
            used to inject faulty font data.
    """
    # This test is unique because it mocks the `pd.read_csv` inside `utils` to test
    # the real `get_font_meta` function's ability to handle bad data.
    # We don't mock `get_font_meta` itself here.

    # Mock the return of `pd.read_csv` in `utils.py` to simulate reading a fonts.csv
    # with a row that contains NaN.
    mock_fonts_df_with_nan = pd.DataFrame({
        'font_path': ['good_font.ttf', 'bad_font.ttf'],
        'supported_chars': ['abc', float('nan')],
        'label': ['regular', 'regular'],
        'num_chars': [3, 0]
    })
    mock_utils_read_csv.return_value = mock_fonts_df_with_nan

    # Mock other dependencies for BaseDataGenerator initialization
    mock_get_charsets.return_value = (set('abc'), set('a'), set('b'))
    mock_gen_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})

    # Configure the mock renderer to return a tuple
    mock_renderer.return_value.render.return_value = (MagicMock(), {})

    # Initialize the generator. This will call the *real*, fixed `get_font_meta`,
    # which in turn calls our mocked `pd.read_csv`.
    generator = SyntheticDataGenerator()

    # Assert that the bad font was dropped and is not in the font map or dataframe
    bad_font_path = 'bad_font.ttf'
    assert bad_font_path not in generator.font_map
    assert not any(generator.fonts_df['font_path'] == bad_font_path)

    # The `process` method should now run without raising an exception.
    # We patch `get_random_words` and `get_random_font` to avoid unrelated errors.
    with patch.object(generator, 'get_random_words', return_value=['a']):
        with patch.object(generator, 'get_random_font', return_value='good_font.ttf'):
            generator.process()


@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
def test_generator_raises_value_error_for_unsupported_chars(
    mock_budoux, mock_gen_read_csv, mock_get_charsets, mock_renderer, mock_get_font_meta
):
    """Tests that the generator raises an error for unsupported characters.

    This test ensures that the `SyntheticDataGenerator` correctly identifies
    when a given input text contains characters that are not supported by any
    of the available fonts. When such a situation occurs, the generator is
    expected to raise a `ValueError` to prevent attempting to render text that
    would result in missing or incorrect glyphs.

    The test sets up a mock font environment where no font supports the
    character 'X', and then attempts to process a text containing 'X',
    asserting that the expected `ValueError` is raised.

    Args:
        mock_budoux: Mock for the BudouX parser.
        mock_gen_read_csv: Mock for `pd.read_csv` in the generator module.
        mock_get_charsets: Mock for the `get_charsets` utility.
        mock_renderer: Mock for the `Renderer` class.
        mock_get_font_meta: Mock for `get_font_meta` to provide a controlled
            set of fonts and supported characters.
    """
    # Mock dependencies for SyntheticDataGenerator initialization
    mock_get_charsets.return_value = (set('abc'), set('a'), set('b'))
    mock_gen_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})
    mock_budoux.return_value.parse.return_value = ['abcX']

    # Mock the return of `get_font_meta`. The mock fonts do not support 'X'.
    mock_fonts_df = pd.DataFrame({
        'font_path': ['font1.ttf', 'font2.ttf'],
        'supported_chars': ['ab', 'c'],
        'label': ['regular', 'regular'],
        'num_chars': [2, 1]
    })
    mock_font_map = {
            'font1.ttf': set('ab'),
            'font2.ttf': set('c'),
    }
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    # Configure the mock renderer to return a tuple
    mock_renderer.return_value.render.return_value = (MagicMock(), {})

    # Initialize the generator
    generator = SyntheticDataGenerator()

    # The text 'abcX' contains 'X', which is not supported by any of the mock
    # fonts. The `process` method should raise a ValueError.
    with pytest.raises(ValueError, match="Text contains unsupported characters: X"):
        generator.process(text='abcX')
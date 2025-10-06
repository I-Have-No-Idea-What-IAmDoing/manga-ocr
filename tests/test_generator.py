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


@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
def test_generator_raises_value_error_for_unsupported_chars(
    mock_budoux, mock_gen_read_csv, mock_get_charsets, mock_renderer, mock_get_font_meta
):
    """Test that SyntheticDataGenerator raises ValueError for unsupported characters."""
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


@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
def test_add_random_furigana(mock_budoux, mock_read_csv, mock_get_charsets, mock_renderer, mock_get_font_meta):
    """Tests the add_random_furigana method with various inputs."""
    mock_get_charsets.return_value = (
        set('abc一二三' + 'あいう'),
        set('あいう'),
        set()
    )
    mock_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})
    # Provide a mock fonts_df with the 'label' column to prevent AttributeError
    mock_get_font_meta.return_value = (pd.DataFrame({'label': ['regular']}), {})
    generator = SyntheticDataGenerator(renderer=mock_renderer)

    # Test case 1: Add furigana to kanji
    with patch('numpy.random.uniform', return_value=0.0), \
         patch('numpy.random.choice', side_effect=['hiragana', ['a', 'b']]), \
         patch('numpy.random.normal', return_value=2.0):
        result = generator.add_random_furigana("一二三", word_prob=1.0)
        assert result == '<ruby>一二三<rt>ab</rt></ruby>'

    # Test case 2: No furigana when word_prob is 0
    result = generator.add_random_furigana("一", word_prob=0.0)
    assert result == '一'

    # Test case 3: Group short ASCII
    with patch('numpy.random.uniform', return_value=0.0):
        result = generator.add_random_furigana("abc")
        assert result == '<span style="text-combine-upright: all">abc</span>'

    # Test case 4: Don't group long ASCII
    result = generator.add_random_furigana("abcd")
    assert result == 'abcd'


@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
def test_words_to_lines(mock_budoux, mock_read_csv, mock_get_charsets, mock_renderer, mock_get_font_meta):
    """Tests the words_to_lines method."""
    mock_get_charsets.return_value = (set(), set(), set())
    mock_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})
    # Provide a mock fonts_df with the 'label' column to prevent AttributeError
    mock_get_font_meta.return_value = (pd.DataFrame({'label': ['regular']}), {})
    generator = SyntheticDataGenerator(renderer=mock_renderer)

    # Test basic line breaking
    with patch('numpy.random.poisson', return_value=10):
        lines = generator.words_to_lines(["this", "is", "a", "long", "line"])
        assert lines == ["thisisa", "longline"]

    # Test single line
    with patch('numpy.random.poisson', return_value=20):
        lines = generator.words_to_lines(["short", "line"])
        assert lines == ["shortline"]

    # Test empty input
    lines = generator.words_to_lines([])
    assert lines == []
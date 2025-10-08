"""Extended tests for the synthetic data generator.

This module provides a comprehensive suite of unit tests for the
`SyntheticDataGenerator` class, covering its initialization, text processing,
font selection, and styling capabilities. These tests use extensive mocking to
isolate the generator's logic and verify its behavior under various conditions.
"""

from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, Mock

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.env import FONTS_ROOT

@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
def test_generator_initialization(mock_renderer, mock_budoux, mock_read_csv, mock_get_charsets, mock_get_font_meta):
    """Tests that the SyntheticDataGenerator initializes correctly.

    This test verifies that the `SyntheticDataGenerator` constructor
    successfully loads all its required assets and initializes its internal
    state as expected. It uses mocks for all external dependencies, including
    file I/O and helper utilities, to ensure that the test is isolated and
    focuses solely on the initialization logic of the generator.

    Args:
        mock_renderer: Mock for the `Renderer` class.
        mock_budoux: Mock for the BudouX parser.
        mock_read_csv: Mock for `pd.read_csv` to simulate loading data.
        mock_get_charsets: Mock for the `get_charsets` utility.
        mock_get_font_meta: Mock for the `get_font_meta` utility.
    """
    # Mock dependencies
    mock_get_font_meta.return_value = (
        pd.DataFrame({'font_path': ['dummy.ttf'], 'label': ['regular'], 'num_chars': [1]}),
        {'dummy.ttf': {'a'}}
    )
    mock_get_charsets.return_value = ({'a'}, {'a'}, {'a'})
    mock_read_csv.return_value = pd.DataFrame({'len': [1], 'p': [1.0]})
    mock_budoux.return_value = MagicMock()

    # Initialize the generator
    generator = SyntheticDataGenerator()

    # Assert that the generator is initialized with the correct attributes
    assert generator.renderer is not None
    assert generator.parser is not None
    assert not generator.fonts_df.empty
    assert generator.font_map
    assert generator.vocab
    assert generator.hiragana
    assert generator.katakana
    assert not generator.len_to_p.empty


@pytest.fixture
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
def generator(mock_renderer, mock_budoux, mock_read_csv, mock_get_charsets, mock_get_font_meta):
    """Provides a pytest fixture for a `SyntheticDataGenerator` instance.

    This fixture creates a fully mocked `SyntheticDataGenerator` that can be
    injected into test functions. It sets up mock returns for all external
    dependencies, including font metadata, character sets, and the text
    parser, allowing for isolated and predictable testing of the generator's
    methods.

    Args:
        mock_renderer: Mock for the `Renderer` class.
        mock_budoux: Mock for the BudouX parser.
        mock_read_csv: Mock for `pd.read_csv`.
        mock_get_charsets: Mock for the `get_charsets` utility.
        mock_get_font_meta: Mock for the `get_font_meta` utility.

    Returns:
        An instance of `SyntheticDataGenerator` with mocked dependencies,
        ready for testing.
    """
    font1_path_rel = 'font1.ttf'
    font2_path_rel = 'font2.ttf'
    font3_path_rel = 'font3.ttf'
    font1_path_abs = str(Path(FONTS_ROOT) / font1_path_rel)
    font2_path_abs = str(Path(FONTS_ROOT) / font2_path_rel)
    font3_path_abs = str(Path(FONTS_ROOT) / font3_path_rel)

    mock_get_font_meta.return_value = (
        pd.DataFrame({
            'font_path': [font1_path_rel, font2_path_rel, font3_path_rel],
            'label': ['regular', 'common', 'special'],
            'num_chars': [10, 20, 4001]
        }),
        {font1_path_abs: set('abcde'), font2_path_abs: set('fghij'), font3_path_abs: set('xyz')}
    )
    mock_get_charsets.return_value = (set('abcdefghijxyz'), set('a'), set('b'))
    mock_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})

    mock_parser = MagicMock()
    mock_parser.parse.side_effect = lambda x: x.split()
    mock_budoux.return_value = mock_parser

    mock_renderer_instance = mock_renderer.return_value
    mock_renderer_instance.render.return_value = (MagicMock(), {'font_path': font1_path_abs})

    return SyntheticDataGenerator(renderer=mock_renderer_instance)

def test_process_with_given_text(generator):
    """Tests the `process` method with a predefined text string.

    This test verifies that the `process` method correctly handles a given
    text input. It ensures that the method renders the text, returns the
    correct ground truth, and uses the specified font.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    img, text_gt, params = generator.process('abcde', override_css_params={'font_path': str(Path(FONTS_ROOT) / 'font1.ttf')})
    assert img is not None
    assert text_gt == 'abcde'
    assert params['font_path'] == str(Path(FONTS_ROOT) / 'font1.ttf')
    generator.renderer.render.assert_called()

def test_process_with_random_text(generator):
    """Tests the `process` method's ability to generate random text.

    This test verifies that when no text is provided to the `process` method,
    it correctly falls back to generating random words. It ensures that an
    image is generated and that the returned ground truth text matches the
    randomly generated text.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    font1_path_abs = str(Path(FONTS_ROOT) / 'font1.ttf')
    with patch.object(generator, 'get_random_font', return_value=font1_path_abs), \
         patch.object(generator, 'get_random_words', return_value=['abc']):
        img, text_gt, params = generator.process()
        assert img is not None
        assert text_gt == 'abc'
        assert 'font_path' in params
        generator.renderer.render.assert_called()

def test_get_random_words(generator):
    """Tests the `get_random_words` method for generating random text.

    This test ensures that the `get_random_words` method returns a non-empty
    list of strings, which represent the randomly generated words.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    words = generator.get_random_words(vocab=list('abc'))
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    assert len(''.join(words)) > 0

def test_split_into_words(generator):
    """Tests the `split_into_words` method's text splitting logic.

    This test verifies that the `split_into_words` method correctly uses the
    mocked BudouX parser to split a given string into a list of words.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    words = generator.split_into_words('test text')
    assert words == ['test', 'text']
    generator.parser.parse.assert_called_with('test text')

def test_words_to_lines(generator):
    """Tests the `words_to_lines` method for formatting words into lines.

    This test ensures that the `words_to_lines` method correctly arranges a
    list of words into a list of lines, preserving the original text.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    lines = generator.words_to_lines(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    assert isinstance(lines, list)
    assert ''.join(lines) == 'abcdefg'

def test_add_random_furigana(generator):
    """Tests the `add_random_furigana` method for adding ruby text and styling.

    This test covers various scenarios for the `add_random_furigana` method,
    including adding furigana to kanji, applying Tate-Chu-Yoko styling to
    ASCII characters, and handling cases where no styling is applied. It uses
    extensive patching to control the random outcomes and isolate the logic.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    # Test with kanji and furigana
    with patch('manga_ocr_dev.synthetic_data_generator.generator.is_kanji', return_value=True), \
         patch('manga_ocr_dev.synthetic_data_generator.generator.is_ascii', return_value=False), \
         patch('numpy.random.uniform', return_value=0.1), \
         patch('numpy.random.choice', side_effect=['hiragana', ['a']]), \
         patch('numpy.random.normal', return_value=0.4), \
         patch('numpy.random.randint', return_value=1):
        result = generator.add_random_furigana('日本語', word_prob=1.0)
        assert '<ruby>日本語<rt>a</rt></ruby>' in result

    # Test with ASCII combination
    with patch('manga_ocr_dev.synthetic_data_generator.generator.is_kanji', return_value=False), \
         patch('manga_ocr_dev.synthetic_data_generator.generator.is_ascii', return_value=True), \
         patch('numpy.random.uniform', return_value=0.1):
        result = generator.add_random_furigana('AB')
        assert result == '<span style="text-combine-upright: all">AB</span>'

    # Test with long ASCII (no combination)
    with patch('manga_ocr_dev.synthetic_data_generator.generator.is_kanji', return_value=False), \
         patch('manga_ocr_dev.synthetic_data_generator.generator.is_ascii', return_value=True), \
         patch('numpy.random.uniform', return_value=0.9):
        result = generator.add_random_furigana('ABCD')
        assert result == 'ABCD'

    # Test with no furigana applied
    with patch('manga_ocr_dev.synthetic_data_generator.generator.is_kanji', return_value=True), \
         patch('manga_ocr_dev.synthetic_data_generator.generator.is_ascii', return_value=False), \
         patch('numpy.random.uniform', return_value=0.9):
        result = generator.add_random_furigana('日本語', word_prob=0.0)
        assert result == '日本語'

def test_is_font_supporting_text(generator):
    """Tests the `is_font_supporting_text` method.

    This test verifies that the `is_font_supporting_text` method correctly
    checks if a given font supports all characters in a string. It checks
    both supported and unsupported characters, as well as whitespace handling.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    font_path = str(Path(FONTS_ROOT) / 'font1.ttf')
    assert generator.is_font_supporting_text(font_path, 'abc')
    assert not generator.is_font_supporting_text(font_path, 'xyz')
    assert generator.is_font_supporting_text(font_path, ' a b ')

def test_get_random_font(generator):
    """Tests the `get_random_font` method's font selection logic.

    This test ensures that `get_random_font` can select a font that supports
    a given text, and that it raises a `ValueError` when no suitable font can
    be found.

    Args:
        generator: The mocked `SyntheticDataGenerator` fixture.
    """
    font1_path_abs = str(Path(FONTS_ROOT) / 'font1.ttf')
    font3_path_abs = str(Path(FONTS_ROOT) / 'font3.ttf')

    # Test with text that has full support
    with patch.object(generator, 'is_font_supporting_text', return_value=True):
        font_path = generator.get_random_font(text='a')
        assert font_path is not None

    # Test with text that has no full support (should raise ValueError)
    with patch.object(generator, 'is_font_supporting_text', return_value=False), \
         pytest.raises(ValueError, match="Text contains unsupported characters"):
        generator.get_random_font(text='unsupported')
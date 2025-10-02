import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, Mock

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.env import FONTS_ROOT

@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
def test_generator_initialization(mock_renderer, mock_budoux, mock_read_csv, mock_get_charsets, mock_get_font_meta):
    """
    Tests that the SyntheticDataGenerator initializes correctly with mocked dependencies.
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
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.budoux.load_default_japanese_parser')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
def generator(mock_renderer, mock_budoux, mock_read_csv, mock_get_charsets, mock_get_font_meta):
    """
    Provides a SyntheticDataGenerator instance with mocked dependencies for testing.
    """
    font1_path_rel = 'font1.ttf'
    font2_path_rel = 'font2.ttf'
    font3_path_rel = 'font3.ttf'
    font1_path_abs = str(FONTS_ROOT / font1_path_rel)
    font2_path_abs = str(FONTS_ROOT / font2_path_rel)
    font3_path_abs = str(FONTS_ROOT / font3_path_rel)

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
    img, text_gt, params = generator.process('abcde', override_css_params={'font_path': str(FONTS_ROOT / 'font1.ttf')})
    assert img is not None
    assert text_gt == 'abcde'
    assert params['font_path'] == str(FONTS_ROOT / 'font1.ttf')
    generator.renderer.render.assert_called()

def test_process_with_random_text(generator):
    font1_path_abs = str(FONTS_ROOT / 'font1.ttf')
    with patch.object(generator, 'get_random_font', return_value=font1_path_abs), \
         patch.object(generator, 'get_random_words', return_value=['abc']):
        img, text_gt, params = generator.process()
        assert img is not None
        assert text_gt == 'abc'
        assert 'font_path' in params
        generator.renderer.render.assert_called()

def test_get_random_words(generator):
    words = generator.get_random_words(vocab=list('abc'))
    assert isinstance(words, list)
    assert all(isinstance(word, str) for word in words)
    assert len(''.join(words)) > 0

def test_split_into_words(generator):
    words = generator.split_into_words('test text')
    assert words == ['test', 'text']
    generator.parser.parse.assert_called_with('test text')

def test_words_to_lines(generator):
    lines = generator.words_to_lines(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    assert isinstance(lines, list)
    assert ''.join(lines) == 'abcdefg'

def test_add_random_furigana(generator):
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
    font_path = str(FONTS_ROOT / 'font1.ttf')
    assert generator.is_font_supporting_text(font_path, 'abc')
    assert not generator.is_font_supporting_text(font_path, 'xyz')
    assert generator.is_font_supporting_text(font_path, ' a b ')

def test_get_random_font(generator):
    font1_path_abs = str(FONTS_ROOT / 'font1.ttf')
    font3_path_abs = str(FONTS_ROOT / 'font3.ttf')

    # Test with text that has full support
    with patch.object(generator, 'is_font_supporting_text', return_value=True):
        font_path = generator.get_random_font(text='a')
        assert font_path is not None

    # Test with text that has no full support (should fallback)
    with patch.object(generator, 'is_font_supporting_text', return_value=False):
        font_path = generator.get_random_font(text='unsupported')
        assert font_path == font3_path_abs

def test_process_with_unsupported_chars(generator):
    """
    Tests that `process` removes characters not supported by the selected font.
    """
    font_path = str(FONTS_ROOT / 'font1.ttf')
    img, text_gt, params = generator.process('abcdexyz', override_css_params={'font_path': font_path})
    assert text_gt == 'abcde'
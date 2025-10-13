import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2

@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
def test_generator_handles_nan_text(
    mock_budoux, mock_gen_read_csv, mock_get_charsets, mock_get_font_meta
):
    """Tests that the generator can handle NaN values for text input."""
    # Mock dependencies for SyntheticDataGeneratorV2 initialization
    mock_get_charsets.return_value = (set('abc'), set('a'), set('b'))
    mock_gen_read_csv.return_value = pd.DataFrame({'len': [10], 'p': [1.0]})
    mock_budoux.return_value.parse.return_value = ['abc']

    # Mock the return of `get_font_meta`
    mock_fonts_df = pd.DataFrame({
        'font_path': ['font1.ttf'],
        'supported_chars': ['abc'],
        'label': ['regular'],
        'num_chars': [3]
    })
    mock_font_map = {
            'font1.ttf': set('abc'),
    }
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    # Initialize the generator
    generator = SyntheticDataGeneratorV2()

    # The `process` method should now run without raising an exception when text is NaN.
    # We patch `get_random_words` to ensure the output is predictable.
    with patch.object(generator, 'get_random_words', return_value=['a', 'b']):
        img, text_gt, params = generator.process(text=np.nan)
        assert text_gt == 'ab'
        assert img is not None
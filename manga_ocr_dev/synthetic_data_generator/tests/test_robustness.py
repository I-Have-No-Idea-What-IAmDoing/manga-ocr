"""Tests for the robustness of the synthetic data generator.

This module contains tests for the `SyntheticDataGenerator` class, focusing
on its robustness to various failures and edge cases. These tests verify that
the generator can handle rendering failures, augmentation failures, and
invalid font characters gracefully.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from pathlib import Path
import re
from PIL import Image

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.common.composer import Composer
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer


@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.Renderer')
def test_furigana_generation_with_limited_vocab(mock_renderer, mock_get_font_meta, mock_get_charsets, mock_read_csv, mock_budoux):
    """Tests that furigana generation only uses characters from the font's vocabulary."""
    mock_fonts_df = MagicMock()
    mock_font_map = {'font.ttf': set('ab')}
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)
    mock_get_charsets.return_value = (set('a'), set('b'), set())
    mock_read_csv.return_value = pd.DataFrame({'len': [1], 'p': [1.0]})
    mock_renderer.return_value.render.return_value = (np.zeros((10, 10, 4), dtype=np.uint8), {})

    generator = SyntheticDataGenerator()
    generator.hiragana = 'a'
    generator.katakana = 'b'

    # The furigana should only contain 'a' and 'b', which are in the font's vocab
    line = generator.add_random_furigana('X', vocab=set('ab'))

    # Extract furigana text using regex
    furigana_text = ''.join(re.findall(r'<rt>(.*?)</rt>', line))

    # Assert that all characters in the furigana are in the allowed vocabulary
    assert all(c in 'ab' for c in furigana_text)


@patch('pathlib.Path.absolute')
@patch('pathlib.Path.as_uri')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.Html2Image')
def test_renderer_handles_screenshot_failure(mock_html2image, mock_as_uri, mock_absolute):
    """Tests that the renderer gracefully handles a screenshot failure."""
    mock_html2image.return_value.screenshot_as_bytes.side_effect = Exception("Screenshot failed")
    mock_absolute.return_value = Path('dummy/path')
    mock_as_uri.return_value = 'file://dummy/path'
    renderer = Renderer()
    img, params = renderer.render(['test'], override_css_params={'font_path': 'dummy.ttf'})
    assert img is None


@patch('manga_ocr_dev.synthetic_data_generator.common.composer.get_background_df')
@patch('manga_ocr_dev.synthetic_data_generator.common.composer.A.Compose')
def test_composer_handles_augmentation_failure(mock_compose, mock_get_background_df):
    """Tests that the composer gracefully handles an augmentation failure."""
    mock_compose.side_effect = Exception("Augmentation failed")
    mock_get_background_df.return_value = pd.DataFrame([{'path': 'dummy_path.jpg'}])
    composer = Composer(background_dir='dummy_dir')

    # Use a real image for the mock to ensure np.array and other operations work
    mock_pil_image = Image.new('RGBA', (100, 100))

    with patch('manga_ocr_dev.synthetic_data_generator.common.composer.Image.open', return_value=mock_pil_image):
        result = composer(np.zeros((10, 10, 4), dtype=np.uint8), {})
        assert result is not None  # The main point is that it doesn't crash

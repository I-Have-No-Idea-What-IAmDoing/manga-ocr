import os
import numpy as np
from unittest.mock import patch
import pandas as pd
import cv2
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from fontTools.ttLib import TTFont

from manga_ocr_dev.env import FONTS_ROOT


@patch('manga_ocr_dev.synthetic_data_generator.renderer.Html2Image.screenshot_as_bytes')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df')
def test_synthetic_data_generator(mock_get_background_df, mock_get_charsets, mock_read_csv, mock_get_font_meta, mock_imread, mock_screenshot):
    """
    Tests that the synthetic data generator can successfully produce an image-text pair
    and correctly filters characters not present in the font.
    """
    mock_get_background_df.return_value = pd.DataFrame([{'path': 'dummy.jpg', 'h': 100, 'w': 100, 'ratio': 1.0}])
    mock_get_charsets.return_value = (np.array(['t', 'e', 's', ' ']), np.array([]), np.array([]))
    mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [1]})
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_screenshot.return_value = cv2.imencode('.png', np.zeros((100, 100, 4), dtype=np.uint8))[1].tobytes()

    browser_executable = os.environ.get('CHROME_EXECUTABLE_PATH',
                                        '/home/jules/.cache/ms-playwright/chromium-1181/chrome-linux/chrome')
    os.environ['CHROME_EXECUTABLE_PATH'] = browser_executable
    font_path = str(FONTS_ROOT / 'NotoSansJP-Regular.ttf')

    # Get the character set for the font to use in the mock
    font = TTFont(font_path)
    valid_chars = set()
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            for code in cmap.cmap:
                valid_chars.add(chr(code))

    mock_fonts_df = pd.DataFrame([{
        'font_path': font_path,
        'label': 'regular',
        'num_chars': len(valid_chars)
    }])
    mock_font_map = {font_path: valid_chars}
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    input_text = 'test text'
    # The generator will filter out any characters from the input text that are not
    # in the font's character set.
    expected_text = "".join([c for c in input_text if c in valid_chars])

    with Renderer(browser_executable=browser_executable) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        img, text, params = generator.process(
            input_text,
            override_css_params={'font_path': font_path}
        )

        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert isinstance(text, str)
        assert text == expected_text


@patch('manga_ocr_dev.synthetic_data_generator.renderer.Html2Image.screenshot_as_bytes')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
@patch('manga_ocr_dev.synthetic_data_generator.generator.pd.read_csv')
@patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df')
def test_synthetic_data_generator_random_text(mock_get_background_df, mock_get_charsets, mock_read_csv, mock_get_font_meta, mock_imread, mock_screenshot):
    """
    Tests that the synthetic data generator can successfully produce an image-text pair
    with randomly generated text.
    """
    mock_get_background_df.return_value = pd.DataFrame([{'path': 'dummy.jpg', 'h': 100, 'w': 100, 'ratio': 1.0}])
    mock_get_charsets.return_value = (np.array(['t', 'e', 's', ' ']), np.array(['t']), np.array(['e']))
    mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [10]})
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_screenshot.return_value = cv2.imencode('.png', np.zeros((100, 100, 4), dtype=np.uint8))[1].tobytes()

    browser_executable = os.environ.get('CHROME_EXECUTABLE_PATH',
                                        '/home/jules/.cache/ms-playwright/chromium-1181/chrome-linux/chrome')
    os.environ['CHROME_EXECUTABLE_PATH'] = browser_executable
    font_path = str(FONTS_ROOT / 'NotoSansJP-Regular.ttf')
    rel_font_path = 'NotoSansJP-Regular.ttf'

    valid_chars = set('test ')
    mock_fonts_df = pd.DataFrame([{
        'font_path': rel_font_path,
        'label': 'regular',
        'num_chars': len(valid_chars)
    }])
    mock_font_map = {font_path: valid_chars}
    mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

    with Renderer(browser_executable=browser_executable) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        img, text, params = generator.process(text=None)

        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert isinstance(text, str)
        # Check that the generated text only contains valid characters
        assert all(c in valid_chars for c in text.replace('\n', ''))
        assert len(text) > 0
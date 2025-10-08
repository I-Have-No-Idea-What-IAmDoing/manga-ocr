"""Integration tests for the synthetic data generator.

This module contains integration-style tests for the `SyntheticDataGenerator`,
verifying its ability to produce complete image-text pairs. These tests use
mocking for file I/O and browser interactions but test the generator's core
logic in a more integrated fashion.
"""

import os
import numpy as np
from unittest.mock import patch
import pandas as pd
import cv2
from pathlib import Path
import unittest
import tempfile
import shutil
from PIL import Image

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from fontTools.ttLib import TTFont

from manga_ocr_dev.env import FONTS_ROOT


class TestIntegrationSyntheticDataGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.backgrounds_dir = Path(self.temp_dir)

        # Create a dummy background image
        self.dummy_bg_path = self.backgrounds_dir / "dummy.jpg"
        dummy_bg_img = Image.new('RGB', (100, 100), color = 'red')
        dummy_bg_img.save(self.dummy_bg_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.subprocess.Popen')
    @patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome', return_value='dummy_chrome_path')
    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Html2Image.screenshot_as_bytes')
    @patch('manga_ocr_dev.synthetic_data_generator.common.composer.cv2.imread')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
    @patch('manga_ocr_dev.synthetic_data_generator.common.composer.get_background_df')
    def test_synthetic_data_generator_with_given_text(self, mock_get_background_df, mock_get_charsets, mock_read_csv, mock_get_font_meta, mock_imread, mock_screenshot, mock_find_chrome, mock_popen):
        """
        Tests the data generator's ability to produce an image from a given text.
        """
        mock_get_background_df.return_value = pd.DataFrame([{'path': self.dummy_bg_path, 'h': 100, 'w': 100, 'ratio': 1.0}])
        mock_get_charsets.return_value = (np.array(['t', 'e', 's', ' ']), np.array([]), np.array([]))
        mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [1]})
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_screenshot.return_value = cv2.imencode('.png', np.zeros((100, 100, 4), dtype=np.uint8))[1].tobytes()

        browser_executable = os.environ.get('CHROME_EXECUTABLE_PATH')
        font_path_abs = str(Path(FONTS_ROOT) / 'NotoSansJP-Regular.ttf')
        font_path_rel = 'NotoSansJP-Regular.ttf'

        font = TTFont(font_path_abs)
        valid_chars = set()
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                for code in cmap.cmap:
                    valid_chars.add(chr(code))

        mock_fonts_df = pd.DataFrame([{'font_path': font_path_rel, 'label': 'regular', 'num_chars': len(valid_chars)}])
        mock_font_map = {font_path_rel: valid_chars}
        mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

        input_text = 'test text'
        expected_text = "".join([c for c in input_text if c in valid_chars])

        with Renderer(browser_executable=browser_executable) as renderer:
            generator = SyntheticDataGenerator(renderer=renderer, background_dir=self.backgrounds_dir)
            with patch.object(generator.composer, '_is_low_contrast', return_value=False):
                img, text, params = generator.process(
                    input_text,
                    override_css_params={'font_path': font_path_rel}
                )

                self.assertIsInstance(img, np.ndarray)
                self.assertGreater(img.shape[0], 0)
                self.assertGreater(img.shape[1], 0)
                self.assertIsInstance(text, str)
                self.assertEqual(text, expected_text)

    @patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.subprocess.Popen')
    @patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome', return_value='dummy_chrome_path')
    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Html2Image.screenshot_as_bytes')
    @patch('manga_ocr_dev.synthetic_data_generator.common.composer.cv2.imread')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
    @patch('manga_ocr_dev.synthetic_data_generator.common.composer.get_background_df')
    def test_synthetic_data_generator_with_random_text(self, mock_get_background_df, mock_get_charsets, mock_read_csv, mock_get_font_meta, mock_imread, mock_screenshot, mock_find_chrome, mock_popen):
        """
        Tests the data generator's ability to produce an image from random text.
        """
        mock_get_background_df.return_value = pd.DataFrame([{'path': self.dummy_bg_path, 'h': 100, 'w': 100, 'ratio': 1.0}])
        mock_get_charsets.return_value = (np.array(['t', 'e', 's', ' ']), np.array(['t']), np.array(['e']))
        mock_read_csv.return_value = pd.DataFrame({'p': [1.0], 'len': [10]})
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_screenshot.return_value = cv2.imencode('.png', np.zeros((100, 100, 4), dtype=np.uint8))[1].tobytes()

        browser_executable = os.environ.get('CHROME_EXECUTABLE_PATH')
        rel_font_path = 'NotoSansJP-Regular.ttf'

        valid_chars = set('test ')
        mock_fonts_df = pd.DataFrame([{'font_path': rel_font_path, 'label': 'regular', 'num_chars': len(valid_chars)}])
        mock_font_map = {rel_font_path: valid_chars}
        mock_get_font_meta.return_value = (mock_fonts_df, mock_font_map)

        with Renderer(browser_executable=browser_executable) as renderer:
            generator = SyntheticDataGenerator(renderer=renderer, background_dir=self.backgrounds_dir)
            with patch.object(generator.composer, '_is_low_contrast', return_value=False):
                img, text, params = generator.process(text=None)

                self.assertIsInstance(img, np.ndarray)
                self.assertGreater(img.shape[0], 0)
                self.assertGreater(img.shape[1], 0)
                self.assertIsInstance(text, str)
                self.assertTrue(all(c in valid_chars for c in text.replace('\n', '')))
                self.assertGreater(len(text), 0)
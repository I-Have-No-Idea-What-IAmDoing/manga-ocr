"""Basic tests for the SyntheticDataGenerator.

This module provides a `unittest.TestCase` suite for the `SyntheticDataGenerator`,
covering its basic functionality. These tests ensure that the generator can be
initialized and can process both predefined and randomly generated text.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer


class TestSyntheticDataGenerator(unittest.TestCase):
    """A test suite for the SyntheticDataGenerator class."""

    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_font_meta')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.get_charsets')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.pd.read_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.budoux.load_default_japanese_parser')
    def setUp(self, mock_budoux, mock_read_csv, mock_get_charsets, mock_get_font_meta):
        """Sets up a mocked environment for testing the generator.

        This method is called before each test. It initializes a
        `SyntheticDataGenerator` instance with all its external dependencies
        mocked. This provides a clean, isolated environment for each test case.

        Args:
            mock_budoux: Mock for the BudouX parser.
            mock_read_csv: Mock for `pd.read_csv`.
            mock_get_charsets: Mock for the `get_charsets` utility.
            mock_get_font_meta: Mock for the `get_font_meta` utility.
        """
        mock_get_font_meta.return_value = (pd.DataFrame({'font_path': ['dummy.ttf'], 'label': ['regular']}), {'dummy.ttf': {'a'}})
        mock_get_charsets.return_value = ({'a'}, {'a'}, {'a'})
        mock_read_csv.return_value = pd.DataFrame({'len': [1], 'p': [1.0]})

        # FIX: Configure the mock parser to return the input text as a word
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = lambda x: [x]
        mock_budoux.return_value = mock_parser

        self.mock_renderer = MagicMock(spec=Renderer)
        self.mock_renderer.render.return_value = (MagicMock(), {'font_path': 'dummy.ttf'})
        self.generator = SyntheticDataGenerator(renderer=self.mock_renderer)


    def test_synthetic_data_generator_with_given_text(self):
        """Tests that the generator can process a given text.

        This test verifies that the `process` method correctly handles a
        predefined text input, calling the renderer and returning the expected
        ground truth text.
        """
        with patch.object(self.generator, 'get_random_font', return_value='dummy.ttf'):
            img, text_gt, params = self.generator.process('a')
            self.assertIsNotNone(img)
            self.assertEqual(text_gt, 'a')
            self.mock_renderer.render.assert_called()


    def test_synthetic_data_generator_with_random_text(self):
        """Tests that the generator can process a random text.

        This test ensures that the `process` method can generate random text
        when no input text is provided. It verifies that the renderer is

        called and that the returned ground truth matches the mock random text.
        """
        with patch.object(self.generator, 'get_random_words', return_value=['a']):
            with patch.object(self.generator, 'get_random_font', return_value='dummy.ttf'):
                img, text_gt, params = self.generator.process()
                self.assertIsNotNone(img)
                self.assertEqual(text_gt, 'a')
                self.mock_renderer.render.assert_called()
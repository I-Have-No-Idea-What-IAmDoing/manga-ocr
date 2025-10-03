import unittest
from unittest.mock import MagicMock, patch, call, ANY
import numpy as np
import sys
from pathlib import Path
import pandas as pd

# Add project root to path to allow sibling imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.env import FONTS_ROOT


class TestSyntheticDataGenerator(unittest.TestCase):
    @patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
    @patch('pandas.read_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
    def setUp(self, mock_get_charsets, mock_read_csv, mock_get_font_meta):
        # Mock the dependencies that read from the filesystem
        mock_get_charsets.return_value = (set("a字b"), ["あ"], ["ア"])
        mock_read_csv.return_value = pd.DataFrame({'len': [1], 'p': [1.0]})

        # Provide a more realistic mock for get_font_meta
        self.font_path = 'dummy.ttf'
        mock_fonts_df = pd.DataFrame({'font_path': [self.font_path], 'label': ['common'], 'num_chars': [3]})
        self.mock_font_map = {str(FONTS_ROOT / self.font_path): set("a字b")}
        mock_get_font_meta.return_value = (mock_fonts_df, self.mock_font_map)

        # Mock the renderer to avoid actual image generation
        self.mock_renderer = MagicMock(spec=Renderer)
        self.mock_renderer.render.return_value = (np.zeros((1, 1)), {})
        self.generator = SyntheticDataGenerator(renderer=self.mock_renderer)

        # Manually set the vocab and charsets as they are used in the test
        self.generator.vocab = set("a字b")
        self.generator.hiragana = ["あ"]
        self.generator.katakana = ["ア"]

    def test_add_random_furigana_order(self):
        # This test is designed to fail with the buggy implementation
        # and pass with the corrected one.
        np.random.seed(0)  # for reproducible furigana

        # Test case 1: ASCII then Kanji
        line1 = "a字"
        processed_line1 = self.generator.add_random_furigana(line1, word_prob=1.0)
        a_pos1 = processed_line1.find('a')
        kanji_pos1 = processed_line1.find('字')
        self.assertLess(a_pos1, kanji_pos1, f"For input '{line1}', 'a' should appear before '字' in the output: '{processed_line1}'")

        # Test case 2: Kanji then ASCII
        line2 = "字a"
        processed_line2 = self.generator.add_random_furigana(line2, word_prob=1.0)
        a_pos2 = processed_line2.find('a')
        kanji_pos2 = processed_line2.find('字')
        self.assertLess(kanji_pos2, a_pos2, f"For input '{line2}', '字' should appear before 'a' in the output: '{processed_line2}'")

    def test_process_unsupported_text_raises_error(self):
        # Test that providing text with unsupported characters raises a ValueError
        self.generator.font_map[str(FONTS_ROOT / self.font_path)] = set("a字b")
        with patch.object(self.generator, 'get_random_font', return_value=str(FONTS_ROOT / self.font_path)):
            with self.assertRaises(ValueError):
                self.generator.process(text="xyz")

    def test_words_to_lines(self):
        # Test the line splitting logic
        words = ["this", "is", "a", "long", "line"]
        with patch('numpy.random.poisson', return_value=11):
            lines = self.generator.words_to_lines(words)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], "thisisalong")
            self.assertEqual(lines[1], "line")

    @patch('numpy.random.randint')
    @patch('numpy.random.uniform')
    @patch('numpy.random.choice')
    @patch('numpy.random.rand')
    def test_random_css_params(self, mock_rand, mock_choice, mock_uniform, mock_randint):
        # Mock the random functions to control the output of get_random_css_params
        mock_rand.side_effect = [0.6, 0.8, 0.1, 0.1]  # vertical, text_color, text_orientation, letter_spacing
        mock_randint.return_value = 48  # font_size
        mock_uniform.side_effect = [0.6, -0.02]  # line_height, letter_spacing
        mock_choice.side_effect = ["stroke", 2]  # effect, stroke_size

        params = Renderer.get_random_css_params()

        self.assertEqual(params['font_size'], 48)
        self.assertTrue(params['vertical'])
        self.assertAlmostEqual(params['line_height'], 0.6)
        self.assertEqual(params['text_color'], 'white')
        self.assertEqual(params['text_orientation'], 'upright')
        self.assertAlmostEqual(params['letter_spacing'], -0.02)
        self.assertEqual(params['stroke_size'], 2)
        self.assertEqual(params['stroke_color'], 'black')


if __name__ == '__main__':
    unittest.main()
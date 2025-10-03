import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
from pathlib import Path
import pandas as pd

# Add project root to path to allow sibling imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator


class TestSyntheticDataGenerator(unittest.TestCase):
    @patch('manga_ocr_dev.synthetic_data_generator.generator.get_font_meta')
    @patch('pandas.read_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.generator.get_charsets')
    def setUp(self, mock_get_charsets, mock_read_csv, mock_get_font_meta):
        # Mock the dependencies that read from the filesystem
        mock_get_charsets.return_value = (set("a字b"), ["あ"], ["ア"])
        mock_read_csv.return_value = pd.DataFrame({'len': [1], 'p': [1.0]})

        # Provide a more realistic mock for get_font_meta
        mock_fonts_df = pd.DataFrame({'label': ['common']})
        mock_get_font_meta.return_value = (mock_fonts_df, {})

        # Mock the renderer to avoid actual image generation
        self.mock_renderer = MagicMock()
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


if __name__ == '__main__':
    unittest.main()
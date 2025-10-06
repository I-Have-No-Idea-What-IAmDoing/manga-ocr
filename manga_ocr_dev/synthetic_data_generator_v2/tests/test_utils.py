import sys
import unittest
import tempfile
import shutil
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from unittest.mock import patch
import numpy as np

from manga_ocr_dev.synthetic_data_generator_v2.utils import get_background_df, is_kanji, is_hiragana, is_katakana, is_ascii, get_charsets, get_font_meta


@patch('manga_ocr_dev.synthetic_data_generator_v2.utils.ASSETS_PATH')
class TestAssetLoadingFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create dummy vocab.csv
        self.vocab_path = self.temp_dir / "vocab.csv"
        vocab_content = "char\n猫\nあ\nア\nA"
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            f.write(vocab_content)

        # Create dummy fonts.csv
        self.fonts_csv_path = self.temp_dir / "fonts.csv"
        fonts_content = "font_path,supported_chars\nfont1.ttf,猫あ\nfont2.otf,Aア"
        with open(self.fonts_csv_path, "w", encoding="utf-8") as f:
            f.write(fonts_content)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_charsets(self, mock_assets_path):
        """Test that get_charsets correctly loads and categorizes characters."""
        mock_assets_path.__truediv__.return_value = self.vocab_path

        vocab, hiragana, katakana = get_charsets(vocab_path=self.vocab_path)

        self.assertTrue(np.array_equal(vocab, ['猫', 'あ', 'ア', 'A']))
        self.assertTrue(np.array_equal(hiragana, ['あ']))
        self.assertTrue(np.array_equal(katakana, ['ア']))

    @patch('manga_ocr_dev.synthetic_data_generator_v2.utils.FONTS_ROOT', new_callable=lambda: Path('/fake/fonts'))
    def test_get_font_meta(self, mock_fonts_root, mock_assets_path):
        """Test that get_font_meta correctly loads font metadata and creates a font map."""
        mock_assets_path.__truediv__.return_value = self.fonts_csv_path

        df, font_map = get_font_meta()

        self.assertEqual(len(df), 2)
        self.assertIn('/fake/fonts/font1.ttf', df['font_path'].values)

        expected_map = {
            str(Path('/fake/fonts/font1.ttf')): {'猫', 'あ'},
            str(Path('/fake/fonts/font2.otf')): {'A', 'ア'}
        }
        self.assertEqual(font_map, expected_map)


class TestCharTypeFunctions(unittest.TestCase):
    def test_is_kanji(self):
        self.assertTrue(is_kanji("猫"))
        self.assertFalse(is_kanji("A"))
        self.assertFalse(is_kanji("あ"))
        self.assertFalse(is_kanji("ア"))
        self.assertFalse(is_kanji("猫猫"))
        self.assertFalse(is_kanji(""))
        self.assertFalse(is_kanji(None))
        self.assertFalse(is_kanji(123))

    def test_is_hiragana(self):
        self.assertTrue(is_hiragana("あ"))
        self.assertFalse(is_hiragana("A"))
        self.assertFalse(is_hiragana("猫"))
        self.assertFalse(is_hiragana("ア"))
        self.assertFalse(is_hiragana("ああ"))
        self.assertFalse(is_hiragana(""))
        self.assertFalse(is_hiragana(None))
        self.assertFalse(is_hiragana(123))

    def test_is_katakana(self):
        self.assertTrue(is_katakana("ア"))
        self.assertFalse(is_katakana("A"))
        self.assertFalse(is_katakana("猫"))
        self.assertFalse(is_katakana("あ"))
        self.assertFalse(is_katakana("アア"))
        self.assertFalse(is_katakana(""))
        self.assertFalse(is_katakana(None))
        self.assertFalse(is_katakana(123))

    def test_is_ascii(self):
        self.assertTrue(is_ascii("A"))
        self.assertTrue(is_ascii("!"))
        self.assertTrue(is_ascii("7"))
        self.assertFalse(is_ascii("猫"))
        self.assertFalse(is_ascii("あ"))
        self.assertFalse(is_ascii("ア"))
        self.assertFalse(is_ascii("AA"))
        self.assertFalse(is_ascii(""))
        self.assertFalse(is_ascii(None))
        self.assertFalse(is_ascii(123))


class TestGetBackgroundDf(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Create dummy files with valid and invalid names
        (self.temp_dir / "valid_bg_0_100_50_200.png").touch()
        (self.temp_dir / "another_valid_bg_10_20_30_40.txt").touch()
        (self.temp_dir / "invalid_bg.png").touch()
        (self.temp_dir / "invalid_bg_1_2_3.jpg").touch()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parses_valid_filenames_and_skips_invalid(self):
        """Test that get_background_df correctly parses valid filenames and skips invalid ones."""
        df = get_background_df(self.temp_dir)

        # Should only have 2 entries for the valid filenames
        self.assertEqual(len(df), 2)
        self.assertIsInstance(df, pd.DataFrame)

        # Verify the data for the first valid file
        record1 = df[df['path'].str.contains("valid_bg_0_100_50_200")]
        self.assertEqual(record1.iloc[0]['h'], 100)  # ymax - ymin = 100 - 0
        self.assertEqual(record1.iloc[0]['w'], 150)  # xmax - xmin = 200 - 50
        self.assertEqual(record1.iloc[0]['ratio'], 1.5) # w / h = 150 / 100

        # Verify the data for the second valid file
        record2 = df[df['path'].str.contains("another_valid_bg_10_20_30_40")]
        self.assertEqual(record2.iloc[0]['h'], 10) # ymax - ymin = 20 - 10
        self.assertEqual(record2.iloc[0]['w'], 10) # xmax - xmin = 40 - 30
        self.assertEqual(record2.iloc[0]['ratio'], 1.0) # w / h = 10 / 10

    def test_empty_directory(self):
        """Test that get_background_df returns an empty DataFrame for an empty directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        df = get_background_df(empty_dir)
        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
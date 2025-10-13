import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.common.exceptions import SkipSample
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT
import manga_ocr_dev.synthetic_data_generator.common.utils as utils_module
import manga_ocr_dev.synthetic_data_generator.common.base_generator as base_generator_module

class TestSyntheticDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.assets_dir = cls.temp_dir / "assets"
        cls.fonts_dir = cls.temp_dir / "fonts"
        cls.assets_dir.mkdir(exist_ok=True)
        cls.fonts_dir.mkdir(exist_ok=True)

        cls.patcher_assets_utils = patch.object(utils_module, 'ASSETS_PATH', cls.assets_dir)
        cls.patcher_fonts_utils = patch.object(utils_module, 'FONTS_ROOT', cls.fonts_dir)
        cls.patcher_assets_base = patch.object(base_generator_module, 'ASSETS_PATH', cls.assets_dir)
        cls.patcher_fonts_base = patch.object(base_generator_module, 'FONTS_ROOT', cls.fonts_dir)

        cls.patcher_assets_utils.start()
        cls.patcher_fonts_utils.start()
        cls.patcher_assets_base.start()
        cls.patcher_fonts_base.start()

        cls.create_dummy_files()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        cls.patcher_assets_utils.stop()
        cls.patcher_fonts_utils.stop()
        cls.patcher_assets_base.stop()
        cls.patcher_fonts_base.stop()

    @classmethod
    def create_dummy_files(cls):
        vocab_df = pd.DataFrame({'char': ['t', 'e', 's', 'ト', 'ス', 'テ', 'あ', 'い', 'う', 'え', 'お', 'カ', 'キ', 'ク', 'A', 'B', 'C', '1', '2', '3', '漢', '字', 'v', 'i', 'b', 'l']})
        vocab_df.to_csv(cls.assets_dir / "vocab.csv", index=False)
        len_to_p_df = pd.DataFrame({'len': [4], 'p': [1.0]})
        len_to_p_df.to_csv(cls.assets_dir / "len_to_p.csv", index=False)
        real_font_path = PROJECT_FONTS_ROOT / "NotoSansJP-Regular.ttf"
        temp_font_path = cls.fonts_dir / "NotoSansJP-Regular.ttf"
        if not temp_font_path.exists():
            shutil.copy(real_font_path, temp_font_path)
        fonts_df = pd.DataFrame({'font_path': [temp_font_path.name], 'supported_chars': ['tesトス_テあいうえおABC123漢字vibl'], 'label': ['common']})
        fonts_df.to_csv(cls.assets_dir / "fonts.csv", index=False)

    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
    @patch('manga_ocr_dev.synthetic_data_generator.generator.SyntheticDataGenerator._process')
    def test_process_retry(self, mock_process, mock_renderer):
        """
        Test that process retries on SkipSample exception.
        """
        generator = SyntheticDataGenerator(renderer=mock_renderer)

        # Simulate that _process fails 3 times and then succeeds
        mock_process.side_effect = [
            SkipSample("Test skip 1"),
            (None, "dummy_meta", {}),
            SkipSample("Test skip 3"),
            ("dummy_image", "dummy_meta", {})
        ]

        img, meta, _ = generator.process("test")

        self.assertEqual(mock_process.call_count, 4)
        self.assertEqual(img, "dummy_image")
        self.assertEqual(meta, "dummy_meta")

    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
    @patch('manga_ocr_dev.synthetic_data_generator.generator.SyntheticDataGenerator._process')
    def test_process_fail_after_retries(self, mock_process, mock_renderer):
        """
        Test that process raises SkipSample after 4 failed attempts.
        """
        generator = SyntheticDataGenerator(renderer=mock_renderer)
        mock_process.side_effect = SkipSample("Test skip")

        with self.assertRaises(SkipSample):
            generator.process("test")

        self.assertEqual(mock_process.call_count, 4)

if __name__ == '__main__':
    unittest.main()
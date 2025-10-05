import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import re

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2
import manga_ocr_dev.synthetic_data_generator_v2.generator as generator_module
import manga_ocr_dev.synthetic_data_generator_v2.utils as utils_module
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT


class TestSyntheticDataGeneratorV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.assets_dir = Path(cls.temp_dir) / "assets"
        cls.assets_dir.mkdir()
        cls.fonts_dir = Path(cls.temp_dir) / "fonts"
        cls.fonts_dir.mkdir()
        cls.backgrounds_dir = Path(cls.temp_dir) / "backgrounds"
        cls.backgrounds_dir.mkdir()

        # Create dummy files
        cls.create_dummy_files()

        # Monkey patch the paths
        cls.original_assets_path_generator = generator_module.ASSETS_PATH
        cls.original_fonts_root_generator = generator_module.FONTS_ROOT
        cls.original_assets_path_utils = utils_module.ASSETS_PATH
        cls.original_fonts_root_utils = utils_module.FONTS_ROOT

        generator_module.ASSETS_PATH = cls.assets_dir
        generator_module.FONTS_ROOT = cls.fonts_dir
        utils_module.ASSETS_PATH = cls.assets_dir
        utils_module.FONTS_ROOT = cls.fonts_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

        # Restore original paths
        generator_module.ASSETS_PATH = cls.original_assets_path_generator
        generator_module.FONTS_ROOT = cls.original_fonts_root_generator
        utils_module.ASSETS_PATH = cls.original_assets_path_utils
        utils_module.FONTS_ROOT = cls.original_fonts_root_utils


    @classmethod
    def create_dummy_files(cls):
        # Dummy vocab.csv
        vocab_df = pd.DataFrame({'char': ['あ', 'い', 'う', 'え', 'お', 'A', 'B', 'C', '1', '2', '3', '漢', '字']})
        vocab_df.to_csv(cls.assets_dir / "vocab.csv", index=False)

        # Dummy len_to_p.csv
        len_to_p_df = pd.DataFrame({'len': [1, 2, 3], 'p': [0.3, 0.4, 0.3]})
        len_to_p_df.to_csv(cls.assets_dir / "len_to_p.csv", index=False)

        # Copy a real font to the temp dir to ensure tests are environment-independent
        real_font_path = PROJECT_FONTS_ROOT / "NotoSansJP-Regular.ttf"
        temp_font_path = cls.fonts_dir / "NotoSansJP-Regular.ttf"
        shutil.copy(real_font_path, temp_font_path)

        fonts_df = pd.DataFrame({
            'font_path': [temp_font_path.name],  # Use the relative name of the copied font
            'supported_chars': ['あいうえおABC123漢字tes'],
            'label': ['common']
        })
        fonts_df.to_csv(cls.assets_dir / "fonts.csv", index=False)

        # Dummy background
        dummy_bg = np.zeros((200, 200, 3), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(dummy_bg).save(cls.backgrounds_dir / "dummy_bg_0_100_0_100.png")

    def test_initialization(self):
        """Test that the generator initializes without errors."""
        try:
            generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
            self.assertIsNotNone(generator)
        except Exception as e:
            self.fail(f"SyntheticDataGeneratorV2 initialization failed with {e}")

    def test_process_simple_text(self):
        """Test processing a simple text string."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        img, text_gt, params = generator.process("あいうえお")
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        self.assertEqual(text_gt, "あいうえお")

    def test_process_random_text(self):
        """Test processing with random text generation."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        img, text_gt, params = generator.process()
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        self.assertIsInstance(text_gt, str)

    def test_vertical_rendering(self):
        """Test vertical text rendering."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        img, _, _ = generator.process("あいう", override_params={'vertical': True})
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], img.shape[1])

    def test_horizontal_rendering(self):
        """Test horizontal text rendering."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        img, _, _ = generator.process("あいう", override_params={'vertical': False})
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[1], img.shape[0])

    def test_furigana_rendering(self):
        """Test furigana rendering."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        original_add_random_furigana = generator.add_random_furigana
        def mock_add_random_furigana(line, word_prob, vocab):
            return [('furigana', '漢字', 'かんじ')]
        generator.add_random_furigana = mock_add_random_furigana

        img, _, _ = generator.process("漢字")
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)

        generator.add_random_furigana = original_add_random_furigana

    def test_tcy_rendering(self):
        """Test tate-chū-yoko rendering."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        original_add_random_furigana = generator.add_random_furigana
        def mock_add_random_furigana(line, word_prob, vocab):
            return [('tcy', '12')]
        generator.add_random_furigana = mock_add_random_furigana

        img, _, _ = generator.process("12", override_params={'vertical': True})
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)

        generator.add_random_furigana = original_add_random_furigana

    def test_grayscale_color(self):
        """Test that text is rendered in a grayscale color."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        _, _, params = generator.process("test")
        color = params['color']
        match = re.match(r'#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})', color)
        self.assertTrue(match)
        r, g, b = [int(c, 16) for c in match.groups()]
        self.assertEqual(r, g)
        self.assertEqual(g, b)
        self.assertLessEqual(r, 100)

    def test_font_size_control(self):
        """Test that font size is within the specified range."""
        generator = SyntheticDataGeneratorV2(background_dir=None, min_font_size=40, max_font_size=50)
        _, _, params = generator.process("test")
        font_size = params['font_size']
        self.assertGreaterEqual(font_size, 40)
        self.assertLess(font_size, 50)

    def test_target_size(self):
        """Test that the final image is resized to the target size."""
        target_size = (128, 128)
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir, target_size=target_size)
        img, _, _ = generator.process("test")
        self.assertEqual(img.shape[0], target_size[1])
        self.assertEqual(img.shape[1], target_size[0])

if __name__ == '__main__':
    unittest.main()
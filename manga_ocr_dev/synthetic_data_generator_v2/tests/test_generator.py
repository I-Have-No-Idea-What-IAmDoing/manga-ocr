import sys
from pathlib import Path
import unittest
from unittest.mock import patch
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

        cls.create_dummy_files()

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
        generator_module.ASSETS_PATH = cls.original_assets_path_generator
        generator_module.FONTS_ROOT = cls.original_fonts_root_generator
        utils_module.ASSETS_PATH = cls.original_assets_path_utils
        utils_module.FONTS_ROOT = cls.original_fonts_root_utils

    def setUp(self):
        """Patch augmentations to be disabled to ensure a deterministic test environment."""
        self.compose_patcher = patch('albumentations.Compose', lambda transforms: lambda image: {'image': image})
        self.mock_compose = self.compose_patcher.start()

    def tearDown(self):
        self.compose_patcher.stop()

    @classmethod
    def create_dummy_files(cls):
        vocab_df = pd.DataFrame({'char': ['あ', 'い', 'う', 'え', 'お', 'カ', 'キ', 'ク', 'A', 'B', 'C', '1', '2', '3', '漢', '字', 't', 'e', 's', 'v', 'i', 'b', 'l']})
        vocab_df.to_csv(cls.assets_dir / "vocab.csv", index=False)
        len_to_p_df = pd.DataFrame({'len': [1, 2, 3], 'p': [0.3, 0.4, 0.3]})
        len_to_p_df.to_csv(cls.assets_dir / "len_to_p.csv", index=False)
        real_font_path = PROJECT_FONTS_ROOT / "NotoSansJP-Regular.ttf"
        temp_font_path = cls.fonts_dir / "NotoSansJP-Regular.ttf"
        shutil.copy(real_font_path, temp_font_path)
        fonts_df = pd.DataFrame({
            'font_path': [temp_font_path.name],
            'supported_chars': ['あいうえおABC123漢字tesvibl'],
            'label': ['common']
        })
        fonts_df.to_csv(cls.assets_dir / "fonts.csv", index=False)
        # Use a white background for high contrast by default
        dummy_bg = np.full((200, 200, 3), 255, dtype=np.uint8)
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
        # Use black text for high contrast against the default white background
        img, text_gt, params = generator.process("あいうえお", override_params={'color': '#000000'})
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)
        self.assertGreater(img.shape[1], 0)
        self.assertEqual(text_gt, "あいうえお")

    def test_process_random_text(self):
        """Test processing with random text generation."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        img, text_gt, params = generator.process(override_params={'color': '#000000'})
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
        """Test furigana rendering by forcing it with a mock."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        with unittest.mock.patch('numpy.random.uniform', return_value=0.0):
            img, _, _ = generator.process("漢字")
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)

    def test_tcy_rendering(self):
        """Test tate-chū-yoko rendering by forcing it with a mock."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        with unittest.mock.patch('numpy.random.uniform', return_value=0.0):
            img, _, _ = generator.process("12", override_params={'vertical': True})
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(img.shape[0], 0)

    def test_grayscale_color_bias(self):
        """Test that text is rendered in a grayscale color biased to extremes."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        for _ in range(20):
            _, _, params = generator.process("test")
            color = params['color']
            match = re.match(r'#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})', color)
            self.assertTrue(match, f"Color '{color}' does not match hex format")
            r, g, b = [int(c, 16) for c in match.groups()]
            self.assertEqual(r, g)
            self.assertEqual(g, b)
            is_dark = r <= 40
            is_light = r >= 215
            self.assertTrue(is_dark or is_light, f"Grayscale value {r} is not in the biased extremes")

    def test_cropping_does_not_cut_text(self):
        """Test that the final crop does not cut off the text overlay."""
        black_bg = np.zeros((500, 500, 3), dtype=np.uint8)
        from PIL import Image
        black_bg_path = self.backgrounds_dir / "black_bg_0_500_0_500.png"
        Image.fromarray(black_bg).save(black_bg_path)
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir)
        generator.composer.background_df = pd.DataFrame([{'path': str(black_bg_path)}])
        img, _, _ = generator.process("visible", override_params={'color': '#FFFFFF'})
        self.assertEqual(np.max(img), 255, "The white text seems to be cropped out or altered.")

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
        # Explicitly set font_size to ensure the rendered text is not too small and rejected.
        # A larger font size is used to be certain it passes the min_text_height check.
        img, _, _ = generator.process("test", override_params={'color': '#000000', 'font_size': 50})
        self.assertIsNotNone(img, "generator.process() returned None unexpectedly.")
        self.assertEqual(img.shape[0], target_size[1])
        self.assertEqual(img.shape[1], target_size[0])

    def test_min_output_size(self):
        """Test that the final image is upscaled to the min_output_size."""
        min_size = 300
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir, min_output_size=min_size)
        img, _, _ = generator.process("test", override_params={'color': '#000000'})
        self.assertGreaterEqual(min(img.shape[:2]), min_size)

    @patch('numpy.random.rand', return_value=0.8) # Mock to prevent drawing a bubble
    def test_legibility_check_discards_small_text(self, mock_rand):
        """Test that samples with too small text are discarded."""
        generator = SyntheticDataGeneratorV2(background_dir=self.backgrounds_dir, min_font_size=5, max_font_size=6)
        img, _, _ = generator.process("t", override_params={'color': '#FFFFFF'})
        self.assertIsNone(img, "Sample with very small text was not discarded")

    def test_stroke_effect(self):
        """Test that the stroke effect is applied correctly."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        override_params = {'effect': 'stroke', 'stroke_width': 2, 'stroke_color': '#FF0000'}
        img, _, _ = generator.process("test", override_params=override_params)
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(np.sum(img), 0)

    def test_glow_effect(self):
        """Test that the glow effect is applied correctly."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        override_params = {'effect': 'glow', 'shadow_blur': 5, 'shadow_color': '#00FF00', 'shadow_offset': (2, 2)}
        img, _, _ = generator.process("test", override_params=override_params)
        self.assertIsInstance(img, np.ndarray)
        self.assertGreater(np.sum(img), 0)

    def test_effect_params(self):
        """Test that effect parameters are generated correctly."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        with unittest.mock.patch('numpy.random.choice', return_value='stroke'):
            params = generator.get_random_render_params()
            self.assertEqual(params['effect'], 'stroke')
            self.assertIn('stroke_width', params)
            self.assertIn('stroke_color', params)
        with unittest.mock.patch('numpy.random.choice', return_value='glow'):
            params = generator.get_random_render_params()
            self.assertEqual(params['effect'], 'glow')
            self.assertIn('shadow_blur', params)
            self.assertIn('shadow_color', params)
            self.assertIn('shadow_offset', params)


if __name__ == '__main__':
    unittest.main()
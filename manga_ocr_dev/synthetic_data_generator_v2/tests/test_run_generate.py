import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import tempfile
import shutil
import json

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2
from manga_ocr_dev.synthetic_data_generator_v2.run_generate import worker_fn, run
import manga_ocr_dev.synthetic_data_generator_v2.run_generate as run_generate_module
import manga_ocr_dev.synthetic_data_generator_v2.generator as generator_module
import manga_ocr_dev.synthetic_data_generator_v2.utils as utils_module
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT


class TestRunGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.synthetic_data_root = cls.temp_dir / "synthetic_data"
        cls.assets_dir = cls.temp_dir / "assets"
        cls.fonts_dir = cls.temp_dir / "fonts"

        cls.lines_dir = cls.synthetic_data_root / "lines"
        cls.lines_dir.mkdir(parents=True)
        cls.backgrounds_dir = cls.temp_dir / "backgrounds"  # Use a separate temp dir for backgrounds
        cls.backgrounds_dir.mkdir()
        cls.assets_dir.mkdir(exist_ok=True)
        cls.fonts_dir.mkdir(exist_ok=True)

        # Monkey patch the paths
        cls.original_data_synthetic_root = run_generate_module.DATA_SYNTHETIC_ROOT
        cls.original_background_dir = run_generate_module.BACKGROUND_DIR
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.synthetic_data_root
        run_generate_module.BACKGROUND_DIR = cls.backgrounds_dir

        cls.original_assets_path_generator = generator_module.ASSETS_PATH
        cls.original_fonts_root_generator = generator_module.FONTS_ROOT
        cls.original_assets_path_utils = utils_module.ASSETS_PATH
        cls.original_fonts_root_utils = utils_module.FONTS_ROOT

        generator_module.ASSETS_PATH = cls.assets_dir
        generator_module.FONTS_ROOT = cls.fonts_dir
        utils_module.ASSETS_PATH = cls.assets_dir
        utils_module.FONTS_ROOT = cls.fonts_dir

        cls.create_dummy_files()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

        # Restore original paths
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.original_data_synthetic_root
        generator_module.ASSETS_PATH = cls.original_assets_path_generator
        generator_module.FONTS_ROOT = cls.original_fonts_root_generator
        utils_module.ASSETS_PATH = cls.original_assets_path_utils
        utils_module.FONTS_ROOT = cls.original_fonts_root_utils

    @classmethod
    def create_dummy_files(cls):
        # Dummy lines.csv
        lines_df = pd.DataFrame({'source': ['corpus'], 'id': ['corpus_001'], 'line': ['テスト']})
        lines_df.to_csv(cls.lines_dir / "0000.csv", index=False)

        # Dummy vocab.csv
        vocab_df = pd.DataFrame({'char': ['t', 'e', 's', 'ト', 'ス']})
        vocab_df.to_csv(cls.assets_dir / "vocab.csv", index=False)

        # Dummy len_to_p.csv
        len_to_p_df = pd.DataFrame({'len': [4], 'p': [1.0]})
        len_to_p_df.to_csv(cls.assets_dir / "len_to_p.csv", index=False)

        # Copy a real font to the temp dir
        real_font_path = PROJECT_FONTS_ROOT / "NotoSansJP-Regular.ttf"
        temp_font_path = cls.fonts_dir / "NotoSansJP-Regular.ttf"
        shutil.copy(real_font_path, temp_font_path)

        fonts_df = pd.DataFrame({'font_path': [temp_font_path.name], 'supported_chars': ['tesトス_テ'], 'label': ['common']})
        fonts_df.to_csv(cls.assets_dir / "fonts.csv", index=False)

        # Dummy background (white)
        dummy_bg = np.full((200, 200, 3), 255, dtype=np.uint8)
        from PIL import Image
        Image.fromarray(dummy_bg).save(cls.backgrounds_dir / "dummy_bg_0_200_0_200.png")

    def test_worker_fn_debug_mode(self):
        """Test that the worker function correctly saves debug info."""
        temp_out_dir = self.temp_dir / "out_worker"
        temp_out_dir.mkdir()
        temp_debug_dir = self.temp_dir / "debug_worker"
        temp_debug_dir.mkdir()

        run_generate_module.OUT_DIR = temp_out_dir
        run_generate_module.DEBUG_DIR = temp_debug_dir

        generator = SyntheticDataGeneratorV2(background_dir=None)
        args = (0, 'test_source', 'test_id_123', 'test')

        worker_fn(args, generator, debug=True)

        debug_file = temp_debug_dir / "test_id_123.json"
        self.assertTrue(debug_file.exists())

        with open(debug_file, 'r') as f:
            debug_data = json.load(f)

        self.assertIn('font_path', debug_data)
        self.assertIsInstance(debug_data['font_path'], str)

    @patch('manga_ocr_dev.synthetic_data_generator_v2.run_generate.thread_map', side_effect=lambda func, args, **kwargs: [func(arg) for arg in args])
    def test_run_creates_output_files(self, mock_thread_map):
        """Test that the main run function creates output images and metadata."""
        run(package=0, n_random=1, n_limit=2)

        output_img_dir = self.synthetic_data_root / "img_v2" / "0000"
        output_meta_dir = self.synthetic_data_root / "meta_v2"

        self.assertTrue(output_img_dir.exists())
        self.assertTrue(output_meta_dir.exists())

        # Should be 2 images (1 from corpus, 1 random)
        self.assertEqual(len(list(output_img_dir.glob('*.jpg'))), 2)

        meta_file = output_meta_dir / "0000.csv"
        self.assertTrue(meta_file.exists())

        df = pd.read_csv(meta_file)
        self.assertEqual(len(df), 2)

if __name__ == '__main__':
    unittest.main()
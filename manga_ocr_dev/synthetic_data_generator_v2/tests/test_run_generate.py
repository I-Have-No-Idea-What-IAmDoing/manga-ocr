import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2
from manga_ocr_dev.synthetic_data_generator_v2.run_generate import worker_fn, NumpyEncoder
import manga_ocr_dev.synthetic_data_generator_v2.run_generate as run_generate_module
import manga_ocr_dev.synthetic_data_generator_v2.generator as generator_module
import manga_ocr_dev.synthetic_data_generator_v2.utils as utils_module
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT


class TestRunGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.assets_dir = Path(cls.temp_dir) / "assets"
        cls.assets_dir.mkdir()
        cls.fonts_dir = Path(cls.temp_dir) / "fonts"
        cls.fonts_dir.mkdir()
        cls.out_dir = Path(cls.temp_dir) / "out"
        cls.out_dir.mkdir()
        cls.debug_dir = Path(cls.temp_dir) / "debug"
        cls.debug_dir.mkdir()

        # Create dummy files
        cls.create_dummy_files()

        # Monkey patch the paths
        cls.original_assets_path_generator = generator_module.ASSETS_PATH
        cls.original_fonts_root_generator = generator_module.FONTS_ROOT
        cls.original_assets_path_utils = utils_module.ASSETS_PATH
        cls.original_fonts_root_utils = utils_module.FONTS_ROOT

        run_generate_module.OUT_DIR = cls.out_dir
        run_generate_module.DEBUG_DIR = cls.debug_dir
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
        vocab_df = pd.DataFrame({'char': ['t', 'e', 's']})
        vocab_df.to_csv(cls.assets_dir / "vocab.csv", index=False)

        # Dummy len_to_p.csv
        len_to_p_df = pd.DataFrame({'len': [4], 'p': [1.0]})
        len_to_p_df.to_csv(cls.assets_dir / "len_to_p.csv", index=False)

        # Copy a real font to the temp dir to ensure tests are environment-independent
        real_font_path = PROJECT_FONTS_ROOT / "NotoSansJP-Regular.ttf"
        temp_font_path = cls.fonts_dir / "NotoSansJP-Regular.ttf"
        shutil.copy(real_font_path, temp_font_path)

        fonts_df = pd.DataFrame({
            'font_path': [temp_font_path.name],
            'supported_chars': ['tes'],
            'label': ['common']
        })
        fonts_df.to_csv(cls.assets_dir / "fonts.csv", index=False)

    def test_worker_fn_debug_mode(self):
        """Test that the worker function correctly saves debug info."""
        generator = SyntheticDataGeneratorV2(background_dir=None)
        args = (0, 'test_source', 'test_id_123', 'test')

        worker_fn(args, generator, debug=True)

        debug_file = self.debug_dir / "test_id_123.json"
        self.assertTrue(debug_file.exists())

        with open(debug_file, 'r') as f:
            debug_data = json.load(f)

        self.assertIn('font_path', debug_data)
        self.assertIsInstance(debug_data['font_path'], str)

if __name__ == '__main__':
    unittest.main()
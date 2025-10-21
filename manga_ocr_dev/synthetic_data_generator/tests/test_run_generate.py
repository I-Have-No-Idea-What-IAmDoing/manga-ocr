import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import tempfile
import shutil
import json

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator.run_generate import run, worker_fn
import manga_ocr_dev.synthetic_data_generator.run_generate as run_generate_module
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2


class TestRunGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.synthetic_data_root = cls.temp_dir / "synthetic_data"
        cls.assets_dir = cls.temp_dir / "assets"
        cls.fonts_dir = cls.temp_dir / "fonts"

        cls.lines_dir = cls.synthetic_data_root / "lines"
        cls.lines_dir.mkdir(parents=True)
        cls.backgrounds_dir = cls.temp_dir / "backgrounds"
        cls.backgrounds_dir.mkdir()
        cls.assets_dir.mkdir(exist_ok=True)
        cls.fonts_dir.mkdir(exist_ok=True)

        # Monkey patch the paths in the run script itself
        cls.original_data_synthetic_root = run_generate_module.DATA_SYNTHETIC_ROOT
        cls.original_background_dir = run_generate_module.BACKGROUND_DIR
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.synthetic_data_root
        run_generate_module.BACKGROUND_DIR = cls.backgrounds_dir

        cls.patcher_assets_utils = patch('manga_ocr_dev.synthetic_data_generator.common.utils.ASSETS_PATH', cls.assets_dir)
        cls.patcher_fonts_utils = patch('manga_ocr_dev.synthetic_data_generator.common.utils.FONTS_ROOT', cls.fonts_dir)
        cls.patcher_assets_base = patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.ASSETS_PATH', cls.assets_dir)
        cls.patcher_fonts_base = patch('manga_ocr_dev.synthetic_data_generator.common.base_generator.FONTS_ROOT', cls.fonts_dir)

        cls.patcher_assets_utils.start()
        cls.patcher_fonts_utils.start()
        cls.patcher_assets_base.start()
        cls.patcher_fonts_base.start()

        cls.create_dummy_files()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        # Restore original paths
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.original_data_synthetic_root
        run_generate_module.BACKGROUND_DIR = cls.original_background_dir
        cls.patcher_assets_utils.stop()
        cls.patcher_fonts_utils.stop()
        cls.patcher_assets_base.stop()
        cls.patcher_fonts_base.stop()

    @classmethod
    def create_dummy_files(cls):
        lines_df = pd.DataFrame({'source': ['corpus'], 'id': ['corpus_001'], 'line': ['テスト']})
        lines_df.to_csv(cls.lines_dir / "0000.csv", index=False)
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
        dummy_bg = np.full((200, 200, 3), 255, dtype=np.uint8)
        from PIL import Image
        Image.fromarray(dummy_bg).save(cls.backgrounds_dir / "dummy_bg_0_200_0_200.png")

    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.worker_fn')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map', side_effect=lambda func, args, **kwargs: [func(arg) for arg in args])
    @patch('manga_ocr_dev.synthetic_data_generator.common.composer.Composer._is_low_contrast', return_value=False)
    def test_run_pictex(self, mock_is_low_contrast, mock_thread_map, mock_worker_fn):
        """Test that the main run function creates output files for the pictex renderer."""
        output_img_dir = self.synthetic_data_root / "img" / "0000"
        output_meta_dir = self.synthetic_data_root / "meta"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = output_meta_dir / "0000.csv"

        def side_effect(args, generator, renderer_type, debug):
            i, source, id_, text = args
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img_path = output_img_dir / f"{id_}.jpg"
            from PIL import Image
            Image.fromarray(img).save(img_path)

            with open(meta_file, 'a') as f:
                f.write(f'{source},{id_},{text},{False},dummy.ttf\n')

            return source, id_, text, False, 'dummy.ttf'

        mock_worker_fn.side_effect = side_effect

        with open(meta_file, 'w') as f:
            f.write('source,id,text,vertical,font_path\n')

        run(renderer='pictex', package=0, n_random=1, n_limit=2, max_workers=1)
        self.assertTrue(output_img_dir.exists())
        self.assertTrue(output_meta_dir.exists())
        self.assertEqual(len(list(output_img_dir.glob('*.jpg'))), 2)
        self.assertTrue(meta_file.exists())
        df = pd.read_csv(meta_file)
        self.assertEqual(len(df), 2)

    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.worker_fn')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map', side_effect=lambda func, args, **kwargs: [func(arg) for arg in args])
    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
    def test_run_html(self, MockRenderer, mock_thread_map, mock_worker_fn):
        """Test that the main run function creates output files for the html renderer."""

        output_img_dir = self.synthetic_data_root / "img" / "0000"
        output_meta_dir = self.synthetic_data_root / "meta"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = output_meta_dir / "0000.csv"

        def side_effect(args, generator, renderer_type, debug):
            i, source, id_, text = args
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img_path = output_img_dir / f"{id_}.jpg"
            from PIL import Image
            Image.fromarray(img).save(img_path)

            with open(meta_file, 'a') as f:
                f.write(f'{source},{id_},{text},{False},dummy.ttf\n')

            return source, id_, text, False, 'dummy.ttf'

        mock_worker_fn.side_effect = side_effect
        mock_renderer_instance = MagicMock()
        MockRenderer.return_value.__enter__.return_value = mock_renderer_instance

        with open(meta_file, 'w') as f:
            f.write('source,id,text,vertical,font_path\n')

        run(renderer='html', package=0, n_random=1, n_limit=2, max_workers=1)

        self.assertTrue(output_img_dir.exists())
        self.assertTrue(output_meta_dir.exists())
        self.assertEqual(len(list(output_img_dir.glob('*.jpg'))), 2)
        self.assertTrue(meta_file.exists())
        df = pd.read_csv(meta_file)
        self.assertEqual(len(df), 2)

    def test_worker_fn_debug_mode_pictex(self):
        """Test that the worker function correctly saves debug info for pictex."""
        temp_out_dir = self.temp_dir / "out_worker_pictex"
        temp_out_dir.mkdir()
        temp_debug_dir = self.temp_dir / "debug_worker_pictex"
        temp_debug_dir.mkdir()

        run_generate_module.OUT_DIR = temp_out_dir
        run_generate_module.DEBUG_DIR = temp_debug_dir

        generator = SyntheticDataGeneratorV2(background_dir=None)
        args = (0, 'test_source', 'test_id_123', 'test')

        worker_fn(args, generator, renderer_type='pictex', debug=True)

        debug_file = temp_debug_dir / "test_id_123.json"
        self.assertTrue(debug_file.exists())
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
        self.assertIn('font_path', debug_data)
        self.assertIsInstance(debug_data['font_path'], str)

    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
    def test_worker_fn_debug_mode_html(self, MockRenderer):
        """Test that the worker function correctly saves debug info for html."""
        temp_out_dir = self.temp_dir / "out_worker_html"
        temp_out_dir.mkdir()
        temp_debug_dir = self.temp_dir / "debug_worker_html"
        temp_debug_dir.mkdir()

        run_generate_module.OUT_DIR = temp_out_dir
        run_generate_module.DEBUG_DIR = temp_debug_dir

        mock_renderer_instance = MagicMock()
        dummy_img = np.zeros((100, 100, 4), dtype=np.uint8)
        dummy_img[:, :, 3] = 255
        mock_renderer_instance.render.return_value = (dummy_img, {'vertical': False, 'font_path': 'dummy.ttf', 'html': '<html></html>', 'text_color': '#000000'})

        generator = SyntheticDataGenerator(background_dir=None, renderer=mock_renderer_instance)
        args = (0, 'test_source', 'test_id_456', 'test')

        worker_fn(args, generator, renderer_type='html', debug=True)

        debug_file_json = temp_debug_dir / "test_id_456.json"
        debug_file_html = temp_debug_dir / "test_id_456.html"
        self.assertTrue(debug_file_json.exists())
        self.assertTrue(debug_file_html.exists())
        with open(debug_file_json, 'r') as f:
            debug_data = json.load(f)
        self.assertIn('font_path', debug_data)


if __name__ == '__main__':
    unittest.main()

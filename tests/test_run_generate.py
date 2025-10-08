"""Tests for the main synthetic data generation script.

This module contains tests for the `run_generate.py` script, which orchestrates
the entire synthetic data generation process. The tests cover the parallel
worker function and the main `run` function, ensuring that data is processed
correctly and that exceptions are handled properly.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import unittest
import tempfile
import shutil
import json

from manga_ocr_dev.synthetic_data_generator.run_generate import worker_fn, run
import manga_ocr_dev.synthetic_data_generator.run_generate as run_generate_module
from manga_ocr_dev.env import FONTS_ROOT as PROJECT_FONTS_ROOT


class TestRunGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.synthetic_data_root = cls.temp_dir / "synthetic_data"
        cls.lines_dir = cls.synthetic_data_root / "lines"
        cls.lines_dir.mkdir(parents=True)

        # Create dummy lines file
        lines_df = pd.DataFrame({'source': ['corpus'], 'id': ['corpus_001'], 'line': ['テスト']})
        lines_df.to_csv(cls.lines_dir / "0000.csv", index=False)

        # Monkey patch the DATA_SYNTHETIC_ROOT in the run script
        cls.original_data_synthetic_root = run_generate_module.DATA_SYNTHETIC_ROOT
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.synthetic_data_root

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)
        run_generate_module.DATA_SYNTHETIC_ROOT = cls.original_data_synthetic_root

    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
    def test_worker_fn(self, mock_imwrite):
        mock_generator = MagicMock()
        mock_generator.process.return_value = (
            MagicMock(), 'test_text', {'font_path': 'dummy.ttf', 'vertical': True}
        )
        with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
            args = (0, 'source', 'id_001', 'text')
            result = worker_fn(args, mock_generator, 'html')
        mock_generator.process.assert_called_once_with('text')
        mock_imwrite.assert_called_once()
        self.assertEqual(result, ('source', 'id_001', 'test_text', True, 'dummy.ttf'))

    @patch('pathlib.Path.exists', return_value=True)
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.DataFrame.to_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.Path.mkdir')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.SyntheticDataGeneratorV2')
    def test_run_pictex(self, mock_gen_v2, mock_thread_map, mock_mkdir, mock_to_csv, mock_exists):
        mock_thread_map.return_value = [('test_source', 'test_id', 'test_line', True, 'dummy.ttf')]
        run(renderer='pictex', package=0, n_random=0, n_limit=1, max_workers=1)
        mock_gen_v2.assert_called_once()
        mock_thread_map.assert_called_once()
        mock_to_csv.assert_called_once()

    @patch('pathlib.Path.exists', return_value=True)
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.pd.DataFrame.to_csv')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.Path.mkdir')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.thread_map')
    @patch('manga_ocr_dev.synthetic_data_generator.renderer.Renderer')
    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.SyntheticDataGenerator')
    def test_run_html(self, mock_gen_v1, mock_renderer, mock_thread_map, mock_mkdir, mock_to_csv, mock_exists):
        mock_thread_map.return_value = [('test_source', 'test_id', 'test_line', True, 'dummy.ttf')]
        run(renderer='html', package=0, n_random=0, n_limit=1, max_workers=1)
        mock_renderer.assert_called_once()
        mock_gen_v1.assert_called_once()
        mock_thread_map.assert_called_once()
        mock_to_csv.assert_called_once()

    @patch('manga_ocr_dev.synthetic_data_generator.run_generate.cv2.imwrite')
    @patch('builtins.print')
    def test_worker_fn_exception_handling(self, mock_print, mock_imwrite):
        mock_generator = MagicMock()
        mock_generator.process.side_effect = Exception("Test exception")
        with patch('manga_ocr_dev.synthetic_data_generator.run_generate.OUT_DIR', Path('/dummy/out')):
            args = (0, 'source', 'id_001', 'text')
            with pytest.raises(Exception, match="Test exception"):
                worker_fn(args, mock_generator, 'html')
        mock_print.assert_called()

if __name__ == '__main__':
    unittest.main()
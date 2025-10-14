"""Tests for the MangaDataset class.

This module contains unit tests for the `MangaDataset` class, which is
responsible for loading, augmenting, and preparing data for training the OCR
model. The tests cover the dataset's initialization, data loading from
different sources, and the `__getitem__` method's functionality.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import torch
import numpy as np
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.config.schemas import DatasetConfig, DatasetSourceConfig, DatasetTrainConfig, DatasetEvalConfig, AugmentationConfig, AugmentationProbabilities

class TestMangaDataset(unittest.TestCase):
    """A test suite for the MangaDataset class."""

    def setUp(self):
        """Sets up a mocked environment for testing the MangaDataset.

        This method is called before each test. It initializes a `MangaDataset`
        instance with all its external dependencies mocked. This provides a
        clean, isolated environment for each test case, with a mock processor
        and a sample dataset configuration.
        """
        self.processor = MagicMock()
        self.processor.tokenizer = MagicMock()
        self.processor.tokenizer.pad_token_id = 0
        self.processor.tokenizer.decode.return_value = 'test'
        self.processor.tokenizer.return_value = MagicMock(input_ids=[1, 2, 3])
        self.processor.feature_extractor = MagicMock(return_value=MagicMock(pixel_values=torch.randn(3, 224, 224)))
        self.max_target_length = 128

        self.dataset_config = DatasetConfig(
            train=DatasetTrainConfig(sources=[
                DatasetSourceConfig(type='synthetic', params={'packages': [1]}),
                DatasetSourceConfig(type='manga109', params={'split': 'train'})
            ]),
            eval=DatasetEvalConfig(sources=[]),
            augmentations=AugmentationConfig(
                medium=[],
                heavy=[],
                probabilities=AugmentationProbabilities(medium=0.5, heavy=0.5)
            ),
            augment=True
        )

    @patch('manga_ocr_dev.training.dataset.pd.read_csv')
    @patch('manga_ocr_dev.training.dataset.Path.is_dir', return_value=True)
    def test_initialization(self, mock_is_dir, mock_read_csv):
        """Tests that the dataset initializes correctly and loads data from all sources."""
        mock_read_csv.return_value = pd.DataFrame({'id': ['1'], 'text': ['test'], 'crop_path': ['path/to/crop'], 'split': ['train']})
        dataset = MangaDataset(self.processor, self.dataset_config, self.max_target_length)
        self.assertIsInstance(dataset, MangaDataset)
        self.assertEqual(len(dataset.data), 2)

    @patch('manga_ocr_dev.training.dataset.pd.read_csv')
    @patch('manga_ocr_dev.training.dataset.Path.is_dir', return_value=True)
    def test_load_synthetic_data(self, mock_is_dir, mock_read_csv):
        """Tests the loading of synthetic data from specified packages."""
        mock_read_csv.return_value = pd.DataFrame({'id': ['1'], 'text': ['test'], 'crop_path': ['path/to/crop'], 'split': ['train']})
        dataset = MangaDataset(self.processor, self.dataset_config, self.max_target_length)
        synthetic_data = dataset.load_synthetic_data(packages=[1])
        self.assertEqual(len(synthetic_data), 1)
        self.assertEqual(synthetic_data.iloc[0]['text'], 'test')

    @patch('manga_ocr_dev.training.dataset.pd.read_csv')
    def test_load_manga109_data(self, mock_read_csv):
        """Tests the loading of data from the Manga109 dataset for a specific split."""
        mock_read_csv.return_value = pd.DataFrame({'text': ['test'], 'crop_path': ['path/to/crop'], 'split': ['train']})
        dataset = MangaDataset(self.processor, self.dataset_config, self.max_target_length)
        manga109_data = dataset.load_manga109_data(split='train')
        self.assertEqual(len(manga109_data), 1)
        self.assertEqual(manga109_data.iloc[0]['text'], 'test')

    @patch('manga_ocr_dev.training.dataset.cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    @patch('manga_ocr_dev.training.dataset.pd.read_csv')
    @patch('manga_ocr_dev.training.dataset.Path.is_dir', return_value=True)
    def test_getitem(self, mock_is_dir, mock_read_csv, mock_imread):
        """Tests the __getitem__ method for correct data preprocessing and augmentation."""
        mock_read_csv.return_value = pd.DataFrame({'id': ['1'], 'text': ['test'], 'crop_path': ['path/to/crop'], 'split': ['train']})
        dataset = MangaDataset(self.processor, self.dataset_config, self.max_target_length)
        item = dataset[0]
        self.assertIn('pixel_values', item)
        self.assertIn('labels', item)
        self.assertEqual(item['pixel_values'].shape, (3, 224, 224))
        self.assertEqual(item['labels'].shape, (3,))

    @patch('manga_ocr_dev.training.dataset.pd.read_csv')
    @patch('manga_ocr_dev.training.dataset.Path.is_dir', return_value=True)
    def test_disable_augmentations(self, mock_is_dir, mock_read_csv):
        """Tests that the `disable_augmentations` method correctly turns off augmentation."""
        mock_read_csv.return_value = pd.DataFrame({'id': ['1'], 'text': ['test'], 'crop_path': ['path/to/crop'], 'split': ['train']})
        dataset = MangaDataset(self.processor, self.dataset_config, self.max_target_length)
        dataset.disable_augmentations()
        self.assertFalse(dataset.config.augment)


if __name__ == '__main__':
    unittest.main()
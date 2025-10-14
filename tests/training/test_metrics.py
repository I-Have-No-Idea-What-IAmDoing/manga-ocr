"""Tests for the evaluation metrics computation.

This module contains unit tests for the `Metrics` class, which is responsible
for calculating and logging evaluation metrics such as Character Error Rate
(CER) and accuracy during the model training process.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from manga_ocr_dev.training.metrics import Metrics

class TestMetrics(unittest.TestCase):
    """A test suite for the `Metrics` class."""

    def setUp(self):
        """Sets up a mocked environment and common objects for all tests.

        This method is called before each test and prepares a standard set of
        mocks for the processor, CER metric, and the `pred` object that is
        passed to the `compute_metrics` method. This ensures a consistent and
        controlled environment for testing.
        """
        # Mock for the processor and its tokenizer
        self.mock_processor = MagicMock()
        self.mock_processor.tokenizer.pad_token_id = 0
        self.mock_processor.tokenizer.batch_decode.side_effect = [
            ["pred 1", "pred 2"],  # Decoded predictions
            ["label 1", "label 2"]  # Decoded labels
        ]

        # Mock for the CER metric
        self.mock_cer_metric = MagicMock()
        self.mock_cer_metric.compute.return_value = 0.5

        # Mock for the pred object
        self.mock_pred = MagicMock()
        self.mock_pred.predictions = np.array([[1], [2]])
        self.mock_pred.label_ids = np.array([[-100, 1], [2, -100]])
        self.mock_pred.inputs = np.random.rand(2, 3, 224, 224) # Mock pixel values

    @patch('manga_ocr_dev.training.metrics.evaluate.load')
    def test_init(self, mock_evaluate_load):
        """Tests that the Metrics class initializes correctly."""
        mock_evaluate_load.return_value = self.mock_cer_metric
        metrics = Metrics(self.mock_processor)
        mock_evaluate_load.assert_called_once_with("cer")
        self.assertEqual(metrics.cer_metric, self.mock_cer_metric)

    @patch('manga_ocr_dev.training.metrics.wandb')
    @patch('manga_ocr_dev.training.metrics.tensor_to_image')
    @patch('manga_ocr_dev.training.metrics.evaluate.load')
    def test_compute_metrics(self, mock_evaluate_load, mock_tensor_to_image, mock_wandb):
        """Tests the core logic of `compute_metrics`, including CER, accuracy, and wandb logging."""
        # Setup mocks
        mock_evaluate_load.return_value = self.mock_cer_metric
        mock_tensor_to_image.return_value = "mock_image"

        # Initialize Metrics and call compute_metrics
        metrics = Metrics(self.mock_processor)
        results = metrics.compute_metrics(self.mock_pred)

        # Assertions for metric calculation
        self.assertEqual(results['cer'], 0.5)
        self.assertEqual(results['accuracy'], 0.0) # "pred1" != "label1"

        # Check that CER was computed with normalized strings
        self.mock_cer_metric.compute.assert_called_once()
        args, kwargs = self.mock_cer_metric.compute.call_args
        self.assertTrue(np.array_equal(kwargs['predictions'], np.array(["pred1", "pred2"])))
        self.assertTrue(np.array_equal(kwargs['references'], np.array(["label1", "label2"])))


        # Assertions for wandb logging
        self.assertEqual(mock_tensor_to_image.call_count, 2)
        mock_wandb.log.assert_called_once()
        log_args = mock_wandb.log.call_args[0][0]
        self.assertIn("eval/samples", log_args)
        self.assertEqual(len(log_args["eval/samples"]), 2)

    @patch('manga_ocr_dev.training.metrics.wandb')
    @patch('manga_ocr_dev.training.metrics.tensor_to_image')
    @patch('manga_ocr_dev.training.metrics.evaluate.load')
    def test_compute_metrics_no_inputs(self, mock_evaluate_load, mock_tensor_to_image, mock_wandb):
        """Tests that wandb logging is correctly skipped when inputs are `None`."""
        mock_evaluate_load.return_value = self.mock_cer_metric
        self.mock_pred.inputs = None

        metrics = Metrics(self.mock_processor)
        metrics.compute_metrics(self.mock_pred)

        mock_tensor_to_image.assert_not_called()
        mock_wandb.log.assert_not_called()

    @patch('manga_ocr_dev.training.metrics.wandb')
    @patch('manga_ocr_dev.training.metrics.tensor_to_image')
    @patch('manga_ocr_dev.training.metrics.evaluate.load')
    def test_compute_metrics_cer_error(self, mock_evaluate_load, mock_tensor_to_image, mock_wandb):
        """Tests that the CER is gracefully set to 0 if the computation fails."""
        self.mock_cer_metric.compute.side_effect = Exception("CER Error")
        mock_evaluate_load.return_value = self.mock_cer_metric

        metrics = Metrics(self.mock_processor)
        results = metrics.compute_metrics(self.mock_pred)

        self.assertEqual(results['cer'], 0)
        self.assertEqual(results['accuracy'], 0.0)
        mock_wandb.log.assert_called_once()
        mock_tensor_to_image.assert_called()
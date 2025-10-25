"""Tests for the model construction script.

This module contains unit tests for the `get_model.py` script, which is
responsible for constructing the `VisionEncoderDecoderModel` and its
corresponding processor. The tests cover the model's initialization,
configuration, and layer truncation logic.
"""

import unittest
from unittest.mock import patch, MagicMock
from pydantic_settings import SettingsConfigDict
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.config.schemas import AppConfig

# Underscore prefix to prevent pytest from collecting it as a test class
class _TestAppConfig:
    def __init__(self):
        self.app = MagicMock()

class TestGetModel(unittest.TestCase):
    """A test suite for the `get_model` function."""

    def _setup_mocks(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM):
        """Helper function to set up common mocks for the tests in this suite.

        This function configures a standard set of mocks for the Hugging Face
        classes that are used by `get_model`. This helps to reduce code
        duplication across the test methods.

        Args:
            MockAutoImageProcessor: Mock for `AutoImageProcessor`.
            MockAutoTokenizer: Mock for `AutoTokenizer`.
            MockAutoConfig: Mock for `AutoConfig`.
            MockAutoModel: Mock for `AutoModel`.
            MockAutoModelForCausalLM: Mock for `AutoModelForCausalLM`.

        Returns:
            A tuple containing the mocked encoder config, decoder config, and
            decoder model, which can be used for further assertions.
        """
        MockAutoImageProcessor.from_pretrained.return_value = MagicMock()

        mock_tok = MagicMock()
        mock_tok.cls_token_id = 101
        mock_tok.pad_token_id = 0
        mock_tok.sep_token_id = 102
        MockAutoTokenizer.from_pretrained.return_value = mock_tok

        encoder = MagicMock()
        encoder.config.to_dict.return_value = {'model_type': 'bert'}
        MockAutoModel.from_pretrained.return_value = encoder

        decoder_config = MagicMock()
        decoder_config.model_type = 'bert'
        decoder_config.to_dict.return_value = {'model_type': 'bert'}

        MockAutoConfig.from_pretrained.return_value = decoder_config

        mock_decoder = MagicMock()
        mock_decoder.bert.encoder.layer = [1, 2, 3, 4, 5, 6]  # A list of layers
        mock_decoder.config = decoder_config
        MockAutoModelForCausalLM.from_pretrained.return_value = mock_decoder

        return encoder.config, decoder_config, mock_decoder

    @patch('manga_ocr_dev.training.get_model.VisionEncoderDecoderModel')
    @patch('manga_ocr_dev.training.get_model.AutoModelForCausalLM')
    @patch('manga_ocr_dev.training.get_model.AutoModel')
    @patch('manga_ocr_dev.training.get_model.AutoConfig')
    @patch('manga_ocr_dev.training.get_model.AutoTokenizer')
    @patch('manga_ocr_dev.training.get_model.AutoImageProcessor')
    def test_get_model(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM, MockVLM):
        """Tests that `get_model` correctly constructs a model and processor."""
        _, decoder_config, _ = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)

        config = _TestAppConfig()
        config.app.model.encoder_name = "google/vit-base-patch16-224-in21k"
        config.app.model.decoder_name = "bert-base-uncased"
        model, processor = get_model(config.app.model)

        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)
        MockAutoImageProcessor.from_pretrained.assert_called_once_with(config.app.model.encoder_name, use_fast=True)
        MockAutoTokenizer.from_pretrained.assert_called_once_with(config.app.model.decoder_name)
        MockAutoModel.from_pretrained.assert_called_once_with(config.app.model.encoder_name)
        MockAutoModelForCausalLM.from_pretrained.assert_called_once_with(
            config.app.model.decoder_name, config=decoder_config
        )
        self.assertTrue(decoder_config.add_cross_attention)

    @patch('manga_ocr_dev.training.get_model.VisionEncoderDecoderModel')
    @patch('manga_ocr_dev.training.get_model.AutoModelForCausalLM')
    @patch('manga_ocr_dev.training.get_model.AutoModel')
    @patch('manga_ocr_dev.training.get_model.AutoConfig')
    @patch('manga_ocr_dev.training.get_model.AutoTokenizer')
    @patch('manga_ocr_dev.training.get_model.AutoImageProcessor')
    def test_get_model_with_layer_truncation(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM, mock_ved_model):
        """Tests that `get_model` correctly truncates the decoder's layers when configured."""
        _, decoder_config, mock_decoder = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)

        config = _TestAppConfig()
        config.app.model.num_decoder_layers = 2

        model, processor = get_model(config.app.model)

        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)
        self.assertEqual(len(mock_decoder.bert.encoder.layer), 2)
        self.assertEqual(decoder_config.num_hidden_layers, 2)

    @patch('manga_ocr_dev.training.get_model.VisionEncoderDecoderModel')
    @patch('manga_ocr_dev.training.get_model.AutoModelForCausalLM')
    @patch('manga_ocr_dev.training.get_model.AutoModel')
    @patch('manga_ocr_dev.training.get_model.AutoConfig')
    @patch('manga_ocr_dev.training.get_model.AutoTokenizer')
    @patch('manga_ocr_dev.training.get_model.AutoImageProcessor')
    def test_get_model_with_unsupported_truncation(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM, mock_ved_model):
        """Tests that `get_model` raises a `ValueError` for an unsupported model type during truncation."""
        _, decoder_config, _ = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)
        decoder_config.model_type = 'unsupported'

        config = _TestAppConfig()
        config.app.model.num_decoder_layers = 2

        with self.assertRaisesRegex(ValueError, "Unsupported model_type for layer truncation: unsupported"):
            get_model(config.app.model)

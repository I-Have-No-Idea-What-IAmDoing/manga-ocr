import unittest
from unittest.mock import patch, MagicMock
from pydantic_settings import SettingsConfigDict
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.config.schemas import AppConfig

# Underscore prefix to prevent pytest from collecting it as a test class
class _TestAppConfig(AppConfig):
    model_config = SettingsConfigDict(
        yaml_file='manga_ocr_dev/training/config.yaml',
        env_file_encoding="utf-8",
        extra="ignore",
    )

class TestGetModel(unittest.TestCase):
    def _setup_mocks(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM):
        """Helper function to set up common mocks for tests."""
        MockAutoImageProcessor.from_pretrained.return_value = MagicMock()

        mock_tok = MagicMock()
        mock_tok.cls_token_id = 101
        mock_tok.pad_token_id = 0
        mock_tok.sep_token_id = 102
        MockAutoTokenizer.from_pretrained.return_value = mock_tok

        encoder_config = MagicMock()
        encoder_config.to_dict.return_value = {'model_type': 'bert'}

        decoder_config = MagicMock()
        decoder_config.model_type = 'bert'
        decoder_config.to_dict.return_value = {'model_type': 'bert'}

        MockAutoConfig.from_pretrained.side_effect = [encoder_config, decoder_config]

        mock_decoder = MagicMock()
        mock_decoder.bert.encoder.layer = [1, 2, 3, 4, 5, 6]  # A list of layers
        MockAutoModelForCausalLM.from_config.return_value = mock_decoder

        MockAutoModel.from_config.return_value = MagicMock()

        return encoder_config, decoder_config, mock_decoder

    @patch('manga_ocr_dev.training.get_model.VisionEncoderDecoderModel')
    @patch('manga_ocr_dev.training.get_model.AutoModelForCausalLM')
    @patch('manga_ocr_dev.training.get_model.AutoModel')
    @patch('manga_ocr_dev.training.get_model.AutoConfig')
    @patch('manga_ocr_dev.training.get_model.AutoTokenizer')
    @patch('manga_ocr_dev.training.get_model.AutoImageProcessor')
    def test_get_model(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM, MockVLM):
        """Test that get_model correctly constructs a model and processor."""
        encoder_config, _, _ = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)

        config = _TestAppConfig()
        model, processor = get_model(config.model)

        self.assertIsNotNone(model)
        self.assertIsNotNone(processor)
        MockAutoImageProcessor.from_pretrained.assert_called_once_with(config.model.encoder_name, use_fast=True)
        MockAutoTokenizer.from_pretrained.assert_called_once_with(config.model.decoder_name)
        self.assertEqual(MockAutoConfig.from_pretrained.call_count, 2)
        MockAutoModel.from_config.assert_called_once_with(encoder_config)

    @patch('manga_ocr_dev.training.get_model.VisionEncoderDecoderModel')
    @patch('manga_ocr_dev.training.get_model.AutoModelForCausalLM')
    @patch('manga_ocr_dev.training.get_model.AutoModel')
    @patch('manga_ocr_dev.training.get_model.AutoConfig')
    @patch('manga_ocr_dev.training.get_model.AutoTokenizer')
    @patch('manga_ocr_dev.training.get_model.AutoImageProcessor')
    def test_get_model_with_layer_truncation(self, MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM, mock_ved_model):
        """Test that get_model correctly truncates decoder layers."""
        _, decoder_config, mock_decoder = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)

        config = _TestAppConfig()
        config.model.num_decoder_layers = 2

        model, processor = get_model(config.model)

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
        """Test that get_model raises ValueError for unsupported model type for truncation."""
        _, decoder_config, _ = self._setup_mocks(MockAutoImageProcessor, MockAutoTokenizer, MockAutoConfig, MockAutoModel, MockAutoModelForCausalLM)
        decoder_config.model_type = 'unsupported'

        config = _TestAppConfig()
        config.model.num_decoder_layers = 2

        with self.assertRaisesRegex(ValueError, "Unsupported model_type for layer truncation: unsupported"):
            get_model(config.model)
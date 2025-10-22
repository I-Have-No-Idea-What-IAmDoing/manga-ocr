"""Tests for the Pydantic configuration schemas.

This module contains unit tests for the Pydantic models defined in
`manga_ocr_dev.training.config.schemas`. These tests ensure that the
configuration schemas correctly validate input data, handle missing fields,
apply default values, and process field aliases as expected.
"""

import unittest
import pytest
from pydantic import ValidationError
import copy

from manga_ocr_dev.training.config import schemas

# A complete and valid configuration for testing
VALID_CONFIG = {
    "model": {
        "encoder_name": "google/vit-base-patch16-224-in21k",
        "decoder_name": "bert-base-uncased",
        "max_len": 512,
        "num_decoder_layers": 6,
    },
    "dataset": {
        "augment": True,
        "train": {
            "sources": [{"type": "synthetic", "params": {"num_samples": 1000}}]
        },
        "eval": {
            "sources": [{"type": "manga109", "params": {"split": "test"}}]
        },
        "augmentations": {
            "medium": [{"name": "GaussianBlur", "params": {"blur_limit": 3}}],
            "heavy": [{"name": "RandomBrightnessContrast", "params": {"p": 0.5}}],
            "probabilities": {"medium": 0.7, "heavy": 0.1},
        },
    },
    "training": {
        "per_device_train_batch_size": 8,
        "num_train_epochs": 3,
        "fp16": True,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "dataloader_num_workers": 4,
        "logging_steps": 100,
        "report_to": "wandb",
        "save_steps": 500,
        "eval_steps": 500,
    },
}


class TestConfigSchemas(unittest.TestCase):
    """A test suite for the configuration Pydantic schemas."""

    def test_valid_config_parses_correctly(self):
        """Tests that a complete and valid configuration is parsed without errors."""
        try:
            schemas.AppConfig(app=copy.deepcopy(VALID_CONFIG))
        except ValidationError as e:
            self.fail(f"Valid configuration failed to parse: {e}")

    def test_missing_required_field_raises_error(self):
        """Tests that omitting a required field raises a `ValidationError`."""
        invalid_config = copy.deepcopy(VALID_CONFIG)
        del invalid_config["model"]
        with self.assertRaises(ValidationError):
            schemas.AppConfig(app=invalid_config)

    def test_incorrect_data_type_raises_error(self):
        """Tests that providing an incorrect data type for a field raises a `ValidationError`."""
        invalid_config = copy.deepcopy(VALID_CONFIG)
        invalid_config["training"]["num_train_epochs"] = "three"  # Should be an int
        with self.assertRaises(ValidationError):
            schemas.AppConfig(app=invalid_config)

    def test_default_values_are_applied(self):
        """Tests that default values are correctly applied for optional fields."""
        config_dict = copy.deepcopy(VALID_CONFIG)
        # Remove a field that has a default value to test if it's applied
        del config_dict["training"]["report_to"]

        # This should parse without error
        config = schemas.AppConfig(app=config_dict)

        # Check if the default value is set
        self.assertEqual(config.app.training.report_to, "wandb")

    def test_field_aliases_work_correctly(self):
        """Tests that Pydantic field aliases are correctly handled during parsing."""
        config_dict = copy.deepcopy(VALID_CONFIG)

        # Use the alias for a field
        config_dict["training"]["per_device_train_batch_size"] = 16

        config = schemas.AppConfig(app=config_dict)

        # The aliased field should be populated correctly
        self.assertEqual(config.app.training.batch_size, 16)

    def test_augmentation_probabilities_defaults(self):
        """Tests that augmentation probabilities use defaults if not specified."""
        config_dict = copy.deepcopy(VALID_CONFIG)
        del config_dict["dataset"]["augmentations"]["probabilities"]

        config = schemas.AppConfig(app=config_dict)

        self.assertIsNotNone(config.app.dataset.augmentations.probabilities)
        self.assertEqual(config.app.dataset.augmentations.probabilities.medium, 0.8)
        self.assertEqual(config.app.dataset.augmentations.probabilities.heavy, 0.02)

    def test_torch_compile_option(self):
        """Tests that the torch_compile option is handled correctly."""
        # Test default value
        config_dict_default = copy.deepcopy(VALID_CONFIG)
        config_default = schemas.AppConfig(app=config_dict_default)
        self.assertFalse(config_default.app.training.torch_compile)

        # Test setting to True
        config_dict_true = copy.deepcopy(VALID_CONFIG)
        config_dict_true["training"]["torch_compile"] = True
        config_true = schemas.AppConfig(app=config_dict_true)
        self.assertTrue(config_true.app.training.torch_compile)


if __name__ == "__main__":
    unittest.main()
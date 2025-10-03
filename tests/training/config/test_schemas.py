import unittest
import pytest
from pydantic import ValidationError
import copy

from manga_ocr_dev.training.config import schemas

# A complete and valid configuration for testing
VALID_CONFIG = {
    "run_name": "test_run",
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
    def test_valid_config_parses_correctly(self):
        """Ensure a valid configuration is parsed without errors."""
        try:
            schemas.AppConfig(**copy.deepcopy(VALID_CONFIG))
        except ValidationError as e:
            self.fail(f"Valid configuration failed to parse: {e}")

    def test_missing_required_field_raises_error(self):
        """Test that a missing required field raises a validation error."""
        invalid_config = copy.deepcopy(VALID_CONFIG)
        del invalid_config["model"]
        with self.assertRaises(ValidationError):
            schemas.AppConfig(**invalid_config)

    def test_incorrect_data_type_raises_error(self):
        """Test that an incorrect data type raises a validation error."""
        invalid_config = copy.deepcopy(VALID_CONFIG)
        invalid_config["training"]["num_train_epochs"] = "three"  # Should be an int
        with self.assertRaises(ValidationError):
            schemas.AppConfig(**invalid_config)

    def test_default_values_are_applied(self):
        """Check that default values are correctly set for optional fields."""
        config_dict = copy.deepcopy(VALID_CONFIG)
        # Remove a field that has a default value to test if it's applied
        del config_dict["training"]["report_to"]

        # This should parse without error
        config = schemas.AppConfig(**config_dict)

        # Check if the default value is set
        self.assertEqual(config.training.report_to, "wandb")

    def test_field_aliases_work_correctly(self):
        """Verify that Pydantic field aliases are correctly handled."""
        config_dict = copy.deepcopy(VALID_CONFIG)

        # Use the alias for a field
        config_dict["training"]["per_device_train_batch_size"] = 16

        config = schemas.AppConfig(**config_dict)

        # The aliased field should be populated correctly
        self.assertEqual(config.training.batch_size, 16)

    def test_augmentation_probabilities_defaults(self):
        """Test that augmentation probabilities use defaults if not specified."""
        config_dict = copy.deepcopy(VALID_CONFIG)
        del config_dict["dataset"]["augmentations"]["probabilities"]

        config = schemas.AppConfig(**config_dict)

        self.assertIsNotNone(config.dataset.augmentations.probabilities)
        self.assertEqual(config.dataset.augmentations.probabilities.medium, 0.8)
        self.assertEqual(config.dataset.augmentations.probabilities.heavy, 0.02)

if __name__ == "__main__":
    unittest.main()
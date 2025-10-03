"""Handles loading and validation of the training configuration.

This module provides the `load_config` function, which reads the main `config.yaml`
file, parses it, and validates its structure and types using the Pydantic
schemas defined in `schemas.py`. It also loads a default configuration instance
for easy access throughout the training package.
"""

import yaml
from pathlib import Path
from manga_ocr_dev.training.config.schemas import AppConfig, TrainingConfig


def load_config(config_path: Path) -> AppConfig:
    """Loads a YAML configuration file and merges it with environment variables.

    This function orchestrates the loading of configuration from multiple sources,
    ensuring a consistent and validated configuration object. The layering is as follows,
    with later sources overriding earlier ones:

    1.  Default values defined in the Pydantic schemas.
    2.  Values from the specified YAML configuration file.
    3.  Values from environment variables (prefixed with `MANGA_OCR_TRAINING_`) or a `.env` file.

    Args:
        config_path (Path): The path to the YAML configuration file.

    Returns:
        An `AppConfig` object representing the final, validated configuration.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Pop training config from yaml dict to handle it separately
    training_yaml = config_dict.pop("training", {})

    # Load training config from environment variables and .env file
    training_env = TrainingConfig()

    # Merge YAML and environment configs, with environment taking precedence
    # The `model_dump` method with `exclude_unset=True` returns only the fields
    # that were explicitly set, e.g., from environment variables.
    merged_training_config_data = {
        **training_yaml,
        **training_env.model_dump(exclude_unset=True),
    }
    config_dict["training"] = merged_training_config_data

    return AppConfig(**config_dict)


# Load the default config from the default path for easy access
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
config = load_config(CONFIG_PATH)
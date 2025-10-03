"""Handles loading and validation of the training configuration.

This module provides the `load_config` function, which reads the main `config.yaml`
file, parses it, and validates its structure and types using the Pydantic
schemas defined in `schemas.py`. It also loads a default configuration instance
for easy access throughout the training package.
"""

import yaml
from pathlib import Path
from manga_ocr_dev.training.config.schemas import AppConfig

def load_config(config_path: str) -> AppConfig:
    """Loads a YAML configuration file and validates it with Pydantic schemas.

    This function opens and parses a specified YAML file, then uses the `AppConfig`
    Pydantic model to validate the configuration's structure and data types. This
    ensures that the configuration is well-formed before it is used by the
    training pipeline.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        An `AppConfig` object representing the validated configuration.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)

# Load the default config from the default path for easy access
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
config = load_config(str(CONFIG_PATH))
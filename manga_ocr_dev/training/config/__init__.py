"""Handles loading and validation of the training configuration.

This module provides the `load_config` function, which leverages `pydantic-settings`
to build a configuration object from multiple sources, including YAML files,
environment variables, and `.env` files.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from manga_ocr_dev.training.config.schemas import AppConfig

# Default config path, resolved relative to this file's location to ensure
# it works correctly regardless of the current working directory.
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Loads and validates the training configuration.

    This function instantiates the `AppConfig` settings object, which automatically
    loads configuration from the following sources, in order of precedence:

    1.  Arguments passed to this function (e.g., a specific config path).
    2.  Environment variables.
    3.  `.env` file.
    4.  YAML configuration file (the one specified by `config_path` or the
        default `config.yaml`).
    5.  Default values defined in the Pydantic schemas.

    Args:
        config_path: An optional path to a YAML configuration file. If not
            provided, the default `config.yaml` located alongside the training
            package will be used.

    Returns:
        An `AppConfig` object representing the final, validated configuration.
    """
    # Use the provided config_path, or fall back to the default.
    effective_config_path = config_path or DEFAULT_CONFIG_PATH

    if not effective_config_path.is_file():
        # Raise a more informative error if the config file doesn't exist.
        raise FileNotFoundError(
            f"Configuration file not found at: {effective_config_path}"
        )

    # Pass the path to the AppConfig constructor via the _settings_config argument.
    # This ensures that pydantic-settings knows where to load the YAML file from.
    return AppConfig(
        _settings_config=SettingsConfigDict(yaml_file=effective_config_path)
    )
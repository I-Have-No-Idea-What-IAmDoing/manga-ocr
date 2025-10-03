"""This file defines the environment variables and paths for development.

This module centralizes the configuration of directory paths used throughout
the manga_ocr_dev scripts, making it easier to manage the project structure.
All paths are constructed relative to the project's root directory.
"""

from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
"""The root directory of the project."""

ASSETS_PATH = ROOT_DIR / "assets"
"""The path to the directory containing shared assets like vocabularies and data."""

FONTS_ROOT = ROOT_DIR / "fonts"
"""The path to the directory where font files are stored for data generation."""

DATA_SYNTHETIC_ROOT = ROOT_DIR / "assets"
"""The root for synthetic data, expected to contain 'meta' and 'img' subdirectories."""

BACKGROUND_DIR = ROOT_DIR / "manga_ocr_dev" / "data" / "background"
"""The directory where generated background images for synthetic data are saved."""

MANGA109_ROOT = ROOT_DIR / "assets"
"""The root for the Manga109 dataset, expected to contain 'data.csv' and a 'crops' subdirectory."""

TRAIN_ROOT = ROOT_DIR / "manga_ocr_dev" / "out"
"""The directory where training outputs, such as models and logs, are saved."""
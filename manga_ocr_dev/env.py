"""
This file defines the environment variables and paths used in the development scripts.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

ASSETS_PATH = ROOT_DIR / "assets"
FONTS_ROOT = ROOT_DIR / "fonts"
DATA_SYNTHETIC_ROOT = ROOT_DIR / "assets"
BACKGROUND_DIR = ROOT_DIR / "tmp" / "backgrounds"
MANGA109_ROOT = ROOT_DIR / "assets"
TRAIN_ROOT = ROOT_DIR / "out"
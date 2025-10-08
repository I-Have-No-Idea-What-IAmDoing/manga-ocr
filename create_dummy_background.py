"""Generates a placeholder background image for the data generator.

This script creates a simple 1024x1024 black image and saves it to the
directory specified by `BACKGROUND_DIR` from `manga_ocr_dev.env`.

The primary purpose of this utility is to prevent errors in the synthetic
data generation pipeline, which expects at least one background image to be
present. Running this script ensures that the `Composer` class can always
find a background, even if real background images have not yet been added.

The filename `dummy_0_1024_0_1024.png` is deliberately formatted to be
compatible with the `get_background_df` function, which parses image
dimensions from the filename.

Usage:
    python create_dummy_background.py
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path to allow importing from manga_ocr_dev
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from manga_ocr_dev.env import BACKGROUND_DIR


# Create the directory for the dummy background images
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

# Create a black image
dummy_image = np.zeros((1024, 1024, 3), dtype=np.uint8)

# Save the image with a filename that matches the expected format
cv2.imwrite(str(Path(BACKGROUND_DIR) / "dummy_0_1024_0_1024.png"), dummy_image)

print("Dummy background image created successfully in the correct directory.")
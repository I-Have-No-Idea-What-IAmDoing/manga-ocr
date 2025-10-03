"""Creates a dummy background image for the synthetic data generator.

This script generates a simple black 1024x1024 image and saves it to the
`BACKGROUND_DIR`. Its primary purpose is to serve as a placeholder, ensuring
that the data generation pipeline does not fail if the background directory
is empty. The filename is formatted to be compatible with the background
loading logic, which expects dimensions to be encoded in the name.
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
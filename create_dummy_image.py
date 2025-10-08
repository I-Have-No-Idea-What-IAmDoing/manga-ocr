"""Creates a dummy background image.

This script generates a simple 600x800 black image and saves it to
`assets/backgrounds/dummy_background.png`.

This utility can be used to create a placeholder background image. Note that
the main data generation pipeline uses a different directory and filename
format for backgrounds (see `create_dummy_background.py`), so this script may
be intended for other testing or development purposes.

Usage:
    python create_dummy_image.py
"""
from PIL import Image
import os

# Create a 600x800 black image using PIL
img = Image.new('RGB', (600, 800), color = 'black')

# Ensure the output directory exists, creating it if necessary
os.makedirs("assets/backgrounds", exist_ok=True)

# Save the dummy image to the specified path
img.save("assets/backgrounds/dummy_background.png")

print("Dummy image created at assets/backgrounds/dummy_background.png")
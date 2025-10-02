import cv2
import numpy as np
import os

# Create a directory for the dummy background images
os.makedirs("tmp/backgrounds", exist_ok=True)

# Create a black image
dummy_image = np.zeros((1024, 1024, 3), dtype=np.uint8)

# Save the image with a filename that matches the expected format
cv2.imwrite("tmp/backgrounds/dummy_0_1024_0_1024.png", dummy_image)

print("Dummy background image created successfully.")
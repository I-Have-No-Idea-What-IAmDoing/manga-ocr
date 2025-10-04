from PIL import Image
import os

# Create a black image
img = Image.new('RGB', (600, 800), color = 'black')

# Ensure the directory exists
os.makedirs("assets/backgrounds", exist_ok=True)

# Save the image
img.save("assets/backgrounds/dummy_background.png")
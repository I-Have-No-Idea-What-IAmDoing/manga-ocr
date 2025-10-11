"""Image augmentation functions for the synthetic data generator.

This module provides a set of functions to apply various augmentations to
generated images, increasing the diversity and robustness of the training data.
These augmentations simulate real-world conditions such as out-of-focus text,
compression artifacts, and perspective distortion.
"""

import numpy as np
from PIL import Image
import io
from scipy.ndimage import gaussian_filter
import cv2

def apply_blur(image, sigma):
    """Applies Gaussian blur to an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        sigma (float): The standard deviation for the Gaussian kernel.

    Returns:
        np.ndarray: The blurred image.
    """
    return gaussian_filter(image, sigma=sigma)

def apply_jpeg_compression(image, quality):
    """Applies JPEG compression to an image.

    This function simulates the artifacts introduced by JPEG compression by
    saving the image to an in-memory buffer with a specified quality level
    and then reloading it.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        quality (int): The JPEG quality level (1-100).

    Returns:
        np.ndarray: The image with JPEG compression artifacts.
    """
    pil_image = Image.fromarray(image)
    if pil_image.mode == 'RGBA':
        # JPEG does not support alpha, so convert to RGB
        # Create a white background and paste the image on it
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3]) # 3 is the alpha channel
        pil_image = background

    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    return np.array(compressed_image)

def apply_perspective_transform(image, magnitude):
    """Applies a perspective transformation to an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        magnitude (float): The magnitude of the perspective distortion.

    Returns:
        np.ndarray: The transformed image.
    """
    height, width = image.shape[:2]

    # Define the original and new corner points for the perspective transform
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Introduce random shifts to the corner points
    dw = width * magnitude * (np.random.rand(4) - 0.5)
    dh = height * magnitude * (np.random.rand(4) - 0.5)

    pts2 = np.float32([
        [0 + dw[0], 0 + dh[0]],
        [width + dw[1], 0 + dh[1]],
        [0 + dw[2], height + dh[2]],
        [width + dw[3], height + dh[3]]
    ])

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return transformed_image
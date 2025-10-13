"""Handles the composition of text images onto backgrounds.

This module defines the `Composer` class, which is responsible for taking a
rendered text image and overlaying it onto a background image. The composition
process includes several optional steps, such as adding a speech bubble around
the text, applying various augmentations to the background, and performing a
final crop to ensure the text is well-positioned and legible.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import albumentations as A

from manga_ocr_dev.synthetic_data_generator.common.utils import get_background_df


class Composer:
    """Composes text images with backgrounds and optional speech bubbles.

    This class manages the process of overlaying a generated text image onto a
    randomly selected background. It can also create a speech bubble to enclose
    the text, apply a series of augmentations to the background to increase
    variety, and perform a final smart crop to produce the output image.

    Attributes:
        background_df (pd.DataFrame): A DataFrame containing metadata about the
            available background images.
        target_size (tuple[int, int], optional): The final desired output size
            (width, height) of the image. If None, the size is not fixed.
        min_output_size (int, optional): The minimum size for the smallest
            dimension of the output image.
    """
    def __init__(self, background_dir, target_size=None, min_output_size=None, max_output_size=950):
        """Initializes the Composer with a directory of background images.

        Args:
            background_dir (str or Path): The path to the directory containing
                background images.
            target_size (tuple[int, int], optional): The final output size
                (width, height) to which the image will be resized. Defaults to
                None.
            min_output_size (int, optional): The minimum size for the smallest
                dimension of the output image. Defaults to None.
            max_output_size (int, optional): The maximum size for the largest
                dimension of the output image. Defaults to 950.
        """
        # Load metadata for available background images
        self.background_df = get_background_df(background_dir)
        # Set the target size for the final composed image
        self.target_size = target_size
        # Set the minimum allowed size for the smallest dimension of the output
        self.min_output_size = min_output_size
        # Set the maximum allowed size for the largest dimension of the output
        self.max_output_size = max_output_size

    def draw_bubble(self, width, height, text_color, padding=20, radius=20):
        """Draws a high-contrast, rounded-rectangle speech bubble.

        This method creates a speech bubble that is guaranteed to have high
        contrast with the text. It checks the brightness of the text color and
        chooses either a black or white bubble fill accordingly.

        Args:
            width (int): The internal width of the bubble.
            height (int): The internal height of the bubble.
            text_color (str): The hex color string of the text.
            padding (int): The padding to add around the content area.
            radius (int): The corner radius of the rounded rectangle.

        Returns:
            Image.Image: A Pillow `Image` object representing the bubble.
        """
        # Create a transparent RGBA image to serve as the canvas for the bubble
        bubble = Image.new('RGBA', (width + padding * 2, height + padding * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(bubble)

        # Determine if the text color is light or dark to choose a contrasting bubble color
        try:
            # Calculate a simple brightness value by summing the RGB components
            r, g, b = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            brightness = r + g + b
        except (ValueError, TypeError):
            brightness = 0  # Default to assuming dark text on error

        is_light_text = brightness > 382  # Threshold for determining if a color is light

        # Set the bubble's fill and outline colors based on text brightness
        if is_light_text:
            # For light text, create a dark bubble
            grayscale = np.random.randint(0, 16)
            fill = (grayscale, grayscale, grayscale, 255)
            outline = 'white'
        else:
            # For dark text, create a light bubble
            grayscale = np.random.randint(245, 256)
            fill = (grayscale, grayscale, grayscale, 255)
            outline = 'black'

        # Draw the rounded rectangle for the speech bubble
        draw.rounded_rectangle(
            (0, 0, width + padding * 2 - 1, height + padding * 2 - 1),
            radius=radius,
            fill=fill,
            outline=outline,
            width=3
        )
        return bubble

    def __call__(self, text_image_np, params):
        """Composes the text image with a background and an optional bubble.

        This method takes a NumPy array of a text image, decides whether to add
        a speech bubble, and then overlays it onto a randomly selected and
        augmented background image. It performs scaling and positioning to
        ensure the text is legible and well-placed, then applies a final crop.

        Args:
            text_image_np (np.ndarray): The input text image as a NumPy array.
            params (dict): A dictionary of rendering parameters, which may be
                used to influence composition decisions.

        Returns:
            np.ndarray or None: The composed final image as a NumPy array in
            RGB format, or None if the input image is invalid or the resulting
            image is not legible.
        """
        # Return None if the input text image is invalid
        if text_image_np is None or text_image_np.size == 0:
            return None

        # Ensure the text image has an alpha channel for transparency
        if text_image_np.shape[2] == 3:
            text_image_np = cv2.cvtColor(text_image_np, cv2.COLOR_BGR2BGRA)

        text_image = Image.fromarray(text_image_np).convert("RGBA")

        # Randomly decide whether to draw a speech bubble around the text
        draw_bubble = np.random.rand() < 0.45
        # Discard the sample if the rendered text is too small to be legible
        min_text_height = 10
        if text_image.height < min_text_height and not draw_bubble:
            return None

        # If backgrounds are not available, compose the image without one
        if self.background_df.empty:
            composed_image = self._compose_with_bubble_if_needed(text_image, params, draw_bubble)
            final_img_np = np.array(composed_image.convert("RGB"))
        else:
            # If backgrounds are available, attempt to compose with good contrast
            final_img_np = self._compose_with_background_and_retry(text_image, params, draw_bubble)
            if final_img_np is None:
                # Fallback to drawing a bubble if contrast is still low after retries
                composed_image = self._compose_with_bubble_if_needed(text_image, params, True)
                final_img_np = self._compose_with_background_and_retry(composed_image, params, True)

        # Post-processing steps for the final image
        if final_img_np is not None:
            final_img_np = self._resize_and_crop(final_img_np)
            # Convert the final image to grayscale before returning
            final_img_np = cv2.cvtColor(final_img_np, cv2.COLOR_RGB2GRAY)

        return final_img_np

    def _compose_with_bubble_if_needed(self, text_image, params, draw_bubble):
        """Draws a speech bubble around the text image if needed."""
        if draw_bubble:
            bubble_padding = np.random.randint(15, 30)
            bubble_radius = np.random.randint(10, 30)
            bubble_image = self.draw_bubble(
                text_image.width, text_image.height,
                text_color=params.get('color', '#000000'),
                padding=bubble_padding,
                radius=bubble_radius
            )
            bubble_image.paste(text_image, (bubble_padding, bubble_padding), text_image)
            return bubble_image
        return text_image

    def _compose_with_background_and_retry(self, composed_image, params, draw_bubble, retries=3):
        """Attempts to compose an image with a background, retrying on low contrast."""
        for _ in range(retries):
            # Randomly select a background and apply augmentations
            background_path = self.background_df.sample(1).iloc[0].path
            background = self._augment_background(background_path, draw_bubble)

            # Ensure the background is large enough
            background = self._scale_background_if_needed(background, composed_image)

            # Paste the text image onto the background at a random position
            x_offset = np.random.randint(0, background.width - composed_image.width + 1)
            y_offset = np.random.randint(0, background.height - composed_image.height + 1)
            background.paste(composed_image, (x_offset, y_offset), composed_image)

            # If a bubble is not used, check for low contrast
            if not draw_bubble:
                is_low_contrast = self._is_low_contrast(
                    np.array(background.convert("RGB")), np.array(composed_image), x_offset, y_offset
                )
                if not is_low_contrast:
                    # If contrast is good, proceed with this background
                    return self._perform_smart_crop(background, composed_image, x_offset, y_offset)
            else:
                # If a bubble is used, assume good contrast and proceed
                return self._perform_smart_crop(background, composed_image, x_offset, y_offset)

        # If all retries fail, return None to indicate failure
        return None

    def _augment_background(self, background_path, draw_bubble):
        """Applies a series of augmentations to the background image."""
        background = Image.open(background_path).convert("RGB")
        background_np = np.array(background)
        try:
            background_transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.InvertImg(p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.4),
                    contrast_limit=(-0.8, -0.3),
                    p=0.5 if draw_bubble else 1
                ),
                A.Blur(blur_limit=(3, 5), p=0.3),
            ]
            background_np = A.Compose(background_transforms)(image=background_np)["image"]
            return Image.fromarray(background_np).convert("RGBA")
        except Exception:
            return Image.open(background_path).convert("RGBA")

    def _scale_background_if_needed(self, background, composed_image):
        """Scales the background if it's smaller than the composed image."""
        bg_width, bg_height = background.size
        comp_width, comp_height = composed_image.size
        if bg_width <= comp_width or bg_height <= comp_height:
            width_scale = comp_width / bg_width if bg_width > 0 else float('inf')
            height_scale = comp_height / bg_height if bg_height > 0 else float('inf')
            required_scale = max(width_scale, height_scale)
            random_margin_scale = np.random.uniform(1.1, 1.9)
            final_scale_factor = required_scale * random_margin_scale
            if final_scale_factor > 0:
                return ImageOps.scale(background, final_scale_factor, Image.Resampling.LANCZOS)
        return background

    def _perform_smart_crop(self, final_image, composed_image, x_offset, y_offset):
        """Performs a smart crop that includes the text and some context."""
        final_img_np = np.array(final_image.convert("RGB"))
        h, w, _ = final_img_np.shape
        text_x1, text_y1 = x_offset, y_offset
        text_x2, text_y2 = x_offset + composed_image.width, y_offset + composed_image.height
        must_include_x1 = max(0, text_x1 - np.random.randint(10, 50))
        must_include_y1 = max(0, text_y1 - np.random.randint(10, 50))
        must_include_x2 = min(w, text_x2 + np.random.randint(10, 50))
        must_include_y2 = min(h, text_y2 + np.random.randint(10, 50))

        crop_x1 = np.random.randint(0, must_include_x1 + 1)
        crop_y1 = np.random.randint(0, must_include_y1 + 1)
        crop_x2 = np.random.randint(must_include_x2, w + 1)
        crop_y2 = np.random.randint(must_include_y2, h + 1)

        if crop_x1 >= crop_x2:
            crop_x1 = max(0, crop_x2 - 10)
        if crop_y1 >= crop_y2:
            crop_y1 = max(0, crop_y2 - 10)

        if crop_x1 < crop_x2 and crop_y1 < crop_y2:
            return A.Crop(x_min=crop_x1, y_min=crop_y1, x_max=crop_x2, y_max=crop_y2)(image=final_img_np)["image"]
        return final_img_np

    def _resize_and_crop(self, final_img_np):
        """Resizes the image to meet the specified output size constraints."""
        # Resize if smaller than the minimum size
        if self.min_output_size:
            h, w, _ = final_img_np.shape
            if h < self.min_output_size or w < self.min_output_size:
                final_img_np = A.SmallestMaxSize(max_size=self.min_output_size, interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        # Resize if larger than the maximum size
        if self.max_output_size:
            h, w, _ = final_img_np.shape
            if h > self.max_output_size or w > self.max_output_size:
                final_img_np = A.LongestMaxSize(max_size=self.max_output_size, interpolation=cv2.INTER_AREA)(image=final_img_np)["image"]

        # Resize to the final target size if specified
        if self.target_size:
            final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        return final_img_np

        # Resize the image if it's smaller than the minimum allowed size
        if self.min_output_size:
            h, w, _ = final_img_np.shape
            if h < self.min_output_size or w < self.min_output_size:
                final_img_np = A.SmallestMaxSize(max_size=self.min_output_size, interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        # Resize the image if it's larger than the maximum allowed size
        if self.max_output_size:
            h, w, _ = final_img_np.shape
            if h > self.max_output_size or w > self.max_output_size:
                final_img_np = A.LongestMaxSize(max_size=self.max_output_size, interpolation=cv2.INTER_AREA)(image=final_img_np)["image"]

        # Resize the image to the final target size if specified
        if self.target_size:
            final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        # Convert the final image to grayscale before returning
        final_img_np = cv2.cvtColor(final_img_np, cv2.COLOR_RGB2GRAY)
        return final_img_np

    def _is_low_contrast(self, final_img_np, text_image_np, x_offset, y_offset, threshold=0.1):
        """Checks if the text has low contrast with its background using OpenCV.

        This method computes the average intensity of the text pixels and the
        surrounding background pixels. If the absolute difference is below a
        threshold, the image is considered low contrast.

        Args:
            final_img_np (np.ndarray): The final composed image (in RGB).
            text_image_np (np.ndarray): The original text image with an alpha channel.
            x_offset (int): The x-coordinate where the text image was placed.
            y_offset (int): The y-coordinate where the text image was placed.
            threshold (float): The minimum acceptable contrast difference.

        Returns:
            bool: True if the contrast is below the threshold, False otherwise.
        """
        # Convert the final image to grayscale and normalize to [0, 1] for intensity analysis
        final_img_gray = cv2.cvtColor(final_img_np, cv2.COLOR_RGB2GRAY) / 255.0
        # Create a binary mask from the text image's alpha channel
        text_mask = (text_image_np[:, :, 3] > 0).astype(np.uint8)
        # Define the region of interest (ROI) where the text is located
        text_roi = final_img_gray[y_offset:y_offset + text_mask.shape[0], x_offset:x_offset + text_mask.shape[1]]
        # Calculate the average intensity of the text pixels
        text_intensity = np.mean(text_roi[text_mask.astype(bool)])

        # Dilate the text mask to find the surrounding background area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)
        background_mask = (dilated_mask - text_mask).astype(bool)

        # Calculate the average intensity of the background pixels
        background_pixels = text_roi[background_mask]
        if background_pixels.size == 0:
            return False

        background_intensity = np.mean(background_pixels)

        # Ensure that the background and text are not both very dark or very light
        if (text_intensity < 0.1 and background_intensity < 0.1) or \
           (text_intensity > 0.9 and background_intensity > 0.9):
            return True

        # Calculate the contrast as the absolute difference between intensities
        contrast = abs(text_intensity - background_intensity)
        # Return True if the contrast is below the specified threshold
        return contrast < threshold
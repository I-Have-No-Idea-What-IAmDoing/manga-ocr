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

from manga_ocr_dev.synthetic_data_generator_v2.utils import get_background_df


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
    def __init__(self, background_dir, target_size=None, min_output_size=None):
        """Initializes the Composer with a directory of background images.

        Args:
            background_dir (str or Path): The path to the directory containing
                background images.
            target_size (tuple[int, int], optional): The final output size
                (width, height) to which the image will be resized. Defaults to
                None.
            min_output_size (int, optional): The minimum size for the smallest
                dimension of the output image. Defaults to None.
        """
        self.background_df = get_background_df(background_dir)
        self.target_size = target_size
        self.min_output_size = min_output_size

    def draw_bubble(self, width, height, padding=20, radius=20):
        """Draws a white, rounded-rectangle speech bubble.

        This method creates a speech bubble as a Pillow `Image` object with a
        transparent background. The bubble is a rounded rectangle with a black
        outline.

        Args:
            width (int): The internal width of the bubble, before padding.
            height (int): The internal height of the bubble, before padding.
            padding (int): The padding to add around the content area.
            radius (int): The corner radius of the rounded rectangle.

        Returns:
            Image.Image: A Pillow `Image` object representing the bubble.
        """
        bubble = Image.new('RGBA', (width + padding * 2, height + padding * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(bubble)

        if np.random.rand() > 0.8: 
            fill = (255, 255, 255, 255) 
            outline = 'black'
        else:
            fill = (0, 0, 0, 255)
            outline = 'white'
            
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
        if text_image_np is None or text_image_np.size == 0:
            return None

        text_image = Image.fromarray(text_image_np).convert("RGBA")

        # Randomly decide whether to draw a speech bubble around the text.
        draw_bubble = np.random.rand() < 0.7

        if draw_bubble:
            # Create the bubble image with random padding and radius.
            bubble_padding = np.random.randint(15, 30)
            bubble_radius = np.random.randint(10, 30)
            bubble_image = self.draw_bubble(text_image.width, text_image.height, padding=bubble_padding, radius=bubble_radius)

            # Paste the text onto the bubble.
            bubble_image.paste(text_image, (bubble_padding, bubble_padding), text_image)
            composed_image = bubble_image
        else:
            composed_image = text_image

        # Discard the sample if the rendered text is too small to be legible.
        min_text_height = 10 # pixels
        if composed_image.height < min_text_height:
            return None # Discard sample if text is too small

        # If no backgrounds are provided, return the composed image as is,
        # optionally resizing it to the target size.
        if self.background_df.empty:
            final_img_np = np.array(composed_image.convert("RGB"))
            if self.target_size:
                final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]
            return final_img_np

        # Randomly select a background image.
        background_path = self.background_df.sample(1).iloc[0].path
        background = Image.open(background_path).convert("RGB")
        background_np = np.array(background)

        # Define and apply a series of augmentations to the background to
        # increase visual variety.
        background_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.InvertImg(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.4),
                contrast_limit=(-0.8, -0.3),
                # Apply contrast less often if there's a bubble, as high
                # contrast can make the bubble look unnatural.
                p=0.5 if draw_bubble else 1
            ),
            A.Blur(blur_limit=(3, 5), p=0.3),
        ]
        background_np = A.Compose(background_transforms)(image=background_np)["image"]
        background = Image.fromarray(background_np).convert("RGBA")


        bg_width, bg_height = background.size
        comp_width, comp_height = composed_image.size

        # Ensure the background is larger than the composed image, with a random margin.
        if bg_width <= comp_width or bg_height <= comp_height:
            # Determine the required scaling factor to make the background
            # large enough to contain the composed image.
            width_scale = comp_width / bg_width if bg_width > 0 else float('inf')
            height_scale = comp_height / bg_height if bg_height > 0 else float('inf')

            # The background must be scaled by at least the maximum of these two ratios
            # to ensure it can contain the composed image.
            required_scale = max(width_scale, height_scale)

            # Add an additional random scaling factor to create variety and ensure
            # there's always some background visible around the text.
            random_margin_scale = np.random.uniform(1.1, 1.9)
            final_scale_factor = required_scale * random_margin_scale

            if final_scale_factor > 0:
                background = ImageOps.scale(background, final_scale_factor, Image.Resampling.LANCZOS)

        # Randomly determine the position to paste the text overlay.
        x_offset = np.random.randint(0, background.width - composed_image.width + 1)
        y_offset = np.random.randint(0, background.height - composed_image.height + 1)

        background.paste(composed_image, (x_offset, y_offset), composed_image)

        # Check for low contrast, but only if no bubble is drawn, as bubbles
        # provide their own high-contrast background.
        if not draw_bubble and self._is_low_contrast(
            np.array(background.convert("RGB")), np.array(composed_image), x_offset, y_offset
        ):
            return None  # Discard sample if contrast is too low

        final_img_np = np.array(background.convert("RGB"))

        # Perform a final smart crop that ensures the text is not cut off.
        # This defines a bounding box that must be included in the final crop.
        h, w, _ = final_img_np.shape

        text_x1, text_y1 = x_offset, y_offset
        text_x2, text_y2 = x_offset + composed_image.width, y_offset + composed_image.height

        # Define a padded "must-include" region around the text.
        must_include_x1 = max(0, text_x1 - np.random.randint(10, 50))
        must_include_y1 = max(0, text_y1 - np.random.randint(10, 50))
        must_include_x2 = min(w, text_x2 + np.random.randint(10, 50))
        must_include_y2 = min(h, text_y2 + np.random.randint(10, 50))

        # Randomly select crop coordinates that are guaranteed to contain the
        # "must-include" region.
        crop_x1 = np.random.randint(0, must_include_x1 + 1)
        crop_y1 = np.random.randint(0, must_include_y1 + 1)

        crop_x2 = np.random.randint(must_include_x2, w + 1)
        crop_y2 = np.random.randint(must_include_y2, h + 1)

        # Ensure crop coordinates are valid.
        if crop_x1 >= crop_x2:
            crop_x1 = max(0, crop_x2 - 10)
        if crop_y1 >= crop_y2:
            crop_y1 = max(0, crop_y2 - 10)

        final_img_np = A.Crop(x_min=crop_x1, y_min=crop_y1, x_max=crop_x2, y_max=crop_y2)(image=final_img_np)["image"]

        # If a minimum output size is specified, resize the image if it's too small.
        if self.min_output_size:
            h, w, _ = final_img_np.shape
            if h < self.min_output_size or w < self.min_output_size:
                final_img_np = A.SmallestMaxSize(max_size=self.min_output_size, interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        # If the image is too big shrink it
        MAX_SIZE = 950
        h, w, _ = final_img_np.shape
        if h > MAX_SIZE or w > MAX_SIZE:
            final_img_np = A.LongestMaxSize(max_size=MAX_SIZE, interpolation=cv2.INTER_AREA)(image=final_img_np)["image"]

        # If a target size is specified, resize the final image.
        if self.target_size:
            final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        return final_img_np

    def _is_low_contrast(self, final_img_np, text_image_np, x_offset, y_offset, threshold=0.15):
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
        # Convert the final image to grayscale and normalize to [0, 1] for intensity analysis.
        final_img_gray = cv2.cvtColor(final_img_np, cv2.COLOR_RGB2GRAY) / 255.0

        # Create a binary mask from the text image's alpha channel.
        text_mask = (text_image_np[:, :, 3] > 0).astype(np.uint8)

        # Define the region of interest (ROI) where the text is located.
        text_roi = final_img_gray[y_offset:y_offset + text_mask.shape[0], x_offset:x_offset + text_mask.shape[1]]

        # Calculate the average intensity of the text pixels.
        text_intensity = np.mean(text_roi[text_mask.astype(bool)])

        # Dilate the text mask to find the surrounding background area.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)
        background_mask = (dilated_mask - text_mask).astype(bool)

        # Calculate the average intensity of the background pixels.
        background_pixels = text_roi[background_mask]
        if background_pixels.size == 0:
            # Cannot determine background; assume sufficient contrast.
            return False

        background_intensity = np.mean(background_pixels)

        # The contrast is the absolute difference between intensities.
        contrast = abs(text_intensity - background_intensity)

        return contrast < threshold
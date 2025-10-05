import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
import cv2

from manga_ocr_dev.synthetic_data_generator_v2.utils import get_background_df


class Composer:
    def __init__(self, background_dir, target_size=None, min_output_size=None):
        self.background_df = get_background_df(background_dir)
        self.target_size = target_size
        self.min_output_size = min_output_size

    def draw_bubble(self, width, height, padding=20, radius=20):
        """Draws a rounded rectangle bubble."""
        bubble = Image.new('RGBA', (width + padding * 2, height + padding * 2), (255, 255, 255, 0))
        draw = ImageDraw.Draw(bubble)

        draw.rounded_rectangle(
            (0, 0, width + padding * 2 - 1, height + padding * 2 - 1),
            radius=radius,
            fill=(255, 255, 255, 255),
            outline='black',
            width=3
        )
        return bubble

    def __call__(self, text_image_np, params):
        """Composes the text image with a background and an optional bubble."""
        if text_image_np is None or text_image_np.size == 0:
            return None

        text_image = Image.fromarray(text_image_np).convert("RGBA")

        draw_bubble = np.random.rand() < 0.7

        if draw_bubble:
            bubble_padding = np.random.randint(15, 30)
            bubble_radius = np.random.randint(10, 30)
            bubble_image = self.draw_bubble(text_image.width, text_image.height, padding=bubble_padding, radius=bubble_radius)

            bubble_image.paste(text_image, (bubble_padding, bubble_padding), text_image)
            composed_image = bubble_image
        else:
            composed_image = text_image

        if self.background_df.empty:
            final_img_np = np.array(composed_image.convert("RGB"))
            if self.target_size:
                final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]
            return final_img_np

        background_path = self.background_df.sample(1).iloc[0].path
        background = Image.open(background_path).convert("RGB")
        background_np = np.array(background)

        # Apply augmentations to the background
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
        background = Image.fromarray(background_np).convert("RGBA")

        # Dynamic scaling of the text overlay
        scale_factor = np.random.uniform(0.5, 1.0) # Less aggressive scaling
        target_width = int(background.width * scale_factor)

        w, h = composed_image.size
        aspect_ratio = h / w if w > 0 else 0
        target_height = int(target_width * aspect_ratio)

        # Ensure the scaled image fits within the background
        if target_height > background.height * 0.9:
            target_height = int(background.height * 0.9)
            target_width = int(target_height / aspect_ratio) if aspect_ratio > 0 else 0

        if target_width > 0 and target_height > 0:
            composed_image = composed_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Legibility check
        min_text_height = 20 # pixels
        if composed_image.height < min_text_height:
            return None # Discard sample if text is too small

        x_offset = np.random.randint(0, background.width - composed_image.width + 1)
        y_offset = np.random.randint(0, background.height - composed_image.height + 1)

        background.paste(composed_image, (x_offset, y_offset), composed_image)

        final_img_np = np.array(background.convert("RGB"))

        # Less aggressive final random crop that ensures text is not cut off
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

        final_img_np = A.Crop(x_min=crop_x1, y_min=crop_y1, x_max=crop_x2, y_max=crop_y2)(image=final_img_np)["image"]

        if self.min_output_size:
            h, w, _ = final_img_np.shape
            if h < self.min_output_size or w < self.min_output_size:
                final_img_np = A.SmallestMaxSize(max_size=self.min_output_size, interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        if self.target_size:
            final_img_np = A.Resize(height=self.target_size[1], width=self.target_size[0], interpolation=cv2.INTER_LANCZOS4)(image=final_img_np)["image"]

        return final_img_np
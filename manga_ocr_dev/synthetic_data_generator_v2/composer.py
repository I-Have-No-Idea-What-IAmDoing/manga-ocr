import numpy as np
from PIL import Image, ImageDraw

from manga_ocr_dev.synthetic_data_generator_v2.utils import get_background_df


class Composer:
    def __init__(self, background_dir):
        self.background_df = get_background_df(background_dir)

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
            return np.array(composed_image.convert("RGB"))

        background_path = self.background_df.sample(1).iloc[0].path
        background = Image.open(background_path).convert("RGBA")

        max_w = int(background.width * 0.9)
        max_h = int(background.height * 0.9)

        if composed_image.width > max_w or composed_image.height > max_h:
            composed_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        x_offset = np.random.randint(0, background.width - composed_image.width)
        y_offset = np.random.randint(0, background.height - composed_image.height)

        background.paste(composed_image, (x_offset, y_offset), composed_image)

        return np.array(background.convert("RGB"))
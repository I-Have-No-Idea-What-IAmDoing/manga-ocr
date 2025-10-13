"""Core component for generating synthetic manga-style text images.

This script defines the `SyntheticDataGeneratorV2` class, which is responsible
for creating synthetic image-text pairs for training the OCR model. It combines
various elements such as random text generation, font selection, and complex
text styling (e.g., furigana) to produce a diverse and realistic dataset that
mimics the appearance of text in manga.
"""

from pathlib import Path
import numpy as np
from pictex import Canvas, Row, Column, Text, Shadow
from PIL import Image
import io
from scipy.ndimage import rotate

from manga_ocr_dev.env import FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.common.base_generator import BaseDataGenerator
from manga_ocr_dev.synthetic_data_generator.common.composer import Composer
from manga_ocr_dev.synthetic_data_generator.common.utils import (
    is_kanji,
    is_ascii,
)
from .image_augmentations import (
    apply_blur,
    apply_jpeg_compression,
    apply_perspective_transform,
    apply_salt_and_pepper_noise,
)


class SyntheticDataGeneratorV2(BaseDataGenerator):
    """Generates synthetic manga-style text images and their transcriptions.

    This class orchestrates the process of creating synthetic data. It can
    generate random Japanese text or use text from a corpus, then render it
    into an image using a variety of fonts, styles, and effects like furigana
    to mimic the appearance of text in manga.
    """

    def __init__(self, background_dir=None, min_font_size=30, max_font_size=60, target_size=None, min_output_size=None):
        """Initializes the SyntheticDataGenerator.

        Args:
            background_dir (str or Path, optional): Path to the directory
                containing background images.
            min_font_size (int): The minimum font size for text rendering.
            max_font_size (int): The maximum font size for text rendering.
            target_size (tuple[int, int], optional): The final output size
                (width, height) for the composed image.
            min_output_size (int, optional): The minimum size for the smallest
                dimension of the composed image.
        """
        super().__init__()
        # Set the font size range for text rendering
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        # Initialize the composer for blending text with background images
        self.composer = Composer(background_dir, target_size=target_size, min_output_size=min_output_size)

    def get_random_render_params(self):
        """Generates a dictionary of random parameters for text rendering.

        This method creates a set of randomized parameters that control the
        visual appearance of the rendered text. These parameters include text
        orientation, font size, color, and visual effects like stroke or glow.
        This helps to create a diverse dataset that exposes the OCR model to a
        wide range of text styles.

        Returns:
            dict: A dictionary containing randomly generated rendering
            parameters.
        """
        params = {}
        # Randomly choose between vertical and horizontal text layout
        params['vertical'] = np.random.choice([True, False], p=[0.8, 0.2])
        # Randomly select a font size within the specified range
        params['font_size'] = np.random.randint(self.min_font_size, self.max_font_size)

        # Add randomized character and line spacing
        params['letter_spacing'] = np.random.randint(-2, 6)
        params['line_height'] = np.random.uniform(0.9, 1.6)

        # Add randomized text rotation
        if np.random.rand() > 0.5:
            params['rotation'] = np.random.uniform(-15, 15)
        else:
            params['rotation'] = 0

        # Add randomized blur, JPEG quality, and perspective transform
        params['blur_sigma'] = np.random.uniform(0, 1.0) if np.random.rand() < 0.0 else 0
        params['jpeg_quality'] = np.random.randint(50, 101) if np.random.rand() < 0.0 else 100
        params['perspective_magnitude'] = np.random.uniform(0, 0.06) if np.random.rand() < 0.3 else 0
        params['salt_and_pepper_amount'] = np.random.uniform(0, 0.02) if np.random.rand() < 0.1 else 0

        # Randomly choose a text color, either dark or light gray
        if np.random.rand() > 0.30:
            gray_value = np.random.randint(0, 30)
        else:
            gray_value = np.random.randint(225, 256)
        params['color'] = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

        # Randomly select a text effect
        effect = np.random.choice(["stroke", "double_stroke", "shadow", "none"], p=[0.25, 0.1, 0.15, 0.5])
        params['effect'] = effect

        # Helper function to generate a random grayscale hex color
        def get_random_hex_color():
            gray_value = np.random.randint(0, 256)
            return f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

        # Set parameters for the chosen effect
        if effect == "stroke":
            params["stroke_width"] = np.random.choice([1, 2, 3])
            params["stroke_color"] = get_random_hex_color()

        elif effect == "double_stroke":
            params["stroke_width"] = np.random.choice([2, 3, 4])
            params["stroke_color"] = get_random_hex_color()
            params["stroke_width2"] = 1
            params["stroke_color2"] = get_random_hex_color()

        elif effect == "shadow":
            num_shadows = np.random.randint(1, 4)
            shadows = []
            for _ in range(num_shadows):
                shadow = {
                    "offset": (np.random.randint(-5, 6), np.random.randint(-5, 6)),
                    "blur_radius": np.random.choice([2, 5, 10]),
                    "color": get_random_hex_color(),
                }
                shadows.append(shadow)
            params["shadows"] = shadows
        return params

    def process(self, text=None, override_params=None):
        """Processes text to generate a synthetic image and ground truth.

        This method orchestrates the generation of a single synthetic sample.
        It takes optional input text, or generates random text if none is
        provided. It then determines rendering parameters, applies random
        markup (like furigana), renders the text to an image, and composes it
        with a background if a composer is available.

        Args:
            text (str, optional): The text to be rendered. If None, random
                text will be generated. Defaults to None.
            override_params (dict, optional): A dictionary of rendering
                parameters to override the randomly generated ones. Defaults to
                None.

        Returns:
            A tuple containing:
                - np.ndarray: The generated image as a NumPy array.
                - str: The ground truth text.
                - dict: The parameters used for rendering.
        """
        # Get random rendering parameters and apply any overrides
        params = self.get_random_render_params()
        if override_params:
            params.update(override_params)

        # If no text is provided, generate random words using a random font
        if text is None:
            if "font_path" not in params:
                # Loop to ensure a font with a non-empty vocabulary is selected for random text generation.
                # This prevents the creation of samples with no text, which would otherwise be skipped.
                vocab = set()
                while not vocab:
                    font_path = self.get_random_font()
                    vocab = self.font_map.get(font_path, set())
                params["font_path"] = font_path
            else:
                # If a font is provided via override_params, use it.
                font_path = params["font_path"]
                vocab = self.font_map.get(font_path, set())

            # If the vocabulary is empty (e.g., from a bad override font), this will produce empty words.
            # The process will then correctly generate an empty image and skip the sample.
            words = self.get_random_words(vocab)
        # If text is provided, clean it and split it into words
        else:
            text = text.replace("　", " ").replace("…", "...")
            words = self.split_into_words(text)

        # Arrange words into lines to form the ground truth text
        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        # If a font is not already specified, select one that supports the text
        if "font_path" not in params:
            params["font_path"] = self.get_random_font(text_gt)

        # Verify that the selected font supports all characters in the text
        font_path = params.get("font_path")
        if font_path:
            vocab = self.font_map.get(font_path)
            if vocab:
                unsupported_chars = {c for c in text_gt if c not in vocab and not c.isspace()}
                if unsupported_chars:
                    raise ValueError(
                        f"Text contains unsupported characters for font "
                        f"{Path(font_path).name}: {''.join(unsupported_chars)}"
                    )
        else:
            vocab = None

        # If there is no text to render, return an empty image and parameters.
        # This can happen if the input text is empty or contains only whitespace.
        if not text_gt.strip():
            img = self.render([], params)
            return img, "", params

        # Randomly decide whether to add furigana markup to the text.
        # Furigana is a reading aid for Japanese text, consisting of smaller
        # characters printed next to a kanji character to indicate its pronunciation.
        # This helps create more realistic training data for manga.
        lines_with_markup = []
        if np.random.random() < 0.5:
            # The probability of adding furigana to a word is chosen randomly.
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            if lines:
                lines_with_markup = [self.add_random_furigana(line, word_prob) for line in lines]
        else:
            # If not adding furigana, the lines are kept as they are.
            lines_with_markup = [[line] for line in lines]

        # Convert the relative font path to an absolute path for rendering
        relative_font_path = None
        if "font_path" in params and params["font_path"]:
            relative_font_path = params["font_path"]
            params["font_path"] = str(Path(FONTS_ROOT) / params["font_path"])

        # Render the text with markup to an image
        img = self.render(lines_with_markup, params)

        # Apply rotation if specified. This simulates variations in document scanning and alignment.
        if params.get("rotation", 0) != 0:
            # Rotate the image in degrees, expanding the canvas to fit the new dimensions.
            # The background is filled with a transparent color (0).
            # Using order=0 (nearest-neighbor) prevents interpolation artifacts that could
            # interfere with legibility checks or model training.
            img = rotate(img, params["rotation"], reshape=True, cval=0, order=0)

        # Apply perspective transform if specified. This mimics the appearance of text on a curved or angled page.
        if params.get("perspective_magnitude", 0) > 0:
            img = apply_perspective_transform(img, params["perspective_magnitude"])

        # Apply Gaussian blur if specified. This simulates out-of-focus text or low-resolution scanning.
        if params.get("blur_sigma", 0) > 0:
            img = apply_blur(img, params["blur_sigma"])

        # Apply JPEG compression if specified. This introduces compression artifacts common in web images.
        if params.get("jpeg_quality", 100) < 100:
            img = apply_jpeg_compression(img, params["jpeg_quality"])

        # Apply salt and pepper noise if specified. This simulates sensor noise or dust on a scanned page.
        if params.get("salt_and_pepper_amount", 0) > 0:
            img = apply_salt_and_pepper_noise(img, params["salt_and_pepper_amount"])

        # Restore the relative font path in the returned parameters
        if relative_font_path:
            params["font_path"] = relative_font_path

        # If a composer is available, blend the rendered text with a background image
        if self.composer:
            img = self.composer(img, params)
            if img is None:
                return None, None, None

        return img, text_gt, params

    def render(self, lines_with_markup, params):
        """Renders text with markup into a NumPy image array using pictex.

        This method takes text that has been processed and marked up (e.g.,
        with furigana) and uses the `pictex` library to render it into an
        image. It constructs a layout of text components based on the markup
        and rendering parameters, then generates the final image.

        Args:
            lines_with_markup (list[list]): A nested list where each inner list
                contains chunks of a line. Chunks can be strings or tuples
                representing marked-up text.
            params (dict): A dictionary of rendering parameters, including font
                path, size, color, and effects.

        Returns:
            np.ndarray: The rendered image as a NumPy array. Returns an empty
            array if rendering fails or if there is no text to render.
        """
        # Extract rendering parameters from the params dictionary
        font_path = params.get("font_path", "NotoSansJP-Regular.otf")
        font_size = params.get("font_size", 32)
        color = params.get("color", "black")
        vertical = params.get("vertical", False)
        effect = params.get("effect", "none")
        letter_spacing = params.get("letter_spacing", 0)
        line_height = params.get("line_height", 1.2)
        rotation = params.get("rotation", 0)

        # Initialize the pictex canvas with base font and color settings
        canvas = Canvas().font_family(str(font_path)).font_size(font_size).color(color)

        # If there's no text to render, return an empty image
        if not lines_with_markup:
            return np.array([])

        # Helper function to create a pictex Text component with optional effects
        def create_text_component(text, size_multiplier=1.0):
            component = Text(text).font_size(font_size * size_multiplier)
            if effect == "stroke":
                component = component.text_stroke(
                    width=params.get("stroke_width", 1),
                    color=params.get("stroke_color", "white"),
                )
            elif effect == "double_stroke":
                component = component.text_stroke(
                    width=params.get("stroke_width", 2),
                    color=params.get("stroke_color", "white"),
                ).text_stroke(
                    width=params.get("stroke_width2", 1),
                    color=params.get("stroke_color2", "black"),
                )
            elif effect == "shadow":
                shadows = [
                    Shadow(
                        offset=s["offset"],
                        blur_radius=s["blur_radius"],
                        color=s["color"],
                    )
                    for s in params.get("shadows", [])
                ]
                if shadows:
                    component = component.text_shadows(*shadows)
            return component

        # Helper function to create a pictex component from a marked-up chunk
        def create_component(chunk):
            if isinstance(chunk, str):
                return create_text_component(chunk)
            type, *args = chunk
            if type == 'furigana':
                base, ruby = args
                # Create a column for furigana (ruby text above base text)
                return Column(create_text_component(ruby, 0.5), create_text_component(base)).gap(0).horizontal_align("center")
            if type == 'tcy':
                text, = args
                # Create a row for tate-chu-yoko (horizontal text in vertical layout)
                return Row(*[create_text_component(c) for c in text]).gap(0).vertical_align("center") if vertical else create_text_component(text)
            return Text("")

        # Construct the layout based on whether the text is vertical or horizontal
        if vertical:
            line_columns = []
            for line_chunks in lines_with_markup:
                components = [comp for chunk in line_chunks for comp in ([create_text_component(c) for c in chunk] if isinstance(chunk, str) else [create_component(chunk)])]
                line_columns.append(Column(*components).gap(letter_spacing).horizontal_align("center"))
            # Arrange lines as columns in a row, reversing for right-to-left layout
            composed_element = Row(*line_columns[::-1]).gap(int(font_size * (line_height - 1.0))).vertical_align("top")
        else:
            line_rows = [Row(*[create_component(chunk) for chunk in line_chunks]).gap(letter_spacing).vertical_align("bottom") for line_chunks in lines_with_markup]
            # Arrange lines as rows in a column for horizontal layout
            composed_element = Column(*line_rows).gap(int(font_size * (line_height - 1.0))).horizontal_align("left")

        # Render the composed element to an image and convert to a NumPy array
        image = canvas.render(composed_element)
        return image.to_numpy() if image else np.array([])
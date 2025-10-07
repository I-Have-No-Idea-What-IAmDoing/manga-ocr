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

from manga_ocr_dev.env import FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.common.base_generator import BaseDataGenerator
from manga_ocr_dev.synthetic_data_generator.common.composer import Composer
from manga_ocr_dev.synthetic_data_generator.common.utils import (
    is_kanji,
    is_ascii,
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
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        if background_dir:
            self.composer = Composer(background_dir, target_size=target_size, min_output_size=min_output_size)
        else:
            self.composer = None

    def get_random_render_params(self):
        """Generates a dictionary of random parameters for text rendering."""
        params = {}
        params['vertical'] = np.random.choice([True, False], p=[0.8, 0.2])
        params['font_size'] = np.random.randint(self.min_font_size, self.max_font_size)

        if np.random.rand() > 0.25:
            gray_value = np.random.randint(0, 30)
        else:
            gray_value = np.random.randint(225, 256)
        params['color'] = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

        effect = np.random.choice(["stroke", "glow", "none"], p=[0.35, 0.15, 0.5])
        params['effect'] = effect

        def get_random_hex_color():
            gray_value = np.random.randint(0, 256)
            return f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

        if effect == "stroke":
            params["stroke_width"] = np.random.choice([1, 2, 3])
            params["stroke_color"] = get_random_hex_color()
        elif effect == "glow":
            params["shadow_blur"] = np.random.choice([2, 5, 10])
            params["shadow_color"] = get_random_hex_color()
            params['shadow_offset'] = (0, 0)
        return params

    def process(self, text=None, override_params=None):
        """Processes text to generate a synthetic image and ground truth."""
        params = self.get_random_render_params()
        if override_params:
            params.update(override_params)

        if text is None:
            if "font_path" not in params:
                font_path = self.get_random_font()
                vocab = self.font_map.get(font_path, set())
                params["font_path"] = font_path
            else:
                font_path = params["font_path"]
                vocab = self.font_map.get(font_path, set())
            words = self.get_random_words(vocab)
        else:
            text = text.replace("　", " ").replace("…", "...")
            words = self.split_into_words(text)

        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        if "font_path" not in params:
            params["font_path"] = self.get_random_font(text_gt)

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

        if not text_gt.strip():
            img = self.render([], params)
            return img, "", params

        lines_with_markup = []
        if np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            if lines:
                lines_with_markup = [self.add_random_furigana(line, word_prob, vocab) for line in lines]
        else:
            lines_with_markup = [[line] for line in lines]

        relative_font_path = None
        if "font_path" in params and params["font_path"]:
            relative_font_path = params["font_path"]
            params["font_path"] = str(Path(FONTS_ROOT) / params["font_path"])

        img = self.render(lines_with_markup, params)

        if relative_font_path:
            params["font_path"] = relative_font_path

        if self.composer:
            img = self.composer(img, params)

        return img, text_gt, params

    def render(self, lines_with_markup, params):
        """Renders text with markup into a NumPy image array using pictex."""
        font_path = params.get("font_path", "NotoSansJP-Regular.otf")
        font_size = params.get("font_size", 32)
        color = params.get("color", "black")
        vertical = params.get("vertical", False)
        effect = params.get("effect", "none")

        canvas = Canvas().font_family(str(font_path)).font_size(font_size).color(color)

        if not lines_with_markup:
            return np.array(canvas.render(""))

        def create_text_component(text, size_multiplier=1.0):
            component = Text(text).font_size(font_size * size_multiplier)
            if effect == "stroke":
                component = component.text_stroke(
                    width=params.get("stroke_width", 1),
                    color=params.get("stroke_color", "white"),
                )
            elif effect == "glow":
                shadow = Shadow(
                    offset=params.get("shadow_offset", (0, 0)),
                    blur_radius=params.get("shadow_blur", 5),
                    color=params.get("shadow_color", "black"),
                )
                component = component.text_shadows(shadow)
            return component

        def create_component(chunk):
            if isinstance(chunk, str):
                return create_text_component(chunk)
            type, *args = chunk
            if type == 'furigana':
                base, ruby = args
                return Column(create_text_component(ruby, 0.5), create_text_component(base)).gap(0).horizontal_align("center")
            if type == 'tcy':
                text, = args
                return Row(*[create_text_component(c) for c in text]).gap(0).vertical_align("center") if vertical else create_text_component(text)
            return Text("")

        if vertical:
            line_columns = []
            for line_chunks in lines_with_markup:
                components = [comp for chunk in line_chunks for comp in ([create_text_component(c) for c in chunk] if isinstance(chunk, str) else [create_component(chunk)])]
                line_columns.append(Column(*components).gap(0).horizontal_align("center"))
            composed_element = Row(*line_columns[::-1]).gap(font_size // 2).vertical_align("top")
        else:
            line_rows = [Row(*[create_component(chunk) for chunk in line_chunks]).gap(0).vertical_align("bottom") for line_chunks in lines_with_markup]
            composed_element = Column(*line_rows).gap(font_size // 4).horizontal_align("left")

        image = canvas.render(composed_element)
        return image.to_numpy() if image else np.array([])
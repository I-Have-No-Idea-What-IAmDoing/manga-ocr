"""Core component for generating synthetic manga-style text images.

This script defines the `SyntheticDataGenerator` class, which is responsible
for creating synthetic image-text pairs for training the OCR model. It combines
various elements such as random text generation, font selection, and complex
text styling (e.g., furigana) to produce a diverse and realistic dataset that
mimics the appearance of text in manga.
"""

from pathlib import Path
import numpy as np

from manga_ocr_dev.env import FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.common.base_generator import BaseDataGenerator
from manga_ocr_dev.synthetic_data_generator.common.composer import Composer
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.common.utils import (
    is_ascii,
    is_kanji,
)


class SyntheticDataGenerator(BaseDataGenerator):
    """Generates synthetic manga-style text images and their transcriptions.

    This class orchestrates the process of creating synthetic data. It can
    generate random Japanese text or use text from a corpus, then render it
    into an image using a variety of fonts, styles, and effects like furigana
    to mimic the appearance of text in manga.

    Attributes:
        renderer (Renderer): The renderer instance used to create images from
            text.
        composer (Composer): The composer instance for background composition.
    """

    def __init__(self, background_dir=None, target_size=None, min_output_size=None, renderer=None):
        """Initializes the SyntheticDataGenerator.

        This involves loading all necessary assets for data generation.

        Args:
            background_dir (str or Path, optional): Path to the directory
                containing background images.
            target_size (tuple[int, int], optional): The final output size
                (width, height) for the composed image.
            min_output_size (int, optional): The minimum size for the smallest
                dimension of the composed image.
            renderer (Renderer, optional): An instance of the `Renderer` class.
        """
        super().__init__()
        # Initialize the renderer, creating a new instance if one is not provided
        self.renderer = renderer if renderer else Renderer()
        # Initialize the composer for background images if a directory is provided
        if background_dir:
            self.composer = Composer(background_dir, target_size=target_size, min_output_size=min_output_size)
        else:
            self.composer = None

    def process(self, text=None, override_css_params=None):
        """
        Generates a single synthetic image-text pair with a retry mechanism.
        If a sample generation fails, it will be retried up to 3 times before being skipped.
        A failure is defined as either an exception being raised or the returned image being None.
        """
        for i in range(4):
            try:
                # Attempt to generate the sample
                result = self._process(text, override_css_params)
                # Check if the image was successfully generated (not None)
                if result[0] is not None:
                    return result
                # If the image is None on the last retry, return the failed result
                if i == 3:
                    return result
            except Exception as e:
                # If an exception occurs on the last retry, re-raise it
                if i == 3:
                    raise e

    def _process(self, text=None, override_css_params=None):
        """Generates a single synthetic image-text pair.

        This method generates random text (if not provided), renders it to an
        image with randomized styling, and then composes it with a background.

        Args:
            text (str, optional): The source text to render.
            override_css_params (dict, optional): A dictionary of CSS
                parameters to override the default randomized styles.

        Returns:
            A tuple containing:
                - np.ndarray: The rendered image as a NumPy array.
                - str: The ground truth text.
                - dict: A dictionary of the CSS parameters used for rendering.
        """
        if override_css_params is None:
            override_css_params = {}

        # If no text is provided, generate random words using a random font
        if text is None:
            if "font_path" not in override_css_params:
                font_path = self.get_random_font()
                vocab = self.font_map[font_path]
                override_css_params["font_path"] = font_path
            else:
                font_path = override_css_params["font_path"]
                vocab = self.font_map[font_path]
            words = self.get_random_words(vocab)
        # If text is provided, clean and split it into words
        else:
            text = text.replace("　", " ").replace("…", "...")
            words = self.split_into_words(text)

        # Arrange words into lines and create the ground truth text
        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        # If a font is not specified, select one that supports the characters in the text
        if "font_path" not in override_css_params:
            try:
                # Attempt to find a random font that supports all characters in the text
                override_css_params["font_path"] = self.get_random_font(text_gt)
            except ValueError:
                # If no suitable font is found, return an empty image and text to skip the sample
                img, params = self.renderer.render([], override_css_params)
                return img, "", params

        # Verify that the selected font supports all characters in the text.
        # This is a fallback mechanism in case the provided font is not suitable.
        font_path = override_css_params.get("font_path")
        vocab = self.font_map.get(font_path)
        if vocab:
            unsupported_chars = {c for c in text_gt if c not in vocab and not c.isspace()}
            if unsupported_chars:
                original_vocab = vocab
                try:
                    # If the current font is missing characters, try to find a fallback font
                    font_path = self.get_random_font(text_gt)
                    override_css_params["font_path"] = font_path
                    vocab = self.font_map.get(font_path)
                except ValueError:
                    # If no fallback font is found, strip the unsupported characters from the text
                    vocab = original_vocab
                    text_gt = "".join([c for c in text_gt if c in vocab or c.isspace()])
                    lines = self.words_to_lines(self.split_into_words(text_gt))
        else:
            # If the font has no associated vocabulary, assume it supports all characters
            vocab = None

        # If there is no text to render, return an empty image and text
        if not text_gt.strip():
            img, params = self.renderer.render([], override_css_params)
            return img, "", params

        # Randomly decide whether to add furigana to the text
        if np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            if lines:
                lines = [self.add_random_furigana(line, word_prob, vocab) for line in lines]

        # Convert the relative font path to an absolute path for the renderer
        relative_font_path = None
        if "font_path" in override_css_params:
            relative_font_path = override_css_params["font_path"]
            override_css_params["font_path"] = str(Path(FONTS_ROOT) / override_css_params["font_path"])

        # Render the text to an image
        img, params = self.renderer.render(lines, override_css_params)

        # Restore the relative font path in the returned parameters
        if relative_font_path:
            params["font_path"] = relative_font_path

        # If a composer is available, compose the rendered text onto a background
        if self.composer:
            if 'text_color' in params:
                params['color'] = params['text_color']
            img = self.composer(img, params)

        return img, text_gt, params

    def add_random_furigana(self, line, word_prob=1.0, vocab=None):
        """Adds random furigana to kanji characters in a line of text.

        This method processes a line of text, identifies sequences of kanji
        characters, and randomly adds furigana (ruby text) to them. This is
        used to generate more realistic and diverse training data that mimics
        the appearance of text in manga. It also handles special styling for
        short ASCII sequences.

        Args:
            line (str): The line of text to process.
            word_prob (float, optional): The probability of adding furigana to
                a group of kanji characters. Defaults to 1.0.
            vocab (set, optional): A set of allowed characters for generating
                furigana. If None, the default vocabulary is used. Defaults to
                None.

        Returns:
            str: The processed line with HTML-like tags for furigana and other
            styling.
        """
        if vocab is None:
            vocab = self.vocab

        # Ensure that the character sources for furigana are subsets of the font's vocabulary
        hiragana_in_vocab = set(self.hiragana).intersection(vocab)
        katakana_in_vocab = set(self.katakana).intersection(vocab)

        def flush_kanji_group(group):
            """Processes a group of consecutive kanji characters, adding furigana."""
            if not group:
                return ""
            # Randomly decide whether to add furigana
            if np.random.uniform() < word_prob:
                # Determine the length and character set for the furigana
                furigana_len = int(np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(group))

                # Use only characters available in the current font for furigana
                char_source_map = {
                    "hiragana": hiragana_in_vocab if hiragana_in_vocab else vocab,
                    "katakana": katakana_in_vocab if katakana_in_vocab else vocab,
                    "all": vocab,
                }
                char_source_key = np.random.choice(
                    ["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05]
                )
                char_source = char_source_map[char_source_key]

                # If the character source is empty, do not add furigana
                if not char_source:
                    return group

                # Generate the furigana and wrap it in ruby tags
                furigana = "".join(np.random.choice(list(char_source), furigana_len))
                return f"<ruby>{group}<rt>{furigana}</rt></ruby>"
            return group

        def flush_ascii_group(group):
            """Processes a group of consecutive ASCII characters, adding styling."""
            if not group:
                return ""
            # For short ASCII sequences, randomly apply tate-chu-yoko styling
            if len(group) <= 3 and np.random.uniform() < 0.7:
                return f'<span style="text-combine-upright: all">{group}</span>'
            return group

        processed, kanji_group, ascii_group = "", "", ""
        # Iterate through the line character by character to identify and group character types
        for c in line:
            if is_kanji(c):
                # If an ASCII group is active, process it before starting a new kanji group
                if ascii_group:
                    processed += flush_ascii_group(ascii_group)
                    ascii_group = ""
                kanji_group += c
            elif is_ascii(c):
                # If a kanji group is active, process it before starting a new ASCII group
                if kanji_group:
                    processed += flush_kanji_group(kanji_group)
                    kanji_group = ""
                ascii_group += c
            else:
                # If the character is neither kanji nor ASCII, process any active groups
                if kanji_group:
                    processed += flush_kanji_group(kanji_group)
                    kanji_group = ""
                if ascii_group:
                    processed += flush_ascii_group(ascii_group)
                    ascii_group = ""
                processed += c

        # Process any remaining character groups at the end of the line
        if kanji_group:
            processed += flush_kanji_group(kanji_group)
        if ascii_group:
            processed += flush_ascii_group(ascii_group)

        return processed
"""Core component for generating synthetic manga-style text images.

This script defines the `SyntheticDataGeneratorV2` class, which is responsible
for creating synthetic image-text pairs for training the OCR model. It combines
various elements such as random text generation, font selection, and complex
text styling (e.g., furigana) to produce a diverse and realistic dataset that
mimics the appearance of text in manga.
"""

from pathlib import Path

import budoux
import numpy as np
import pandas as pd
from pictex import Canvas, Row, Column, Text, Shadow

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator_v2.composer import Composer
from manga_ocr_dev.synthetic_data_generator_v2.utils import (
    get_charsets,
    get_font_meta,
    is_kanji,
    is_ascii,
)


class SyntheticDataGeneratorV2:
    """Generates synthetic manga-style text images and their transcriptions.

    This class orchestrates the process of creating synthetic data. It can
    generate random Japanese text or use text from a corpus, then render it
    into an image using a variety of fonts, styles, and effects like furigana
    to mimic the appearance of text in manga.

    Attributes:
        vocab (set[str]): The set of all characters supported by the generator.
        hiragana (set[str]): A subset of `vocab` containing hiragana characters.
        katakana (set[str]): A subset of `vocab` containing katakana characters.
        len_to_p (pd.DataFrame): A DataFrame defining the probability
            distribution of text lengths, used for generating random text.
        parser (budoux.Parser): A parser for splitting Japanese text into
            semantically coherent chunks.
        fonts_df (pd.DataFrame): A DataFrame with metadata about available fonts.
        font_map (dict[str, set[str]]): A mapping from font paths to the set of
            characters each font supports.
        font_labels (list[str]): A list of font categories ('common', 'special',
            etc.) used for weighted sampling.
        font_p (np.ndarray): An array of sampling probabilities corresponding
            to `font_labels`.
    """

    def __init__(self, background_dir=None, min_font_size=30, max_font_size=60, target_size=None, min_output_size=None):
        """Initializes the SyntheticDataGenerator.

        This involves loading all necessary assets for data generation,
        including character sets from `vocab.csv`, font metadata from
        `fonts.csv`, and the BudouX text parser.

        Args:
            background_dir (str or Path, optional): Path to the directory
                containing background images. If provided, a Composer will be
                initialized. Defaults to None.
            min_font_size (int): The minimum font size for text rendering.
            max_font_size (int): The maximum font size for text rendering.
            target_size (tuple[int, int], optional): The final output size
                (width, height) for the composed image. Passed to the
                Composer. Defaults to None.
            min_output_size (int, optional): The minimum size for the smallest
                dimension of the composed image. Passed to the Composer.
                Defaults to None.
        """
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self.parser = budoux.load_default_japanese_parser()
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self.get_font_labels_prob()
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        if background_dir:
            self.composer = Composer(background_dir, target_size=target_size, min_output_size=min_output_size)
        else:
            self.composer = None

    def get_random_render_params(self):
        """Generates a dictionary of random parameters for text rendering.

        This method creates a set of randomized parameters that control the
        visual appearance of the generated text, including orientation
        (vertical/horizontal), font size, color, and special effects like
        strokes or glows.

        Returns:
            dict: A dictionary containing rendering parameters such as
            'vertical', 'font_size', 'color', and effect-specific parameters.
        """
        params = {}
        params['vertical'] = np.random.choice([True, False], p=[0.8, 0.2])
        params['font_size'] = np.random.randint(self.min_font_size, self.max_font_size)

        # Bias text color towards black or white for better contrast, which is
        # common in manga.
        if np.random.rand() < 0.3:
            # Dark extreme (0-30)
            gray_value = np.random.randint(0, 30)
        else:
            # Light extreme (225-255)
            gray_value = np.random.randint(225, 256)

        params['color'] = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

        # Randomly select a text effect. Strokes are common, glows are less
        # common, and no effect is also frequent.
        effect = np.random.choice(
            ["stroke", "glow", "none"], p=[0.4, 0.15, 0.45]
        )
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

    def get_font_labels_prob(self):
        """Gets font labels and their sampling probabilities from metadata.

        The probabilities are based on predefined weights for 'common',
        'regular', and 'special' font labels, which allows for controlling
        the frequency of different font types in the generated data.

        Returns:
            A tuple containing:
                - list[str]: A list of unique font labels.
                - np.ndarray: A NumPy array of corresponding sampling
                  probabilities.
        """
        labels = {
            "common": 0.2,
            "regular": 0.75,
            "special": 0.05,
        }
        labels = {k: labels[k] for k in self.fonts_df.label.unique()}
        p = np.array(list(labels.values()))
        p = p / p.sum()
        labels = list(labels.keys())
        return labels, p

    def process(self, text=None, override_params=None):
        """Processes text to generate a synthetic image and ground truth.

        This is the main method for the data generation workflow. It takes
        optional input text, or generates random text if none is provided.
        It then handles font selection, markup generation (like furigana),
        rendering the text to an image, and finally composing it with a
        background if a composer is available.

        Args:
            text (str, optional): The input text to render. If None, random
                words will be generated. Defaults to None.
            override_params (dict, optional): A dictionary of rendering
                parameters to override the randomly generated ones. Defaults
                to None.

        Returns:
            A tuple containing:
                - np.ndarray or None: The final generated image as a NumPy
                  array, or None on failure.
                - str: The ground truth text corresponding to the image.
                - dict: The dictionary of parameters used for generation.

        Raises:
            ValueError: If the input text contains characters that are not
                supported by the selected font.
        """
        params = self.get_random_render_params()
        if override_params:
            params.update(override_params)

        # If no text is provided, generate random words. Otherwise, parse the
        # input text.
        if text is None:
            # If a font is not already specified, select one randomly to
            # determine the available character set for word generation.
            if "font_path" not in params:
                font_path = self.get_random_font()
                vocab = self.font_map[font_path]
                params["font_path"] = font_path
            else:
                font_path = params["font_path"]
                vocab = self.font_map[font_path]

            words = self.get_random_words(vocab)

        else:
            # Preprocess the input text.
            text = text.replace("　", " ")
            text = text.replace("…", "...")
            words = self.split_into_words(text)

        # Arrange the words into lines.
        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        # If a font hasn't been chosen yet, select one that supports the
        # generated text.
        if "font_path" not in params:
            params["font_path"] = self.get_random_font(text_gt)

        # Verify that the chosen font supports all characters in the text.
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
            params["font_path"] = str(
                Path(FONTS_ROOT) / params["font_path"]
            )

        img = self.render(lines_with_markup, params)

        if relative_font_path:
            params["font_path"] = relative_font_path

        if self.composer:
            img = self.composer(img, params)

        return img, text_gt, params

    def add_random_furigana(self, line, word_prob=1.0, vocab=None):
        """Adds furigana and other markup to a line of text.

        This method processes a line of text, identifying groups of kanji and
        ASCII characters. It then randomly applies furigana (ruby text) to
        kanji groups and Tate-Chu-Yoko (TCY) markup to short ASCII groups.
        The output is a list of chunks, where each chunk is either a string
        or a tuple representing marked-up text.

        Args:
            line (str): The line of text to process.
            word_prob (float): The probability of applying furigana to a
                kanji group.
            vocab (set[str], optional): The character set to use for
                generating furigana text. If None, the generator's default
                vocabulary is used.

        Returns:
            list: A list of processed chunks. Strings are plain text, while
            tuples like ('furigana', base, ruby) or ('tcy', text) represent
            text with markup.
        """
        if vocab is None:
            vocab = self.vocab

        def flush_kanji_group(group):
            if not group:
                return []
            # Randomly decide whether to add furigana to the current group of
            # kanji characters.
            if np.random.uniform() < word_prob:
                # Determine the length of the furigana, which is a random
                # multiplier of the kanji group's length.
                furigana_len = int(
                    np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(group)
                )
                # Choose the character set for the furigana (hiragana is most
                # common).
                char_source = np.random.choice(
                    ["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05]
                )
                char_source = {
                    "hiragana": self.hiragana,
                    "katakana": self.katakana,
                    "all": vocab,
                }[char_source]
                furigana = "".join(np.random.choice(list(char_source), furigana_len))
                # Return a markup tuple for furigana.
                return [('furigana', group, furigana)]
            else:
                # If no furigana is added, return the kanji group as a plain string.
                return [group]

        def flush_ascii_group(group):
            if not group:
                return []
            # For short ASCII groups, randomly apply Tate-Chu-Yoko (TCY), which
            # renders horizontal text within a vertical layout.
            if len(group) <= 3 and np.random.uniform() < 0.7:
                return [('tcy', group)]
            else:
                return [group]

        processed_chunks = []
        kanji_group = ""
        ascii_group = ""

        # Iterate through the line character by character to identify and group
        # consecutive kanji or ASCII characters.
        for c in line:
            if is_kanji(c):
                if ascii_group:
                    processed_chunks.extend(flush_ascii_group(ascii_group))
                    ascii_group = ""
                kanji_group += c
            elif is_ascii(c):
                if kanji_group:
                    processed_chunks.extend(flush_kanji_group(kanji_group))
                    kanji_group = ""
                ascii_group += c
            else:
                # If the character is neither kanji nor ASCII, flush any active
                # groups and append the character as a plain string.
                if kanji_group:
                    processed_chunks.extend(flush_kanji_group(kanji_group))
                    kanji_group = ""
                if ascii_group:
                    processed_chunks.extend(flush_ascii_group(ascii_group))
                    ascii_group = ""
                processed_chunks.append(c)

        # Flush any remaining groups at the end of the line.
        if kanji_group:
            processed_chunks.extend(flush_kanji_group(kanji_group))
        if ascii_group:
            processed_chunks.extend(flush_ascii_group(ascii_group))

        # Re-join adjacent string chunks for cleaner output.
        final_chunks = []
        current_string = ""
        for chunk in processed_chunks:
            if isinstance(chunk, str):
                current_string += chunk
            else:
                if current_string:
                    final_chunks.append(current_string)
                    current_string = ""
                final_chunks.append(chunk)
        if current_string:
            final_chunks.append(current_string)

        return final_chunks

    def render(self, lines_with_markup, params):
        """Renders text with markup into a NumPy image array.

        This method uses the `pictex` library to render lines of text that
        may include markup for furigana and TCY. It constructs a canvas and
        assembles text components based on the provided parameters for font,
        size, color, orientation, and effects.

        Args:
            lines_with_markup (list[list]): A list of lines, where each line
                is a list of chunks (strings or markup tuples).
            params (dict): A dictionary of rendering parameters, including
                'font_path', 'font_size', 'color', 'vertical', and effect
                details.

        Returns:
            np.ndarray: The rendered image as a NumPy array. Returns an empty
            array if the rendering fails or if there's no text to render.
        """
        font_path = params.get("font_path", "NotoSansJP-Regular.otf")
        font_size = params.get("font_size", 32)
        color = params.get("color", "black")
        vertical = params.get("vertical", False)
        effect = params.get("effect", "none")

        canvas = (
            Canvas()
            .font_family(str(font_path))
            .font_size(font_size)
            .color(color)
            .padding(10)
        )

        if not lines_with_markup:
            return np.array(canvas.render(""))

        # Helper function to create a basic text component with optional effects.
        def create_text_component(text, size_multiplier=1.0):
            component = Text(text).font_size(font_size * size_multiplier)
            if effect == "stroke":
                component = component.text_stroke(
                    width=params.get("stroke_width", 1),
                    color=params.get("stroke_color", "white"),
                )
            elif effect == "glow":
                offset_x, offset_y = params.get("shadow_offset", (0, 0))
                shadow = Shadow(
                    offset=(offset_x, offset_y),
                    blur_radius=params.get("shadow_blur", 5),
                    color=params.get("shadow_color", "black"),
                )
                component = component.text_shadows(shadow)
            return component

        # Helper function to create a component from a chunk, handling markup.
        def create_component(chunk):
            if isinstance(chunk, str):
                return create_text_component(chunk)

            type, *args = chunk
            # Handle 'furigana' markup by creating a column of ruby over base text.
            if type == 'furigana':
                base, ruby = args
                furigana_text = create_text_component(ruby, size_multiplier=0.5)
                base_text = create_text_component(base)
                return Column(furigana_text, base_text).gap(0).horizontal_align("center")

            # Handle 'tcy' (Tate-Chu-Yoko) markup.
            if type == 'tcy':
                text, = args
                # In vertical mode, render TCY as a horizontal row of characters.
                if vertical:
                    return Row(*[create_text_component(c) for c in text]).gap(0).vertical_align("center")
                else:
                    return create_text_component(text)

            return Text("")

        if vertical:
            # For vertical layout, each line is a column of characters/components.
            line_columns = []
            for line_chunks in lines_with_markup:
                components = []
                for chunk in line_chunks:
                    # Deconstruct strings into individual character components for
                    # vertical rendering.
                    if isinstance(chunk, str):
                        components.extend([create_text_component(c) for c in chunk])
                    else:
                        components.append(create_component(chunk))
                line_columns.append(Column(*components).gap(0).horizontal_align("center"))

            # The final layout is a row of these columns, arranged right-to-left.
            composed_element = Row(*line_columns[::-1]).gap(font_size // 2).vertical_align("top")
        else:
            # For horizontal layout, each line is a row of components.
            line_rows = []
            for line_chunks in lines_with_markup:
                components = [create_component(chunk) for chunk in line_chunks]
                line_rows.append(Row(*components).gap(0).vertical_align("bottom"))

            # The final layout is a column of these rows.
            composed_element = Column(*line_rows).gap(font_size // 4).horizontal_align("left")

        image = canvas.render(composed_element)
        if image is None:
            return np.array([])  # Return empty array on failure
        return image.to_numpy()

    def get_random_words(self, vocab):
        """Generates a list of random words from a given vocabulary.

        The total length of the generated text is determined by a probability
        distribution loaded from `len_to_p.csv`.

        Args:
            vocab (list or set): A collection of characters to sample from.

        Returns:
            list[str]: A list of randomly generated words.
        """
        vocab = list(vocab)
        max_text_len = np.random.choice(self.len_to_p["len"], p=self.len_to_p["p"])

        words = []
        text_len = 0
        while True:
            word = "".join(np.random.choice(vocab, np.random.randint(1, 4)))
            words.append(word)
            text_len += len(word)
            if text_len >= max_text_len:
                break

        return words

    def split_into_words(self, text):
        """Splits a given text into semantically coherent words.

        This method uses the BudouX parser to split Japanese text into words.
        It also truncates the list of words based on a randomly determined
        maximum text length.

        Args:
            text (str): The input text to be split.

        Returns:
            list[str]: A list of words.
        """
        max_text_len = np.random.choice(self.len_to_p["len"], p=self.len_to_p["p"])

        words = []
        text_len = 0
        for word in self.parser.parse(text):
            words.append(word)
            text_len += len(word)
            if text_len >= max_text_len:
                break

        return words

    def words_to_lines(self, words):
        """Arranges a list of words into lines with a maximum length.

        This method implements a simple line-breaking algorithm. It joins
        words into lines, ensuring that each line does not exceed a randomly
        determined maximum length.

        Args:
            words (list[str]): A list of words to be arranged into lines.

        Returns:
            list[str]: A list of strings, where each string is a formatted line.
        """
        text = "".join(words)

        # Determine a dynamic maximum line length using a Poisson distribution
        # to create more natural-looking, varied line lengths.
        max_num_lines = 10
        min_line_len = len(text) // max_num_lines if text else 0
        max_line_len = 20
        if min_line_len > max_line_len:
            max_line_len = min_line_len + 5
        max_line_len = np.clip(np.random.poisson(10), min_line_len, max_line_len)

        # A simple greedy algorithm for line breaking.
        lines = []
        current_line = ""
        for word in words:
            # If adding the next word exceeds the max length, start a new line.
            if len(current_line) + len(word) > max_line_len:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line += word
        # Add the last line if it's not empty.
        if current_line:
            lines.append(current_line)

        return lines

    def is_font_supporting_text(self, font_path, text):
        """Checks if a given font supports all characters in a text.

        Args:
            font_path (str): The path to the font file.
            text (str): The text to check for character support.

        Returns:
            bool: True if the font supports all non-whitespace characters in
            the text, False otherwise.
        """
        chars = self.font_map[font_path]
        for c in text:
            if c.isspace():
                continue
            if c not in chars:
                return False
        return True

    def get_random_font(self, text=None):
        """Selects a random font, optionally filtered by text support.

        This method randomly selects a font based on predefined label
        probabilities ('common', 'regular', 'special'). If `text` is provided,
        it ensures the selected font can render all characters in the text.

        Args:
            text (str, optional): The text that the font must support.
                Defaults to None.

        Returns:
            str: The file path of the selected font.

        Raises:
            ValueError: If no font can be found that supports all characters
                in the provided text.
        """
        label = np.random.choice(self.font_labels, p=self.font_p)
        df = self.fonts_df[self.fonts_df.label == label]
        if text is None:
            return df.sample(1).iloc[0].font_path
        valid_mask = df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))
        if not valid_mask.any():
            valid_mask = self.fonts_df.font_path.apply(
                lambda x: self.is_font_supporting_text(x, text)
            )
            df = self.fonts_df
            if not valid_mask.any():
                unsupported_chars = {
                    c for c in text if not self.is_char_supported_by_any_font(c)
                }
                raise ValueError(
                    f"Text contains unsupported characters: {''.join(unsupported_chars)}"
                )
        return df[valid_mask].sample(1).iloc[0].font_path

    def is_char_supported_by_any_font(self, char):
        """Checks if a character is supported by at least one available font.

        Args:
            char (str): The character to check.

        Returns:
            bool: True if at least one font in the `font_map` supports the
            character, False otherwise.
        """
        return any(char in font_chars for font_chars in self.font_map.values())
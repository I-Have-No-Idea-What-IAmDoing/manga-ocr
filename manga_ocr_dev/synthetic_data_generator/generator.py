import budoux
import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.utils import (
    get_font_meta,
    get_charsets,
    is_ascii,
    is_kanji,
)


class SyntheticDataGenerator:
    """Generates synthetic manga-style text images and their transcriptions.

    This class orchestrates the process of creating synthetic data for training
    the Manga OCR model. It can generate random Japanese text or use provided
    text, then renders it into an image using various fonts, styles, and
    effects like furigana to mimic the appearance of text in manga.

    Attributes:
        vocab (set[str]): The set of all characters supported by the generator.
        hiragana (set[str]): A set of hiragana characters.
        katakana (set[str]): A set of katakana characters.
        len_to_p (pd.DataFrame): A DataFrame defining the probability
            distribution of text lengths.
        parser (budoux.Parser): A parser for splitting Japanese text into
            semantically correct units.
        fonts_df (pd.DataFrame): A DataFrame with metadata about available fonts.
        font_map (dict[str, set[str]]): A mapping from font paths to the set of
            characters each font supports.
        font_labels (list[str]): A list of font categories (e.g., 'common').
        font_p (np.ndarray): An array of sampling probabilities for each font
            category.
        renderer (Renderer): The renderer instance used to create images from text.
    """
    def __init__(self, renderer=None):
        """Initializes the SyntheticDataGenerator.

        Args:
            renderer (Renderer, optional): An instance of the `Renderer` class.
                If not provided, a new one will be created.
        """
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self.parser = budoux.load_default_japanese_parser()
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self.get_font_labels_prob()
        self.renderer = renderer if renderer else Renderer()

    def process(self, text=None, override_css_params=None):
        """Generates a single image-text pair.

        This method can either use a provided source text or generate random
        Japanese text. It then renders the text into an image with randomized
        styling.

        Args:
            text (str, optional): The source text to render. If None, random
                text will be generated. Defaults to None.
            override_css_params (dict, optional): A dictionary of CSS parameters
                to override the default rendering styles. Defaults to None.

        Returns:
            tuple[np.ndarray, str, dict]: A tuple containing the rendered
            image as a NumPy array, the ground truth text, and a dictionary of
            the CSS parameters used for rendering.
        """

        if override_css_params is None:
            override_css_params = {}

        if text is None:
            # if using random text, choose font first,
            # and then generate text using only characters supported by that font
            if "font_path" not in override_css_params:
                font_path = self.get_random_font()
                vocab = self.font_map[font_path]
                override_css_params["font_path"] = font_path
            else:
                font_path = override_css_params["font_path"]
                vocab = self.font_map[font_path]

            words = self.get_random_words(vocab)

        else:
            text = text.replace("　", " ")
            text = text.replace("…", "...")
            words = self.split_into_words(text)

        lines = self.words_to_lines(words)
        text_gt = "\n".join(lines)

        if "font_path" not in override_css_params:
            override_css_params["font_path"] = self.get_random_font(text_gt)

        font_path = override_css_params.get("font_path")
        if font_path:
            vocab = self.font_map.get(font_path)

            # remove unsupported characters
            lines = ["".join([c for c in line if c in vocab]) for line in lines]
            text_gt = "\n".join(lines)
        else:
            vocab = None

        if np.random.random() < 0.5:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])

            lines = [self.add_random_furigana(line, word_prob, vocab) for line in lines]

        img, params = self.renderer.render(lines, override_css_params)
        return img, text_gt, params

    def get_random_words(self, vocab):
        """Generates a list of random words from a given vocabulary.

        The length of the generated text is determined by a probability
        distribution of text lengths.

        Args:
            vocab (list or set): A collection of characters to be used for
                generating words.

        Returns:
            list: A list of randomly generated words.
        """
        vocab = list(vocab)
        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)

        words = []
        text_len = 0
        while True:
            word = "".join(np.random.choice(vocab, np.random.randint(1, 4)))
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def split_into_words(self, text):
        """Splits a given text into words using the BudouX parser.

        The function also truncates the text to a random length based on a
        predefined probability distribution.

        Args:
            text (str): The input text to be split.

        Returns:
            list: A list of words.
        """
        max_text_len = np.random.choice(self.len_to_p.len, p=self.len_to_p.p)

        words = []
        text_len = 0
        for word in self.parser.parse(text):
            words.append(word)
            text_len += len(word)
            if text_len + len(word) >= max_text_len:
                break

        return words

    def words_to_lines(self, words):
        """Converts a list of words into a list of lines.

        This function concatenates words and splits them into lines of a
        randomized maximum length.

        Args:
            words (list): A list of words.

        Returns:
            list: A list of strings, where each string is a line of text.
        """
        text = "".join(words)

        max_num_lines = 10
        min_line_len = len(text) // max_num_lines
        max_line_len = 20
        max_line_len = np.clip(np.random.poisson(6), min_line_len, max_line_len)
        lines = []
        line = ""
        for word in words:
            line += word
            if len(line) >= max_line_len:
                lines.append(line)
                line = ""
        if line:
            lines.append(line)

        return lines

    def add_random_furigana(self, line, word_prob=1.0, vocab=None):
        """Adds random furigana to kanji characters in a line of text.

        This function processes a line of text and adds furigana (ruby text)
        to kanji characters with a given probability. It can also combine
        short ASCII character sequences.

        Args:
            line (str): The input line of text.
            word_prob (float, optional): The probability of adding furigana to
                a kanji group. Defaults to 1.0.
            vocab (list or set, optional): The vocabulary to use for
                generating furigana characters. If None, the default
                vocabulary is used. Defaults to None.

        Returns:
            str: The processed line with HTML-like tags for furigana and other
            styling.
        """
        if vocab is None:
            vocab = self.vocab

        processed = ""
        kanji_group = ""
        ascii_group = ""
        for i, c in enumerate(line):
            if is_kanji(c):
                c_type = "kanji"
                kanji_group += c
            elif is_ascii(c):
                c_type = "ascii"
                ascii_group += c
            else:
                c_type = "other"

            if c_type != "kanji" or i == len(line) - 1:
                if kanji_group:
                    if np.random.uniform() < word_prob:
                        furigana_len = int(np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(kanji_group))
                        char_source = np.random.choice(["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05])
                        char_source = {
                            "hiragana": self.hiragana,
                            "katakana": self.katakana,
                            "all": vocab,
                        }[char_source]
                        furigana = "".join(np.random.choice(char_source, furigana_len))
                        processed += f"<ruby>{kanji_group}<rt>{furigana}</rt></ruby>"
                    else:
                        processed += kanji_group
                    kanji_group = ""

            if c_type != "ascii" or i == len(line) - 1:
                if ascii_group:
                    if len(ascii_group) <= 3 and np.random.uniform() < 0.7:
                        processed += f'<span style="text-combine-upright: all">{ascii_group}</span>'
                    else:
                        processed += ascii_group
                    ascii_group = ""

            if c_type == "other":
                processed += c

        return processed

    def is_font_supporting_text(self, font_path, text):
        """Checks if a given font supports all characters in a text.
        Args:
            font_path (str): The path to the font file.
            text (str): The text to check.
        Returns:
            bool: True if the font supports all characters in the text,
            False otherwise.
        """
        font_path_abs = str(FONTS_ROOT / font_path)
        chars = self.font_map[font_path_abs]
        for c in text:
            if c.isspace():
                continue
            if c not in chars:
                return False
        return True

    def get_font_labels_prob(self):
        """Gets the font labels and their sampling probabilities.
        The probabilities are based on predefined weights for 'common',
        'regular', and 'special' font labels.
        Returns:
            tuple: A tuple containing:
                - list: A list of unique font labels present in the fonts
                  metadata.
                - np.ndarray: A NumPy array of corresponding probabilities.
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

    def get_random_font(self, text=None):
        """Selects a random font.
        If text is provided, it attempts to select a font that supports all
        characters in the text. If no such font is found, it falls back to
        selecting from fonts that have a large number of characters.
        Args:
            text (str, optional): The text for which to find a supporting
                font. Defaults to None.
        Returns:
            str: The path to the selected font file.
        """
        label = np.random.choice(self.font_labels, p=self.font_p)
        df = self.fonts_df[self.fonts_df.label == label]
        if text is None:
            return str(FONTS_ROOT / df.sample(1).iloc[0].font_path)
        valid_mask = df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))
        if not valid_mask.any():
            # if text contains characters not supported by any font, just pick some of the more capable fonts
            df = self.fonts_df
            valid_mask = df.num_chars >= 4000
        return str(FONTS_ROOT / df[valid_mask].sample(1).iloc[0].font_path)
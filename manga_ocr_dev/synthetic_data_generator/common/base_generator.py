"""Base class for synthetic data generators.

This module defines the `BaseDataGenerator` class, which contains the common
logic for text processing, font selection, and word generation shared between
the different synthetic data generator implementations. This helps to avoid
code duplication and makes the system more modular and maintainable.
"""

from pathlib import Path

import budoux
import numpy as np
import pandas as pd

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.common.utils import (
    get_charsets,
    get_font_meta,
    is_ascii,
    is_kanji,
)


class BaseDataGenerator:
    """A base class for synthetic data generators.

    This class provides the core functionalities for generating synthetic text,
    including loading character sets and font metadata, splitting text into
    words and lines, and adding random furigana. It is designed to be
    extended by specific generator implementations that handle the actual
    rendering process.

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

    def __init__(self):
        """Initializes the BaseDataGenerator.

        This involves loading all necessary assets for data generation,
        including character sets from `vocab.csv`, font metadata from
        `fonts.csv`, and the BudouX text parser.
        """
        self.vocab, self.hiragana, self.katakana = get_charsets()
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        self.parser = budoux.load_default_japanese_parser()
        self.fonts_df, self.font_map = get_font_meta()
        self.font_labels, self.font_p = self.get_font_labels_prob()

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

        max_num_lines = 10
        min_line_len = len(text) // max_num_lines if text else 0
        max_line_len = 20
        if min_line_len > max_line_len:
            max_line_len = min_line_len + 5
        max_line_len = np.clip(np.random.poisson(10), min_line_len, max_line_len)

        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) > max_line_len:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line += word
        if current_line:
            lines.append(current_line)

        return lines

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
            if np.random.uniform() < word_prob:
                furigana_len = int(
                    np.clip(np.random.normal(1.5, 0.5), 1, 4) * len(group)
                )
                char_source = np.random.choice(
                    ["hiragana", "katakana", "all"], p=[0.8, 0.15, 0.05]
                )
                char_source = {
                    "hiragana": self.hiragana,
                    "katakana": self.katakana,
                    "all": vocab,
                }[char_source]
                furigana = "".join(np.random.choice(list(char_source), furigana_len))
                return [('furigana', group, furigana)]
            else:
                return [group]

        def flush_ascii_group(group):
            if not group:
                return []
            if len(group) <= 3 and np.random.uniform() < 0.7:
                return [('tcy', group)]
            else:
                return [group]

        processed_chunks = []
        kanji_group = ""
        ascii_group = ""

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
                if kanji_group:
                    processed_chunks.extend(flush_kanji_group(kanji_group))
                    kanji_group = ""
                if ascii_group:
                    processed_chunks.extend(flush_ascii_group(ascii_group))
                    ascii_group = ""
                processed_chunks.append(c)

        if kanji_group:
            processed_chunks.extend(flush_kanji_group(kanji_group))
        if ascii_group:
            processed_chunks.extend(flush_ascii_group(ascii_group))

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

    def is_font_supporting_text(self, font_path, text):
        """Checks if a given font supports all characters in a text.

        Args:
            font_path (str): The path to the font file.
            text (str): The text to check for character support.

        Returns:
            bool: True if the font supports all non-whitespace characters in
            the text, False otherwise.
        """
        chars = self.font_map.get(font_path)
        if not chars:
            return False
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

        # Use relative paths for checking support
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
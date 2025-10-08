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
from pykakasi import kakasi

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
        # Load character sets (full vocabulary, hiragana, katakana)
        self.vocab, self.hiragana, self.katakana = get_charsets()
        # Load the probability distribution for text lengths
        self.len_to_p = pd.read_csv(ASSETS_PATH / "len_to_p.csv")
        # Initialize the BudouX parser for semantic text splitting
        self.parser = budoux.load_default_japanese_parser()

        # Initialize the pykakasi converter for furigana generation
        self.kakasi = kakasi()

        # Load font metadata and the character support map
        self.fonts_df, self.font_map = get_font_meta()
        # Determine font labels and their sampling probabilities for weighted selection
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
        # Define the base weights for each font category
        labels = {
            "common": 0.2,
            "regular": 0.75,
            "special": 0.05,
        }
        # Filter the labels to include only those present in the loaded fonts
        labels = {k: labels[k] for k in self.fonts_df.label.unique()}
        # Convert the weights to a NumPy array and normalize them to sum to 1
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
        # Determine the maximum length of the text to be generated based on the probability distribution
        max_text_len = np.random.choice(self.len_to_p["len"], p=self.len_to_p["p"])

        words = []
        text_len = 0
        # Continuously generate words until the total text length reaches the maximum
        while True:
            # Generate a random word with a length between 1 and 3 characters
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
        # Determine the maximum text length from the probability distribution
        max_text_len = np.random.choice(self.len_to_p["len"], p=self.len_to_p["p"])

        words = []
        text_len = 0
        # Use the BudouX parser to split the text into words
        for word in self.parser.parse(text):
            words.append(word)
            text_len += len(word)
            # Stop once the total text length reaches the maximum
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

        # Determine a dynamic maximum line length based on the total text length
        max_num_lines = 10
        min_line_len = len(text) // max_num_lines if text else 0
        max_line_len = 20
        if min_line_len > max_line_len:
            max_line_len = min_line_len + 5
        # Add randomness to the line length for more variety
        max_line_len = np.clip(np.random.poisson(10), min_line_len, max_line_len)

        lines = []
        current_line = ""
        # Iterate through the words and arrange them into lines
        for word in words:
            # If adding the next word exceeds the max line length, start a new line
            if len(current_line) + len(word) > max_line_len:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line += word
        # Add the last line to the list
        if current_line:
            lines.append(current_line)

        return lines

    def add_random_furigana(self, line, word_prob=1.0):
        """Adds furigana and other markup to a line of text using pykakasi.

        This method processes a line of text, converting kanji to their phonetic
        (hiragana) readings. It then randomly applies furigana markup to the
        kanji. It also handles Tate-Chu-Yoko (TCY) markup for short ASCII groups.
        The output is a list of chunks, where each chunk is either a string
        or a tuple representing marked-up text.

        Args:
            line (str): The line of text to process.
            word_prob (float): The probability of applying furigana to a
                kanji group.

        Returns:
            list: A list of processed chunks. Strings are plain text, while
            tuples like ('furigana', base, ruby) or ('tcy', text) represent
            text with markup.
        """
        processed_chunks = []
        # Use kakasi to convert the line into a list of dictionaries,
        # each containing the original text, hiragana, and other forms.
        converted = self.kakasi.convert(line)

        for item in converted:
            original = item['orig']
            hiragana = item['hira']

            # Check if the original text contains any kanji characters.
            if any(is_kanji(c) for c in original):
                # Randomly decide whether to add furigana to this kanji group.
                if np.random.uniform() < word_prob:
                    # Create a furigana markup tuple.
                    processed_chunks.append(('furigana', original, hiragana))
                else:
                    # If not adding furigana, just append the original kanji text.
                    processed_chunks.append(original)
            # Check if the text is a short ASCII sequence (e.g., numbers, acronyms).
            elif any(is_ascii(c) for c in original) and len(original) <= 3:
                # Randomly apply Tate-Chu-Yoko (TCY) markup.
                if np.random.uniform() < 0.7:
                    processed_chunks.append(('tcy', original))
                else:
                    processed_chunks.append(original)
            else:
                # For all other text (hiragana, katakana, punctuation), append as is.
                processed_chunks.append(original)

        # The following logic to merge consecutive string chunks is retained
        # to ensure the output format is clean and optimized.
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
        # Get the set of supported characters for the given font
        chars = self.font_map.get(font_path)
        if not chars:
            return False
        # Check each character in the text, skipping whitespace
        for c in text:
            if c.isspace():
                continue
            # If any character is not in the font's supported set, return False
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
        # Select a font category based on the predefined probabilities
        label = np.random.choice(self.font_labels, p=self.font_p)
        # Filter the font DataFrame to include only fonts of the selected category
        df = self.fonts_df[self.fonts_df.label == label]
        # If no text is provided, randomly sample one font from the filtered DataFrame
        if text is None:
            return df.sample(1).iloc[0].font_path

        # If text is provided, find all fonts in the category that support it
        valid_mask = df.font_path.apply(lambda x: self.is_font_supporting_text(x, text))

        # If no supporting font is found in the category, search all fonts
        if not valid_mask.any():
            valid_mask = self.fonts_df.font_path.apply(
                lambda x: self.is_font_supporting_text(x, text)
            )
            df = self.fonts_df
            # If still no supporting font is found, raise an error
            if not valid_mask.any():
                unsupported_chars = {
                    c for c in text if not self.is_char_supported_by_any_font(c)
                }
                raise ValueError(
                    f"Text contains unsupported characters: {''.join(unsupported_chars)}"
                )
        # Randomly sample one font from the list of valid fonts
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
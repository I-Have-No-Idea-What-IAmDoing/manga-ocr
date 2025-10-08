"""Utility functions for the synthetic data generation pipeline.

This module provides helper functions that support the data generation process.
This includes functions for loading and processing assets like fonts and
backgrounds, as well as character-level utilities for identifying different
Japanese character types (kanji, hiragana, katakana) and ASCII characters.
"""

from pathlib import Path
import pandas as pd
import unicodedata

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    """Creates a DataFrame of background images and their metadata.

    This function iterates through a directory of background images, parses
    the dimensions from their filenames, and compiles the information into a
    pandas DataFrame. The resulting DataFrame is used by the `Renderer` to
    select appropriate backgrounds for compositing text.

    The filenames are expected to be in a specific format where the last four
    parts of the filename (split by '_') are ymin, ymax, xmin, and xmax.

    Args:
        background_dir (Path): The path to the directory containing the
            background images.

    Returns:
        A pandas DataFrame with columns 'path', 'h', 'w', and 'ratio',
        containing the path, height, width, and aspect ratio of each
        background image.
    """
    background_df = []
    if background_dir is None:
        return pd.DataFrame(background_df)

    for path in Path(background_dir).iterdir():
        if not path.is_file():
            continue
        try:
            ymin, ymax, xmin, xmax = [int(v) for v in path.stem.split("_")[-4:]]
            h = ymax - ymin
            w = xmax - xmin
            ratio = w / h if h > 0 else 0
            background_df.append({"path": str(path), "h": h, "w": w, "ratio": ratio})
        except (ValueError, IndexError):
            print(f"Could not parse dimensions from filename: {path.name}")
            continue

    return pd.DataFrame(background_df)


def is_kanji(ch):
    """Checks if a character is a Japanese kanji character.

    This function determines if a given character is a CJK Unified Ideograph,
    which is the block that contains kanji characters, by checking its
    Unicode name. It handles non-string inputs gracefully.

    Args:
        ch (str): The character to check. Must be a single character.

    Returns:
        True if the character is a kanji, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch)
    except (TypeError, ValueError):
        return False


def is_hiragana(ch):
    """Checks if a character is a Japanese hiragana character.

    This function determines if a given character is a hiragana character by
    checking its Unicode name. It handles non-string inputs gracefully.

    Args:
        ch (str): The character to check. Must be a single character.

    Returns:
        True if the character is a hiragana, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "HIRAGANA" in unicodedata.name(ch)
    except (TypeError, ValueError):
        return False


def is_katakana(ch):
    """Checks if a character is a Japanese katakana character.

    This function determines if a given character is a katakana character by
    checking its Unicode name. It handles non-string inputs gracefully.

    Args:
        ch (str): The character to check. Must be a single character.

    Returns:
        True if the character is a katakana, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "KATAKANA" in unicodedata.name(ch)
    except (TypeError, ValueError):
        return False


def is_ascii(ch):
    """Checks if a character is an ASCII character.

    This function determines if a given character is within the ASCII range
    (0-127) by checking its ordinal value. It handles non-string inputs
    gracefully.

    Args:
        ch (str): The character to check. Must be a single character.

    Returns:
        True if the character is an ASCII character, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return ord(ch) < 128
    except (TypeError, ValueError):
        return False


def get_charsets(vocab_path=None):
    """Loads and categorizes character sets from a vocabulary file.

    This function reads a vocabulary from a specified CSV file and separates
    the characters into different sets: the full vocabulary, hiragana, and
    katakana. These sets are used for various purposes in the data generation
    pipeline, such as generating random furigana.

    Args:
        vocab_path (str or Path, optional): The path to the vocabulary CSV
            file. If not provided, it defaults to `assets/vocab.csv`.

    Returns:
        A tuple containing three NumPy arrays:
            - The full vocabulary of characters.
            - The subset of hiragana characters.
            - The subset of katakana characters.
    """
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    hiragana = vocab[[is_hiragana(c) for c in vocab]]
    katakana = vocab[[is_katakana(c) for c in vocab]]
    return vocab, hiragana, katakana


def get_font_meta():
    """Loads font metadata and creates a character support map.

    This function reads the `fonts.csv` file from the `ASSETS_PATH`, which
    contains metadata about the fonts used for synthetic data generation. The
    paths in the CSV are expected to be relative to the `FONTS_ROOT`.

    Returns:
        A tuple containing:
            - A pandas DataFrame with the font metadata. The `font_path` column
              contains paths as they appear in the CSV.
            - A dictionary mapping each relative font path to a set of
              characters supported by that font.
    """
    df = pd.read_csv(ASSETS_PATH / "fonts.csv")
    df = df.dropna()

    # The font paths in the CSV are relative. The generators are responsible
    # for resolving them to absolute paths when needed.
    font_map = {row.font_path: set(row.supported_chars) for row in df.itertuples()}

    return df, font_map
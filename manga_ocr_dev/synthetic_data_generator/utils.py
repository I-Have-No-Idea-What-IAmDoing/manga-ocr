"""Utility functions for the synthetic data generator.

This module provides helper functions for character type checking, asset loading,
and metadata processing, which are used throughout the data generation pipeline.
"""
import pandas as pd
from pathlib import Path

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def is_kanji(char):
    """Checks if a character is a Kanji character."""
    if len(char) != 1:
        return False
    return '\u4e00' <= char <= '\u9fff'


def is_hiragana(char):
    """Checks if a character is a Hiragana character."""
    if len(char) != 1:
        return False
    return '\u3040' <= char <= '\u309f'


def is_katakana(char):
    """Checks if a character is a Katakana character."""
    if len(char) != 1:
        return False
    return '\u30a0' <= char <= '\u30ff'


def is_ascii(char):
    """Checks if a character is an ASCII character."""
    if len(char) != 1:
        return False
    return char.isascii()


def get_background_df(bg_dir):
    """Parses background image filenames to create a DataFrame with metadata."""
    bg_dir = Path(bg_dir)
    records = []
    for path in bg_dir.glob('*.png'):
        try:
            ymin, ymax, xmin, xmax = map(int, path.stem.split('_')[-4:])
            h = ymax - ymin
            w = xmax - xmin
            ratio = h / w if w > 0 else 0
            records.append({'path': path, 'h': h, 'w': w, 'ratio': ratio})
        except (ValueError, IndexError):
            # Ignore files that don't match the naming convention
            continue
    return pd.DataFrame(records)


def get_charsets():
    """Loads character sets from the vocab file."""
    path = ASSETS_PATH / 'vocab.csv'
    if not path.exists():
        raise FileNotFoundError(f"Vocab file not found at {path}")
    df = pd.read_csv(path)
    chars = df['char'].tolist()
    vocab = set(chars)
    hiragana = {c for c in chars if is_hiragana(c)}
    katakana = {c for c in chars if is_katakana(c)}
    return vocab, hiragana, katakana


def get_font_meta():
    """Loads font metadata and creates a font map."""
    path = FONTS_ROOT / 'fonts.csv'
    if not path.exists():
        raise FileNotFoundError(f"Font metadata file not found at {path}")
    df = pd.read_csv(path)
    df['font_path'] = df['font_path'].apply(lambda p: FONTS_ROOT / p)
    df['supported_chars'] = df['supported_chars'].apply(set)
    font_map = df.set_index('font_path')['supported_chars'].to_dict()
    return df, font_map
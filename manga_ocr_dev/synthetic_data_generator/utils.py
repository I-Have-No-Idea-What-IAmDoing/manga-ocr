import pandas as pd
import unicodedata

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    """Creates a DataFrame of background images and their metadata.

    This function iterates through a directory of background images, parses
    the dimensions from their filenames, and compiles the information into a
    pandas DataFrame.

    The filenames are expected to be in a specific format where the last four
    parts of the filename (split by '_') are ymin, ymax, xmin, and xmax.

    Args:
        background_dir (Path): The path to the directory containing the
            background images.

    Returns:
        pd.DataFrame: A DataFrame with columns 'path', 'h', 'w', and 'ratio',
        containing the path, height, width, and aspect ratio of each
        background image.
    """
    background_df = []
    for path in background_dir.iterdir():
        ymin, ymax, xmin, xmax = [int(v) for v in path.stem.split("_")[-4:]]
        h = ymax - ymin
        w = xmax - xmin
        ratio = w / h

        background_df.append(
            {
                "path": str(path),
                "h": h,
                "w": w,
                "ratio": ratio,
            }
        )
    background_df = pd.DataFrame(background_df)
    return background_df


def is_kanji(ch):
    """Checks if a character is a kanji.

    This function determines if a given character is a CJK Unified Ideograph
    by checking its Unicode name.

    Args:
        ch (str): The character to check.

    Returns:
        bool: True if the character is a kanji, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch)
    except:
        return False

def is_hiragana(ch):
    """Checks if a character is a hiragana.

    This function determines if a given character is a hiragana character
    by checking its Unicode name.

    Args:
        ch (str): The character to check.

    Returns:
        bool: True if the character is a hiragana, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "HIRAGANA" in unicodedata.name(ch)
    except:
        return False


def is_katakana(ch):
    """Checks if a character is a katakana.

    This function determines if a given character is a katakana character
    by checking its Unicode name.

    Args:
        ch (str): The character to check.

    Returns:
        bool: True if the character is a katakana, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return "KATAKANA" in unicodedata.name(ch)
    except:
        return False

def is_ascii(ch):
    """Checks if a character is an ASCII character.

    This function determines if a given character is within the ASCII range
    (0-127) by checking its ordinal value.

    Args:
        ch (str): The character to check.

    Returns:
        bool: True if the character is an ASCII character, False otherwise.
    """
    try:
        if len(str(ch)) > 1:
            return False
        return ord(ch) < 128
    except:
        return False

def get_charsets(vocab_path=None):
    """Loads character sets from a vocabulary file.

    This function reads a vocabulary from a CSV file and separates it into
    hiragana, katakana, and the full vocabulary.

    Args:
        vocab_path (str or Path, optional): The path to the vocabulary CSV
            file. If not provided, it defaults to the path specified in
            `ASSETS_PATH`.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The full vocabulary.
            - np.ndarray: The hiragana character set.
            - np.ndarray: The katakana character set.
    """
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    hiragana = vocab[[is_hiragana(c) for c in vocab]]
    katakana = vocab[[is_katakana(c) for c in vocab]]
    return vocab, hiragana, katakana


def get_font_meta():
    """Loads font metadata from the fonts CSV file.

    This function reads 'fonts.csv' from the `ASSETS_PATH`, constructs the
    full paths to the font files, and creates a mapping from font paths to
    the set of characters they support.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with font metadata.
            - dict: A dictionary mapping font paths to sets of supported
              characters.
    """
    df = pd.read_csv(ASSETS_PATH / "fonts.csv")
    df.font_path = df.font_path.apply(lambda x: str(FONTS_ROOT / x))
    df = df.dropna()
    font_map = {row.font_path: set(row.supported_chars) for row in df.itertuples()}
    return df, font_map
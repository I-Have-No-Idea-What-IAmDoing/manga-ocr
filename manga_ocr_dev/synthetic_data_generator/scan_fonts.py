"""Scans font files to generate metadata about character support.

This script is a utility for preprocessing fonts to be used in synthetic data
generation. It iterates through all supported font files in the `FONTS_ROOT`
directory, determines which characters from the project's vocabulary are
supported by each font, and saves this information to `fonts.csv` in the
`ASSETS_PATH` directory. This metadata is crucial for the data generator to
select appropriate fonts for rendering text.
"""
from pathlib import Path

import PIL
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm.contrib.concurrent import process_map

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT

vocab = None


def has_glyph(font, glyph):
    """Checks if a font has a glyph for a specific character.

    This function iterates through the color map (cmap) tables of a `TTFont`
    object to determine if a glyph for the given character exists. This is a
    more reliable way to check for character support than just trying to
    render the character.

    Args:
        font (TTFont): An instance of a `fontTools.ttLib.TTFont` object.
        glyph (str): The character to check for. Must be a single character.

    Returns:
        True if a glyph for the character is found in any of the font's
        cmap tables, False otherwise.
    """
    # Iterate through all character map (cmap) tables in the font
    for table in font["cmap"].tables:
        try:
            # Check if the Unicode code point of the character exists in the cmap
            if ord(glyph) in table.cmap.keys():
                return True
        except Exception:
            # Handle potential errors with malformed cmap tables
            return False
    # If the glyph is not found in any cmap table, return False
    return False


def process_font(font_path):
    """Determines the set of supported characters for a given font file.

    This function checks for the presence of a glyph for each character in a
    predefined vocabulary. It then attempts to render the character to ensure
    it's not a blank or placeholder glyph, as some fonts render unsupported
    characters as placeholder shapes (e.g., rectangles).

    Args:
        font_path (str or Path): The path to the font file to be processed.

    Returns:
        A string containing all the supported characters found in the font,
        concatenated together. Returns an empty string if the font cannot be
        processed.
    """
    global vocab
    # Load the vocabulary if it hasn't been loaded yet
    if vocab is None:
        vocab = pd.read_csv(ASSETS_PATH / "vocab.csv").char.values

    try:
        font_path = str(font_path)
        # Load the font using fontTools for glyph checking and PIL for rendering
        ttfont = TTFont(font_path)
        pil_font = ImageFont.truetype(font_path, 24)

        supported_chars = []

        # Iterate through each character in the vocabulary to check for support
        for char in vocab:
            # First, check if a glyph for the character exists in the font's cmap
            if not has_glyph(ttfont, char):
                continue

            # Render the character to an image to verify it's not a blank or placeholder glyph
            image = PIL.Image.new("L", (40, 40), 255)
            draw = ImageDraw.Draw(image)
            draw.text((10, 0), char, 0, font=pil_font)
            # If the rendered image is completely white, the character is not actually supported
            if (np.array(image) != 255).sum() == 0:
                continue

            supported_chars.append(char)

        return "".join(supported_chars)
    except Exception as e:
        # Catch and report any errors that occur during font processing
        print(f"Error while processing {font_path}: {e}")
        return ""


def main():
    """Scans all fonts in a directory and generates a font metadata CSV file.

    This function finds all supported font files (e.g., .ttf, .otf) in the
    `FONTS_ROOT` directory, processes them in parallel to determine which
    characters they support, and then saves this information to `fonts.csv` in
    the `ASSETS_PATH` directory.

    The output CSV contains the font path, the list of supported characters,
    the total number of supported characters, and a default 'regular' label,
    which can be manually updated later for weighted font sampling.
    """
    # Define the input directory for fonts and the output path for the CSV
    path_in = Path(FONTS_ROOT)
    out_path = ASSETS_PATH / "fonts.csv"

    # Define the set of supported font file extensions
    suffixes = {".TTF", ".otf", ".ttc", ".ttf"}
    # Recursively find all font files in the input directory
    font_paths = [path for path in path_in.glob("**/*") if path.suffix in suffixes]

    # Process each font in parallel to determine its supported characters
    data = process_map(process_font, font_paths, max_workers=16)

    # Convert the absolute font paths to paths relative to the FONTS_ROOT
    font_paths = [str(path.relative_to(FONTS_ROOT)) for path in font_paths]
    # Create a DataFrame to store the font metadata
    data = pd.DataFrame({"font_path": font_paths, "supported_chars": data})
    # Calculate the number of supported characters for each font
    data["num_chars"] = data.supported_chars.str.len()
    # Add a default label for font weighting, which can be modified manually
    data["label"] = "regular"
    # Save the DataFrame to the output CSV file
    data.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
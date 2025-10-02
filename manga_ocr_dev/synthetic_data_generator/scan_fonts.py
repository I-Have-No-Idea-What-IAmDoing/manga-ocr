import PIL
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from tqdm.contrib.concurrent import process_map

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT

vocab = None


def has_glyph(font, glyph):
    for table in font["cmap"].tables:
        try:
            if ord(glyph) in table.cmap.keys():
                return True
        except:
            return False
    return False


def process(font_path):
    """Determines the list of supported characters for a given font.

    This function checks for the presence of a glyph for each character in a
    predefined vocabulary and then attempts to render the character to ensure
    it's not a blank or placeholder glyph. This is not always perfect as some
    fonts may render unsupported characters as placeholder shapes (e.g.,
    rectangles).

    Args:
        font_path (str or Path): The path to the font file.

    Returns:
        str: A string containing all the supported characters found in the font.
    """
    global vocab
    if vocab is None:
        vocab = pd.read_csv(ASSETS_PATH / "vocab.csv").char.values

    try:
        font_path = str(font_path)
        ttfont = TTFont(font_path)
        pil_font = ImageFont.truetype(font_path, 24)

        supported_chars = []

        for char in vocab:
            if not has_glyph(ttfont, char):
                continue

            image = PIL.Image.new("L", (40, 40), 255)
            draw = ImageDraw.Draw(image)
            draw.text((10, 0), char, 0, font=pil_font)
            if (np.array(image) != 255).sum() == 0:
                continue

            supported_chars.append(char)

        supported_chars = "".join(supported_chars)
    except Exception as e:
        print(f"Error while processing {font_path}: {e}")
        supported_chars = ""

    return supported_chars


def main():
    """Scans all fonts in the `FONTS_ROOT` directory to generate a metadata CSV.

    This function finds all supported font files (e.g., .ttf, .otf) in the
    `FONTS_ROOT` directory, processes them in parallel to determine which
    characters they support, and then saves this information to 'fonts.csv'
    in the `ASSETS_PATH` directory.

    The output CSV contains the font path, the list of supported characters,
    the number of supported characters, and a default 'regular' label.
    """
    path_in = FONTS_ROOT
    out_path = ASSETS_PATH / "fonts.csv"

    suffixes = {".TTF", ".otf", ".ttc", ".ttf"}
    font_paths = [path for path in path_in.glob("**/*") if path.suffix in suffixes]

    data = process_map(process, font_paths, max_workers=16)

    font_paths = [str(path.relative_to(FONTS_ROOT)) for path in font_paths]
    data = pd.DataFrame({"font_path": font_paths, "supported_chars": data})
    data["num_chars"] = data.supported_chars.str.len()
    data["label"] = "regular"
    data.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

import sys
import time
from pathlib import Path

import fire
import numpy as np
import pyperclip
from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger

from manga_ocr import MangaOcr


def are_images_identical(img1, img2):
    """Checks if two PIL Image objects are identical.

    This function first checks if either image is None, then converts the images
    to RGB format and compares their NumPy array representations.

    Args:
        img1 (Image.Image | None): The first image.
        img2 (Image.Image | None): The second image.

    Returns:
        bool: True if the images are identical, False otherwise.
    """
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1.convert("RGB"))
    img2 = np.array(img2.convert("RGB"))

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(mocr, img_or_path, write_to):
    """Processes an image with MangaOcr and writes the result.

    This function takes an image, performs OCR on it using the provided
    MangaOcr instance, and then writes the recognized text to the specified
    destination (either the clipboard or a file).

    Args:
        mocr (MangaOcr): An initialized MangaOcr instance.
        img_or_path (str | Path | Image.Image): The image to process.
        write_to (str): The destination for the recognized text. Can be
            "clipboard" or a path to a .txt file.

    Raises:
        ValueError: If `write_to` is not "clipboard" or a path to a .txt file.
    """
    t0 = time.time()
    text = mocr(img_or_path)
    t1 = time.time()

    logger.info(f"Text recognized in {t1 - t0:0.03f} s: {text}")

    if write_to == "clipboard":
        pyperclip.copy(text)
    else:
        write_to = Path(write_to)
        if write_to.suffix != ".txt":
            raise ValueError('write_to must be either "clipboard" or a path to a text file')

        with write_to.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def get_path_key(path):
    """Creates a unique key for a file path.

    The key consists of the path itself and its last modification time. This is
    used to uniquely identify a file and detect changes.

    Args:
        path (Path): The file path.

    Returns:
        tuple[Path, float]: A tuple containing the path and its modification time.
    """
    return path, path.lstat().st_mtime


def run(
    read_from="clipboard",
    write_to="clipboard",
    pretrained_model_name_or_path="kha-white/manga-ocr-base",
    force_cpu=False,
    delay_secs=0.1,
    verbose=False,
):
    """Runs OCR in the background, monitoring for new images.

    This function continuously checks for new images in the specified source
    (clipboard or a directory) and performs OCR on them. The recognized text is
    then written to the specified destination.

    Args:
        read_from (str, optional): The source to read images from. Can be
            "clipboard" or a directory path. Defaults to "clipboard".
        write_to (str, optional): The destination for the recognized text. Can be
            "clipboard" or a .txt file path. Defaults to "clipboard".
        pretrained_model_name_or_path (str, optional): The name or path of the
            pretrained model. Defaults to "kha-white/manga-ocr-base".
        force_cpu (bool, optional): If True, forces CPU usage even if a GPU is
            available. Defaults to False.
        delay_secs (float, optional): The delay in seconds between checking for
            new images. Defaults to 0.1.
        verbose (bool, optional): If True, enables verbose logging. Defaults to
            False.

    Raises:
        NotImplementedError: If writing to the clipboard is attempted on Wayland
            without `wl-clipboard` installed.
        ValueError: If `read_from` is not "clipboard" or a valid directory.
    """

    mocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

    if sys.platform not in ("darwin", "win32") and write_to == "clipboard":
        # Check if the system is using Wayland
        import os

        if os.environ.get("WAYLAND_DISPLAY"):
            # Check if the wl-clipboard package is installed
            if os.system("which wl-copy > /dev/null") == 0:
                pyperclip.set_clipboard("wl-clipboard")
            else:
                msg = (
                    "Your session uses wayland and does not have wl-clipboard installed. "
                    "Install wl-clipboard for write in clipboard to work."
                )
                raise NotImplementedError(msg)

    if read_from == "clipboard":
        from PIL import ImageGrab

        logger.info("Reading from clipboard")

        img = None
        while True:
            old_img = img

            try:
                img = ImageGrab.grabclipboard()
            except OSError as error:
                if not verbose and "cannot identify image file" in str(error):
                    # Pillow error when clipboard hasn't changed since last grab (Linux)
                    pass
                elif not verbose and "target image/png not available" in str(error):
                    # Pillow error when clipboard contains text (Linux, X11)
                    pass
                else:
                    logger.warning("Error while reading from clipboard ({})".format(error))
            else:
                if isinstance(img, Image.Image) and not are_images_identical(img, old_img):
                    process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)

    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.info(f"Reading from directory {read_from}")

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(get_path_key(path))

        while True:
            for path in read_from.iterdir():
                path_key = get_path_key(path)
                if path_key not in old_paths:
                    old_paths.add(path_key)

                    try:
                        img = Image.open(path)
                        img.load()
                    except (UnidentifiedImageError, OSError) as e:
                        logger.warning(f"Error while reading file {path}: {e}")
                    else:
                        process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)


if __name__ == "__main__":
    fire.Fire(run)
import os
import sys
import traceback
from functools import partial
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import cv2
import fire
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

OUT_DIR = None


def f(args, generator):
    """Processes a single data sample, generating and saving an image.

    This function is designed to be used as a worker in a parallel processing
    setup. It takes a tuple of arguments and a `SyntheticDataGenerator`
    instance, generates an image from text, saves it to a file, and returns
    metadata about the generated sample.

    Args:
        args (tuple): A tuple containing the index, source, ID, and text for
            the data sample.
        generator (SyntheticDataGenerator): An instance of the
            `SyntheticDataGenerator` class used to generate the image.

    Returns:
        tuple: A tuple containing metadata about the generated sample:
        (source, id, text_gt, vertical, font_path).
    """
    try:
        i, source, id_, text = args
        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text)

        cv2.imwrite(str(OUT_DIR / filename), img)

        font_path = Path(params["font_path"]).relative_to(FONTS_ROOT)
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret

    except Exception:
        print(traceback.format_exc())
        raise


def run(package=0, n_random=10000, n_limit=None, max_workers=14, cdp_port=9222):
    """Generates a package of synthetic data.

    This function orchestrates the generation of a data package, which
    consists of rendered images and a metadata CSV file. It reads lines of
    text from a source CSV, adds a specified number of random text samples,
    and then uses a pool of workers to generate an image for each line.

    The generated images are saved in a subdirectory of
    `DATA_SYNTHETIC_ROOT/img`, and the metadata is saved to
    `DATA_SYNTHETIC_ROOT/meta`.

    Args:
        package (int, optional): The number of the data package to generate.
            This is used to determine the input and output file paths.
            Defaults to 0.
        n_random (int, optional): The number of samples with random text to
            generate and add to the package. Defaults to 10000.
        n_limit (int, optional): If specified, limits the total number of
            generated samples. This is useful for debugging. Defaults to None.
        max_workers (int, optional): The maximum number of worker threads to
            use for parallel processing. Defaults to 14.
        cdp_port (int, optional): The port for the Chrome DevTools Protocol,
            used by the renderer. Defaults to 9222.
    """

    package = f"{package:04d}"
    lines = pd.read_csv(DATA_SYNTHETIC_ROOT / f"lines/{package}.csv")
    random_lines = pd.DataFrame(
        {
            "source": "random",
            "id": [f"random_{package}_{i}" for i in range(n_random)],
            "line": None,
        }
    )
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)
    args = [(i, *values) for i, values in enumerate(lines.values)]

    global OUT_DIR
    OUT_DIR = DATA_SYNTHETIC_ROOT / "img" / package
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    browser_executable = os.environ.get('CHROME_EXECUTABLE_PATH')
    with Renderer(cdp_port=cdp_port, browser_executable=browser_executable) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        f_with_generator = partial(f, generator=generator)
        data = thread_map(f_with_generator, args, max_workers=max_workers, desc=f"Processing package {package}")

    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = DATA_SYNTHETIC_ROOT / f"meta/{package}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
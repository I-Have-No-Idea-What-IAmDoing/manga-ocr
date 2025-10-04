"""Main executable for the synthetic data generation pipeline.

This script orchestrates the generation of a complete package of synthetic data.
It combines a corpus of text with randomly generated text, then uses a pool of
worker threads to render a styled image for each line. The generated images
and their corresponding metadata are saved to the `assets` directory, ready to
be used for training the OCR model.

This script is designed to be run from the command line using `fire`.
"""

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


def worker_fn(args, generator):
    """A worker function for processing a single data sample in parallel.

    This function is designed to be used with `thread_map`. It takes a tuple of
    arguments and a `SyntheticDataGenerator` instance, generates a synthetic
    image from text, saves it to a file, and returns metadata about the
    generated sample.

    Args:
        args (tuple[int, str, str, str]): A tuple containing the index, source,
            ID, and text for the data sample.
        generator (SyntheticDataGenerator): An initialized instance of the
            `SyntheticDataGenerator` to be used for image generation.

    Returns:
        A tuple containing metadata about the generated sample:
        (source, id, text_gt, vertical, font_path).
    """
    try:
        i, source, id_, text = args
        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text)

        if img is None:
            print(f"Skipping render for text: {text}")
            return None

        cv2.imwrite(str(Path(OUT_DIR) / filename), img)

        font_path = params["font_path"]
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret

    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None
    except Exception:
        print(traceback.format_exc())
        raise


def run(package=0, n_random=10000, n_limit=None, max_workers=14, cdp_port=9222):
    """Generates a package of synthetic data, including images and metadata.

    This function orchestrates the generation of a complete data package. It
    reads lines of text from a source CSV file, adds a specified number of
    randomly generated text samples, and then uses a pool of worker threads
    to generate a styled image for each line of text.

    The generated images are saved in a subdirectory of `assets/img`, and the
    corresponding metadata is saved to a CSV file in `assets/meta`.

    Args:
        package (int, optional): The numerical ID of the data package to
            generate. This corresponds to the input file `assets/lines/{package:04d}.csv`.
            Defaults to 0.
        n_random (int, optional): The number of samples with randomly
            generated text to create and add to the package. Defaults to 10000.
        n_limit (int | None, optional): If specified, limits the total number
            of generated samples to this value for quick tests. Defaults to None.
        max_workers (int, optional): The maximum number of worker threads to
            use for parallel image generation. Defaults to 14.
        cdp_port (int, optional): The port for the Chrome DevTools Protocol,
            used by the underlying renderer. Defaults to 9222.
    """

    package_id = f"{package:04d}"
    lines_path = Path(DATA_SYNTHETIC_ROOT) / f"lines/{package_id}.csv"
    if not lines_path.exists():
        raise FileNotFoundError(f"Lines file not found: {lines_path}")

    lines = pd.read_csv(lines_path)
    random_lines = pd.DataFrame(
        {
            "source": "random",
            "id": [f"random_{package_id}_{i}" for i in range(n_random)],
            "line": None,
        }
    )
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)
    args = [(i, *values) for i, values in enumerate(lines.values)]

    global OUT_DIR
    OUT_DIR = Path(DATA_SYNTHETIC_ROOT) / "img" / package_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    browser_executable = os.environ.get("CHROME_EXECUTABLE_PATH")
    with Renderer(cdp_port=cdp_port, browser_executable=browser_executable) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        f_with_generator = partial(worker_fn, generator=generator)
        results = thread_map(
            f_with_generator, args, max_workers=max_workers, desc=f"Processing package {package_id}"
        )

    data = [res for res in results if res is not None]
    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = Path(DATA_SYNTHETIC_ROOT) / f"meta/{package_id}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
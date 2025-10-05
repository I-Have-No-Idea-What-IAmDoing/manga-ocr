import json
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
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2

OUT_DIR = None
DEBUG_DIR = None


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, obj):
        """Serializes NumPy types and Path objects to JSON-compatible formats.

        This method is an extension of the default JSON encoder. It handles
        the conversion of NumPy's generic types (like `np.int64`) to standard
        Python numbers, NumPy arrays to Python lists, and `pathlib.Path`
        objects to strings.

        Args:
            obj: The object to serialize.

        Returns:
            A JSON-serializable representation of the object.
        """
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)


def worker_fn(args, generator, debug=False):
    """A worker function for processing a single data sample in parallel."""
    try:
        i, source, id_, text = args
        if debug:
            print(f"Processing sample {id_}: '{text}'")

        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text)

        if img is None:
            print(f"Skipping render for text: {text}")
            return None

        img_path = Path(OUT_DIR) / filename
        cv2.imwrite(str(img_path), img)
        if debug:
            print(f"  - Saved image to {img_path}")

        if debug:
            # In debug mode, save the generation parameters to a JSON file.
            debug_info = params.copy()
            # Convert Path objects to strings for JSON serialization, as they
            # are not natively supported by the json module.
            for key, value in debug_info.items():
                if isinstance(value, Path):
                    debug_info[key] = str(value)
            json_path = Path(DEBUG_DIR) / f"{id_}.json"
            json_path.write_text(
                json.dumps(debug_info, indent=4, cls=NumpyEncoder), encoding="utf-8"
            )
            print(f"  - Saved params to {json_path}")

        font_path = params.get("font_path")
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret

    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None
    except Exception:
        print(traceback.format_exc())
        raise


def run(
    package=0, n_random=10000, n_limit=None, max_workers=14, debug=False,
    min_font_size=40, max_font_size=60, target_size=None, min_output_size=None
):
    """Generates a package of synthetic data, including images and metadata.

    This function orchestrates the synthetic data generation process. It reads
    a list of text lines from a specified package file, generates additional
    random text samples, and then uses a pool of workers to process each
    sample. The generated images are saved to an output directory, and a
    metadata CSV file is created with information about each sample.

    Args:
        package (int): The ID of the data package to process. This corresponds
            to a `lines/{package_id:04d}.csv` file.
        n_random (int): The number of additional random text samples to
            generate.
        n_limit (int, optional): If specified, the total number of samples
            (from the file and random) will be limited to this number.
        max_workers (int): The maximum number of worker threads to use for
            parallel processing.
        debug (bool): If True, enables debug mode, which saves additional
            parameter information for each generated sample.
        min_font_size (int): The minimum font size for text rendering.
        max_font_size (int): The maximum font size for text rendering.
        target_size (tuple[int, int] or str, optional): The final output size
            (width, height) for the composed image. Can be a tuple of ints or
            a comma-separated string from the CLI.
        min_output_size (int, optional): The minimum size for the smallest
            dimension of the composed image.
    """

    # Explicitly cast numeric types to handle string inputs from CLI
    package = int(package)
    n_random = int(n_random)
    if n_limit is not None:
        n_limit = int(n_limit)
    max_workers = int(max_workers)
    min_font_size = int(min_font_size)
    max_font_size = int(max_font_size)
    if min_output_size is not None:
        min_output_size = int(min_output_size)

    if isinstance(target_size, str):
        target_size = tuple(map(int, target_size.split(',')))

    package_id = f"{package:04d}"
    lines_path = Path(DATA_SYNTHETIC_ROOT) / f"lines/{package_id}.csv"
    if not lines_path.exists():
        raise FileNotFoundError(f"Lines file not found: {lines_path}")

    # Load text lines from the specified package file.
    lines = pd.read_csv(lines_path)
    # Create a DataFrame for generating additional random text samples.
    random_lines = pd.DataFrame(
        {
            "source": "random",
            "id": [f"random_{package_id}_{i}" for i in range(n_random)],
            "line": None,
        }
    )
    # Combine the lines from the file with the random sample placeholders.
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)
    # Prepare the arguments for the worker function.
    args = [(i, *values) for i, values in enumerate(lines.values)]

    # Set up the output directories for the images and debug info.
    global OUT_DIR, DEBUG_DIR
    OUT_DIR = Path(DATA_SYNTHETIC_ROOT) / "img_v2" / package_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if debug:
        DEBUG_DIR = Path(DATA_SYNTHETIC_ROOT) / "debug_v2" / package_id
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    background_dir = Path(DATA_SYNTHETIC_ROOT) / "backgrounds"
    generator = SyntheticDataGeneratorV2(
        background_dir=background_dir,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        target_size=target_size,
        min_output_size=min_output_size,
    )
    f_with_generator = partial(worker_fn, generator=generator, debug=debug)
    results = thread_map(
        f_with_generator,
        args,
        max_workers=max_workers,
        desc=f"Processing package {package_id}",
    )

    data = [res for res in results if res is not None]
    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = Path(DATA_SYNTHETIC_ROOT) / f"meta_v2/{package_id}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
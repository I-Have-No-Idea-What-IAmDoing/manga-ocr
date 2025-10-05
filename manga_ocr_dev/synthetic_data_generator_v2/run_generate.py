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
            debug_info = params.copy()
            # Convert Path objects to strings for JSON serialization
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
    min_font_size=30, max_font_size=60, target_size=None, min_output_size=None
):
    """Generates a package of synthetic data, including images and metadata."""

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
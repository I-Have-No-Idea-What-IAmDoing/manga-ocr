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

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
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


def worker_fn(args, generator, renderer_type, debug=False):
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
            if renderer_type == 'html':
                html = debug_info.pop("html", "")
                html_path = Path(DEBUG_DIR) / f"{id_}.html"
                html_path.write_text(html, encoding="utf-8")
                print(f"  - Saved HTML to {html_path}")

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
        vertical = params.get("vertical", False)
        ret = source, id_, text_gt, vertical, str(font_path)
        return ret

    except ValueError as e:
        print(f"Skipping due to error: {e}")
        return None
    except Exception:
        print(traceback.format_exc())
        raise


def run(
    renderer='pictex',
    package=0,
    n_random=10000,
    n_limit=None,
    max_workers=14,
    debug=False,
    min_font_size=40,
    max_font_size=60,
    target_size=None,
    min_output_size=None,
    cdp_port=9222,
):
    """Generates a package of synthetic data, including images and metadata."""
    package = int(package)
    n_random = int(n_random)
    if n_limit is not None:
        n_limit = int(n_limit)
    max_workers = int(max_workers)

    if renderer not in ['pictex', 'html']:
        raise ValueError("`renderer` must be either 'pictex' or 'html'")

    package_id = f"{package:04d}"
    lines_path = Path(DATA_SYNTHETIC_ROOT) / f"lines/{package_id}.csv"
    if not lines_path.exists():
        raise FileNotFoundError(f"Lines file not found: {lines_path}")

    lines = pd.read_csv(lines_path)
    random_lines = pd.DataFrame(
        {"source": "random", "id": [f"random_{package_id}_{i}" for i in range(n_random)], "line": None}
    )
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)
    args = [(i, *values) for i, values in enumerate(lines.values)]

    global OUT_DIR, DEBUG_DIR
    version_str = 'v2' if renderer == 'pictex' else 'v1'
    OUT_DIR = Path(DATA_SYNTHETIC_ROOT) / f"img_{version_str}" / package_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if debug:
        DEBUG_DIR = Path(DATA_SYNTHETIC_ROOT) / f"debug_{version_str}" / package_id
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    background_dir = Path(BACKGROUND_DIR)

    if isinstance(target_size, str):
        target_size = tuple(map(int, target_size.split(',')))
    if min_output_size is not None:
        min_output_size = int(min_output_size)

    if renderer == 'pictex':
        generator = SyntheticDataGeneratorV2(
            background_dir=background_dir,
            min_font_size=int(min_font_size),
            max_font_size=int(max_font_size),
            target_size=target_size,
            min_output_size=min_output_size,
        )
        f_with_generator = partial(worker_fn, generator=generator, renderer_type=renderer, debug=debug)
        results = thread_map(f_with_generator, args, max_workers=max_workers, desc=f"Processing package {package_id} (pictex)")
    else:
        from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
        browser_executable = os.environ.get("CHROME_EXECUTABLE_PATH")
        with Renderer(cdp_port=int(cdp_port), browser_executable=browser_executable, debug=debug) as renderer_instance:
            generator = SyntheticDataGenerator(
                background_dir=background_dir,
                renderer=renderer_instance,
                target_size=target_size,
                min_output_size=min_output_size,
            )
            f_with_generator = partial(worker_fn, generator=generator, renderer_type=renderer, debug=debug)
            results = thread_map(f_with_generator, args, max_workers=max_workers, desc=f"Processing package {package_id} (html)")

    data = [res for res in results if res is not None]
    if not data:
        print("No data generated.")
        return

    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = Path(DATA_SYNTHETIC_ROOT) / f"meta_{version_str}/{package_id}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
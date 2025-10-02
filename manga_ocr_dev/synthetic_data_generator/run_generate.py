import traceback
from functools import partial
from pathlib import Path

import cv2
import fire
import pandas as pd
from tqdm.contrib.concurrent import thread_map

import sys

sys.path.append("J:/Applications/manga-ocr/")
from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

OUT_DIR = None


def f(args, generator):
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
        return None, None, None, None, None


def run(package=0, n_random=10000, n_limit=None, max_workers=14, cdp_port=9222):
    """
    :param package: number of data package to generate
    :param n_random: how many samples with random text to generate
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    :param cdp_port: port for chrome devtools protocol
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

    with Renderer(cdp_port=cdp_port) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        f_with_generator = partial(f, generator=generator)
        data = thread_map(f_with_generator, args, max_workers=max_workers, desc=f"Processing package {package}")

    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = DATA_SYNTHETIC_ROOT / f"meta/{package}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)


"""Generates background images from manga pages for synthetic data.

This script processes the Manga109 dataset to extract background images that
can be used for generating synthetic training data. It works by identifying
regions on manga pages that do not contain text or comic frames, and then
randomly cropping these regions to serve as backgrounds for rendered text.
The generated images are saved to the `BACKGROUND_DIR`.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT, BACKGROUND_DIR


def find_rectangle(mask, y, x, aspect_ratio_range=(0.33, 3.0)):
    """Finds the largest possible rectangle in an unmasked area of a mask.

    Starting from a given seed point (y, x), this function expands a rectangle
    outwards until it hits a masked area (where the mask value is True) or the
    edge of the mask. The expansion stops if the aspect ratio of the rectangle
    goes outside the specified range.

    Args:
        mask (np.ndarray): A 2D boolean NumPy array where True values indicate
            masked (forbidden) areas.
        y (int): The starting y-coordinate for the expansion.
        x (int): The starting x-coordinate for the expansion.
        aspect_ratio_range (tuple, optional): A tuple `(min_ratio, max_ratio)`
            specifying the acceptable range for the rectangle's aspect ratio
            (width / height). Defaults to (0.33, 3.0).

    Returns:
        tuple[int, int, int, int]: A tuple containing the coordinates of the
        found rectangle in the format (ymin, ymax, xmin, xmax).
    """
    ymin_ = ymax_ = y
    xmin_ = xmax_ = x

    ymin = ymax = xmin = xmax = None

    while True:
        if ymin is None:
            ymin_ -= 1
            if ymin_ == 0 or mask[ymin_, xmin_:xmax_].any():
                ymin = ymin_

        if ymax is None:
            ymax_ += 1
            if ymax_ == mask.shape[0] - 1 or mask[ymax_, xmin_:xmax_].any():
                ymax = ymax_

        if xmin is None:
            xmin_ -= 1
            if xmin_ == 0 or mask[ymin_:ymax_, xmin_].any():
                xmin = xmin_

        if xmax is None:
            xmax_ += 1
            if xmax_ == mask.shape[1] - 1 or mask[ymin_:ymax_, xmax_].any():
                xmax = xmax_

        h = ymax_ - ymin_
        w = xmax_ - xmin_
        if h > 1 and w > 1:
            ratio = w / h
            if ratio < aspect_ratio_range[0] or ratio > aspect_ratio_range[1]:
                return ymin_, ymax_, xmin_, xmax_

        if None not in (ymin, ymax, xmin, xmax):
            return ymin, ymax, xmin, xmax


def generate_backgrounds(crops_per_page=5, min_size=40):
    """Extracts and saves background image crops from Manga109 pages.

    This function iterates through the pages of the Manga109 dataset, creates
    a mask to exclude text boxes and panel frames, and then extracts random
    rectangular crops from the remaining unmasked areas. These crops are saved
    as PNG files and are used as backgrounds in the synthetic data generation
    pipeline.

    Args:
        crops_per_page (int, optional): The number of random background crops
            to generate from each manga page. Defaults to 5.
        min_size (int, optional): The minimum size (both width and height) for
            a cropped image to be considered valid and saved. Defaults to 40.
    """
    data = pd.read_csv(Path(MANGA109_ROOT) / "data.csv")
    frames_df = pd.read_csv(Path(MANGA109_ROOT) / "frames.csv")

    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    page_paths = data.page_path.unique()
    for page_path in tqdm(page_paths):
        page = cv2.imread(str(Path(MANGA109_ROOT) / page_path))
        mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in data[data.page_path == page_path].itertuples():
            mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        frames_mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in frames_df[frames_df.page_path == page_path].itertuples():
            frames_mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        mask = mask | ~frames_mask

        if mask.all():
            continue

        unmasked_points = np.stack(np.where(~mask), axis=1)
        for i in range(crops_per_page):
            p = unmasked_points[np.random.randint(0, unmasked_points.shape[0])]
            y, x = p
            ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
            crop = page[ymin:ymax, xmin:xmax]

            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = (
                    "_".join(Path(page_path).with_suffix("").parts[-2:])
                    + f"_{ymin}_{ymax}_{xmin}_{xmax}.png"
                )
                cv2.imwrite(str(Path(BACKGROUND_DIR) / out_filename), crop)


if __name__ == "__main__":
    generate_backgrounds()
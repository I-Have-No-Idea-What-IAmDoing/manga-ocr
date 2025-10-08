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
        found rectangle in the format (ymin, ymax, xmin, xmax), where ymax and
        xmax are exclusive.
    """
    h, w = mask.shape
    ymin, ymax = y, y
    xmin, xmax = x, x

    # Iteratively expand the rectangle until it hits a masked area or the image boundary
    while True:
        last_ymin, last_ymax, last_xmin, last_xmax = ymin, ymax, xmin, xmax

        # Try to expand vertically (up and down)
        can_expand_up = ymin > 0 and not mask[ymin - 1, xmin:xmax+1].any()
        can_expand_down = ymax + 1 < h and not mask[ymax + 1, xmin:xmax+1].any()

        if can_expand_up:
            ymin -= 1
        if can_expand_down:
            ymax += 1

        # Try to expand horizontally (left and right)
        can_expand_left = xmin > 0 and not mask[ymin:ymax+1, xmin - 1].any()
        can_expand_right = xmax + 1 < w and not mask[ymin:ymax+1, xmax + 1].any()

        if can_expand_left:
            xmin -= 1
        if can_expand_right:
            xmax += 1

        # If no expansion occurred in this iteration, the rectangle is at its maximum size
        if (ymin, ymax, xmin, xmax) == (last_ymin, last_ymax, last_xmin, last_xmax):
            break

        # Check if the aspect ratio is still within the valid range
        rect_h = ymax - ymin + 1
        rect_w = xmax - xmin + 1
        if rect_h > 1 and rect_w > 1:
            ratio = rect_w / rect_h
            if not (aspect_ratio_range[0] <= ratio <= aspect_ratio_range[1]):
                # If the aspect ratio is invalid, revert to the last valid rectangle and return
                return last_ymin, last_ymax + 1, last_xmin, last_xmax + 1

    # Return the coordinates of the final rectangle
    return ymin, ymax + 1, xmin, xmax + 1


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

    # Get unique page paths from the dataset
    page_paths = data.page_path.unique()

    # Process each page to extract backgrounds
    for page_path in tqdm(page_paths):
        # Load the manga page image
        page = cv2.imread(str(Path(MANGA109_ROOT) / page_path))

        # Create a mask for text boxes
        mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in data[data.page_path == page_path].itertuples():
            mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        # Create a mask for comic panel frames
        frames_mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in frames_df[frames_df.page_path == page_path].itertuples():
            frames_mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        # Combine the masks, marking text boxes and areas outside frames as invalid
        mask = mask | ~frames_mask

        # If the entire page is masked, skip to the next one
        if mask.all():
            continue

        # Get the coordinates of all unmasked points
        unmasked_points = np.stack(np.where(~mask), axis=1)

        # Generate a specified number of crops from the page
        for i in range(crops_per_page):
            # Pick a random unmasked point as a seed
            p = unmasked_points[np.random.randint(0, unmasked_points.shape[0])]
            y, x = p

            # Find the largest possible rectangle around the seed point
            ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
            crop = page[ymin:ymax, xmin:xmax]

            # If the crop is large enough, save it to the background directory
            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = (
                    "_".join(Path(page_path).with_suffix("").parts[-2:])
                    + f"_{ymin}_{ymax}_{xmin}_{xmax}.png"
                )
                cv2.imwrite(str(Path(BACKGROUND_DIR) / out_filename), crop)


if __name__ == "__main__":
    generate_backgrounds()
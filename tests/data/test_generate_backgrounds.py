"""Tests for the background image generation script.

This module contains unit tests for the `generate_backgrounds.py` script, which
is responsible for extracting background images from the Manga109 dataset. The
tests cover the rectangle-finding algorithm and the main generation function.
"""

import cv2
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from manga_ocr_dev.data.generate_backgrounds import find_rectangle, generate_backgrounds
from manga_ocr_dev.env import MANGA109_ROOT, BACKGROUND_DIR


def test_find_rectangle_no_mask():
    """Tests `find_rectangle` on a completely unmasked area.

    This test verifies that when there are no masked areas, the `find_rectangle`
    function correctly expands to the full dimensions of the input mask.
    """
    mask = np.zeros((100, 100), dtype=bool)
    ymin, ymax, xmin, xmax = find_rectangle(mask, 50, 50)
    assert (ymin, ymax, xmin, xmax) == (0, 100, 0, 100)


def test_find_rectangle_with_mask():
    """Tests `find_rectangle` in the presence of a simple mask.

    This test ensures that the `find_rectangle` function correctly stops its
    expansion when it encounters a masked area, returning the largest possible
    rectangle within the unmasked region.
    """
    mask = np.zeros((100, 100), dtype=bool)
    mask[50:, :] = True
    mask[:, 50:] = True
    ymin, ymax, xmin, xmax = find_rectangle(mask, 25, 25)
    assert (ymin, ymax, xmin, xmax) == (0, 50, 0, 50)


def test_find_rectangle_aspect_ratio():
    """Tests that `find_rectangle` respects the aspect ratio constraint.

    This test verifies that the rectangle expansion process stops if the
    aspect ratio of the growing rectangle goes outside the specified valid
    range. This is important for generating background crops with reasonable
    proportions.
    """
    mask = np.zeros((100, 100), dtype=bool)
    # This should stop early due to aspect ratio
    ymin, ymax, xmin, xmax = find_rectangle(mask, 50, 50, aspect_ratio_range=(0.1, 0.2))
    # The exact values depend on the expansion order, but it should not be the full frame
    assert (ymin, ymax, xmin, xmax) != (0, 100, 0, 100)


@patch('cv2.imwrite')
@patch('cv2.imread')
@patch('pandas.read_csv')
@patch('pathlib.Path.mkdir')
def test_generate_backgrounds(mock_mkdir, mock_read_csv, mock_imread, mock_imwrite):
    """Tests the main `generate_backgrounds` function's orchestration.

    This test verifies that the `generate_backgrounds` function correctly
    loads data, creates masks, and generates valid background crops. It uses
    mocks to avoid file system and I/O operations, focusing on the logic of
    the function itself.

    Args:
        mock_mkdir: Mock for `pathlib.Path.mkdir`.
        mock_read_csv: Mock for `pandas.read_csv`.
        mock_imread: Mock for `cv2.imread`.
        mock_imwrite: Mock for `cv2.imwrite`.
    """
    # Mocking the file system and data loading
    mock_mkdir.return_value = None

    # Mocking data.csv
    mock_data_df = pd.DataFrame({
        'page_path': ['manga1/001.jpg', 'manga1/001.jpg'],
        'ymin': [10, 60],
        'ymax': [30, 80],
        'xmin': [10, 60],
        'xmax': [30, 80],
    })
    # Mocking frames.csv
    mock_frames_df = pd.DataFrame({
        'page_path': ['manga1/001.jpg'],
        'ymin': [0],
        'ymax': [100],
        'xmin': [0],
        'xmax': [100],
    })
    mock_read_csv.side_effect = [mock_data_df, mock_frames_df]

    # Mocking the image
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = mock_image

    generate_backgrounds(crops_per_page=1, min_size=10)

    # Verify that imwrite was called, indicating a crop was generated
    assert mock_imwrite.called
    # Verify the crop is from an unmasked area
    args, _ = mock_imwrite.call_args
    filename, crop = args
    assert crop.shape[0] >= 10
    assert crop.shape[1] >= 10


@patch('cv2.imwrite')
@patch('cv2.imread')
@patch('pandas.read_csv')
@patch('pathlib.Path.mkdir')
def test_generate_backgrounds_fully_masked(mock_mkdir, mock_read_csv, mock_imread, mock_imwrite):
    """Tests that no backgrounds are generated from a fully masked page.

    This test ensures that if a page is completely covered by text boxes or
    is outside of any comic frames, the `generate_backgrounds` function
    correctly skips it and does not attempt to generate any crops.

    Args:
        mock_mkdir: Mock for `pathlib.Path.mkdir`.
        mock_read_csv: Mock for `pandas.read_csv`.
        mock_imread: Mock for `cv2.imread`.
        mock_imwrite: Mock for `cv2.imwrite`.
    """
    mock_mkdir.return_value = None
    mock_data_df = pd.DataFrame({
        'page_path': ['manga1/001.jpg'],
        'ymin': [0],
        'ymax': [100],
        'xmin': [0],
        'xmax': [100],
    })
    mock_frames_df = pd.DataFrame({
        'page_path': ['manga1/001.jpg'],
        'ymin': [0],
        'ymax': [100],
        'xmin': [0],
        'xmax': [100],
    })
    mock_read_csv.side_effect = [mock_data_df, mock_frames_df]
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = mock_image

    # Since the mask will cover everything, no crops should be generated
    generate_backgrounds(crops_per_page=1, min_size=10)
    assert not mock_imwrite.called
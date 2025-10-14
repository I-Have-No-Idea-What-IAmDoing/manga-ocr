"""Tests for the training utility functions.

This module contains unit tests for the helper functions in `training/utils.py`,
which are used for various tasks in the training pipeline, such as converting
tensors to images for visualization.
"""

import torch
import numpy as np

from manga_ocr_dev.training.utils import tensor_to_image


def test_tensor_to_image():
    """Tests the `tensor_to_image` function for correct tensor conversion.

    This test verifies that the `tensor_to_image` function correctly converts a
    PyTorch image tensor (typically in C, H, W format with values in [-1, 1])
    back into a standard NumPy image format (H, W, C with values in [0, 255]).
    """
    # Create a sample tensor in the expected format (C, H, W) and range [-1, 1]
    tensor = torch.tensor(
        [
            [[-1.0, 0.0], [0.5, 1.0]],  # Channel 1
            [[-0.5, 0.25], [0.75, -0.25]],  # Channel 2
            [[0.0, 1.0], [-1.0, 0.0]],  # Channel 3
        ]
    )

    # Expected output after denormalization and transpose
    # Denormalization: (x + 1) / 2 * 255
    # Transpose: (C, H, W) -> (H, W, C)
    expected_array = np.array(
        [
            [
                [0, 63, 127],  # ((-1+1)/2*255, (-0.5+1)/2*255, (0+1)/2*255)
                [127, 159, 255],  # ((0+1)/2*255, (0.25+1)/2*255, (1+1)/2*255)
            ],
            [
                [191, 223, 0],  # ((0.5+1)/2*255, (0.75+1)/2*255, (-1+1)/2*255)
                [255, 95, 127],  # ((1+1)/2*255, (-0.25+1)/2*255, (0+1)/2*255)
            ],
        ],
        dtype=np.uint8,
    )

    # Call the function
    result = tensor_to_image(tensor)

    # Assert shape, dtype, and content
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected_array)


def test_tensor_to_image_clipping():
    """Tests that `tensor_to_image` correctly clips values outside the [-1, 1] range.

    This test ensures that if the input tensor contains values outside the
    expected [-1, 1] range, they are correctly clipped to the boundaries
    before denormalization. This prevents underflow or overflow issues and
    ensures the output is a valid image.
    """
    # Create a tensor with values outside the [-1, 1] range to test clipping
    tensor = torch.tensor(
        [
            [[-2.0, 2.0]],  # Channel 1
            [[-1.5, 1.5]],  # Channel 2
            [[-1.0, 1.0]],  # Channel 3
        ]
    )

    # Expected output after clipping and denormalization
    # Values <-2.0, -1.5> should clip to -1.0, resulting in 0.
    # Values <2.0, 1.5> should clip to 1.0, resulting in 255.
    expected_array = np.array(
        [
            [
                [0, 0, 0],  # ((-1+1)/2*255, (-1+1)/2*255, (-1+1)/2*255)
                [255, 255, 255],  # ((1+1)/2*255, (1+1)/2*255, (1+1)/2*255)
            ]
        ],
        dtype=np.uint8,
    )

    # Call the function
    result = tensor_to_image(tensor)

    # Assert shape, dtype, and content
    assert result.shape == (1, 2, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected_array)
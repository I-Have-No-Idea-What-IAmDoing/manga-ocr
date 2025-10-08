"""Tests for the synthetic data rendering engine.

This module contains unit tests for the `Renderer` class and its associated
utility functions, which are responsible for generating synthetic text images.
These tests verify the functionality of text rendering, background composition,
and various image manipulation helpers.
"""

import numpy as np
import pytest
import unittest
from unittest.mock import patch, MagicMock
import cv2

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css, crop_by_alpha, blend, rounded_rectangle

@pytest.fixture
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome')
def renderer(mock_find_chrome):
    """Provides a `Renderer` instance with mocked dependencies for testing.

    This pytest fixture initializes the `Renderer` while mocking its external
    dependencies, such as browser discovery and background data loading. This
    allows for isolated testing of the renderer's logic.

    Returns:
        A `Renderer` instance ready for testing.
    """
    mock_find_chrome.return_value = 'dummy_chrome_path'
    r = Renderer()
    r.hti = MagicMock()
    return r

def test_renderer_initialization(renderer):
    """Tests that the Renderer initializes correctly."""
    assert renderer.hti is not None

def test_get_random_css_params():
    """Tests the generation of random CSS parameters."""
    params = Renderer.get_random_css_params()
    assert isinstance(params, dict)
    assert 'font_size' in params

def test_get_css():
    """Tests the `get_css` function for correct CSS string generation."""
    css = get_css(font_size=12, font_path='dummy.ttf', vertical=True, glow_size=1, stroke_size=1, letter_spacing=0.1, text_orientation='upright')
    assert 'font-size: 12px;' in css
    assert 'writing-mode: vertical-rl;' in css
    assert 'text-shadow' in css
    assert 'letter-spacing' in css
    assert 'text-orientation' in css


def test_get_css_transparent_background():
    """Tests that get_css generates CSS for a transparent background."""
    css = get_css(font_size=12, font_path='dummy.ttf', background_color='transparent')
    assert 'background-color: transparent;' in css
    assert 'html, body {' in css


def test_crop_by_alpha():
    """Tests the `crop_by_alpha` function with various image scenarios."""
    # Test case 1: Standard crop with margin
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    content_rgba = (128, 150, 170, 255)
    img[20:80, 30:70, :] = content_rgba
    cropped_img = crop_by_alpha(img, margin=10)
    assert cropped_img.shape == (60 + 20, 40 + 20, 4)
    # Check that the content area is correct
    content_area = cropped_img[10:-10, 10:-10, :]
    assert np.array_equal(content_area, np.full((60, 40, 4), content_rgba))
    # Check that the margin is black and fully transparent
    assert np.all(cropped_img[0:10, :, :] == 0)  # Top margin
    assert np.all(cropped_img[-10:, :, :] == 0)  # Bottom margin
    assert np.all(cropped_img[:, 0:10, :] == 0)  # Left margin
    assert np.all(cropped_img[:, -10:, :] == 0)  # Right margin

    # Test case 2: Crop with no margin
    cropped_img_no_margin = crop_by_alpha(img, margin=0)
    assert cropped_img_no_margin.shape == (60, 40, 4)
    assert np.array_equal(cropped_img_no_margin, np.full((60, 40, 4), content_rgba))

    # Test case 3: Fully transparent image
    img_transparent = np.zeros((100, 100, 4), dtype=np.uint8)
    cropped_transparent = crop_by_alpha(img_transparent)
    assert cropped_transparent.shape == (1, 1, 4)
    assert np.all(cropped_transparent == 0)

    # Test case 4: Fully opaque image with non-uniform content
    img_opaque = np.zeros((100, 100, 4), dtype=np.uint8)
    img_opaque[:, :, 0] = 50
    img_opaque[:, :, 1] = 100
    img_opaque[:, :, 2] = 150
    img_opaque[:, :, 3] = 255
    cropped_opaque = crop_by_alpha(img_opaque, margin=0)
    assert cropped_opaque.shape == (100, 100, 4)
    assert np.array_equal(cropped_opaque, img_opaque)

    # Test case 5: Image with content touching the border
    img_border = np.zeros((100, 100, 4), dtype=np.uint8)
    img_border[0, :, 0:3] = 128  # Give the line some color
    img_border[0, :, 3] = 255    # Top edge with non-zero alpha
    cropped_border = crop_by_alpha(img_border, margin=0)
    # The image should be cropped to the single line of content
    assert cropped_border.shape == (1, 100, 4)
    # Verify the content of the cropped line
    expected_content = img_border[0:1, :, :]
    assert np.array_equal(cropped_border, expected_content)

def test_blend():
    """Tests the `blend` function for correct alpha compositing."""
    fg = np.zeros((100, 100, 4), dtype=np.uint8)
    fg[:, :, 3] = 128 # 50% transparent
    bg = np.full((100, 100, 4), 255, dtype=np.uint8) # white background, with alpha

    blended_img = blend(fg, bg)
    assert blended_img.shape == (100, 100, 4)
    assert np.all(blended_img[0, 0] == [127, 127, 127, 255])

def test_rounded_rectangle():
    """Tests the `rounded_rectangle` drawing function."""
    shape = (100, 200)
    radius = 20
    color = (255, 0, 0, 255)
    result_img = rounded_rectangle(shape, radius, color)
    assert result_img.shape == (100, 200, 4)
    assert np.any(result_img > 0)

def test_get_css_stroke():
    """Tests that get_css generates a correct stroke effect."""
    css = get_css(
        font_size=48,
        font_path='dummy.ttf',
        stroke_size=1,
        stroke_color='black'
    )
    # A correct implementation should create a hard-edged stroke.
    # This can be done with multiple shadows at different offsets and 0 blur.
    expected_shadows = [
        "-1px -1px 0 black", "1px -1px 0 black", "-1px 1px 0 black", "1px 1px 0 black",
        "-1px 0px 0 black", "1px 0px 0 black", "0px -1px 0 black", "0px 1px 0 black"
    ]

    for shadow in expected_shadows:
        assert shadow in css

    # Also, ensure that the glow effect is not present for a stroke.
    assert "0 0 1px" not in css

def test_get_css_glow():
    """Tests that get_css generates a correct glow effect."""
    css = get_css(
        font_size=48,
        font_path='dummy.ttf',
        glow_size=5,
        glow_color='blue'
    )
    assert 'text-shadow: 0 0 5px blue;' in css


class TestRendererParams(unittest.TestCase):
    @patch("numpy.random.choice")
    @patch("numpy.random.randint")
    @patch("numpy.random.uniform")
    @patch("numpy.random.rand")
    def test_get_random_css_params_uses_new_ranges(
        self, mock_rand, mock_uniform, mock_randint, mock_choice
    ):
        """Verify that get_random_css_params uses the updated ranges for improved readability."""
        # Arrange
        # Mock random values to control the parameters generated
        # The order of calls to rand() is: vertical, text_color, text_orientation, letter_spacing
        mock_rand.side_effect = [0.6, 0.4, 0.6, 0.1]
        mock_uniform.side_effect = [1.8, 0.05]  # line_height, letter_spacing
        mock_randint.return_value = 80  # font_size
        mock_choice.side_effect = ["stroke", 2]  # effect, stroke_size

        # Act
        params = Renderer.get_random_css_params()

        # Assert
        # Check font size is called with the new range
        mock_randint.assert_called_once_with(48, 96)
        self.assertEqual(params["font_size"], 80)

        # Check line height is called with the new range
        mock_uniform.assert_any_call(1.4, 2.0)
        self.assertEqual(params["line_height"], 1.8)

        # Check that the effect probabilities are updated
        effect_call_args, effect_call_kwargs = mock_choice.call_args_list[0]
        self.assertEqual(effect_call_args[0], ["stroke", "glow", "none"])
        np.testing.assert_array_equal(
            effect_call_kwargs["p"], [0.4, 0.15, 0.45]
        )
        self.assertEqual(params["text_color"], "black")
        self.assertEqual(params["stroke_color"], "white")

        # Check that stroke size is chosen from the new range
        stroke_size_call_args, _ = mock_choice.call_args_list[1]
        self.assertEqual(stroke_size_call_args[0], [1, 2, 3])
        self.assertEqual(params["stroke_size"], 2)
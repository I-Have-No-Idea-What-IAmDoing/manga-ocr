"""Tests for the synthetic data rendering engine.

This module contains unit tests for the `Renderer` class and its associated
utility functions, which are responsible for generating synthetic text images.
These tests verify the functionality of text rendering, background composition,
and various image manipulation helpers.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css, crop_by_alpha, blend, rounded_rectangle

@pytest.fixture
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df')
def renderer(mock_get_background_df, mock_find_chrome):
    """Provides a `Renderer` instance with mocked dependencies for testing.

    This pytest fixture initializes the `Renderer` while mocking its external
    dependencies, such as browser discovery and background data loading. This
    allows for isolated testing of the renderer's logic.

    Returns:
        A `Renderer` instance ready for testing.
    """
    mock_find_chrome.return_value = 'dummy_chrome_path'
    mock_get_background_df.return_value = MagicMock()
    r = Renderer()
    r.hti = MagicMock()
    return r

def test_renderer_initialization(renderer):
    """Tests that the Renderer initializes correctly."""
    assert renderer.hti is not None
    assert renderer.background_df is not None

def test_render_text(renderer):
    """
    Tests the `render_text` method for correct image and parameter output.

    This test ensures that `render_text` successfully produces an image with
    the correct format (BGRA) and returns a dictionary of the CSS parameters
    used for rendering.
    """
    renderer.hti.screenshot_as_bytes.return_value = cv2.imencode('.png', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()

    lines = ['test']
    img, params = renderer.render_text(lines, override_css_params={'font_path': 'dummy.ttf'})

    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 4
    assert isinstance(params, dict)

def test_get_random_css_params():
    """Tests the generation of random CSS parameters."""
    params = Renderer.get_random_css_params()
    assert isinstance(params, dict)
    assert 'font_size' in params

def test_render_background(renderer):
    """Tests the `render_background` method for compositing images."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)

    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread') as mock_imread:
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'

        result_img = renderer.render_background(img)
        assert isinstance(result_img, np.ndarray)

def test_get_css():
    """Tests the `get_css` function for correct CSS string generation."""
    css = get_css(font_size=12, font_path='dummy.ttf', vertical=True, shadow_size=1, stroke_size=1, letter_spacing=0.1, text_orientation='upright')
    assert 'font-size: 12px;' in css
    assert 'writing-mode: vertical-rl;' in css
    assert 'text-shadow' in css
    assert 'letter-spacing' in css
    assert 'text-orientation' in css

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
    bg = np.full((100, 100, 3), 255, dtype=np.uint8) # white background

    blended_img = blend(fg, bg)
    assert blended_img.shape == (100, 100, 3)
    assert np.all(blended_img[0, 0] == [127, 127, 127])

def test_rounded_rectangle():
    """Tests the `rounded_rectangle` drawing function."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result_img = rounded_rectangle(img, (10, 10), (90, 90), radius=0.5, color=(255, 255, 255), thickness=-1)
    assert np.any(result_img > 0)

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

        params = {'text_color': 'black'}
        result_img = renderer.render_background(img, params)
        assert isinstance(result_img, np.ndarray)

def test_get_css():
    """Tests the `get_css` function for correct CSS string generation."""
    css = get_css(font_size=12, font_path='dummy.ttf', vertical=True, glow_size=1, stroke_size=1, letter_spacing=0.1, text_orientation='upright')
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


def test_render_background_contrast_adjustment_and_prescaling(renderer):
    """
    Tests that the background is inverted for better contrast and that oversized
    images are pre-scaled.
    """
    # 1. Setup
    renderer.max_size = 500
    # Text image is larger than max_size to trigger pre-scaling
    img = np.zeros((600, 600, 4), dtype=np.uint8)
    img[100:500, 100:500, 3] = 255  # Opaque text area

    dark_bg = np.full((100, 100, 3), 20, dtype=np.uint8)
    params = {'text_color': 'black'}

    # 2. Mocks
    # Mock file system and random operations to make the test deterministic
    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread', return_value=dark_bg), \
         patch('numpy.random.random', return_value=0.9), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.crop_by_alpha', side_effect=lambda x, margin: x), \
         patch('albumentations.Compose') as mock_compose, \
         patch('albumentations.LongestMaxSize') as mock_scaler, \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.blend') as mock_blend:

        # Mock the transformations to have predictable outputs
        # The scaler will reduce the image to max_size, preserving an alpha channel
        scaled_img_with_alpha = np.zeros((500, 500, 4), dtype=np.uint8)
        scaled_img_with_alpha[:, :, 3] = 255
        mock_scaler.return_value.return_value = {'image': scaled_img_with_alpha}
        # The background transform will resize the background to match the (padded) text image
        # After pre-scaling to 500 and padding with 0.1*500 on each side, the size is 600.
        padded_size = 600
        mock_compose.return_value = lambda **kwargs: {'image': cv2.resize(kwargs['image'], (padded_size, padded_size))}
        # The final blend will return a simple black image
        mock_blend.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        # 3. Execution
        with patch('numpy.random.uniform', return_value=0.1): # Control padding
             renderer.render_background(img, params)

        # 4. Assertions
        # Assert that pre-scaling was called because the image was too large
        mock_scaler.assert_called_with(500)

        # Assert that the background was inverted
        # The background passed to the first blend call should be inverted (bright)
        # because the original background was dark, and the text was black.
        first_call_args, _ = mock_blend.call_args_list[0]
        blended_background = first_call_args[1]
        assert blended_background.mean() > 230 # Should be 235 (255-20)


def test_render_background_no_prescaling(renderer):
    """Tests that pre-scaling is skipped for images smaller than max_size."""
    # 1. Setup
    renderer.max_size = 500
    # Text image is smaller than max_size
    img_size = 400
    img = np.zeros((img_size, img_size, 4), dtype=np.uint8)
    params = {'text_color': 'black'}

    # 2. Mocks
    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread'), \
         patch('numpy.random.random', return_value=0.9), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.crop_by_alpha', side_effect=lambda x, margin: x), \
         patch('numpy.random.uniform', return_value=0.1), \
         patch('albumentations.Compose') as mock_compose, \
         patch('albumentations.LongestMaxSize') as mock_scaler, \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.blend', return_value=np.zeros((10, 10, 3), dtype=np.uint8)):

        # Calculate expected size after padding.
        pad_ratio = 0.1
        padded_size = int(img_size + (img_size * pad_ratio) + (img_size * pad_ratio))
        mock_compose.return_value = lambda **kwargs: {'image': np.zeros((padded_size, padded_size, 3))}

        # 3. Execution
        renderer.render_background(img, params)

        # 4. Assertions
        # Assert that pre-scaling was NOT called
        mock_scaler.assert_not_called()

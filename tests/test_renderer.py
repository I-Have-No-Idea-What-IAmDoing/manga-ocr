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
def renderer(mock_get_background_df, mock_find_chrome, tmp_path):
    """Provides a `Renderer` instance with mocked dependencies for testing.

    This pytest fixture initializes the `Renderer` while mocking its external
    dependencies, such as browser discovery and background data loading. This
    allows for isolated testing of the renderer's logic.

    Returns:
        A `Renderer` instance ready for testing.
    """
    mock_find_chrome.return_value = 'dummy_chrome_path'
    mock_get_background_df.return_value = MagicMock()
    # Use a real tempfile.TemporaryDirectory to avoid issues with path handling
    r = Renderer(debug=True)
    r.hti = MagicMock()
    r.hti.temp_path = str(tmp_path)
    return r

def test_renderer_initialization(renderer):
    """Tests that the Renderer initializes correctly."""
    assert renderer.hti is not None
    assert renderer.background_df is not None

def test_render_text_transparent(renderer):
    """
    Tests the `_render_text_transparent` method for correct image and parameter output.

    This test ensures that `_render_text_transparent` successfully produces an image with
    the correct format (BGRA) and returns a dictionary of the CSS parameters
    used for rendering.
    """
    renderer.hti.screenshot_as_bytes.return_value = cv2.imencode('.png', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()

    lines = ['test']
    img, params = renderer._render_text_transparent(lines, override_css_params={'font_path': 'dummy.ttf'})

    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 4
    assert isinstance(params, dict)


def test_get_random_css_params():
    """Tests the generation of random CSS parameters."""
    params = Renderer.get_random_css_params()
    assert isinstance(params, dict)
    assert 'font_size' in params


@patch('os.path.exists', return_value=True)
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imwrite')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread')
@patch('os.remove')
def test_render_final_image(mock_remove, mock_imread, mock_imwrite, mock_exists, renderer, tmp_path):
    """Tests the `_render_final_image` method for rendering text on a background."""
    # Setup
    text_img = np.zeros((50, 50, 4), dtype=np.uint8)
    text_img[:, :, 3] = 255  # Opaque text area
    lines = ['test']
    params = {
        'font_path': 'dummy.ttf', 'font_size': 12, 'text_color': 'black', 'line_height': 1.5,
        'vertical': False, 'glow_size': 0, 'stroke_size': 0, 'letter_spacing': None,
        'text_orientation': None
    }
    renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock the second rendering pass
    final_img_bytes = cv2.imencode('.png', np.full((100, 100, 3), 255, dtype=np.uint8))[1].tobytes()
    with patch.object(renderer.executor, 'submit') as mock_submit, \
         patch('numpy.random.uniform', return_value=0.1): # for padding
        mock_submit.return_value.result.return_value = final_img_bytes
        result_img = renderer._render_final_image(text_img, lines, params)

    # Assertions
    assert isinstance(result_img, np.ndarray)
    mock_imwrite.assert_called_once()
    mock_remove.assert_called_once()
    assert result_img is not None


def test_get_css():
    """Tests the `get_css` function for correct CSS string generation."""
    css = get_css(font_size=12, font_path='dummy.ttf', vertical=True, glow_size=1, stroke_size=1, letter_spacing=0.1, text_orientation='upright')
    assert 'font-size: 12px;' in css
    assert 'writing-mode: vertical-rl;' in css
    assert 'text-shadow' in css
    assert 'letter-spacing' in css
    assert 'text-orientation' in css

    # Test new parameters
    css_with_bg = get_css(
        font_size=12,
        font_path='dummy.ttf',
        background_image_uri='file:///dummy.png',
        padding=(10, 20)
    )
    assert "background-image: url('file:///dummy.png');" in css_with_bg
    assert "padding: 10px 20px;" in css_with_bg
    assert "box-sizing: border-box;" in css_with_bg

def test_get_css_transparent_background():
    """Tests that get_css generates CSS for a transparent background."""
    css = get_css(font_size=12, font_path='dummy.ttf', background_color='transparent')
    assert 'background-color: transparent;' in css
    assert 'html {' in css
    assert 'body {' in css


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


@patch('albumentations.Compose')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imwrite')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread')
@patch('os.remove')
def test_render_final_image_contrast_adjustment(mock_remove, mock_imread, mock_imwrite, mock_compose, renderer, tmp_path):
    """
    Tests that the background is inverted for better contrast before being used
    in the final render.
    """
    # 1. Setup
    text_img = np.zeros((50, 50, 4), dtype=np.uint8)
    text_img[:, :, 3] = 255
    dark_bg = np.full((100, 100, 3), 20, dtype=np.uint8)
    params = {
        'text_color': 'black', 'font_path': 'dummy.ttf', 'font_size': 12,
        'line_height': 1.5, 'vertical': False, 'glow_size': 0, 'stroke_size': 0,
        'letter_spacing': None, 'text_orientation': None
    }

    # 2. Mocks
    mock_imread.return_value = dark_bg
    # Mock the Compose pipeline to resize the image, isolating the contrast logic from other transforms
    # Padded size will be 60x60 (50 + 2 * 50 * 0.1)
    mock_compose.return_value = lambda **kwargs: {'image': cv2.resize(kwargs['image'], (60, 60))}
    final_img_bytes = cv2.imencode('.png', np.zeros((60, 60, 3), dtype=np.uint8))[1].tobytes()

    # 3. Execution
    # No bubble (random > 0.7), so contrast check is performed
    with patch('numpy.random.random', return_value=0.9), \
         patch.object(renderer.executor, 'submit') as mock_submit, \
         patch('numpy.random.uniform', return_value=0.1): # for padding
        mock_submit.return_value.result.return_value = final_img_bytes
        result_img = renderer._render_final_image(text_img, ['test'], params)

    # 4. Assertions
    # Assert that the background passed to imwrite was inverted
    written_image = mock_imwrite.call_args[0][1]
    assert written_image.mean() > 230  # Should be inverted (255-20)
    assert result_img.shape[:2] == (60, 60)


from manga_ocr_dev.vendored.html2image.browsers.chrome_cdp import ChromeCDP
import json

@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome')
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.requests.get')
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.create_connection')
def test_chrome_cdp_screenshot_sets_transparent_background(mock_create_conn, mock_requests_get, mock_find_chrome):
    """
    Tests that the modified ChromeCDP screenshot method injects JavaScript
    to set a transparent background before taking the screenshot.
    """
    # Arrange
    mock_find_chrome.return_value = 'dummy_chrome_path'

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{'type': 'page', 'webSocketDebuggerUrl': 'ws://dummy'}]
    mock_requests_get.return_value = mock_response

    mock_ws = MagicMock()
    mock_create_conn.return_value = mock_ws

    cdp = ChromeCDP()

    mock_ws.recv.side_effect = [
        json.dumps({'method': 'Page.loadEventFired'}),
        json.dumps({'result': {'data': 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='}})
    ]

    # Act
    with patch('pathlib.Path.is_absolute', return_value=True):
        cdp.screenshot_as_bytes('dummy_input.html')

    # Assert
    # 1. Check that ws.send was called with Runtime.evaluate with the correct expression
    runtime_evaluate_calls = []
    call_methods = []
    for call in mock_ws.send.call_args_list:
        payload = json.loads(call.args[0])
        method = payload.get('method')
        call_methods.append(method)
        if method == 'Runtime.evaluate':
            runtime_evaluate_calls.append(payload)

    assert len(runtime_evaluate_calls) == 1
    expression = runtime_evaluate_calls[0]['params']['expression']
    assert "document.documentElement.style.background = 'transparent'" in expression
    assert "document.body.style.background = 'transparent'" in expression

    # 2. Check the order of key CDP calls
    try:
        evaluate_index = call_methods.index('Runtime.evaluate')
        screenshot_index = call_methods.index('Page.captureScreenshot')
        assert evaluate_index < screenshot_index
    except (ValueError, IndexError):
        pytest.fail("Expected 'Runtime.evaluate' and 'Page.captureScreenshot' to be called in order.")
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
import albumentations as A

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css, crop_by_alpha, blend, rounded_rectangle

@pytest.fixture
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df')
def renderer(mock_get_background_df, mock_find_chrome):
    """Provides a `Renderer` instance with mocked dependencies for testing."""
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
    img[:, :, 3] = 255

    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread') as mock_imread:
        mock_imread.return_value = np.zeros((200, 200, 3), dtype=np.uint8)
        renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'

        params = {'text_color': 'black'}
        result_img = renderer.render_background(img, params)
        assert isinstance(result_img, np.ndarray)
        assert result_img.size > 0

def test_render_background_uses_cropping_not_resizing(renderer):
    """
    Tests that render_background uses cropping instead of resizing to prevent
    background distortion.
    """
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    params = {'text_color': 'black'}

    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread', return_value=background), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.A.Compose') as mock_compose, \
         patch('numpy.random.random', return_value=0.9): # draw_bubble = False

        # Mock the pad and crop to have predictable sizes
        with patch('numpy.random.uniform', return_value=0.2):
            padded_size = 100 + 2 * int(100 * 0.2)
            mock_compose.return_value.return_value = {'image': np.zeros((padded_size, padded_size, 3))}

            renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'
            renderer.render_background(img, params)

            mock_compose.assert_called_once()
            compose_args, _ = mock_compose.call_args
            transforms = compose_args[0]

            has_resize = any(isinstance(t, A.Resize) for t in transforms)
            assert not has_resize, "A.Resize should not be used in the background transformation pipeline."

            has_pad_if_needed = any(isinstance(t, A.PadIfNeeded) for t in transforms)
            has_random_crop = any(isinstance(t, A.RandomCrop) for t in transforms)
            assert has_pad_if_needed, "A.PadIfNeeded should be used in the background transformation pipeline."
            assert has_random_crop, "A.RandomCrop should be used in the background transformation pipeline."

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
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    content_rgba = (128, 150, 170, 255)
    img[20:80, 30:70, :] = content_rgba
    cropped_img = crop_by_alpha(img, margin=10)
    assert cropped_img.shape == (60 + 20, 40 + 20, 4)
    content_area = cropped_img[10:-10, 10:-10, :]
    assert np.array_equal(content_area, np.full((60, 40, 4), content_rgba))
    assert np.all(cropped_img[0:10, :, :] == 0)
    assert np.all(cropped_img[-10:, :, :] == 0)
    assert np.all(cropped_img[:, 0:10, :] == 0)
    assert np.all(cropped_img[:, -10:, :] == 0)

def test_blend():
    """Tests the `blend` function for correct alpha compositing."""
    fg = np.zeros((100, 100, 4), dtype=np.uint8)
    fg[:, :, 3] = 128
    bg = np.full((100, 100, 3), 255, dtype=np.uint8)
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
    expected_shadows = [
        "-1px -1px 0 black", "1px -1px 0 black", "-1px 1px 0 black", "1px 1px 0 black",
        "-1px 0px 0 black", "1px 0px 0 black", "0px -1px 0 black", "0px 1px 0 black"
    ]
    for shadow in expected_shadows:
        assert shadow in css
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
        mock_rand.side_effect = [0.6, 0.4, 0.6, 0.1]
        mock_uniform.side_effect = [1.8, 0.05]
        mock_randint.return_value = 80
        mock_choice.side_effect = ["stroke", 2]
        params = Renderer.get_random_css_params()
        mock_randint.assert_called_once_with(48, 96)
        self.assertEqual(params["font_size"], 80)
        mock_uniform.assert_any_call(1.4, 2.0)
        self.assertEqual(params["line_height"], 1.8)
        effect_call_args, effect_call_kwargs = mock_choice.call_args_list[0]
        self.assertEqual(effect_call_args[0], ["stroke", "glow", "none"])
        np.testing.assert_array_equal(
            effect_call_kwargs["p"], [0.4, 0.15, 0.45]
        )
        self.assertEqual(params["text_color"], "black")
        self.assertEqual(params["stroke_color"], "white")
        stroke_size_call_args, _ = mock_choice.call_args_list[1]
        self.assertEqual(stroke_size_call_args[0], [1, 2, 3])
        self.assertEqual(params["stroke_size"], 2)

def test_render_background_contrast_adjustment_and_prescaling(renderer):
    """
    Tests that the background is inverted for better contrast and that oversized
    images are pre-scaled.
    """
    renderer.max_size = 500
    img = np.zeros((600, 600, 4), dtype=np.uint8)
    img[100:500, 100:500, 3] = 255
    dark_bg = np.full((100, 100, 3), 20, dtype=np.uint8)
    params = {'text_color': 'black'}
    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread', return_value=dark_bg), \
         patch('numpy.random.random', return_value=0.9), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.crop_by_alpha', side_effect=lambda x, margin: x), \
         patch('albumentations.Compose') as mock_compose, \
         patch('albumentations.LongestMaxSize') as mock_scaler, \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.blend') as mock_blend:
        scaled_img_with_alpha = np.zeros((500, 500, 4), dtype=np.uint8)
        scaled_img_with_alpha[:, :, 3] = 255
        mock_scaler.return_value.return_value = {'image': scaled_img_with_alpha}
        padded_size = 600
        mock_compose.return_value.return_value = {'image': cv2.resize(dark_bg, (padded_size, padded_size))}
        mock_blend.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch('numpy.random.uniform', return_value=0.1):
             renderer.render_background(img, params)
        mock_scaler.assert_called_with(500)
        first_call_args, _ = mock_blend.call_args_list[0]
        blended_background = first_call_args[1]
        assert blended_background.mean() > 230

def test_render_background_no_prescaling(renderer):
    """Tests that pre-scaling is skipped for images smaller than max_size."""
    renderer.max_size = 500
    img_size = 400
    img = np.zeros((img_size, img_size, 4), dtype=np.uint8)
    img[:,:,3] = 255
    params = {'text_color': 'black'}
    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread'), \
         patch('numpy.random.random', return_value=0.9), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.crop_by_alpha', side_effect=lambda x, margin: x), \
         patch('numpy.random.uniform', return_value=0.1), \
         patch('albumentations.Compose') as mock_compose, \
         patch('albumentations.LongestMaxSize') as mock_scaler, \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.blend', return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        pad_ratio = 0.1
        padded_size = int(img_size + (img_size * pad_ratio) + (img_size * pad_ratio))
        mock_compose.return_value.return_value = {'image': np.zeros((padded_size, padded_size, 3))}
        renderer.render_background(img, params)
        mock_scaler.assert_not_called()

def test_render_background_final_random_crop(renderer):
    """Tests that the final random crop is applied correctly."""
    text_img = np.zeros((50, 50, 4), dtype=np.uint8)
    text_img[:, :, 3] = 255
    params = {'text_color': 'black'}
    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread', return_value=np.zeros((100, 100, 3))), \
         patch('numpy.random.random', return_value=0.9), \
         patch('numpy.random.uniform') as mock_uniform:
        mock_uniform.side_effect = [
            0.2, 0.2, 0.2, 0.2,
            0.8, 0.8
        ]
        result_img = renderer.render_background(text_img, params)
    assert result_img.shape == (56, 56, 3)

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
    with patch('pathlib.Path.is_absolute', return_value=True):
        cdp.screenshot_as_bytes('dummy_input.html')
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
    try:
        evaluate_index = call_methods.index('Runtime.evaluate')
        screenshot_index = call_methods.index('Page.captureScreenshot')
        assert evaluate_index < screenshot_index
    except (ValueError, IndexError):
        pytest.fail("Expected 'Runtime.evaluate' and 'Page.captureScreenshot' to be called in order.")
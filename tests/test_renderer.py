"""Tests for the synthetic data rendering engine.

This module contains unit tests for the `Renderer` class and its associated
utility functions, which are responsible for generating synthetic text images.
These tests verify the functionality of text rendering, background composition,
and various image manipulation helpers.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import cv2

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css
from manga_ocr_dev.env import FONTS_ROOT


@pytest.fixture
@patch("manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome")
@patch("manga_ocr_dev.synthetic_data_generator.renderer.get_background_df")
def renderer(mock_get_background_df, mock_find_chrome):
    """Provides a `Renderer` instance with mocked dependencies for testing.

    This pytest fixture initializes the `Renderer` while mocking its external
    dependencies, such as browser discovery and background data loading. This
    allows for isolated testing of the renderer's logic.

    Returns:
        A `Renderer` instance ready for testing.
    """
    mock_find_chrome.return_value = "dummy_chrome_path"
    mock_df = MagicMock()
    mock_df.sample.return_value.iloc[0].path = "dummy.png"
    mock_get_background_df.return_value = mock_df
    r = Renderer()
    r.hti = MagicMock()
    return r


def test_renderer_initialization(renderer):
    """Tests that the Renderer initializes correctly."""
    assert renderer.hti is not None
    assert renderer.background_df is not None


def test_render_mocked(renderer):
    """
    Tests the main `render` method with a mocked `_render_html` to check the
    overall pipeline logic without performing an actual render.
    """
    with patch.object(
        renderer, "_render_html", return_value=(np.zeros((200, 200, 3), dtype=np.uint8), {})
    ) as mock_render_html, patch(
        "manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ):
        lines = ["test"]
        override_params = {"font_path": "dummy.ttf"}
        img, params = renderer.render(lines, override_css_params=override_params)

        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2  # Grayscale
        assert isinstance(params, dict)
        mock_render_html.assert_called_once()
        rendered_params = mock_render_html.call_args[0][1]
        assert "background_image_data_uri" in rendered_params
        assert rendered_params["background_image_data_uri"].startswith("data:image/png;base64,")


@pytest.mark.integration
def test_render_integration_with_background():
    """
    Performs a full integration test of the render method, ensuring that a
    background image is correctly rendered. This test is not mocked and
    requires a running Chrome/Chromium instance.
    """
    # Setup: Ensure Chrome is available
    browser_executable = os.environ.get("CHROME_EXECUTABLE_PATH")
    if not browser_executable or not Path(browser_executable).exists():
        pytest.skip("CHROME_EXECUTABLE_PATH not set or invalid, skipping integration test.")

    # Create a dummy background image for the test
    dummy_bg = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)  # Solid blue

    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread', return_value=dummy_bg), \
         patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df') as mock_get_bg:

        mock_df = MagicMock()
        mock_df.sample.return_value.iloc[0].path = 'dummy.png'
        mock_get_bg.return_value = mock_df

        with Renderer(browser_executable=browser_executable) as renderer:
            # We need a real font file for rendering
            font_path = str(FONTS_ROOT / 'NotoSansJP-Regular.ttf')
            if not Path(font_path).exists():
                pytest.skip(f"Required font not found at {font_path}, skipping integration test.")

            lines = ["test"]
            override_params = {
                "font_path": font_path,
                "text_color": "white",
                "draw_bubble": False,  # No bubble to ensure we see the background
            }
            # The actual render call happens here
            img, _ = renderer.render(lines, override_css_params=override_params)

            # Assertions
            assert isinstance(img, np.ndarray)

            # To verify the background, we check if the average color of the
            # rendered image is close to blue. Since the text is white, the
            # average will be a mix, but blue should be the dominant channel.
            # We convert the grayscale image back to BGR to check channels.
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            mean_color = img_bgr.mean(axis=(0, 1))
            assert mean_color[0] > 100  # Blue channel should be high
            assert mean_color[1] < 50   # Green channel should be low
            assert mean_color[2] < 50   # Red channel should be low


def test_get_random_css_params():
    """Tests the generation of random CSS parameters."""
    params = Renderer.get_random_css_params()
    assert isinstance(params, dict)
    assert "font_size" in params
    assert "draw_bubble" in params
    if params["draw_bubble"]:
        assert "bubble_padding" in params


def test_get_css():
    """Tests the `get_css` function for correct CSS string generation."""
    css = get_css(
        font_size=12,
        font_path="dummy.ttf",
        vertical=True,
        glow_size=1,
        stroke_size=1,
        letter_spacing=0.1,
        text_orientation="upright",
        background_image_data_uri="data:image/png;base64,dummy",
        draw_bubble=True,
        bubble_border_color="red",
    )
    assert "font-size: 12px;" in css
    assert "writing-mode: vertical-rl;" in css
    assert "text-shadow" in css
    assert "letter-spacing" in css
    assert "text-orientation" in css
    assert 'background-image: url("data:image/png;base64,dummy")' in css
    assert "border:" in css
    assert "solid red" in css


def test_get_css_stroke():
    """Tests that get_css generates a correct stroke effect."""
    css = get_css(font_size=48, font_path="dummy.ttf", stroke_size=1, stroke_color="black")
    expected_shadows = [
        "-1px -1px 0 black", "1px -1px 0 black", "-1px 1px 0 black", "1px 1px 0 black",
        "-1px 0px 0 black", "1px 0px 0 black", "0px -1px 0 black", "0px 1px 0 black"
    ]
    for shadow in expected_shadows:
        assert shadow in css
    assert "0 0 1px" not in css


def test_get_css_glow():
    """Tests that get_css generates a correct glow effect."""
    css = get_css(font_size=48, font_path="dummy.ttf", glow_size=5, glow_color="blue")
    assert "text-shadow: 0 0 5px blue;" in css


class TestRendererParams(unittest.TestCase):
    @patch("numpy.random.choice")
    @patch("numpy.random.randint")
    @patch("numpy.random.uniform")
    @patch("numpy.random.rand")
    def test_get_random_css_params_with_bubble(
        self, mock_rand, mock_uniform, mock_randint, mock_choice
    ):
        """Verify that get_random_css_params generates bubble params when draw_bubble is True."""
        mock_rand.side_effect = [0.6, 0.4, 0.4, 0.6, 0.1, 0.9]
        mock_uniform.return_value = 1.8
        mock_randint.side_effect = [80, 20, 30, 3]
        mock_choice.side_effect = ["stroke", 2]

        params = Renderer.get_random_css_params()

        self.assertTrue(params["draw_bubble"])
        self.assertIn("bubble_padding", params)
        self.assertEqual(params["bubble_background_color"], "white")

    @patch("numpy.random.rand")
    def test_get_random_css_params_no_bubble(self, mock_rand):
        """Verify that get_random_css_params does not generate bubble params when draw_bubble is False."""
        mock_rand.side_effect = [0.6, 0.4, 0.8, 0.8, 0.8, 0.1, 0.1]
        with patch("numpy.random.choice"), patch("numpy.random.randint"), patch(
            "numpy.random.uniform"
        ):
            params = Renderer.get_random_css_params()

        self.assertFalse(params["draw_bubble"])
        self.assertNotIn("bubble_padding", params)
        self.assertEqual(params["bubble_background_color"], "transparent")
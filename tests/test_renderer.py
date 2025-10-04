"""Tests for the synthetic data rendering engine.

This module contains unit tests for the `Renderer` class and its associated
utility functions, which are responsible for generating synthetic text images.
These tests verify the functionality of text rendering, background composition,
and various image manipulation helpers.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css


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


def test_render(renderer):
    """
    Tests the main `render` method for a complete rendering pipeline.
    """
    # Mock the _render_html method to avoid actual browser rendering
    with patch.object(
        renderer, "_render_html", return_value=(np.zeros((200, 200, 3), dtype=np.uint8), {})
    ) as mock_render_html, patch(
        "manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread",
        return_value=np.zeros((100, 100, 3), dtype=np.uint8),
    ), patch(
        "manga_ocr_dev.synthetic_data_generator.renderer.cv2.imwrite"
    ):
        lines = ["test"]
        # Provide font_path as it's required by get_css
        override_params = {"font_path": "dummy.ttf"}
        img, params = renderer.render(lines, override_css_params=override_params)

        # Assertions
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 2  # Grayscale
        assert isinstance(params, dict)
        mock_render_html.assert_called_once()
        # Check that background_image_path was added to the params for rendering
        rendered_params = mock_render_html.call_args[0][1]
        assert "background_image_path" in rendered_params
        assert isinstance(rendered_params["background_image_path"], Path)


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
        background_image_path=Path("bg.png"),
        draw_bubble=True,
        bubble_border_color="red",
    )
    assert "font-size: 12px;" in css
    assert "writing-mode: vertical-rl;" in css
    assert "text-shadow" in css
    assert "letter-spacing" in css
    assert "text-orientation" in css
    assert "background-image: url" in css
    assert "border:" in css
    assert "solid red" in css


def test_get_css_stroke():
    """Tests that get_css generates a correct stroke effect."""
    css = get_css(font_size=48, font_path="dummy.ttf", stroke_size=1, stroke_color="black")
    # A correct implementation should create a hard-edged stroke.
    # This can be done with multiple shadows at different offsets and 0 blur.
    expected_shadows = [
        "-1px -1px 0 black",
        "1px -1px 0 black",
        "-1px 1px 0 black",
        "1px 1px 0 black",
        "-1px 0px 0 black",
        "1px 0px 0 black",
        "0px -1px 0 black",
        "0px 1px 0 black",
    ]

    for shadow in expected_shadows:
        assert shadow in css

    # Also, ensure that the glow effect is not present for a stroke.
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
        # Arrange
        # rand() calls: vertical, text_color, draw_bubble, text_orientation, letter_spacing
        mock_rand.side_effect = [
            0.6,
            0.4,
            0.4,
            0.6,
            0.1,
        ]  # draw_bubble is True (0.4 < 0.7)
        mock_uniform.return_value = 1.8  # line_height
        mock_randint.side_effect = [
            80,
            20,
            30,
            3,
        ]  # font_size, bubble_padding, bubble_border_radius, bubble_border_width
        mock_choice.side_effect = ["stroke", 2]  # effect, stroke_size

        # Act
        params = Renderer.get_random_css_params()

        # Assert
        self.assertTrue(params["draw_bubble"])
        self.assertIn("bubble_padding", params)
        self.assertIn("bubble_border_radius", params)
        self.assertIn("bubble_border_width", params)
        self.assertIn("bubble_background_color", params)
        self.assertIn("bubble_border_color", params)
        self.assertEqual(
            params["bubble_background_color"], "white"
        )  # text is black

    @patch("numpy.random.rand")
    def test_get_random_css_params_no_bubble(self, mock_rand):
        """Verify that get_random_css_params does not generate bubble params when draw_bubble is False."""
        # Arrange
        # rand() calls: vertical, text_color, draw_bubble, text_orientation, letter_spacing
        mock_rand.side_effect = [
            0.6,
            0.4,
            0.8,
            0.8,
            0.8,
        ]  # draw_bubble is False (0.8 > 0.7)

        # Act
        with patch("numpy.random.choice"), patch("numpy.random.randint"), patch(
            "numpy.random.uniform"
        ):
            params = Renderer.get_random_css_params()

        # Assert
        self.assertFalse(params["draw_bubble"])
        self.assertNotIn("bubble_padding", params)
        self.assertNotIn("bubble_border_radius", params)
        self.assertNotIn("bubble_border_width", params)
        self.assertEqual(params["bubble_background_color"], "transparent")
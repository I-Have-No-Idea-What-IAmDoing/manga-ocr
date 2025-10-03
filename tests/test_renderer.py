import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import cv2

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, get_css, crop_by_alpha, blend, rounded_rectangle


@pytest.fixture
@patch('manga_ocr_dev.vendored.html2image.browsers.chrome_cdp.find_chrome')
@patch('manga_ocr_dev.synthetic_data_generator.renderer.get_background_df')
def renderer(mock_get_background_df, mock_find_chrome):
    """
    Provides a Renderer instance with mocked dependencies for testing.
    """
    mock_find_chrome.return_value = 'dummy_chrome_path'
    mock_get_background_df.return_value = MagicMock()
    r = Renderer()
    r.hti = MagicMock()
    return r


def test_renderer_initialization(renderer):
    assert renderer.hti is not None
    assert renderer.background_df is not None


def test_render_text(renderer):
    """
    Tests the render_text method.
    """
    renderer.hti.screenshot_as_bytes.return_value = cv2.imencode('.png', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()

    lines = ['test']
    img, params = renderer.render_text(lines, override_css_params={'font_path': 'dummy.ttf'})

    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 4
    assert isinstance(params, dict)


def test_get_random_css_params():
    params = Renderer.get_random_css_params()
    assert isinstance(params, dict)
    assert 'font_size' in params


def test_render_background(renderer):
    img = np.zeros((100, 100, 4), dtype=np.uint8)

    with patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread') as mock_imread:
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'

        result_img = renderer.render_background(img)
        assert isinstance(result_img, np.ndarray)


def test_get_css():
    css = get_css(font_size=12, font_path='dummy.ttf', vertical=True, shadow_size=1, stroke_size=1, letter_spacing=0.1, text_orientation='upright')
    assert 'font-size: 12px;' in css
    assert 'writing-mode: vertical-rl;' in css
    assert 'text-shadow' in css
    assert 'letter-spacing' in css
    assert 'text-orientation' in css


def test_crop_by_alpha():
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
    fg = np.zeros((100, 100, 4), dtype=np.uint8)
    fg[:, :, 3] = 128 # 50% transparent
    bg = np.full((100, 100, 3), 255, dtype=np.uint8) # white background

    blended_img = blend(fg, bg)
    assert blended_img.shape == (100, 100, 3)
    assert np.all(blended_img[0, 0] == [127, 127, 127])


def test_rounded_rectangle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result_img = rounded_rectangle(img, (10, 10), (90, 90), radius=0.5, color=(255, 255, 255), thickness=-1)
    assert np.any(result_img > 0)


def test_render(renderer):
    """Tests the main render method orchestration."""
    renderer.render_text = MagicMock(return_value=(np.zeros((100, 100, 4), dtype=np.uint8), {}))
    renderer.render_background = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

    lines = ["test"]
    img, params = renderer.render(lines)

    renderer.render_text.assert_called_once_with(lines, None)
    renderer.render_background.assert_called_once()

    assert img.ndim == 2  # Should be grayscale
    assert isinstance(params, dict)


def test_render_text_bgr_input(renderer):
    """Tests that render_text correctly handles 3-channel BGR images."""
    bgr_img = np.zeros((100, 100, 3), dtype=np.uint8)
    renderer.hti.screenshot_as_bytes.return_value = cv2.imencode('.png', bgr_img)[1].tobytes()

    lines = ['test']
    img, params = renderer.render_text(lines, override_css_params={'font_path': 'dummy.ttf'})

    assert img.shape == (100, 100, 4)


@patch('numpy.random.choice')
@patch('numpy.random.rand')
def test_get_random_css_params_stroke(mock_rand, mock_choice):
    """Tests that get_random_css_params correctly generates stroke parameters."""
    mock_rand.side_effect = [0, 0]  # vertical=True, text_orientation=True
    mock_choice.side_effect = ['stroke', 5]
    params = Renderer.get_random_css_params()
    assert 'stroke_size' in params
    assert params['stroke_size'] == 5
    assert 'stroke_color' in params
    assert 'shadow_size' not in params


@patch('numpy.random.choice')
@patch('numpy.random.rand')
def test_get_random_css_params_shadow(mock_rand, mock_choice):
    """Tests that get_random_css_params correctly generates shadow parameters."""
    mock_rand.side_effect = [0.1, 0.2, 0.9]  # vertical=True, text_orientation=True, shadow_color='black'
    mock_choice.side_effect = ['shadow', 10]
    params = Renderer.get_random_css_params()
    assert 'shadow_size' in params
    assert params['shadow_size'] == 10
    assert params['shadow_color'] == 'black'
    assert 'stroke_size' not in params


def test_render_background_small_image(renderer):
    """Tests that render_background handles very small images correctly."""
    img = np.zeros((5, 5, 4), dtype=np.uint8)
    result_img = renderer.render_background(img)
    assert result_img.shape == (10, 10, 3)
    assert np.all(result_img == 0)


@patch('manga_ocr_dev.synthetic_data_generator.renderer.cv2.imread')
@patch('numpy.random.random')
@patch('numpy.random.uniform')
def test_render_background_with_bubble(mock_uniform, mock_random, mock_imread, renderer):
    """Tests the render_background path with a bubble."""
    # Set mock return values
    mock_uniform.return_value = 0.2
    mock_random.return_value = 0.6  # Ensures draw_bubble is True
    mock_imread.return_value = np.full((200, 200, 3), 255, dtype=np.uint8)
    renderer.background_df.sample.return_value.iloc[0].path = 'dummy.png'

    # Prepare input image
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[10:40, 10:40, 3] = 255  # Content is 30x30

    # Calculate expected intermediate sizes
    # After crop_by_alpha, img is 30x30.
    # m0 = int(min(30, 30) * uniform(0.2, 0.4)) -> mocked to 30 * 0.2 = 6
    # Padded shape is (30 + 6*2, 30 + 6*2) = (42, 42)
    padded_size = 42
    bubble_mock = np.zeros((padded_size, padded_size, 4), dtype=np.uint8)
    renderer.create_bubble = MagicMock(return_value=bubble_mock)

    # Run the function
    result_img = renderer.render_background(img)

    # Assertions
    renderer.create_bubble.assert_called_once()
    # Check that create_bubble was called with the correct shape
    assert renderer.create_bubble.call_args[0][0] == (padded_size, padded_size, 4)
    assert result_img is not None
    assert result_img.shape[2] == 3


def test_create_bubble(renderer):
    """Tests that create_bubble generates a valid bubble image."""
    shape = (200, 200, 4)
    margin = 30
    bubble = renderer.create_bubble(shape, margin)
    assert bubble.shape == shape
    assert bubble.dtype == np.uint8
    assert np.any(bubble > 0)
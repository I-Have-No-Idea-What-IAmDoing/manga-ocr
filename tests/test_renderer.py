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
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[20:80, 20:80, 3] = 255
    cropped_img = crop_by_alpha(img, margin=10)
    assert cropped_img.shape[0] == 80
    assert cropped_img.shape[1] == 80

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

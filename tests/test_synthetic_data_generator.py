import numpy as np
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.env import FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.utils import get_font_meta


def test_synthetic_data_generator():
    """
    Tests that the synthetic data generator can successfully produce an image-text pair
    and correctly filters characters not present in the font.
    """
    browser_executable = '/home/jules/.cache/ms-playwright/chromium-1181/chrome-linux/chrome'
    dummy_font_path = str(FONTS_ROOT / 'dummy_font.ttf')

    # Get the character set for the dummy font
    _, font_map = get_font_meta()
    font_key = 'dummy_font.ttf'
    valid_chars = font_map.get(font_key, set())

    input_text = 'test text'
    # The generator will filter out any characters from the input text that are not
    # in the font's character set.
    expected_text = "".join([c for c in input_text if c in valid_chars])

    with Renderer(browser_executable=browser_executable) as renderer:
        generator = SyntheticDataGenerator(renderer=renderer)
        img, text, params = generator.process(
            input_text,
            override_css_params={'font_path': dummy_font_path}
        )

        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert isinstance(text, str)
        assert text == expected_text
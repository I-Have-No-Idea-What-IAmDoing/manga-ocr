import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator.common.composer import Composer


class TestComposer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backgrounds_dir = self.temp_dir / "backgrounds"
        self.backgrounds_dir.mkdir()

        # Create a high-contrast dummy background for general tests
        self.dummy_bg_path = self.backgrounds_dir / "dummy_bg_0_100_0_100.png"
        dummy_bg = Image.new('RGB', (100, 100), 'white')
        dummy_bg.save(self.dummy_bg_path)

        # Create a dummy text image (black text)
        self.dummy_text_image = np.zeros((50, 80, 4), dtype=np.uint8)
        self.dummy_text_image[10:40, 10:70, 3] = 255  # Opaque text with a transparent border

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that the composer initializes correctly."""
        composer = Composer(self.backgrounds_dir)
        self.assertIsNotNone(composer.background_df)
        self.assertFalse(composer.background_df.empty)

    def test_bubble_drawing(self):
        """Test the bubble drawing functionality."""
        composer = Composer(self.backgrounds_dir)
        bubble = composer.draw_bubble(100, 50, text_color='#000000')
        self.assertIsInstance(bubble, Image.Image)
        self.assertTrue(np.any(np.array(bubble)[:, :, 3] > 0))

    def test_high_contrast_bubble(self):
        """Test that the bubble color contrasts with the text color."""
        composer = Composer(self.backgrounds_dir)

        # Dark text should get a light bubble
        bubble_for_dark_text = composer.draw_bubble(100, 50, text_color='#000000')
        bubble_array = np.array(bubble_for_dark_text)
        center_pixel_color = bubble_array[bubble_array.shape[0] // 2, bubble_array.shape[1] // 2]
        self.assertTrue(np.array_equal(center_pixel_color, [255, 255, 255, 255]), "Dark text should get a white bubble")

        # Light text should get a dark bubble
        bubble_for_light_text = composer.draw_bubble(100, 50, text_color='#FFFFFF')
        bubble_array = np.array(bubble_for_light_text)
        center_pixel_color = bubble_array[bubble_array.shape[0] // 2, bubble_array.shape[1] // 2]
        self.assertTrue(np.array_equal(center_pixel_color, [0, 0, 0, 255]), "Light text should get a black bubble")

    def test_composition_with_background(self):
        """Test composing a text image with a background."""
        composer = Composer(self.backgrounds_dir)
        final_image = composer(self.dummy_text_image, {})
        self.assertIsInstance(final_image, np.ndarray)
        self.assertEqual(final_image.ndim, 2)

    @patch('numpy.random.randint')
    def test_dynamic_scaling(self, mock_randint):
        """Test that the background is scaled up deterministically."""
        # Mock randint to return the lower bound, making the crop predictable
        mock_randint.side_effect = lambda low, high: low

        large_text_image = np.zeros((150, 150, 4), dtype=np.uint8)
        large_text_image[:, :, 3] = 255

        composer = Composer(self.backgrounds_dir)
        final_image = composer(large_text_image, {})

        self.assertIsInstance(final_image, np.ndarray)
        # The background (100x100) must scale up to fit the text (150x150).
        # The final cropped image should be larger than the original text image.
        self.assertGreater(final_image.shape[0], large_text_image.shape[0])
        self.assertGreater(final_image.shape[1], large_text_image.shape[1])

    @patch('numpy.random.rand', return_value=0.8)  # Ensure no bubble
    @patch('albumentations.Compose', lambda transforms: lambda image: {'image': image})
    def test_low_contrast_rejection(self, mock_rand):
        """Test that low-contrast images are rejected."""
        # Create a dedicated directory with only a low-contrast background
        low_contrast_dir = self.temp_dir / "low_contrast_bg"
        low_contrast_dir.mkdir()
        black_bg_path = low_contrast_dir / "black_bg_0_200_0_200.png"
        Image.new('RGB', (200, 200), 'black').save(black_bg_path)

        # Create black text
        black_text = np.zeros((50, 100, 4), dtype=np.uint8)
        black_text[10:40, 10:90, 3] = 255  # Opaque text on transparent bg

        # Initialize the composer with only the low-contrast background
        composer = Composer(low_contrast_dir)
        result = composer(black_text, {})
        self.assertIsNone(result, "Low-contrast image (black on black) should be rejected")

    @patch('numpy.random.rand', return_value=0.8)  # Ensure no bubble
    @patch('albumentations.Compose', lambda transforms: lambda image: {'image': image})
    def test_high_contrast_acceptance(self, mock_rand):
        """Test that high-contrast images are accepted."""
        # Background is white from setUp, text is black.
        composer = Composer(self.backgrounds_dir)
        result = composer(self.dummy_text_image, {})
        self.assertIsInstance(result, np.ndarray)

    def test_rejection_of_invalid_input(self):
        """Test that the composer returns None for invalid (None or empty) input."""
        composer = Composer(self.backgrounds_dir)
        self.assertIsNone(composer(None, {}))
        self.assertIsNone(composer(np.array([]), {}))

    @patch('numpy.random.rand', return_value=0.8) # Ensure no bubble is drawn
    def test_rejection_of_small_text(self, mock_rand):
        """Test that text images smaller than the minimum height are rejected."""
        composer = Composer(self.backgrounds_dir)
        small_text_image = np.zeros((5, 5, 4), dtype=np.uint8)
        small_text_image[:, :, 3] = 255
        self.assertIsNone(composer(small_text_image, {}))

    @patch('numpy.random.rand', return_value=0.2) # Ensure bubble is drawn
    def test_composition_without_backgrounds(self, mock_rand):
        """Test composition when no background directory is provided."""
        # Create a temporary directory that is empty
        empty_bg_dir = self.temp_dir / "empty_bg"
        empty_bg_dir.mkdir()

        composer = Composer(empty_bg_dir)
        # With no background, the result should be the text pasted on a bubble
        result = composer(self.dummy_text_image, {})
        self.assertIsInstance(result, np.ndarray)
        # The result should have the shape of the bubble, not the original text image
        self.assertGreater(result.shape[0], self.dummy_text_image.shape[0])
        self.assertGreater(result.shape[1], self.dummy_text_image.shape[1])

    def test_resizing_logic(self):
        """Test the target_size and min_output_size resizing logic."""
        # Test target_size
        composer_target = Composer(self.backgrounds_dir, target_size=(120, 80))
        result_target = composer_target(self.dummy_text_image, {})
        self.assertEqual(result_target.shape[:2], (80, 120)) # (height, width)

        # Test min_output_size (using an image that would otherwise be smaller)
        # We need to ensure the final cropped image is smaller than min_output_size
        # to trigger the resize. We can't control the crop easily, so we'll
        # use a no-background composer to make the output predictable.
        empty_bg_dir = self.temp_dir / "empty_bg_2"
        empty_bg_dir.mkdir()
        composer_min = Composer(empty_bg_dir, min_output_size=200)
        result_min = composer_min(self.dummy_text_image, {})
        self.assertTrue(min(result_min.shape[:2]) >= 200)

    def test_low_contrast_with_no_background_area(self):
        """Test _is_low_contrast when there's no background area to sample."""
        composer = Composer(self.backgrounds_dir)
        final_img = np.zeros((100, 100, 3), dtype=np.uint8)
        text_img = np.zeros((50, 50, 4), dtype=np.uint8)

        # Create a text mask that covers the entire ROI, leaving no background
        text_mask_full = np.ones((50, 50, 4), dtype=np.uint8) * 255

        # The method should return False, assuming sufficient contrast
        is_low = composer._is_low_contrast(final_img, text_mask_full, 0, 0)
        self.assertFalse(is_low)


if __name__ == '__main__':
    unittest.main()
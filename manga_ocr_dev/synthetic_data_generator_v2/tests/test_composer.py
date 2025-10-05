import sys
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from manga_ocr_dev.synthetic_data_generator_v2.composer import Composer


class TestComposer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backgrounds_dir = self.temp_dir / "backgrounds"
        self.backgrounds_dir.mkdir()

        # Create a dummy background image
        dummy_bg = Image.new('RGB', (400, 400), 'blue')
        dummy_bg.save(self.backgrounds_dir / "dummy_bg.png")

        # Create a dummy text image
        self.dummy_text_image = np.zeros((50, 100, 4), dtype=np.uint8)
        self.dummy_text_image[:, :, 3] = 255  # Make it opaque

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
        bubble = composer.draw_bubble(100, 50)
        self.assertIsInstance(bubble, Image.Image)
        # Check for non-transparent pixels, indicating a bubble was drawn
        self.assertTrue(np.any(np.array(bubble)[:, :, 3] > 0))

    def test_composition_with_background(self):
        """Test composing a text image with a background."""
        composer = Composer(self.backgrounds_dir)
        final_image = composer(self.dummy_text_image, {})
        self.assertIsInstance(final_image, np.ndarray)
        self.assertEqual(final_image.ndim, 3)

    def test_composition_with_target_size(self):
        """Test that the output image is resized to the target size."""
        target_size = (150, 150)
        composer = Composer(self.backgrounds_dir, target_size=target_size)
        final_image = composer(self.dummy_text_image, {})
        self.assertEqual(final_image.shape[:2], (target_size[1], target_size[0]))

    def test_composition_with_min_output_size(self):
        """Test that the output image is upscaled to the minimum size."""
        min_size = 500 # Larger than the dummy background
        composer = Composer(self.backgrounds_dir, min_output_size=min_size)
        final_image = composer(self.dummy_text_image, {})
        self.assertGreaterEqual(min(final_image.shape[:2]), min_size)

if __name__ == '__main__':
    unittest.main()
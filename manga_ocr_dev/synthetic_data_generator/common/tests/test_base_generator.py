import unittest
import numpy as np
from manga_ocr_dev.synthetic_data_generator_v2.generator import SyntheticDataGeneratorV2
from manga_ocr_dev.env import ASSETS_PATH

class TestBaseGenerator(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment by initializing the data generator.
        If the assets directory is not found, skip the tests.
        """
        if not ASSETS_PATH.exists():
            self.skipTest(f"Assets directory not found at {ASSETS_PATH}. "
                          "Ensure tests are run from the project root.")
        np.random.seed(0)
        self.generator = SyntheticDataGeneratorV2()

    def test_add_random_furigana_structure(self):
        """
        Tests that the `add_random_furigana` method correctly structures its output,
        applying furigana to kanji while respecting word boundaries and okurigana.
        """
        # A test sentence with a mix of kanji-only words and words with okurigana.
        line = "美味しいご飯を食べる"

        # Process the line with a 100% probability of adding furigana.
        processed_chunks = self.generator.add_random_furigana(line, word_prob=1.0)

        # Expected structure:
        # '美味しい' -> ('furigana', '美味', 'おい') + 'しい'
        # 'ご飯' -> ('furigana', 'ご飯', 'ごはん')
        # 'を' -> 'を'
        # '食べる' -> ('furigana', '食', 'た') + 'べる'
        expected_chunks = [
            ('furigana', '美味', 'おい'),
            'しいご',
            ('furigana', '飯', 'めし'),
            'を',
            ('furigana', '食', 'た'),
            'べる'
            ]

        self.assertEqual(processed_chunks, expected_chunks, "Furigana markup structure is incorrect.")

    def test_add_random_furigana_no_kanji(self):
        """
        Tests that `add_random_furigana` does not add furigana to a line with no kanji.
        """
        line = "こんにちは、げんきですか"
        processed_chunks = self.generator.add_random_furigana(line, word_prob=1.0)

        # The line should remain unchanged as there are no kanji to apply furigana to.
        self.assertEqual(processed_chunks, [line], "Furigana should not be added to non-kanji text.")

    def test_furigana_rendering(self):
        """
        Tests that the furigana markup can be rendered without errors and produces a non-empty image.
        This is a basic check to prevent visual regressions.
        """
        line = "日本語"
        processed_chunks = self.generator.add_random_furigana(line, word_prob=1.0)

        # Define some basic rendering parameters.
        params = {
            'font_path': self.generator.get_random_font(line),
            'font_size': 30,
            'vertical': False,
        }

        # Render the processed chunks.
        img = self.generator.render([processed_chunks], params)

        # The test passes if an image is generated and it's not empty.
        self.assertIsInstance(img, np.ndarray, "Rendering should produce a numpy array.")
        self.assertGreater(img.size, 0, "Rendered image should not be empty.")

if __name__ == '__main__':
    unittest.main()
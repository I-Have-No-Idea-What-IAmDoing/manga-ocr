import unittest
from manga_ocr_dev.synthetic_data_generator.common.exceptions import SkipSample

class TestExceptions(unittest.TestCase):
    def test_skip_sample_exception(self):
        with self.assertRaises(SkipSample):
            raise SkipSample("Test message")

        with self.assertRaises(SkipSample) as cm:
            raise SkipSample("Another test message")
        self.assertEqual(str(cm.exception), "Another test message")
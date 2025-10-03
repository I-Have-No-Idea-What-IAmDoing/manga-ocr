import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent dir to path to import the vendored module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from manga_ocr_dev.vendored.html2image.html2image import Html2Image

class TestHtml2ImageBugs(unittest.TestCase):

    def setUp(self):
        # Mock the browser dependency to isolate the test.
        self.mock_browser_class = MagicMock()
        self.browser_map_patcher = patch.dict(
            'manga_ocr_dev.vendored.html2image.html2image.browser_map',
            {'chrome': self.mock_browser_class, 'chrome-cdp': self.mock_browser_class}
        )
        self.browser_map_patcher.start()

    def tearDown(self):
        self.browser_map_patcher.stop()

    def test_screenshot_with_defaults_processes_no_files(self):
        """
        Verifies that calling screenshot() with default arguments does not
        attempt to process any files. This confirms the mutable default
        argument bug is fixed.
        """
        # 1. Create an instance of Html2Image.
        hti = Html2Image()

        # 2. Mock the methods that would perform file I/O.
        hti.load_file = MagicMock()
        hti.screenshot_loaded_file = MagicMock()
        hti.screenshot_url = MagicMock()
        hti.load_str = MagicMock()

        # 3. Call screenshot() with no arguments.
        paths = hti.screenshot()

        # 4. Assert that no file processing methods were called and no paths were returned.
        hti.load_file.assert_not_called()
        hti.screenshot_loaded_file.assert_not_called()
        hti.screenshot_url.assert_not_called()
        hti.load_str.assert_not_called()
        self.assertEqual(paths, [])

if __name__ == '__main__':
    unittest.main()
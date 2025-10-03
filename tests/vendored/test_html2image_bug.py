import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
import shutil

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
        # Create a temporary directory for test inputs
        self.input_dir = tempfile.mkdtemp()

    def tearDown(self):
        self.browser_map_patcher.stop()
        # Clean up the temporary directory
        shutil.rmtree(self.input_dir)

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

    def test_css_is_applied_to_html_file(self):
        """
        Verifies that CSS from css_str is applied to html_file.
        """
        hti = Html2Image()

        # Mock the method that actually calls the browser
        hti.browser.screenshot = MagicMock()

        # Let's spy on what content is being written to the temp file for screenshotting
        spied_content = ""
        original_load_str = hti.load_str

        def load_str_spy(content, as_filename):
            nonlocal spied_content
            spied_content = content
            original_load_str(content, as_filename)

        hti.load_str = load_str_spy

        # Create a temporary HTML file in our separate input directory
        html_content = "<html><body><h1>Title</h1></body></html>"
        html_filename = "test_input.html"
        html_filepath = os.path.join(self.input_dir, html_filename)
        with open(html_filepath, "w", encoding='utf-8') as f:
            f.write(html_content)

        # CSS to be applied
        css_str = "h1 { color: red; }"

        # Act
        hti.screenshot(html_file=html_filepath, css_str=css_str)

        # Assert
        # Before the fix, `load_str` is not called for `html_file`, so spied_content is empty.
        # After the fix, `load_str` will be called with the combined html and css.
        self.assertIn(css_str, spied_content)
        self.assertIn("<h1>Title</h1>", spied_content)


if __name__ == '__main__':
    unittest.main()
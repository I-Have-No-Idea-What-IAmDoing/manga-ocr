import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

from manga_ocr_dev.data import process_manga109s

DUMMY_XML_FRAMES = """
<annotation>
    <pages>
        <page index="0" width="1200" height="900">
            <frame id="frame1" xmin="10" ymin="20" xmax="110" ymax="120" />
        </page>
    </pages>
</annotation>
"""

DUMMY_XML_CROPS = """
<annotation>
    <pages>
        <page index="0" width="1200" height="900">
            <text id="text1" xmin="30" ymin="40" xmax="80" ymax="60">1</text>
            <text id="text2" xmin="30" ymin="40" xmax="80" ymax="60">2</text>
            <text id="text3" xmin="30" ymin="40" xmax="80" ymax="60">3</text>
            <text id="text4" xmin="30" ymin="40" xmax="80" ymax="60">4</text>
            <text id="text5" xmin="30" ymin="40" xmax="80" ymax="60">5</text>
            <text id="text6" xmin="30" ymin="40" xmax="80" ymax="60">6</text>
            <text id="text7" xmin="30" ymin="40" xmax="80" ymax="60">7</text>
            <text id="text8" xmin="30" ymin="40" xmax="80" ymax="60">8</text>
            <text id="text9" xmin="30" ymin="40" xmax="80" ymax="60">9</text>
            <text id="text10" xmin="30" ymin="40" xmax="80" ymax="60">10</text>
        </page>
    </pages>
</annotation>
"""

class TestProcessManga109s(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.manga109_root = Path(self.tmpdir.name)
        self.manga109s_root = self.manga109_root / "Manga109s_released_2021_02_28"
        self.manga109s_root.mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_get_books(self):
        """Test that get_books reads book list and constructs paths correctly."""
        with patch('manga_ocr_dev.data.process_manga109s.MANGA109_ROOT', self.manga109_root):
            (self.manga109s_root / 'books.txt').write_text('TestBook\n')

            books_df = process_manga109s.get_books()

            self.assertEqual(len(books_df), 1)
            self.assertEqual(books_df.iloc[0]['book'], 'TestBook')
            self.assertTrue(books_df.iloc[0]['annotations'].endswith('annotations/TestBook.xml'))
            self.assertTrue(books_df.iloc[0]['images'].endswith('images/TestBook'))

    @patch('xml.etree.ElementTree.parse')
    @patch('manga_ocr_dev.data.process_manga109s.get_books')
    def test_export_frames(self, mock_get_books, mock_et_parse):
        """Test that export_frames correctly parses XML and saves frame data."""
        with patch('manga_ocr_dev.data.process_manga109s.MANGA109_ROOT', self.manga109_root):
            mock_get_books.return_value = pd.DataFrame([
                {'book': 'TestBook', 'annotations': 'dummy.xml', 'images': 'dummy_images'}
            ])
            mock_et_parse.return_value.getroot.return_value = ET.fromstring(DUMMY_XML_FRAMES)

            process_manga109s.export_frames()

            output_csv = self.manga109_root / 'frames.csv'
            self.assertTrue(output_csv.exists())
            df = pd.read_csv(output_csv)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]['id'], 'frame1')
            self.assertEqual(df.iloc[0]['xmin'], 10)

    @patch('cv2.imwrite')
    @patch('cv2.imread')
    @patch('xml.etree.ElementTree.parse')
    @patch('manga_ocr_dev.data.process_manga109s.get_books')
    def test_export_crops(self, mock_get_books, mock_et_parse, mock_imread, mock_imwrite):
        """Test that export_crops parses XML, processes data, and saves crops."""
        np.random.seed(0)
        with patch('manga_ocr_dev.data.process_manga109s.MANGA109_ROOT', self.manga109_root):
            images_dir = self.manga109s_root / "images" / "TestBook"
            images_dir.mkdir(parents=True)

            mock_get_books.return_value = pd.DataFrame([
                {'book': 'TestBook', 'annotations': 'dummy.xml', 'images': str(images_dir)}
            ])
            mock_et_parse.return_value.getroot.return_value = ET.fromstring(DUMMY_XML_CROPS)
            mock_imread.return_value = np.zeros((900, 1200, 3), dtype=np.uint8)

            process_manga109s.export_crops()

            output_csv = self.manga109_root / 'data.csv'
            self.assertTrue(output_csv.exists())
            df = pd.read_csv(output_csv)
            self.assertEqual(len(df), 10)
            self.assertEqual(df['split'].value_counts()['test'], 1)

            crops_dir = self.manga109_root / 'crops'
            self.assertTrue(crops_dir.is_dir())
            self.assertEqual(mock_imwrite.call_count, 10)

            expected_crop_path = crops_dir / 'text1.png'
            called_paths = [Path(c[0][0]) for c in mock_imwrite.call_args_list]
            self.assertIn(expected_crop_path, called_paths)

if __name__ == '__main__':
    unittest.main()
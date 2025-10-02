import json
from pathlib import Path

from manga_ocr import MangaOcr
from manga_ocr.ocr import post_process

TEST_DATA_ROOT = Path(__file__).parent / "data"


def test_ocr():
    mocr = MangaOcr()

    expected_results = json.loads((TEST_DATA_ROOT / "expected_results.json").read_text(encoding="utf-8"))

    for item in expected_results:
        result = mocr(TEST_DATA_ROOT / "images" / item["filename"])
        assert result == item["result"]


def test_post_process():
    """
    Test the post_process function.
    """
    assert post_process("…") == "．．．"
    assert post_process("・・") == "．．"

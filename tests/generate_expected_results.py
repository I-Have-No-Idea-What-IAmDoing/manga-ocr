import json
from pathlib import Path

from tqdm import tqdm

from manga_ocr import MangaOcr

TEST_DATA_ROOT = Path(__file__).parent / "data"


def generate_expected_results():
    """
    Generates expected OCR results for the test images and saves them to a JSON file.

    This function initializes the MangaOcr model, processes each image in the
    test data directory, and saves the OCR output along with the image filename
    to 'expected_results.json'. This JSON file is then used by the test suite
    to verify the model's performance.
    """
    mocr = MangaOcr()

    results = []

    for path in tqdm(sorted((TEST_DATA_ROOT / "images").iterdir())):
        result = mocr(path)
        results.append({"filename": path.name, "result": result})

    (TEST_DATA_ROOT / "expected_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    generate_expected_results()
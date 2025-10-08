"""A script to generate expected OCR results for the test suite.

This script is used to create a baseline of expected results for the OCR model
on a set of test images. It runs the OCR on each image in the `tests/data/images`
directory and saves the output to `expected_results.json`. This JSON file is
then used by the integration tests to ensure that changes to the model do not
cause unexpected regressions in performance.
"""

import json
from pathlib import Path

from tqdm import tqdm

from manga_ocr import MangaOcr

TEST_DATA_ROOT = Path(__file__).parent / "data"


def generate_expected_results():
    """Generates expected OCR results for test images.

    This function initializes the `MangaOcr` model, iterates through all images
    in the `tests/data/images` directory, and performs OCR on each one. The
    results, consisting of filenames and their corresponding recognized text,
    are then saved to `tests/data/expected_results.json`.

    This script should be run whenever there is a significant, intentional
    change in the OCR model's output, to update the baseline for integration
    tests. The resulting JSON file serves as the ground truth for verifying
    that the model's performance on key examples does not regress unexpectedly.
    """
    mocr = MangaOcr()

    results = []

    for path in tqdm(sorted((Path(TEST_DATA_ROOT) / "images").iterdir())):
        result = mocr(path)
        results.append({"filename": path.name, "result": result})

    (Path(TEST_DATA_ROOT) / "expected_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    generate_expected_results()
"""Processes the CC-100 Japanese text corpus for synthetic data generation.

This script reads the raw Japanese text file from the CC-100 dataset, filters
out short lines, and packages the text into smaller, numbered CSV files. These
CSV files serve as a text source for the synthetic data generator, which uses
them to render realistic text images for training the OCR model.
"""

import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, ASSETS_PATH


def export_lines(num_lines_in_each_package=10000, num_packages=100):
    """Reads the CC-100 corpus and exports it into smaller CSV packages.

    This function processes the `ja.txt` file from the CC-100 dataset. It
    reads the text line by line, filters out lines that are too short (fewer
    than three characters), and groups the remaining lines into multiple CSV
    files, or "packages."

    Each exported CSV is saved to the `ASSETS_PATH/lines` directory and contains
    'source', 'id', and 'line' columns, which are used by the synthetic data
    generator.

    Args:
        num_lines_in_each_package (int, optional): The maximum number of lines
            to include in each CSV package. Defaults to 10000.
        num_packages (int, optional): The total number of packages to create.
            The function will stop after creating this many packages, even if
            more text is available in the source file. Defaults to 100.
    """
    cc100_text_file = DATA_SYNTHETIC_ROOT / "ja.txt"

    id_count = 0
    with open(cc100_text_file, "r", encoding="utf-8") as file:
        for package_count in range(num_packages):
            line_count = 0
            data = []
            for line in tqdm(
                file, desc=f"creating package {package_count:04} of {num_packages}"
            ):
                id_count += 1
                stripped_line = line.strip()
                # skip too short line
                if len(stripped_line) <= 2:
                    continue

                row = {}
                row["source"] = "cc-100"
                row["id"] = f"cc-100_{id_count}"
                row["line"] = stripped_line

                data.append(row)

                line_count += 1
                if line_count >= num_lines_in_each_package:
                    break

            data = pd.DataFrame(data)
            data.to_csv(
                ASSETS_PATH / "lines" / f"{package_count:04}.csv",
                index=False,
                escapechar="\\",
            )


if __name__ == "__main__":
    export_lines()
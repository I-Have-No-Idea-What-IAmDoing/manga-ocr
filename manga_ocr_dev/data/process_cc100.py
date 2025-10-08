"""Processes the CC-100 Japanese text corpus for synthetic data generation.

This script reads the raw Japanese text file from the CC-100 dataset, filters
out short lines, and packages the text into smaller, numbered CSV files. These
CSV files serve as a text source for the synthetic data generator, which uses
them to render realistic text images for training the OCR model.
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, ASSETS_PATH


def export_lines(num_lines_in_each_package=12500, num_packages=160):
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
    cc100_text_file = Path(DATA_SYNTHETIC_ROOT) / "ja.txt"

    # Initialize a counter for generating unique IDs for each line
    id_count = 0

    # Open the raw CC-100 text file for reading
    with open(cc100_text_file, "r", encoding="utf-8") as file:
        # Loop to create the specified number of packages
        for package_count in range(num_packages):
            line_count = 0
            data = []

            # Iterate through lines in the file to build a package
            for line in tqdm(
                file, desc=f"creating package {package_count:04} of {num_packages}"
            ):
                id_count += 1
                stripped_line = line.strip()

                # Skip lines that are too short to be useful for training
                if len(stripped_line) <= 2:
                    continue

                # Create a dictionary for the line with its metadata
                row = {
                    "source": "cc-100",
                    "id": f"cc-100_{id_count}",
                    "line": stripped_line,
                }
                data.append(row)

                # Stop when the package reaches the desired size
                line_count += 1
                if line_count >= num_lines_in_each_package:
                    break

            # Convert the list of dictionaries to a DataFrame
            data = pd.DataFrame(data)

            # Save the package as a CSV file
            output_path = Path(ASSETS_PATH) / "lines" / f"{package_count:04}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(
                output_path,
                index=False,
                escapechar="\\",
            )


if __name__ == "__main__":
    export_lines()
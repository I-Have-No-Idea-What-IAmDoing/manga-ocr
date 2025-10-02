import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("J:/Applications/manga-ocr/")

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, ASSETS_PATH


def export_lines(num_lines_in_each_package=10000, num_packages=100):
    """
    Processes a large Japanese text file from the CC-100 dataset and exports the lines into smaller CSV packages.

    Each line from the input text file is processed, and if it's longer than two characters,
    it's saved into a CSV file along with a source identifier and a unique ID.
    The function creates multiple CSV files, each containing a specified number of lines.

    Args:
        num_lines_in_each_package (int, optional): The maximum number of lines to store in each CSV package.
            Defaults to 10000.
        num_packages (int, optional): The total number of packages (CSV files) to create.
            Defaults to 100.
    """
    cc100_text_file = DATA_SYNTHETIC_ROOT / "ja.txt"

    id_count = 0
    with open(cc100_text_file, 'r', encoding='utf-8') as file:
        for package_count in range(num_packages):
            line_count = 0
            data = []
            for line in tqdm(file, desc=f"creating package {package_count:04} of {num_packages}"):
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
            data.to_csv(ASSETS_PATH/ "lines" / f"{package_count:04}.csv", index=False, escapechar='\\')


if __name__ == "__main__":
    export_lines()
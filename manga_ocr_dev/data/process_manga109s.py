"""Processes the Manga109-s dataset to extract training data and metadata.

This script provides functions to parse the XML annotations of the Manga109-s
dataset. It extracts information about manga pages, comic frames, and text
bounding boxes, and saves this data into structured CSV files. It also generates
cropped images of each text box, which serve as the real-world data for training
the OCR model.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT


def get_books():
    """Retrieves the list of book titles from the Manga109 dataset.

    This function reads the `books.txt` file from the Manga109 dataset's root
    directory to get the list of book titles. It then constructs the full paths
    to the corresponding annotation XML files and image directories for each
    book, organizing them into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - `book`: The title of the manga book.
            - `annotations`: The full path to the XML annotation file.
            - `images`: The full path to the directory containing the book's
              images.
    """
    # Define the root directory of the Manga109-s dataset
    root = Path(MANGA109_ROOT) / "Manga109s_released_2021_02_28"
    # Read the list of book titles from books.txt
    books = (root / "books.txt").read_text().splitlines()
    books = pd.DataFrame(
        {
            "book": books,
            "annotations": [str(root / "annotations" / f"{book}.xml") for book in books],
            "images": [str(root / "images" / book) for book in books],
        }
    )

    return books


def export_frames():
    """Parses XML annotations to extract and save comic frame information.

    This function iterates through each book in the Manga109 dataset, parsing
    its XML annotation file to extract information about the comic frames on
    each page. The extracted data includes page dimensions and the coordinates
    of each frame.

    The collected data is compiled into a single pandas DataFrame and saved as
    `frames.csv` in the `MANGA109_ROOT` directory. This file is used by other
    scripts, such as `generate_backgrounds.py`, to identify non-text areas.
    """
    # Retrieve the list of books in the dataset
    books = get_books()

    data = []
    # Iterate over each book to process its annotations
    for book in tqdm(books.itertuples(), total=len(books)):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        # Find all page elements in the XML
        for page in root.findall("./pages/page"):
            # Find all frame elements within each page
            for frame in page.findall("./frame"):
                # Extract frame attributes and page information
                row = {}
                row["book"] = book.book
                row["page_index"] = int(page.attrib["index"])
                row["page_path"] = str(
                    Path(book.images) / f'{row["page_index"]:03d}.jpg'
                )
                row["page_width"] = int(page.attrib["width"])
                row["page_height"] = int(page.attrib["height"])
                row["id"] = frame.attrib["id"]
                row["xmin"] = int(frame.attrib["xmin"])
                row["ymin"] = int(frame.attrib["ymin"])
                row["xmax"] = int(frame.attrib["xmax"])
                row["ymax"] = int(frame.attrib["ymax"])
                data.append(row)
    # Convert the collected data into a pandas DataFrame
    data = pd.DataFrame(data)

    # Normalize the page paths to be relative to the project structure
    data.page_path = data.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    # Save the DataFrame to a CSV file
    data.to_csv(Path(MANGA109_ROOT) / "frames.csv", index=False)


def export_crops():
    """Extracts text crops and metadata from Manga109 for training.

    This function processes the annotation files for each book in the Manga109
    dataset to extract text-level information, including the bounding box
    coordinates and the transcribed text.

    The extracted metadata is split into training and testing sets (a 90/10
    split) and saved to `data.csv` in the `MANGA109_ROOT` directory.
    Additionally, cropped images of each text box, with a 10-pixel margin,
    are saved as PNG files in the `MANGA109_ROOT/crops` directory. These
    crops serve as the primary source of real-world data for training the
    OCR model.
    """
    # Set up the directory for saving cropped images and define the margin
    crops_root = Path(MANGA109_ROOT) / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    margin = 10

    # Retrieve the list of books
    books = get_books()

    data = []
    # Process each book to extract text annotations
    for book in tqdm(books.itertuples(), total=len(books)):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        # Find all page elements
        for page in root.findall("./pages/page"):
            # Find all text elements within each page
            for text in page.findall("./text"):
                # Extract text attributes and page information
                row = {}
                row["book"] = book.book
                row["page_index"] = int(page.attrib["index"])
                row["page_path"] = str(
                    Path(book.images) / f'{row["page_index"]:03d}.jpg'
                )
                row["page_width"] = int(page.attrib["width"])
                row["page_height"] = int(page.attrib["height"])
                row["id"] = text.attrib["id"]
                row["text"] = text.text
                row["xmin"] = int(text.attrib["xmin"])
                row["ymin"] = int(text.attrib["ymin"])
                row["xmax"] = int(text.attrib["xmax"])
                row["ymax"] = int(text.attrib["ymax"])
                data.append(row)
    # Convert the list of dictionaries to a DataFrame
    data = pd.DataFrame(data)

    # Split the data into training and testing sets (90% train, 10% test)
    n_test = int(0.1 * len(data))
    data["split"] = "train"
    data.loc[data.sample(len(data)).iloc[:n_test].index, "split"] = "test"

    # Define the path for each cropped image
    data["crop_path"] = str(crops_root) + "/" + data.id + ".png"

    # Normalize page and crop paths
    data.page_path = data.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    data.crop_path = data.crop_path.apply(lambda x: "/".join(Path(x).parts[-2:]))
    # Save the consolidated metadata to a CSV file
    data.to_csv(Path(MANGA109_ROOT) / "data.csv", index=False)

    # Group data by page to process images efficiently
    for page_path, boxes in tqdm(
        data.groupby("page_path"), total=data.page_path.nunique()
    ):
        # Load the page image
        img = cv2.imread(str(Path(MANGA109_ROOT) / page_path))

        # Iterate over each text box on the page
        for box in boxes.itertuples():
            # Calculate crop coordinates with a margin, ensuring they are within image bounds
            xmin = max(box.xmin - margin, 0)
            xmax = min(box.xmax + margin, img.shape[1])
            ymin = max(box.ymin - margin, 0)
            ymax = min(box.ymax + margin, img.shape[0])
            # Crop the image
            crop = img[ymin:ymax, xmin:xmax]
            # Define the output path and save the cropped image
            out_path = (crops_root / box.id).with_suffix(".png")
            cv2.imwrite(str(out_path), crop)


if __name__ == "__main__":
    export_frames()
    export_crops()
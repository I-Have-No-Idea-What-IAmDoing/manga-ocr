import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT


def get_books():
    """Retrieves and structures the list of books from the Manga109 dataset.

    This function reads the `books.txt` file from the Manga109 dataset's root
    directory to get the list of book titles. It then constructs the full paths
    to the corresponding annotation XML files and image directories for each
    book, organizing them into a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - `book`: The title of the manga book.
            - `annotations`: The full path to the XML annotation file for the
              book.
            - `images`: The full path to the directory containing the images
              for the book.
    """
    root = MANGA109_ROOT / "Manga109s_released_2021_02_28"
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
    """Parses XML annotations to extract and save frame-level information.

    This function iterates through each book in the Manga109 dataset, parsing
    its XML annotation file to extract information about the comic frames on
    each page. The extracted data includes the book title, page index, page
    dimensions, and the coordinates of each frame.

    The collected data is compiled into a single pandas DataFrame and saved as
    `frames.csv` in the `MANGA109_ROOT` directory, making it available for
    other processing scripts.
    """
    books = get_books()

    data = []
    for book in tqdm(books.itertuples(), total=len(books)):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        for page in root.findall("./pages/page"):
            for frame in page.findall("./frame"):
                row = {}
                row["book"] = book.book
                row["page_index"] = int(page.attrib["index"])
                row["page_path"] = str(Path(book.images) / f'{row["page_index"]:03d}.jpg')
                row["page_width"] = int(page.attrib["width"])
                row["page_height"] = int(page.attrib["height"])
                row["id"] = frame.attrib["id"]
                row["xmin"] = int(frame.attrib["xmin"])
                row["ymin"] = int(frame.attrib["ymin"])
                row["xmax"] = int(frame.attrib["xmax"])
                row["ymax"] = int(frame.attrib["ymax"])
                data.append(row)
    data = pd.DataFrame(data)

    data.page_path = data.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    data.to_csv(MANGA109_ROOT / "frames.csv", index=False)


def export_crops():
    """Extracts text bounding box crops and their metadata from Manga109.

    This function processes the annotation files for each book in the Manga109
    dataset to extract text-level information, including the bounding box
    coordinates and the transcribed text.

    The extracted metadata is split into training and testing sets (a 90/10
    split) and saved to `data.csv` in the `MANGA109_ROOT` directory.
    Additionally, cropped images of each text box, with a 10-pixel margin,
    are saved as PNG files in the `MANGA109_ROOT/crops` directory. These
    crops serve as the real-world data for training the OCR model.
    """
    crops_root = MANGA109_ROOT / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    margin = 10

    books = get_books()

    data = []
    for book in tqdm(books.itertuples(), total=len(books)):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        for page in root.findall("./pages/page"):
            for text in page.findall("./text"):
                row = {}
                row["book"] = book.book
                row["page_index"] = int(page.attrib["index"])
                row["page_path"] = str(Path(book.images) / f'{row["page_index"]:03d}.jpg')
                row["page_width"] = int(page.attrib["width"])
                row["page_height"] = int(page.attrib["height"])
                row["id"] = text.attrib["id"]
                row["text"] = text.text
                row["xmin"] = int(text.attrib["xmin"])
                row["ymin"] = int(text.attrib["ymin"])
                row["xmax"] = int(text.attrib["xmax"])
                row["ymax"] = int(text.attrib["ymax"])
                data.append(row)
    data = pd.DataFrame(data)

    n_test = int(0.1 * len(data))
    data["split"] = "train"
    data.loc[data.sample(len(data)).iloc[:n_test].index, "split"] = "test"

    data["crop_path"] = str(crops_root) + "\\" + data.id + ".png"

    data.page_path = data.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    data.crop_path = data.crop_path.apply(lambda x: "/".join(Path(x).parts[-2:]))
    data.to_csv(MANGA109_ROOT / "data.csv", index=False)

    for page_path, boxes in tqdm(data.groupby("page_path"), total=data.page_path.nunique()):
        img = cv2.imread(str(MANGA109_ROOT / page_path))

        for box in boxes.itertuples():
            xmin = max(box.xmin - margin, 0)
            xmax = min(box.xmax + margin, img.shape[1])
            ymin = max(box.ymin - margin, 0)
            ymax = min(box.ymax + margin, img.shape[0])
            crop = img[ymin:ymax, xmin:xmax]
            out_path = (crops_root / box.id).with_suffix(".png")
            cv2.imwrite(str(out_path), crop)


if __name__ == "__main__":
    export_frames()
    export_crops()
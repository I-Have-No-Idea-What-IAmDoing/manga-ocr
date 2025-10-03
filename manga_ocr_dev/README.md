# Manga OCR Development Environment

This document provides a comprehensive guide for developers working on the Manga OCR project. It covers the project structure, setup instructions, data preprocessing, synthetic data generation, and model training.

## Getting Started

### Prerequisites

-   **Python 3.8+**
-   A virtual environment manager (e.g., `venv`, `conda`) is highly recommended.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kha-white/manga-ocr.git
    cd manga-ocr
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project uses `pip` for dependency management. To install all required packages, including development dependencies, run:
    ```bash
    pip install '.[dev]'
    ```
    *Tip: For faster dependency management, you can use tools like `uv` as a drop-in replacement for `pip`.*

    If you have a GPU, ensure you install the correct version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Documentation

The source code is thoroughly documented using Google Style Python Docstrings. This provides detailed information on the purpose, arguments, and return values of functions, methods, and classes. Developers are encouraged to read the docstrings for a deeper understanding of the codebase.

## Project Structure

The `manga_ocr_dev` directory contains all scripts and resources for developing and training the OCR model.

```
manga_ocr_dev/
├── env.py                     # Defines global paths and constants for the dev environment.
├── data/                      # Scripts for processing raw datasets.
│   ├── generate_backgrounds.py  # Extracts background images from Manga109.
│   ├── process_cc100.py       # Processes and packages text from the CC-100 corpus.
│   └── process_manga109s.py   # Extracts text crops and metadata from Manga109.
├── synthetic_data_generator/  # The pipeline for generating synthetic training data.
│   ├── generator.py           # Orchestrates the generation of synthetic image-text pairs.
│   ├── renderer.py            # Renders text into styled images with backgrounds.
│   ├── run_generate.py        # A runnable script to generate a package of synthetic data.
│   ├── scan_fonts.py          # Scans font files to determine their character support.
│   └── utils.py               # Utility functions for data generation.
├── training/                  # Scripts for training the OCR model.
│   ├── config.yaml            # The main configuration file for training runs.
│   ├── train.py               # The main script to launch a training session.
│   ├── dataset.py             # The PyTorch Dataset class for loading data.
│   ├── get_model.py           # The script for constructing the VisionEncoderDecoderModel.
│   ├── metrics.py             # The class for computing evaluation metrics (CER, accuracy).
│   └── augmentations.py       # Functions for building augmentation pipelines from the config.
└── vendored/                  # Third-party libraries included directly in the project.
    └── html2image/            # A vendored version of the html2image library.
```

## Training Workflow

This section outlines the complete workflow for training a new Manga OCR model.

### 1. Data Preparation

The model is trained on a combination of real data from the Manga109-s dataset and synthetic data.

#### Manga109-s Dataset

1.  **Download the Dataset**: Download the Manga109-s dataset from the [official website](http.www.manga109.org/en/download_s.html).

2.  **Set up Directory**: Unzip the dataset and ensure the directory structure is as follows. The `MANGA109_ROOT` variable in `manga_ocr_dev/env.py` should point to your `assets` directory.
    ```
    assets/
    └── Manga109s_released_2021_02_28/
        ├── annotations/
        ├── images/
        └── books.txt
    ```

3.  **Preprocess the Data**: Run the preprocessing script to extract text crops and metadata. This will create `data.csv`, `frames.csv`, and a `crops` directory inside your `assets` folder.
    ```bash
    python manga_ocr_dev/data/process_manga109s.py
    ```

#### Synthetic Dataset

1.  **Generate Backgrounds**: The synthetic data generator overlays text on background images extracted from manga pages. Run this script to generate a pool of backgrounds from Manga109.
    ```bash
    python manga_ocr_dev/data/generate_backgrounds.py
    ```
    *Note: If you run this without having processed Manga109 first, you can run `create_dummy_background.py` from the root directory to avoid errors.*

2.  **Prepare Fonts**:
    - Place your font files (`.ttf`, `.otf`) in the `fonts/` directory.
    - Scan the fonts to generate `assets/fonts.csv`, which contains metadata about character support.
      ```bash
      python manga_ocr_dev/synthetic_data_generator/scan_fonts.py
      ```
    - (Optional) Manually edit `assets/fonts.csv` to assign labels (`common`, `regular`, `special`) to fonts for weighted sampling.

3.  **Provide a Text Corpus**:
    - The generator uses a text corpus to create realistic text samples. You can use the `process_cc100.py` script or provide your own.
    - Place your text data as CSV files in `assets/lines/`. Each file should have a `line` column. For example, `assets/lines/0000.csv`.

4.  **Run the Generator**: Generate packages of synthetic data. Each package consists of images and a corresponding metadata file.
    ```bash
    python manga_ocr_dev/synthetic_data_generator/run_generate.py --package 0
    ```
    Run this command for each text corpus file you created (e.g., for `0001.csv`, use `--package 1`).

### 2. Configuration

Training behavior is controlled by `manga_ocr_dev/training/config.yaml`. Before training, review and customize this file to set:
-   `run_name`: A unique name for your training run, used for logging.
-   `model`: The encoder/decoder architecture, max length, etc.
-   `dataset`: The data sources to use (synthetic and/or Manga109) and augmentation settings.
-   `training`: Hugging Face `Seq2SeqTrainingArguments`, including learning rate, batch size, and logging steps.

### 3. Launching Training

Once your data is prepared and your configuration is set, start the training process:
```bash
python manga_ocr_dev/training/train.py --config_path manga_ocr_dev/training/config.yaml
```

The script will initialize `wandb` for experiment tracking, create the datasets and model, and start the `Seq2SeqTrainer`. You can monitor the progress and view results in your `wandb` dashboard.
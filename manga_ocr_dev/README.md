# Manga OCR Development Environment

This document provides a comprehensive guide for developers working on the Manga OCR project. It covers the project structure, setup instructions, and the various components of the development environment, including data preprocessing, synthetic data generation, and model training.

## Project Overview

The `manga_ocr_dev` directory contains all the necessary tools and scripts for the development and training of the Manga OCR model. This includes:

- **Data Preprocessing**: Scripts for processing datasets like Manga109-s and preparing them for training.
- **Synthetic Data Generation**: A powerful pipeline for creating synthetic training data, which is crucial for improving the model's robustness.
- **Model Training**: The main training script and related utilities for training the OCR model.

## Getting Started

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) for dependency management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/manga-ocr.git
   cd manga-ocr
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

### Configuration

Before running any of the development scripts, you need to configure the necessary paths in `manga_ocr_dev/env.py`. This file defines the locations for datasets, assets, and training outputs.

## Project Structure

```
assets/                       # Assets (see description below)
manga_ocr/                    # Release code (inference only)
manga_ocr_dev/                # Development code
   env.py                     # Global constants and paths
   data/                      # Data preprocessing scripts
   synthetic_data_generator/  # Synthetic data generation pipeline
   training/                  # Model training scripts
   vendored/                  # Third-party libraries included directly
      html2image/             # Vendored version of the html2image library
```

## Assets

The `assets` directory contains various files required for data generation and training.

### `fonts.csv`
A CSV file with metadata about the fonts used by the synthetic data generator.

- **`font_path`**: Path to the font file, relative to `FONTS_ROOT`.
- **`supported_chars`**: A string of characters supported by the font.
- **`num_chars`**: The number of supported characters.
- **`label`**: `common`, `regular`, or `special` (used for weighted sampling).

You can generate this file for your own fonts using the `manga_ocr_dev/synthetic_data_generator/scan_fonts.py` script.

### `lines_example.csv`
An example of a CSV file used for synthetic data generation.

- **`source`**: The source of the text (e.g., a corpus name).
- **`id`**: A unique ID for the line.
- **`line`**: A line of text from the corpus.

### `len_to_p.csv`
A CSV file mapping text length to its probability of occurrence in manga, used to generate realistically sized text samples.

### `vocab.csv`
A list of all characters supported by the tokenizer.

## Training the OCR Model

1. **Download the Manga109-s Dataset**:
   You can download the dataset from the [official website](http://www.manga109.org/en/download_s.html).

2. **Set up the dataset directory**:
   Ensure your directory structure looks like this, and update the `MANGA109_ROOT` path in `env.py`:
    ```
    <MANGA109_ROOT>/
        Manga109s_released_2021_02_28/
            annotations/
            images/
            books.txt
            readme.txt
    ```

3. **Preprocess the Manga109-s dataset**:
   Run the following script to process the dataset:
   ```bash
   python manga_ocr_dev/data/process_manga109s.py
   ```

4. **(Optional) Generate Synthetic Data**:
   Follow the steps in the "Synthetic Data Generation" section to create additional training data.

5. **Start Training**:
   Run the main training script:
   ```bash
   python manga_ocr_dev/training/train.py --run_name "my_training_run"
   ```

## Synthetic Data Generation

The synthetic data generation pipeline creates image-text pairs that mimic the appearance of text in manga.

### Directory Structure

The pipeline uses the following directory structure within `<DATA_SYNTHETIC_ROOT>`:

```
<DATA_SYNTHETIC_ROOT>/
   img/           # Generated images
      0000/
      0001/
      ...
   lines/         # Lines from a text corpus
      0000.csv
      0001.csv
      ...
   meta/          # Metadata for the generated images
      0000.csv
      0001.csv
      ...
```

### Generation Steps

1. **Generate Backgrounds**:
   ```bash
   python manga_ocr_dev/data/generate_backgrounds.py
   ```

2. **Prepare Fonts**:
   - Place your font files in the `<FONTS_ROOT>` directory.
   - Generate the font metadata by running:
     ```bash
     python manga_ocr_dev/synthetic_data_generator/scan_fonts.py
     ```
   - (Optional) Manually label special fonts in `assets/fonts.csv`.

3. **Provide Text Corpus**:
   Create CSV files with lines of text and place them in `<DATA_SYNTHETIC_ROOT>/lines/`.

4. **Run the Generator**:
   Run the `run_generate.py` script for each data package you want to create:
   ```bash
   python manga_ocr_dev/synthetic_data_generator/run_generate.py --package 0
   python manga_ocr_dev/synthetic_data_generator/run_generate.py --package 1
   ...
   ```
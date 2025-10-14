# Manga OCR

Optical character recognition for Japanese text, with a primary focus on manga. This project uses a custom end-to-end model built with the [Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) framework from Hugging Face Transformers.

Manga OCR is designed to be a high-quality, general-purpose tool for printed Japanese text, robust against scenarios common in manga:
- Both vertical and horizontal text layouts
- Text with furigana (ruby characters)
- Text overlaid on complex images
- A wide variety of fonts and artistic styles
- Low-quality or scanned images

Unlike many OCR models, Manga OCR can recognize multi-line text in a single forward pass, allowing it to process entire text bubbles at once without needing to split them into individual lines.

## See Also

- **[Poricom](https://github.com/bluaxees/Poricom)**: A GUI reader that uses `manga-ocr`.
- **[mokuro](https://github.com/kha-white/mokuro)**: A tool that uses `manga-ocr` to generate an HTML overlay for manga.
- **[Xelieu's Guide](https://rentry.co/lazyXel)**: A comprehensive guide on setting up a reading and mining workflow with `manga-ocr` and `mokuro`.
- **[Development and Training Code](manga_ocr_dev)**: Scripts and resources for training the model and generating synthetic data.
- **[Synthetic Data Generation](manga_ocr_dev/synthetic_data_generator)**: A detailed description of the synthetic data pipeline with examples.

# Installation

You will need **Python 3.8 or newer**. Please note that the latest Python releases may not always be supported immediately due to dependencies like PyTorch, which require time to release compatible versions. Refer to the [PyTorch website](https://pytorch.org/get-started/locally/) for current supported Python versions.

Some users have reported issues with Python installations from the Microsoft Store. If you encounter an `ImportError: DLL load failed while importing fugashi`, we recommend installing Python from the [official website](https://www.python.org/downloads).

If you have a GPU, install PyTorch with CUDA support by following the instructions [here](https://pytorch.org/get-started/locally/#start-locally). Otherwise, this step can be skipped, and the model will run on the CPU.

To install Manga OCR, run:
```bash
pip install manga-ocr
```

## Troubleshooting

- **`ImportError: DLL load failed while importing fugashi`**: This may be caused by a Microsoft Store installation of Python. Try installing from the [official Python website](https.www.python.org/downloads) instead.
- **`mecab-python3` installation issues on ARM**: If you encounter problems on an ARM-based system, try [this workaround](https://github.com/kha-white/manga-ocr/issues/16).

# Usage

## Python API

You can easily integrate Manga OCR into your Python scripts.

```python
from manga_ocr import MangaOcr

# Initialize the model
mocr = MangaOcr()

# Process an image from a file path
text_from_path = mocr('/path/to/your/image.jpg')
print(text_from_path)
```

You can also process a `PIL.Image` object directly:

```python
import PIL.Image
from manga_ocr import MangaOcr

mocr = MangaOcr()
img = PIL.Image.open('/path/to/your/image.jpg')
text_from_image = mocr(img)
print(text_from_image)
```

## Command-Line Interface (CLI)

Manga OCR can run as a background process to monitor for new images and recognize text from them. This is useful for setting up a seamless reading workflow with screen capture tools.

**Workflow Example:**
Capture a screen region with a tool like [ShareX](https://getsharex.com/) or [Flameshot](https://flameshot.org/) -> Save the image to the clipboard -> Manga OCR reads from the clipboard -> Recognized text is written back to the clipboard -> A dictionary tool like [Yomitan](https://github.com/yomidevs/yomitan) reads from the clipboard.

<img src="https://user-images.githubusercontent.com/22717958/150238361-052b95d1-0152-485f-a441-48a957536239.mp4" width="600"/>

- **Read from clipboard, write to clipboard:**
  ```bash
  manga_ocr
  ```

- **Read from a specific folder (e.g., a ShareX screenshot folder):**
  ```bash
  manga_ocr "/path/to/your/screenshot/folder"
  ```
  *Note: When using folder monitoring, any new image saved to that folder will be processed. For a smoother workflow, you can configure your screenshot tool to save images to a dedicated OCR folder without copying them to the clipboard.*

- **For a full list of options:**
  ```bash
  manga_ocr --help
  ```

The first time you run the tool, it will download the model (~400 MB), which may take a few minutes. The OCR is ready to use once the `OCR ready` message appears in the logs.

## Usage Tips

- The model supports multi-line text, but recognition accuracy may decrease with very long text blocks. If a section of a long text fails, try running the OCR on a smaller crop.
- While trained primarily on manga, the model performs well on other printed materials like novels and video games. It is not designed for handwritten text.
- The model will always attempt to find text in an image. Since it has some understanding of Japanese, it may occasionally "hallucinate" realistic-looking text on images with no actual characters.

# Examples

Here are a few examples demonstrating the model's capabilities.

| Image                | Manga OCR Result |
|----------------------|------------------|
| ![](assets/examples/00.jpg) | 素直にあやまるしか |
| ![](assets/examples/01.jpg) | 立川で見た〝穴〟の下の巨大な眼は： |
| ![](assets/examples/02.jpg) | 実戦剣術も一流です |
| ![](assets/examples/03.jpg) | 第３０話重苦しい闇の奥で静かに呼吸づきながら |
| ![](assets/examples/04.jpg) | よかったじゃないわよ！何逃げてるのよ！！早くあいつを退治してよ！ |
| ![](assets/examples/05.jpg) | ぎゃっ |
| ![](assets/examples/06.jpg) | ピンポーーン |
| ![](assets/examples/07.jpg) | ＬＩＮＫ！私達７人の力でガノンの塔の結界をやぶります |
| ![](assets/examples/08.jpg) | ファイアパンチ |
| ![](assets/examples/09.jpg) | 少し黙っている |
| ![](assets/examples/10.jpg) | わかるかな〜？ |
| ![](assets/examples/11.jpg) | 警察にも先生にも町中の人達に！！ |

# Development

This section provides a guide for developers who want to contribute to Manga OCR. It covers setting up the environment, running tests, and understanding the project structure.

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
    The project uses `pip` for dependency management. To install all packages, including development dependencies, run:
    ```bash
    pip install '.[dev]'
    ```
    *Tip: For faster dependency management, you can use tools like `uv` as a drop-in replacement for `pip`.*

    If you have a GPU, ensure you install the correct version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Documentation

The source code is documented using [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). This provides detailed information on the purpose, arguments, and return values of functions, methods, and classes.

We encourage developers to read the docstrings for a deeper understanding of the codebase. When contributing, please ensure that any new code is accompanied by corresponding documentation.

## Running Tests

The project uses `pytest` for testing. To run the full test suite, execute:
```bash
pytest
```

To check test coverage, you can use the `pytest-cov` plugin:
```bash
pytest --cov=manga_ocr
```

## Project Structure

The repository is organized into several key directories:

-   `manga_ocr/`: The main source code for the Manga OCR library.
    -   `ocr.py`: Contains the core `MangaOcr` class that handles model loading and inference.
    -   `run.py`: Implements the command-line interface and background monitoring logic.
-   `manga_ocr_dev/`: Contains all scripts and resources for developing and training the OCR model.
    -   `data/`: Scripts for processing raw datasets like Manga109-s and CC-100 into a usable format.
    -   `synthetic_data_generator/`: The pipeline for generating synthetic training data, including text rendering and image manipulation.
    -   `training/`: Scripts for training the OCR model, including the main training loop, dataset class, and metric computation.
-   `tests/`: Unit tests for the project.
-   `assets/`: Example images, font metadata, and other data required for generation and testing.
-   `fonts/`: A collection of fonts used by the synthetic data generator.

For a more detailed overview of the development environment, please refer to the `manga_ocr_dev/README.md` file.

## Contributing

Contributions are welcome! We appreciate your help in making Manga OCR better. To ensure a smooth process, please review this guide.

### Reporting Bugs

If you encounter a bug, please [open an issue](https://github.com/kha-white/manga-ocr/issues) and provide the following information:
-   Your operating system and Python version.
-   The version of `manga-ocr` you are using (`manga-ocr --version`).
-   A clear and concise description of the bug.
-   Steps to reproduce the issue.
-   The expected behavior and the actual behavior.
-   Any relevant error messages or screenshots.

### Suggesting Enhancements

We are open to suggestions for new features or improvements. Please [open an issue](https://github.com/kha-white/manga-ocr/issues) to discuss your ideas. Provide a clear description of the proposed enhancement and its potential benefits.

### Pull Request Process

1.  **Fork the repository** and clone it to your local machine.
2.  **Create a new branch** for your changes: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bug-fix`.
3.  **Make your changes** to the codebase.
4.  **Follow the coding style**: This project uses [Google Style Python Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for documentation. Please ensure your code is well-commented and includes docstrings for new functions, classes, or methods.
5.  **Run tests** to ensure your changes do not break existing functionality: `pytest`.
6.  **Update documentation** (`README.md` or docstrings) if your changes affect usage or add new features.
7.  **Commit your changes** with a clear and descriptive commit message.
8.  **Push your branch** to your fork: `git push origin feature/your-feature-name`.
9.  **Open a pull request** to the `main` branch of the original repository. Provide a detailed description of your changes and why they are needed.

# Acknowledgments

This project was developed with the use of:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [CC-100](https://data.statmt.org/cc-100/) dataset

# Contact

For any inquiries, please feel free to contact me at kha-white@mail.com.
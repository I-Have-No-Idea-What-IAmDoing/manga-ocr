# Manga OCR

Optical character recognition for Japanese text, with the main focus being Japanese manga.
It uses a custom end-to-end model built with Transformers' [Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) framework. 

Manga OCR can be used as a general purpose printed Japanese OCR, but its main goal was to provide a high quality
text recognition, robust against various scenarios specific to manga:
- both vertical and horizontal text
- text with furigana
- text overlaid on images
- wide variety of fonts and font styles
- low quality images

Unlike many OCR models, Manga OCR supports recognizing multi-line text in a single forward pass,
so that text bubbles found in manga can be processed at once, without splitting them into lines.

See also:
- [Poricom](https://github.com/bluaxees/Poricom), a GUI reader, which uses manga-ocr
- [mokuro](https://github.com/kha-white/mokuro), a tool, which uses manga-ocr to generate an HTML overlay for manga
- [Xelieu's guide](https://rentry.co/lazyXel), a comprehensive guide on setting up a reading and mining workflow with manga-ocr/mokuro (and many other useful tips)
- Development code, including code for training and synthetic data generation: [link](manga_ocr_dev)
- Description of synthetic data generation pipeline + examples of generated images: [link](manga_ocr_dev/synthetic_data_generator)

# Installation

You need Python 3.6 or newer. Please note, that the newest Python release might not be supported due to a PyTorch dependency, which often breaks with new Python releases and needs some time to catch up.
Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for a list of supported Python versions.

Some users have reported problems with Python installed from Microsoft Store. If you see an error:
`ImportError: DLL load failed while importing fugashi: The specified module could not be found.`,
try installing Python from the [official site](https://www.python.org/downloads).

If you want to run with GPU, install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally),
otherwise this step can be skipped.

## Troubleshooting

- `ImportError: DLL load failed while importing fugashi: The specified module could not be found.` - might be because of Python installed from Microsoft Store, try installing Python from the [official site](https://www.python.org/downloads)
- problem with installing `mecab-python3` on ARM architecture - try [this workaround](https://github.com/kha-white/manga-ocr/issues/16)

# Usage

## Python API

```python
from manga_ocr import MangaOcr

mocr = MangaOcr()
text = mocr('/path/to/img')
```

or

```python
import PIL.Image

from manga_ocr import MangaOcr

mocr = MangaOcr()
img = PIL.Image.open('/path/to/img')
text = mocr(img)
```

## Running in the background

Manga OCR can run in the background and process new images as they appear.

You might use a tool like [ShareX](https://getsharex.com/) or [Flameshot](https://flameshot.org/) to manually capture a region of the screen and let the
OCR read it either from the system clipboard, or a specified directory. By default, Manga OCR will write recognized text to clipboard,
from which it can be read by a dictionary like [Yomitan](https://github.com/yomidevs/yomitan).

Clipboard mode on Linux requires `wl-copy` for Wayland sessions or `xclip` for X11 sessions. You can find out which one your system needs by running `echo $XDG_SESSION_TYPE` in the terminal.

Your full setup for reading manga in Japanese with a dictionary might look like this:

capture region with ShareX -> write image to clipboard -> Manga OCR -> write text to clipboard -> Yomitan

https://user-images.githubusercontent.com/22717958/150238361-052b95d1-0152-485f-a441-48a957536239.mp4

- To read images from clipboard and write recognized texts to clipboard, run in command line:
    ```commandline
    manga_ocr
    ```
- To read images from ShareX's screenshot folder, run in command line:
    ```commandline
    manga_ocr "/path/to/sharex/screenshot/folder"
    ```
Note that when running in the clipboard scanning mode, any image that you copy to clipboard will be processed by OCR and replaced
by recognized text. If you want to be able to copy and paste images as usual, you should use the folder scanning mode instead
and define a separate task in ShareX just for OCR, which saves screenshots to some folder without copying them to clipboard.

When running for the first time, downloading the model (~400 MB) might take a few minutes.
The OCR is ready to use after `OCR ready` message appears in the logs.

- To see other options, run in command line:
    ```commandline
    manga_ocr --help
    ```

If `manga_ocr` doesn't work, you might also try replacing it with `python -m manga_ocr`.

## Usage tips

- OCR supports multi-line text, but the longer the text, the more likely some errors are to occur.
  If the recognition failed for some part of a longer text, you might try to run it on a smaller portion of the image.
- The model was trained specifically to handle manga well, but should do a decent job on other types of printed text,
  such as novels or video games. It probably won't be able to handle handwritten text though. 
- The model always attempts to recognize some text on the image, even if there is none.
  Because it uses a transformer decoder (and therefore has some understanding of the Japanese language),
  it might even "dream up" some realistically looking sentences! This shouldn't be a problem for most use cases,
  but it might get improved in the next version.

# Examples

Here are some cherry-picked examples showing the capability of the model. 

| image                | Manga OCR result |
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

# Contact
For any inquiries, please feel free to contact me at kha-white@mail.com

# Development

This section provides a guide for developers who want to contribute to Manga OCR. It covers setting up the environment, running tests, and understanding the project structure.

## Getting Started

### Prerequisites

-   Python 3.8+
-   A virtual environment manager (e.g., `venv`, `conda`) is recommended.

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
    If you have a GPU, ensure you install the correct version of PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

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

-   `manga_ocr/`: The main source code for the Manga OCR library, containing the core OCR functionality and the command-line interface.
-   `manga_ocr_dev/`: Contains all scripts and resources for the development and training of the OCR model.
    -   `data/`: Scripts for processing datasets like Manga109-s and CC-100.
    -   `synthetic_data_generator/`: The pipeline for generating synthetic training data.
    -   `training/`: Scripts for training the OCR model.
-   `tests/`: Unit tests for the project.
-   `assets/`: Example images, fonts metadata, and other assets required for data generation and testing.
-   `fonts/`: A collection of fonts used by the synthetic data generator.

For a more detailed overview of the development environment, please refer to the `manga_ocr_dev/README.md` file.

## Contributing

Contributions are welcome! If you have a suggestion or a bug fix, please open an issue or a pull request.

# Acknowledgments

This project was done with the usage of:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [CC-100](https://data.statmt.org/cc-100/) dataset

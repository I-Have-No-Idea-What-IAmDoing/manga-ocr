import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    """Custom VisionEncoderDecoderModel for Manga OCR.

    This class inherits from both `VisionEncoderDecoderModel` and `GenerationMixin`
    to provide generation capabilities for the OCR model.
    """
    pass


class MangaOcr:
    """A class for performing OCR on manga images.

    This class encapsulates the entire OCR pipeline, including model loading,
    image preprocessing, text generation, and post-processing.

    Attributes:
        processor: The image processor for preparing images for the model.
        tokenizer: The tokenizer for converting token IDs to text.
        model: The underlying OCR model.
    """
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False):
        """Initializes the MangaOcr model.

        This involves loading the pre-trained model, processor, and tokenizer from
        the specified path. It also performs a warm-up run to ensure the model
        is ready for inference.

        Args:
            pretrained_model_name_or_path (str, optional): The name or path of the
                pretrained model to use. Defaults to "kha-white/manga-ocr-base".
            force_cpu (bool, optional): If True, forces the model to run on the CPU,
                even if a GPU is available. Defaults to False.

        Raises:
            FileNotFoundError: If the example image for the warm-up run is not found.
        """
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = MangaOcrModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        # warm up
        logger.info("Warming up MangaOcr model...")
        self(Image.new("RGB", (100, 100), "white"))
        logger.info("MangaOcr model warmed up")

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        """Performs OCR on a given image.

        The image can be provided as a file path or a PIL Image object.

        Args:
            img_or_path (str | Path | Image.Image): The path to the image file or
                a PIL Image object.

        Returns:
            str: The recognized text from the image.

        Raises:
            ValueError: If `img_or_path` is not a valid path or PIL Image.
        """
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {img_or_path}")

        img = img.convert("L").convert("RGB")

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        """Preprocesses an image before feeding it to the model.

        Args:
            img (Image.Image): The PIL Image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image as a PyTorch tensor.
        """
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    """Post-processes the raw text output from the OCR model.

    This function performs several cleaning and normalization steps:
    - Removes all whitespace.
    - Normalizes sequences of dots.
    - Replaces the full-width ellipsis with three dots.
    - Converts half-width characters to full-width.

    Args:
        text (str): The raw text to post-process.

    Returns:
        str: The cleaned and normalized text.
    """
    text = "".join(text.split())
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = text.replace("…", "...")
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
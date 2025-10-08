import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    """A custom VisionEncoderDecoderModel for Manga OCR.

    This class extends the `VisionEncoderDecoderModel` from the `transformers`
    library, adding the `GenerationMixin` to enable text generation. This is
    necessary for the OCR model to be able to decode the visual features into
    a sequence of text.
    """
    pass


class MangaOcr:
    """A class for performing Optical Character Recognition (OCR) on manga images.

    This class encapsulates the entire OCR pipeline, including model loading,
    image preprocessing, text generation, and post-processing. It is designed
    to be easy to use, with a simple interface that takes an image and returns
    the recognized text.

    Attributes:
        processor (ViTImageProcessor): The image processor for preparing images
            for the model. This includes resizing, normalizing, and converting
            the image to a tensor.
        tokenizer (AutoTokenizer): The tokenizer for converting token IDs to
            text. This is used to decode the model's output into a readable
            string.
        model (MangaOcrModel): The underlying OCR model, which is a custom
            `VisionEncoderDecoderModel` that performs the actual OCR task.
    """
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False):
        """Initializes the MangaOcr model.

        This involves loading the pre-trained model, processor, and tokenizer
        from the specified path. It also automatically detects and uses a GPU
        if available, unless `force_cpu` is set to True. A warm-up run is
        performed to ensure the model is ready for inference and to avoid
        delays on the first call.

        Args:
            pretrained_model_name_or_path (str, optional): The name or path of
                the pretrained model to use. This can be a model from the
                Hugging Face Hub or a local path. Defaults to
                "kha-white/manga-ocr-base".
            force_cpu (bool, optional): If True, forces the model to run on the
                CPU, even if a GPU is available. This is useful for debugging or
                for systems without a compatible GPU. Defaults to False.

        """
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = MangaOcrModel.from_pretrained(pretrained_model_name_or_path)

        # Set device to CUDA, MPS, or CPU
        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        # Perform a warm-up run to initialize the model and avoid delays on the first call
        logger.info("Warming up MangaOcr model...")
        self(Image.new("RGB", (100, 100), "white"))
        logger.info("MangaOcr model warmed up")

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        """Performs OCR on a given image.

        This method is the main entry point for performing OCR. It takes an
        image, which can be a file path or a PIL Image object, and returns the
        recognized text. The image is first preprocessed, then fed to the model
        for text generation, and finally the output is post-processed to clean
        it up.

        Args:
            img_or_path (str | Path | Image.Image): The path to the image file
                or a PIL Image object. The image should be in a format that
                can be opened by PIL.

        Returns:
            str: The recognized text from the image, after post-processing.

        Raises:
            ValueError: If `img_or_path` is not a valid file path or a PIL
                Image object.
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

        This method takes a PIL Image, uses the feature extractor to convert
        it into a PyTorch tensor with the correct dimensions and normalization.

        Args:
            img (Image.Image): The PIL Image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image as a PyTorch tensor, ready to
            be fed into the model.
        """
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    """Post-processes the raw text output from the OCR model.

    This function performs several cleaning and normalization steps to improve
    the quality and readability of the recognized text. The steps include:
    - Removing all whitespace characters.
    - Normalizing sequences of dots (e.g., ".." to "...")
    - Replacing the full-width ellipsis character with three dots.
    - Converting half-width (hankaku) characters to full-width (zenkaku).

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
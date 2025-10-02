import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin


class MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    """
    A custom VisionEncoderDecoderModel that also inherits from GenerationMixin.
    """
    pass

class MangaOcr:
    """
    A class for performing OCR on manga images.
    """
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False):
        """
        Initializes the MangaOcr model.

        Args:
            pretrained_model_name_or_path (str, optional): The path to the pretrained model.
                Defaults to "kha-white/manga-ocr-base".
            force_cpu (bool, optional): Whether to force the use of CPU. Defaults to False.

        Raises:
            FileNotFoundError: If the example image is not found.
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

        example_path = Path(__file__).parent.parent / "assets/example.jpg"
        if not example_path.is_file():
            raise FileNotFoundError(f"Missing example image {example_path}")
        self(example_path)

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        """
        Performs OCR on the given image.

        Args:
            img_or_path (str or Path or Image.Image): The path to the image or the image itself.

        Returns:
            str: The recognized text.

        Raises:
            ValueError: If img_or_path is not a path or PIL.Image.
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
        """
        Preprocesses the image before feeding it to the model.

        Args:
            img (Image.Image): The image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image.
        """
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    """
    Post-processes the recognized text.

    Args:
        text (str): The text to post-process.

    Returns:
        str: The post-processed text.
    """
    text = "".join(text.split())
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = text.replace("…", "...")
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
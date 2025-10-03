import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from manga_ocr_dev.env import MANGA109_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.training.config.schemas import DatasetConfig
from manga_ocr_dev.training.augmentations import build_augmentations


class MangaDataset(Dataset):
    """A PyTorch Dataset for loading manga images and text for OCR training.

    This dataset combines synthetically generated text images with real manga
    crops from the Manga109 dataset. It handles data loading, augmentation,
    and preprocessing, making it suitable for training a `VisionEncoderDecoderModel`.

    Attributes:
        processor: A processor that combines a feature extractor and a tokenizer.
        max_target_length (int): The maximum length for tokenized text sequences.
        data (pd.DataFrame): A DataFrame holding the file paths and text for
            each sample in the dataset.
        augment (bool): A flag indicating whether to apply data augmentation.
        transform_medium (A.Compose): The medium-level augmentation pipeline.
        transform_heavy (A.Compose): The heavy-level augmentation pipeline.
    """
    def __init__(
        self,
        processor,
        dataset_config: DatasetConfig,
        max_target_length: int,
        limit_size: Optional[int] = None,
    ):
        """Initializes the MangaDataset.

        Args:
            processor: The processor for feature extraction and tokenization.
            dataset_config (DatasetConfig): The dataset configuration object.
            max_target_length (int): The maximum length for tokenized text sequences.
            limit_size (int | None, optional): If specified, limits the dataset
                to this number of samples. Defaults to None.
        """
        self.processor = processor
        self.max_target_length = max_target_length
        self.config = dataset_config

        data = []
        for source in self.config.train.sources:
            if source.type == 'synthetic':
                data.append(self.load_synthetic_data(**source.params))
            elif source.type == 'manga109':
                data.append(self.load_manga109_data(**source.params))

        self.data = pd.concat(data, ignore_index=True)

        if limit_size:
            self.data = self.data.iloc[:limit_size]

        print(f"Initialized dataset with {len(self.data)} samples.")

        self.transform_medium = build_augmentations(self.config.augmentations.medium)
        self.transform_heavy = build_augmentations(self.config.augmentations.heavy)
        self.aug_probs = self.config.augmentations.probabilities

    def load_synthetic_data(self, packages=None, skip_packages=None):
        """Loads synthetic data from specified packages."""
        
        data = []
        if packages is not None:
            package_ids = {f"{x:04d}" for x in packages}
            glob_pattern = [DATA_SYNTHETIC_ROOT / "meta" / f"{pid}.csv" for pid in package_ids]
        else:
            glob_pattern = sorted((DATA_SYNTHETIC_ROOT / "meta").glob("*.csv"))
            if skip_packages is not None:
                skip_package_ids = {f"{x:04d}" for x in skip_packages}
                glob_pattern = [p for p in glob_pattern if p.stem not in skip_package_ids]

        for path in glob_pattern:
            if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                print(f"Missing image data for package {path}, skipping")
                continue
            df = pd.read_csv(path)
            df = df.dropna()
            df["path"] = df.id.apply(lambda x: str(DATA_SYNTHETIC_ROOT / "img" / path.stem / f"{x}.jpg"))
            df = df[["path", "text"]]
            df["synthetic"] = True
            data.append(df)

        if not data:
            return pd.DataFrame(columns=["path", "text", "synthetic"])

        return pd.concat(data, ignore_index=True)

    def load_manga109_data(self, split):
        """Loads Manga109 data from a specific split."""
        
        df = pd.read_csv(MANGA109_ROOT / "data.csv")
        df = df[df.split == split].reset_index(drop=True)
        df["path"] = df.crop_path.apply(lambda x: str(MANGA109_ROOT / x))
        df = df[["path", "text"]]
        df["synthetic"] = False
        return df

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the given index.

        This method fetches an image and its corresponding text, applies
        augmentations if enabled, and then preprocesses the image and tokenizes
        the text to prepare it for the model.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed 'pixel_values' of
            the image and the 'labels' (tokenized text).
        """
        sample = self.data.loc[idx]
        text = sample.text

        transform = None
        if self.config.augment:
            p_medium = self.aug_probs.medium
            p_heavy = self.aug_probs.heavy
            transform_variant = np.random.choice(
                ["none", "medium", "heavy"],
                p=[1 - p_medium - p_heavy, p_medium, p_heavy],
            )
            if transform_variant == "medium":
                transform = self.transform_medium
            elif transform_variant == "heavy":
                transform = self.transform_heavy

        pixel_values = self.read_image(self.processor, sample.path, transform)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = np.array(labels)
        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }
        return encoding

    @staticmethod
    def read_image(processor, path, transform=None):
        """Reads an image, applies transforms, and extracts pixel values.

        Args:
            processor: The processor for feature extraction.
            path (str or Path): The path to the image file.
            transform (A.Compose, optional): An Albumentations transform
                pipeline to apply. If None, the image is only converted to
                grayscale. Defaults to None.

        Returns:
            torch.Tensor: The preprocessed pixel values of the image.
        """
        img = cv2.imread(str(path))

        if transform is None:
            transform = A.Compose([A.ToGray(always_apply=True)])

        img = transform(image=img)["image"]

        pixel_values = processor.feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


if __name__ == "__main__":
    from manga_ocr_dev.training.get_model import get_model
    from manga_ocr_dev.training.utils import tensor_to_image
    from manga_ocr_dev.training.config import config as app_config

    model, processor = get_model(app_config.model)

    ds = MangaDataset(processor, app_config.dataset, app_config.model.max_len, limit_size=100)

    for i in range(20):
        sample = ds[0]
        img = tensor_to_image(sample["pixel_values"])
        tokens = sample["labels"]
        tokens[tokens == -100] = processor.tokenizer.pad_token_id
        text = "".join(processor.tokenizer.decode(tokens, skip_special_tokens=True).split())

        print(f"{i}:\n{text}\n")
        plt.imshow(img)
        plt.show()
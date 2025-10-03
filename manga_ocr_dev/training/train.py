"""Main training script for the Manga OCR model.

This script orchestrates the entire training process for the vision-encoder-decoder
model. It handles loading the configuration, initializing the model and datasets,
setting up the Hugging Face `Seq2SeqTrainer`, and launching the training and
evaluation loop. The script is designed to be run from the command line, with
the path to the configuration file as an argument.
"""

import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.config import load_config
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics


def main(config_path: str = "manga_ocr_dev/training/config.yaml"):
    """Runs the full training and evaluation pipeline for the Manga OCR model.

    This function performs the following steps:
    1.  Loads the training configuration from the specified YAML file.
    2.  Initializes a `wandb` run for experiment tracking and visualization.
    3.  Constructs the `VisionEncoderDecoderModel` and its associated processor.
    4.  Creates the training and evaluation datasets, disabling augmentations
        for the evaluation set to ensure consistent metrics.
    5.  Sets up the `Seq2SeqTrainingArguments` and the `Seq2SeqTrainer`.
    6.  Starts the training process, which will periodically run evaluation
        and save checkpoints based on the provided configuration.

    Args:
        config_path (str, optional): The path to the training configuration
            file. Defaults to 'manga_ocr_dev/training/config.yaml'.
    """
    config = load_config(config_path)

    wandb.init(project="manga-ocr", name=config.run_name, config=config.dict())

    model, processor = get_model(config.model)

    train_dataset = MangaDataset(
        processor,
        dataset_config=config.dataset,
        max_target_length=config.model.max_len,
    )

    eval_dataset = MangaDataset(
        processor,
        dataset_config=config.dataset,
        max_target_length=config.model.max_len,
    )
    eval_dataset.config.augment = False  # No augmentations for eval

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(TRAIN_ROOT),
        run_name=config.run_name,
        **config.training.dict(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
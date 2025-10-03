import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics
from manga_ocr_dev.training.config import load_config, AppConfig


def main(config_path: str = 'manga_ocr_dev/training/config.yaml'):
    """Main training script for the Manga OCR model.

    This script sets up the model, datasets, and training arguments based
    on the loaded configuration, then initiates the training process using
    the `Seq2SeqTrainer` from the Hugging Face Transformers library.

    Args:
        config_path (str, optional): Path to the training configuration file.
            Defaults to 'manga_ocr_dev/training/config.yaml'.
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
    eval_dataset.config.augment = False # No augmentations for eval

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(TRAIN_ROOT),
        run_name=config.run_name,
        **config.training.dict()
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
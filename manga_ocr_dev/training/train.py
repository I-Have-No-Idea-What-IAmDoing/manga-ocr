import yaml
from pathlib import Path
import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics


class TrainingPipeline:
    """Orchestrates the training process for the Manga OCR model."""
    def __init__(self, config_path: str):
        """Initializes the training pipeline with a configuration file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['model']
        self.dataset_config = self.config['dataset']
        self.training_config = self.config['training']

    def run(self):
        """Main training script for the Manga OCR model.

        This script sets up the model, datasets, and training arguments based
        on the loaded configuration, then initiates the training process using
        the `Seq2SeqTrainer` from the Hugging Face Transformers library.
        """
        wandb.init(project="manga-ocr", name=self.config['run_name'], config=self.config)

        model, processor = get_model(
            encoder_name=self.model_config['encoder_name'],
            decoder_name=self.model_config['decoder_name'],
            max_len=self.model_config['max_len'],
            num_decoder_layers=self.model_config['num_decoder_layers']
        )

        train_dataset = MangaDataset(
            processor,
            sources=self.dataset_config['train']['sources'],
            max_target_length=self.model_config['max_len'],
            augment=self.dataset_config['augment'],
        )

        eval_dataset = MangaDataset(
            processor,
            sources=self.dataset_config['eval']['sources'],
            max_target_length=self.model_config['max_len'],
            augment=False,
        )

        metrics = Metrics(processor)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(TRAIN_ROOT),
            run_name=self.config['run_name'],
            **self.training_config
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


def main(config_path: str = 'manga_ocr_dev/training/config.yaml'):
    """Entry point for the training pipeline.

    Args:
        config_path (str, optional): Path to the training configuration file.
            Defaults to 'manga_ocr_dev/training/config.yaml'.
    """
    pipeline = TrainingPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    fire.Fire(main)
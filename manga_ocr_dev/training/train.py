import fire
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics


def run(
    run_name="debug",
    encoder_name="apple/MobileCLIP2-S2",
    decoder_name="jhu-clsp/mmBERT-base",
    max_len=300,
    num_decoder_layers=3,
    batch_size=64,
    num_epochs=8,
    fp16=True,
):
    """
    Initializes and runs the training process for the Manga OCR model.

    This function sets up the model, datasets, and training arguments, then
    initiates the training using the Seq2SeqTrainer from the Hugging Face
    Transformers library.

    Args:
        run_name (str, optional): A name for the training run, used for logging
            and output directories. Defaults to "debug".
        encoder_name (str, optional): The name or path of the pre-trained encoder model.
            Defaults to "apple/MobileCLIP2-S2".
        decoder_name (str, optional): The name or path of the pre-trained decoder model.
            Defaults to "jhu-clsp/mmBERT-base".
        max_len (int, optional): The maximum length for the generated text sequences.
            Defaults to 300.
        num_decoder_layers (int, optional): The number of layers to use in the decoder.
            If None, the full decoder is used. Defaults to 3.
        batch_size (int, optional): The batch size for training and evaluation.
            Defaults to 64.
        num_epochs (int, optional): The total number of training epochs.
            Defaults to 8.
        fp16 (bool, optional): Whether to use 16-bit floating-point precision (mixed precision)
            for training. Defaults to True.
    """

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[0])
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=16,
        output_dir=TRAIN_ROOT,
        logging_steps=10,
        report_to="none",
        save_steps=20000,
        eval_steps=20000,
        num_train_epochs=num_epochs,
        run_name=run_name,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.feature_extractor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(run)
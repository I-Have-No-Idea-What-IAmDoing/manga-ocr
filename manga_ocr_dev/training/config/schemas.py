"""Pydantic schemas for type-safe training configuration.

This module defines the data structures for the training configuration using
Pydantic models. This ensures that the configuration loaded from the YAML file
is type-safe and validated, preventing common errors and making the
configuration easier to manage and understand. Each class corresponds to a
specific section of the `config.yaml` file.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class ModelConfig(BaseModel):
    """Configuration for the OCR model architecture."""

    encoder_name: str = Field(..., description="The name or path of the pre-trained vision model to use as the encoder.")
    decoder_name: str = Field(..., description="The name or path of the pre-trained language model to use as the decoder.")
    max_len: int = Field(..., description="The maximum sequence length for the decoder.")
    num_decoder_layers: Optional[int] = Field(None, description="If specified, truncates the decoder to this number of layers.")


class DatasetSourceConfig(BaseModel):
    """Configuration for a single data source (e.g., synthetic or Manga109)."""

    type: str = Field(..., description="The type of the data source, e.g., 'synthetic' or 'manga109'.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the data loading function for this source.")


class DatasetTrainConfig(BaseModel):
    """Configuration for the training data sources."""

    sources: List[DatasetSourceConfig] = Field(..., description="A list of data sources to be included in the training set.")


class DatasetEvalConfig(BaseModel):
    """Configuration for the evaluation data sources."""

    sources: List[DatasetSourceConfig] = Field(..., description="A list of data sources to be included in the evaluation set.")


class AugmentationProbabilities(BaseModel):
    """Probabilities for applying different levels of data augmentation."""

    medium: float = Field(0.8, description="The probability of applying the 'medium' augmentation pipeline.")
    heavy: float = Field(0.02, description="The probability of applying the 'heavy' augmentation pipeline.")


class AugmentationConfig(BaseModel):
    """Configuration for the data augmentation pipelines."""

    medium: Optional[List[Dict[str, Any]]] = Field(None, description="A list of Albumentations transforms for the 'medium' pipeline.")
    heavy: Optional[List[Dict[str, Any]]] = Field(None, description="A list of Albumentations transforms for the 'heavy' pipeline.")
    probabilities: AugmentationProbabilities = Field(default_factory=AugmentationProbabilities, description="Probabilities for applying each augmentation pipeline.")


class DatasetConfig(BaseModel):
    """Configuration for the datasets and data augmentations."""

    augment: bool = Field(..., description="A master switch to enable or disable data augmentation for the training set.")
    train: DatasetTrainConfig = Field(..., description="Configuration for the training dataset.")
    eval: DatasetEvalConfig = Field(..., description="Configuration for the evaluation dataset.")
    augmentations: Optional[AugmentationConfig] = Field(None, description="The augmentation pipelines and their probabilities.")


class TrainingConfig(BaseModel):
    """Configuration for the Hugging Face `Seq2SeqTrainingArguments`."""

    batch_size: int = Field(..., alias="per_device_train_batch_size", description="The batch size for training.")
    num_epochs: int = Field(..., alias="num_train_epochs", description="The total number of training epochs.")
    fp16: bool = Field(False, description="Whether to use 16-bit (mixed) precision training.")
    predict_with_generate: bool = Field(True, description="Whether to use `generate` to calculate generative metrics.")
    eval_strategy: str = Field("steps", alias="evaluation_strategy", description="The evaluation strategy to adopt during training.")
    save_strategy: str = Field("steps", alias="save_strategy", description="The checkpoint save strategy to adopt during training.")
    dataloader_num_workers: int = Field(..., alias="dataloader_num_workers", description="The number of workers for the dataloader.")
    logging_steps: int = Field(..., description="Log every `logging_steps` updates.")
    report_to: str = Field("wandb", description="The integration to report results and logs to.")
    save_steps: int = Field(..., description="Save a checkpoint every `save_steps` updates.")
    eval_steps: int = Field(..., description="Run an evaluation every `eval_steps` updates.")
    load_best_model_at_end: bool = Field(True, description="Whether to load the best model found during training at the end.")
    save_total_limit: int = Field(2, description="The total number of checkpoints to keep.")
    include_inputs_for_metrics: bool = Field(True, description="Whether to pass the inputs to `compute_metrics`.")

    model_config = ConfigDict(populate_by_name=True)


class AppConfig(BaseModel):
    """The root configuration object for the entire training application."""

    run_name: str = Field(..., description="A unique name for the training run, used for logging and output directories.")
    model: ModelConfig = Field(..., description="The model configuration.")
    dataset: DatasetConfig = Field(..., description="The dataset configuration.")
    training: TrainingConfig = Field(..., description="The training arguments.")
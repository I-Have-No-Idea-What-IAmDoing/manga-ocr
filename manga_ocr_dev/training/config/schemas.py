"""Pydantic schemas for type-safe training configuration.

This module defines the data structures for the training configuration using
Pydantic models. This ensures that the configuration loaded from the YAML file
is type-safe and validated, preventing common errors and making the
configuration easier to manage and understand. Each class corresponds to a
specific section of the `config.yaml` file.
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


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


class TrainingConfig(BaseSettings):
    """Configuration for the Hugging Face `Seq2SeqTrainingArguments`.

    This class inherits from `pydantic_settings.BaseSettings`, which allows it
    to be configured not only from the YAML file but also from environment
    variables. This is useful for settings that might change between different
    environments, such as the number of dataloader workers.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="MANGA_OCR_TRAINING_",
        populate_by_name=True,
    )

    batch_size: int = Field(64, alias="per_device_train_batch_size", description="The batch size for training.")
    num_epochs: int = Field(8, alias="num_train_epochs", description="The total number of training epochs.")
    fp16: bool = Field(True, description="Whether to use 16-bit (mixed) precision training.")
    predict_with_generate: bool = Field(True, description="Whether to use `generate` to calculate generative metrics.")
    eval_strategy: str = Field("steps", alias="evaluation_strategy", description="The evaluation strategy to adopt during training.")
    save_strategy: str = Field("steps", alias="save_strategy", description="The checkpoint save strategy to adopt during training.")
    dataloader_num_workers: int = Field(16, description="The number of workers for the dataloader.")
    logging_steps: int = Field(10, description="Log every `logging_steps` updates.")
    report_to: str = Field("wandb", description="The integration to report results and logs to.")
    save_steps: int = Field(20000, description="Save a checkpoint every `save_steps` updates.")
    eval_steps: int = Field(20000, description="Run an evaluation every `eval_steps` updates.")
    load_best_model_at_end: bool = Field(True, description="Whether to load the best model found during training at the end.")
    save_total_limit: int = Field(3, description="The total number of checkpoints to keep.")
    include_inputs_for_metrics: bool = Field(True, description="Whether to pass the inputs to `compute_metrics`.")


class AppConfig(BaseSettings):
    """The root configuration object for the entire training application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: ModelConfig = Field(..., description="The model configuration.")
    dataset: DatasetConfig = Field(..., description="The dataset configuration.")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="The training arguments.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Define the priority of configuration sources.
        """
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
import pytest
from pydantic import ValidationError

from manga_ocr_dev.training.config.schemas import (
    AppConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    DatasetSourceConfig,
    DatasetTrainConfig,
    DatasetEvalConfig,
    AugmentationConfig,
    AugmentationProbabilities,
)


def test_model_config():
    """Tests that a valid ModelConfig can be created."""
    config = {
        "encoder_name": "google/vit-base-patch16-224-in21k",
        "decoder_name": "cl-tohoku/bert-base-japanese-char-v2",
        "max_len": 512,
        "num_decoder_layers": 3,
    }
    model_config = ModelConfig(**config)
    assert model_config.encoder_name == config["encoder_name"]
    assert model_config.decoder_name == config["decoder_name"]
    assert model_config.max_len == config["max_len"]
    assert model_config.num_decoder_layers == config["num_decoder_layers"]


def test_model_config_missing_fields():
    """Tests that ModelConfig raises a validation error for missing required fields."""
    with pytest.raises(ValidationError):
        ModelConfig(encoder_name="encoder")  # Missing decoder_name and max_len


def test_dataset_source_config():
    """Tests that a valid DatasetSourceConfig can be created."""
    config = {"type": "synthetic", "params": {"num_samples": 1000}}
    source_config = DatasetSourceConfig(**config)
    assert source_config.type == "synthetic"
    assert source_config.params["num_samples"] == 1000


def test_dataset_train_config():
    """Tests that a valid DatasetTrainConfig can be created."""
    config = {"sources": [{"type": "synthetic"}]}
    train_config = DatasetTrainConfig(**config)
    assert len(train_config.sources) == 1
    assert train_config.sources[0].type == "synthetic"


def test_dataset_eval_config():
    """Tests that a valid DatasetEvalConfig can be created."""
    config = {"sources": [{"type": "manga109"}]}
    eval_config = DatasetEvalConfig(**config)
    assert len(eval_config.sources) == 1
    assert eval_config.sources[0].type == "manga109"


def test_augmentation_probabilities_defaults():
    """Tests that AugmentationProbabilities has correct default values."""
    probs = AugmentationProbabilities()
    assert probs.medium == 0.8
    assert probs.heavy == 0.02


def test_augmentation_config():
    """Tests that a valid AugmentationConfig can be created."""
    config = {
        "medium": [{"name": "Blur"}],
        "heavy": [{"name": "RandomRain"}],
        "probabilities": {"medium": 0.9, "heavy": 0.1},
    }
    aug_config = AugmentationConfig(**config)
    assert aug_config.medium[0]["name"] == "Blur"
    assert aug_config.heavy[0]["name"] == "RandomRain"
    assert aug_config.probabilities.medium == 0.9
    assert aug_config.probabilities.heavy == 0.1


def test_dataset_config():
    """Tests that a valid DatasetConfig can be created."""
    config = {
        "augment": True,
        "train": {"sources": [{"type": "synthetic"}]},
        "eval": {"sources": [{"type": "manga109"}]},
        "augmentations": {"medium": [{"name": "Blur"}]},
    }
    dataset_config = DatasetConfig(**config)
    assert dataset_config.augment is True
    assert dataset_config.train.sources[0].type == "synthetic"
    assert dataset_config.eval.sources[0].type == "manga109"
    assert dataset_config.augmentations.medium[0]["name"] == "Blur"


def test_training_config_aliases():
    """Tests that TrainingConfig correctly uses aliases."""
    config = {
        "per_device_train_batch_size": 8,
        "num_train_epochs": 3,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "dataloader_num_workers": 4,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
    }
    training_config = TrainingConfig(**config)
    assert training_config.batch_size == 8
    assert training_config.num_epochs == 3
    assert training_config.eval_strategy == "epoch"
    assert training_config.save_strategy == "epoch"


def test_training_config_population_by_field_name():
    """Tests that TrainingConfig can be populated by field name."""
    config = {
        "batch_size": 8,
        "num_epochs": 3,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "dataloader_num_workers": 4,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
    }
    training_config = TrainingConfig(**config)
    assert training_config.batch_size == 8
    assert training_config.num_epochs == 3


def test_app_config_valid():
    """Tests that a full, valid AppConfig can be created."""
    config = {
        "run_name": "test_run",
        "model": {
            "encoder_name": "encoder",
            "decoder_name": "decoder",
            "max_len": 128,
        },
        "dataset": {
            "augment": False,
            "train": {"sources": [{"type": "synthetic"}]},
            "eval": {"sources": [{"type": "manga109"}]},
        },
        "training": {
            "per_device_train_batch_size": 8,
            "num_train_epochs": 1,
            "dataloader_num_workers": 2,
            "logging_steps": 5,
            "save_steps": 50,
            "eval_steps": 50,
        },
    }
    app_config = AppConfig(**config)
    assert app_config.run_name == "test_run"
    assert app_config.model.encoder_name == "encoder"
    assert app_config.training.batch_size == 8


def test_app_config_invalid():
    """Tests that AppConfig raises a validation error for invalid data."""
    config = {
        "run_name": "test_run",
        "model": {"encoder_name": "encoder"},  # Missing fields
        "dataset": {"augment": False, "train": {"sources": []}, "eval": {"sources": []}},
        "training": {"per_device_train_batch_size": 8},
    }
    with pytest.raises(ValidationError):
        AppConfig(**config)
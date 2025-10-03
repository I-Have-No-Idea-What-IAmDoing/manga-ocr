from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ModelConfig(BaseModel):
    encoder_name: str
    decoder_name: str
    max_len: int
    num_decoder_layers: Optional[int] = None

class DatasetSourceConfig(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class DatasetTrainConfig(BaseModel):
    sources: List[DatasetSourceConfig]

class DatasetEvalConfig(BaseModel):
    sources: List[DatasetSourceConfig]

class AugmentationProbabilities(BaseModel):
    medium: float = 0.8
    heavy: float = 0.02

class AugmentationConfig(BaseModel):
    medium: Optional[List[Dict[str, Any]]] = None
    heavy: Optional[List[Dict[str, Any]]] = None
    probabilities: AugmentationProbabilities = Field(default_factory=AugmentationProbabilities)

class DatasetConfig(BaseModel):
    augment: bool
    train: DatasetTrainConfig
    eval: DatasetEvalConfig
    augmentations: Optional[AugmentationConfig] = None

class TrainingConfig(BaseModel):
    batch_size: int
    num_epochs: int
    fp16: bool
    predict_with_generate: bool
    eval_strategy: str
    save_strategy: str
    dataloader_num_workers: int
    logging_steps: int
    report_to: str
    save_steps: int
    eval_steps: int
    load_best_model_at_end: bool
    save_total_limit: int
    include_inputs_for_metrics: bool

class AppConfig(BaseModel):
    run_name: str
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSettings:
    name: str
    language: str
    train_path: str
    validation_path: str | None
    test_path: str


@dataclass(frozen=True)
class ModelSettings:
    backbone: str
    fusion_backbone: str
    total_layers: int
    expert_layers: int
    num_experts: int
    top_k: int
    soft_prompt_length: int
    trust_remote_code: bool

    def validate_paper_defaults(self) -> None:
        values = (self.total_layers, self.expert_layers, self.num_experts, self.top_k)
        if values != (40, 5, 4, 2):
            raise ValueError("Paper settings require 40 layers, 5 expert layers, 4 experts, and top-2 routing")


@dataclass(frozen=True)
class EngineeringSettings:
    max_roles: int
    max_source_tokens: int
    max_target_tokens: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation: int
    epochs: int
    seed: int
    dtype: str
    max_new_tokens: int
    num_beams: int
    length_penalty: float


@dataclass(frozen=True)
class ExperimentSettings:
    dataset: DatasetSettings
    model: ModelSettings
    engineering: EngineeringSettings


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config field: {key}")
    return mapping[key]


def load_settings(path: str | Path) -> ExperimentSettings:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    dataset_raw = _require(raw, "dataset")
    model_raw = _require(raw, "model")
    engineering_raw = _require(raw, "engineering")
    dataset = DatasetSettings(
        name=str(_require(dataset_raw, "name")).lower(),
        language=str(_require(dataset_raw, "language")),
        train_path=str(_require(dataset_raw, "train_path")),
        validation_path=dataset_raw.get("validation_path"),
        test_path=str(_require(dataset_raw, "test_path")),
    )
    model = ModelSettings(
        backbone=str(_require(model_raw, "backbone")),
        fusion_backbone=str(_require(model_raw, "fusion_backbone")),
        total_layers=int(_require(model_raw, "total_layers")),
        expert_layers=int(_require(model_raw, "expert_layers")),
        num_experts=int(_require(model_raw, "num_experts")),
        top_k=int(_require(model_raw, "top_k")),
        soft_prompt_length=int(_require(model_raw, "soft_prompt_length")),
        trust_remote_code=bool(model_raw.get("trust_remote_code", False)),
    )
    model.validate_paper_defaults()
    engineering = EngineeringSettings(
        max_roles=int(_require(engineering_raw, "max_roles")),
        max_source_tokens=int(_require(engineering_raw, "max_source_tokens")),
        max_target_tokens=int(_require(engineering_raw, "max_target_tokens")),
        learning_rate=float(_require(engineering_raw, "learning_rate")),
        weight_decay=float(_require(engineering_raw, "weight_decay")),
        batch_size=int(_require(engineering_raw, "batch_size")),
        gradient_accumulation=int(_require(engineering_raw, "gradient_accumulation")),
        epochs=int(_require(engineering_raw, "epochs")),
        seed=int(_require(engineering_raw, "seed")),
        dtype=str(_require(engineering_raw, "dtype")),
        max_new_tokens=int(_require(engineering_raw, "max_new_tokens")),
        num_beams=int(_require(engineering_raw, "num_beams")),
        length_penalty=float(_require(engineering_raw, "length_penalty")),
    )
    if model.top_k > model.num_experts:
        raise ValueError("top_k exceeds num_experts")
    if model.expert_layers >= model.total_layers:
        raise ValueError("expert_layers must be smaller than total_layers")
    if engineering.max_roles < 1:
        raise ValueError("max_roles must be positive")
    for value in (engineering.batch_size, engineering.gradient_accumulation, engineering.epochs, engineering.max_new_tokens, engineering.num_beams, model.soft_prompt_length):
        if value < 1:
            raise ValueError("batch, accumulation, epoch, generation and prompt sizes must be positive")
    if engineering.max_source_tokens < 3 or engineering.max_target_tokens < 2:
        raise ValueError("sequence lengths must allow content and special tokens")
    return ExperimentSettings(dataset, model, engineering)

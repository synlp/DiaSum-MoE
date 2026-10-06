import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import ExperimentSettings
from .data import DialogueCollator, DialogueDataset, load_dataset
from .model import DiaSumMoE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[name]


def dataset_for_split(settings: ExperimentSettings, split: str, override_path: str | None = None) -> DialogueDataset:
    paths = {"train": settings.dataset.train_path, "validation": settings.dataset.validation_path, "test": settings.dataset.test_path}
    if split not in paths:
        raise ValueError(f"Unsupported split: {split}")
    path = override_path or paths[split]
    if path is None:
        raise ValueError(f"No path configured for split: {split}")
    return load_dataset(settings.dataset.name, path)


def build_tokenizer_and_collator(settings: ExperimentSettings, dataset: DialogueDataset) -> tuple[Any, DialogueCollator]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model.backbone,
        trust_remote_code=settings.model.trust_remote_code,
        use_fast=False,
    )
    collator = DialogueCollator(
        tokenizer,
        settings.engineering.max_roles,
        settings.engineering.max_source_tokens,
        settings.engineering.max_target_tokens,
    )
    return tokenizer, collator


def build_model(settings: ExperimentSettings, tokenizer: Any, checkpoint: str | None = None) -> DiaSumMoE:
    model = DiaSumMoE.from_pretrained(settings.model, len(tokenizer), torch_dtype(settings.engineering.dtype))
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


def build_loader(dataset: DialogueDataset, collator: DialogueCollator, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

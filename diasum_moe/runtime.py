import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DialogueCollator, load_dataset
from .model import DiaSumMoE


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name):
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype: {name}")
    return values[name]


def dataset_for_split(settings, split, override_path=None):
    paths = {"train": settings.dataset.train_path, "validation": settings.dataset.validation_path, "test": settings.dataset.test_path}
    if split not in paths:
        raise ValueError(f"Unsupported split: {split}")
    path = override_path or paths[split]
    if path is None:
        raise ValueError(f"No path configured for split: {split}")
    return load_dataset(settings.dataset.name, path)


def build_tokenizer_and_collator(settings, dataset):
    from transformers import AutoTokenizer

    # Use the original LLaMA SentencePiece tokenizer.
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


def build_model(settings, tokenizer, checkpoint=None):
    model = DiaSumMoE.from_pretrained(settings.model, len(tokenizer), torch_dtype(settings.engineering.dtype))
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Checkpoints contain a state dict rather than a pickled model object.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model


def build_loader(dataset, collator, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

import argparse
import json
from pathlib import Path

import torch

from diasum_moe.config import load_settings
from diasum_moe.runtime import build_loader, build_model, build_tokenizer_and_collator, dataset_for_split, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    set_seed(settings.engineering.seed)
    dataset = dataset_for_split(settings, "train", args.train_path)
    tokenizer, collator = build_tokenizer_and_collator(settings, dataset)
    model = build_model(settings, tokenizer)
    device = torch.device(args.device)
    model.to(device)
    loader = build_loader(dataset, collator, settings.engineering.batch_size, True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.engineering.learning_rate,
        weight_decay=settings.engineering.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulation = settings.engineering.gradient_accumulation
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(settings.engineering.epochs):
        model.train()
        total_loss = 0.0
        updates = 0
        for step, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            output = model(batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise FloatingPointError("Non-finite training loss")
            window_start = ((step - 1) // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            (output.loss / window_size).backward()
            total_loss += float(output.loss.detach())
            if step % accumulation == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        checkpoint = output_dir / f"epoch-{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint)
        report = {"epoch": epoch + 1, "loss": total_loss / len(loader), "updates": updates, "checkpoint": str(checkpoint)}
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    train()

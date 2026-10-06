import argparse
import json
from pathlib import Path

import torch

from diasum_moe.config import load_settings
from diasum_moe.runtime import build_loader, build_model, build_tokenizer_and_collator, dataset_for_split, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def generate():
    args = parse_args()
    settings = load_settings(args.config)
    set_seed(settings.engineering.seed)
    dataset = dataset_for_split(settings, args.split, args.data_path)
    tokenizer, collator = build_tokenizer_and_collator(settings, dataset)
    model = build_model(settings, tokenizer, args.checkpoint)
    device = torch.device(args.device)
    model.to(device)
    # Turn off dropout before decoding summaries.
    model.eval()
    loader = build_loader(dataset, collator, 1, False)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for example, batch in zip(dataset.examples, loader):
            batch = batch.to(device)
            sequences, routing, selected = model.generate(
                batch,
                settings.engineering.max_new_tokens,
                settings.engineering.num_beams,
                settings.engineering.length_penalty,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            prediction = tokenizer.decode(sequences[0], skip_special_tokens=True).strip()
            record = {
                "id": example.example_id,
                "prediction": prediction,
                "reference": example.summary,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(destination), "examples": len(dataset)}, ensure_ascii=False))


if __name__ == "__main__":
    generate()

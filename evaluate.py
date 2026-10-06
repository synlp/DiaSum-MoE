import argparse
import json
from pathlib import Path

from diasum_moe.metrics import bertscore_metric, moverscore_metric, overlap_metrics


def read_jsonl(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        raise ValueError(f"Evaluation file is empty: {source}")
    return records


def evaluate_file(path, language, bertscore, moverscore, bert_model):
    records = read_jsonl(path)
    predictions = [str(record["prediction"]) for record in records]
    references = [str(record["reference"]) for record in records]
    metrics = overlap_metrics(predictions, references, language)
    if bertscore:
        metrics["bertscore"] = bertscore_metric(predictions, references, language, bert_model)
    if moverscore:
        metrics["moverscore"] = moverscore_metric(predictions, references)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model")
    parser.add_argument("--moverscore", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    result = evaluate_file(args.predictions, args.language, args.bertscore, args.moverscore, args.bertscore_model)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    evaluate()

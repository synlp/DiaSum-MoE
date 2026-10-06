import argparse
import json

from diasum_moe.data import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dialogsum",), required=True)
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def prepare():
    args = parse_args()
    dataset = load_dataset(args.dataset, args.input)
    result = {"dataset": args.dataset, "examples": len(dataset), "speakers": dataset.speakers}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    prepare()

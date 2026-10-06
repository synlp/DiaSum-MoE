# Dialogue Summarization with Mixture of Experts based on Large Language Models

This repository contains the implementation of [Dialogue Summarization with Mixture of Experts based on Large Language Models](https://aclanthology.org/2024.acl-long.385/), accepted at ACL 2024.

## Requirements

Python 3.10 or later and PyTorch are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Obtain access to [LLaMA-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) under Meta's model license and authenticate Hugging Face before running. The routing backbone and fusion generator load this checkpoint independently. The configuration uses 35 shared layers, four five-layer experts, and hard top-2 selection per utterance.

## Data

The included configuration runs DialogSum with LLaMA-2-13B. Dataset files remain in local data directories.

- **DialogSum:** obtain `DialogSum_Data/dialogsum.train.jsonl`, `dialogsum.dev.jsonl`, and `dialogsum.test.jsonl` from the [official repository](https://github.com/cylnlp/dialogsum). Place them at `data/dialogsum/train.jsonl`, `validation.jsonl`, and `test.jsonl`, respectively, or change the three paths in `configs/dialogsum.yaml`. The loader reads `fname`, speaker-prefixed `dialogue`, and `summary`; official test records with `summary1`, `summary2`, and `summary3` produce three dialogue-reference pairs. Within-utterance line breaks are preserved. Use the original train/dev/test splits.
- **SAMSum:** obtain and unpack `corpus.7z` from the ancillary files of the [original dataset publication](https://arxiv.org/abs/1911.12237). Keep its train, validation, and test JSON files and their `id`, `dialogue`, and `summary` fields. These records use the included English loader schema; set their paths in a copy of the DialogSum configuration.
- **CSDS:** obtain the train/validation/test JSON files through the [official repository](https://github.com/xiaolinAndy/CSDS). Its `Dialogue` field contains speaker and utterance records, and `FinalSumm` contains the overall summary. Prepare speaker-prefixed dialogue text and concatenate the overall-summary segments when converting these files to dialogue-summary records. Preserve the published splits.
- **MC:** follow the dataset access instructions in the [original HET-MC repository](https://github.com/cuhksz-nlp/HET-MC). Preserve its train/test partition, patient/doctor roles, and both summary sections; combine SUM1 and SUM2 for the entire-dialogue target. Use a deterministic 10% subset of the training records for hyperparameter selection, then train the final setting on all training records.

The Chinese CSDS and MC experiments in the paper use Ziya2-13B. Their acquisition and conversion instructions describe the paper datasets; the included run configuration selects the English LLaMA-2 setting.

## Run

```bash
python prepare.py --dataset dialogsum --input data/dialogsum/train.jsonl
python train.py --config configs/dialogsum.yaml --output-dir checkpoints/dialogsum
python generate.py --config configs/dialogsum.yaml --checkpoint checkpoints/dialogsum/epoch-3.pt --split test --output predictions/dialogsum.jsonl
python evaluate.py --predictions predictions/dialogsum.jsonl --language en --output results/dialogsum.json
```

Set sequence limits, optimizer settings, training duration, and decoding options in the configuration. A role marker follows each utterance in the causal routing backbone; the same contextual role state leads that utterance's expert input. An identity straight-through gradient trains the hard router while keeping selected expert states unchanged in forward computation.

Evaluation reports ROUGE-1, ROUGE-2, ROUGE-L, and BLEU. Install `.[contextual-metrics]` and pass `--bertscore` for BERTScore. `--moverscore` uses an installed `moverscore_v2` package and its model resources.

## Structure

```text
DiaSum-MoE/
├── README.md
├── pyproject.toml
├── configs/
│   └── dialogsum.yaml
├── diasum_moe/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   └── runtime.py
├── prepare.py
├── train.py
├── generate.py
└── evaluate.py
```

`diasum_moe/model.py` implements routing, experts, and fusion. The other modules in `diasum_moe/` handle dialogue data, settings, runtime setup, and metrics. `configs/dialogsum.yaml` contains the run configuration. Run the four root-level entry scripts directly.

## Citation

If you use this code, please cite the paper.

```bibtex
@inproceedings{tian2024dialogue,
  title={Dialogue Summarization with Mixture of Experts based on Large Language Models},
  author={Tian, Yuanhe and Xia, Fei and Song, Yan},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7143--7155},
  year={2024}
}
```

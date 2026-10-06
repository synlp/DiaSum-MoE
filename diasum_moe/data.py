import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Utterance:
    speaker: str
    text: str


@dataclass(frozen=True)
class DialogueExample:
    example_id: str
    utterances: tuple[Utterance, ...]
    summary: str
    role_summaries: dict[str, str]


@dataclass
class DialogueBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    utterance_spans: list[list[tuple[int, int]]]
    routing_positions: list[list[int]]
    example_ids: list[str]

    def to(self, device):
        return DialogueBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            target_ids=self.target_ids.to(device),
            target_attention_mask=self.target_attention_mask.to(device),
            utterance_spans=self.utterance_spans,
            routing_positions=self.routing_positions,
            example_ids=self.example_ids,
        )


def _load_records(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        if source.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        else:
            records = json.load(stream)
    if isinstance(records, dict):
        for key in ("data", "records", "examples"):
            if isinstance(records.get(key), list):
                records = records[key]
                break
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Dataset must contain a list of records: {source}")
    return records


def _required_text(record, keys, label):
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Missing {label}; expected one of {list(keys)}")


def _prefixed_dialogue(text):
    utterances = []
    pattern = re.compile(r"^\s*([^:：\n]{1,80})\s*[:：]\s*(.+?)\s*$")
    # DialogSum role markers distinguish new turns from wrapped lines.
    if text.lstrip().startswith("#Person"):
        pattern = re.compile(r"^\s*(#Person\d+#)\s*[:：]\s*(.+?)\s*$")
    for line in text.splitlines():
        if not line.strip():
            continue
        match = pattern.match(line)
        if not match:
            if not utterances:
                raise ValueError(f"Utterance lacks speaker prefix: {line[:80]}")
            previous = utterances[-1]
            utterances[-1] = Utterance(previous.speaker, previous.text + "\n" + line.strip())
            continue
        utterances.append(Utterance(match.group(1).strip(), match.group(2).strip()))
    if not utterances:
        raise ValueError("Dialogue has no utterances")
    return tuple(utterances)


def _record_id(record, index):
    for key in ("id", "fname", "DialogueID", "dialogue_id", "case_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return str(index)


class DialogueDataset(Dataset[DialogueExample]):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @property
    def speakers(self):
        return tuple(sorted({utterance.speaker for example in self.examples for utterance in example.utterances}))


def load_dialogsum(path):
    examples = []
    for index, record in enumerate(_load_records(path)):
        dialogue = _required_text(record, ("dialogue",), "dialogue")
        utterances = _prefixed_dialogue(dialogue)
        example_id = _record_id(record, index)
        if isinstance(record.get("summary"), str) and record["summary"].strip():
            examples.append(DialogueExample(example_id, utterances, record["summary"].strip(), {}))
        else:
            # The official test split supplies three summaries per dialogue.
            for reference_index in (1, 2, 3):
                summary = _required_text(record, (f"summary{reference_index}",), "summary")
                examples.append(DialogueExample(f"{example_id}_ref{reference_index}", utterances, summary, {}))
    return DialogueDataset(examples)


def load_dataset(name, path):
    if name.lower() != "dialogsum":
        raise ValueError("dataset.name must be dialogsum for this configuration")
    return load_dialogsum(path)


class DialogueCollator:
    def __init__(self, tokenizer, max_roles, max_source_tokens, max_target_tokens):
        self.tokenizer = tokenizer
        self.role_tokens = tuple(f"<|role_{index}|>" for index in range(max_roles))
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.role_tokens)})
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def _encode_dialogue(self, example):
        input_ids = []
        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
        spans = []
        routing_positions = []
        # Assign role tokens by first appearance in this dialogue.
        local_roles = {}
        for utterance in example.utterances:
            if utterance.speaker not in local_roles:
                if len(local_roles) == len(self.role_tokens):
                    raise ValueError(f"Dialogue exceeds max_roles: {example.example_id}")
                local_roles[utterance.speaker] = len(local_roles)
            role_id = self.tokenizer.convert_tokens_to_ids(self.role_tokens[local_roles[utterance.speaker]])
            text_ids = self.tokenizer.encode(f"{utterance.speaker}: {utterance.text}", add_special_tokens=False)
            available = self.max_source_tokens - len(input_ids)
            if available < 2:
                break
            content_ids = text_ids[: available - 1]
            # A trailing role token can attend to the complete utterance.
            utterance_ids = content_ids + [role_id]
            if self.tokenizer.eos_token_id is not None and len(utterance_ids) < available:
                utterance_ids.append(self.tokenizer.eos_token_id)
            start = len(input_ids)
            input_ids.extend(utterance_ids)
            # The expert span includes the role token, but excludes EOS.
            expert_end = start + len(content_ids) + 1
            spans.append((start, expert_end))
            routing_positions.append(expert_end - 1)
        if not spans:
            raise ValueError(f"No utterance fits max_source_tokens for {example.example_id}")
        return input_ids, spans, routing_positions

    def __call__(self, examples):
        if not examples:
            raise ValueError("Empty batch")
        encoded = [self._encode_dialogue(example) for example in examples]
        source_length = max(len(item[0]) for item in encoded)
        target_rows = []
        for example in examples:
            # Reserve the last target position for EOS.
            target = self.tokenizer.encode(example.summary, add_special_tokens=False)[: self.max_target_tokens - 1]
            if self.tokenizer.eos_token_id is not None:
                target.append(self.tokenizer.eos_token_id)
            target_rows.append(target)
        target_length = max(len(row) for row in target_rows)
        input_ids = torch.full((len(examples), source_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), source_length), dtype=torch.long)
        target_ids = torch.full((len(examples), target_length), self.tokenizer.pad_token_id, dtype=torch.long)
        target_attention_mask = torch.zeros((len(examples), target_length), dtype=torch.long)
        for index, (source, _, _) in enumerate(encoded):
            input_ids[index, : len(source)] = torch.tensor(source)
            attention_mask[index, : len(source)] = 1
            target_ids[index, : len(target_rows[index])] = torch.tensor(target_rows[index])
            target_attention_mask[index, : len(target_rows[index])] = 1
        return DialogueBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_attention_mask,
            utterance_spans=[item[1] for item in encoded],
            routing_positions=[item[2] for item in encoded],
            example_ids=[example.example_id for example in examples],
        )

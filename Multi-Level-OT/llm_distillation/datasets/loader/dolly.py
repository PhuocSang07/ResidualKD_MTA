import os
import json
import sys
from pathlib import Path
from datasets import Dataset

_REPO_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Default: dolly jsonl nằm trong data/dolly/ của repo, có thể override bằng env var DOLLY_DATA_DIR
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "dolly")
DOLLY_DATA_DIR = os.environ.get("DOLLY_DATA_DIR", _DEFAULT_DATA_DIR)

_SPLIT_FILE = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
}


def tokenize(item, tokenizer, encoder_decoder=False):
    prompt = item.get("prompt", "")
    output = item.get("output", "")

    if not encoder_decoder:
        context_tokens = tokenizer.encode(
            f"{tokenizer.bos_token}{prompt}" if tokenizer.bos_token else prompt,
            add_special_tokens=False,
        )
        answer_tokens = tokenizer.encode(
            f" {output}{tokenizer.eos_token}" if tokenizer.eos_token else f" {output}",
            add_special_tokens=False,
        )
        prompt_tokens = context_tokens + answer_tokens
        labels_tokens = (len(context_tokens) * [-100]) + answer_tokens
        combined = {"input_ids": prompt_tokens, "labels": labels_tokens}
        return dict(combined, attention_mask=[1] * len(prompt_tokens))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(output, add_special_tokens=True, return_tensors="pt")[0]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def get_split(dataset_config, tokenizer, split):
    file_name = _SPLIT_FILE.get(split)
    if file_name is None:
        raise ValueError(f"Unknown split '{split}'. Expected 'train' or 'validation'.")

    data_dir = getattr(dataset_config, "data_dir", None) or DOLLY_DATA_DIR
    file_path = os.path.join(data_dir, file_name)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Dolly {split} file not found: {file_path}\n"
            f"Set env var DOLLY_DATA_DIR to the folder containing train.jsonl / valid.jsonl."
        )

    with open(file_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if dataset_config.training_size < 1:
        records = records[: int(len(records) * dataset_config.training_size)]

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder),
        remove_columns=list(dataset.features),
    )
    return dataset

import os
import sys
from pathlib import Path
from datasets import load_from_disk
from datasets import __file__ as datasets_file
print("datasets package path:", datasets_file)

# Add Multi-Level-OT/ to sys.path so `from llm_distillation.prompt...` resolves correctly
_REPO_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from llm_distillation.prompt.prompt import create_chat_prompt
from llm_distillation.prompt.prompt import create_prompt

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "summary_dialogue"

    if tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Llama-2-7b-chat-hf":
        shot = 3
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/mistral-7B-v0.1":
        shot = 2
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3":
        shot = 2
    elif tokenizer.name_or_path == f"{os.getenv('HOME')}/tiiuae/falcon-7b-instruct":
        shot = 2

    dialogue = item.get('dialogue', item.get('context', ''))
    summary = item.get('summary', item.get('summary_generated', ''))

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            context = dialogue,
            sys_user = True if f"{os.getenv('HOME')}/models/Mistral-7B-Instruct-v0.3" or "/opt/data/private/models/mistral-7B-v0.1" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0,
            context = dialogue,
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {summary}", add_special_tokens=False)
            else:
                answer_tokens = tokenizer.encode(f"{summary}", add_special_tokens=False)
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {summary}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens+answer_tokens
        labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    else:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(summary, add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1]*len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    import pathlib
    _processed_dir = pathlib.Path(__file__).parent.parent / "processed" / "dialogsum"
    dataset = load_from_disk(str(_processed_dir))
    dataset = dataset[split]
    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    return dataset